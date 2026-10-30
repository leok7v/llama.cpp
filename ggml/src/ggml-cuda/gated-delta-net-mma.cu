#include "gated-delta-net-mma.cuh"
#include "mma.cuh"

#include <climits>
#include <mutex>

namespace ggml_cuda_gated_delta_net_mma {
constexpr int D = 128;
using bf16      = nv_bfloat16;

template <int R, int C> struct matrix {
    bf16 hi[R * C], lo[R * C];

    static __device__ __forceinline__ int index(int r, int c) {
#ifdef GGML_USE_HIP
        return r * C + c;
#else
        return 2 * ggml_cuda_mma::swizzle<C / 2, nv_bfloat162>(r * (C / 2) + c / 2, r) + c % 2;
#endif // GGML_USE_HIP
    }

    __device__ __forceinline__ void store2(int r, int c, float2 x) {
#ifdef GGML_USE_MUSA
        store(r, c, x.x);
        store(r, c + 1, x.y);
#else
        const int    i                            = index(r, c);
        const auto   h                            = __float22bfloat162_rn(x);
        const float2 rounded                      = __bfloat1622float2(h);
        *reinterpret_cast<nv_bfloat162 *>(hi + i) = h;
        *reinterpret_cast<nv_bfloat162 *>(lo + i) =
            __float22bfloat162_rn(make_float2(x.x - rounded.x, x.y - rounded.y));
#endif // GGML_USE_MUSA
    }

    __device__ __forceinline__ void store(int r, int c, float x) {
        const int  i = index(r, c);
        const bf16 h = __float2bfloat16(x);
        hi[i]        = h;
        lo[i]        = __float2bfloat16(x - __bfloat162float(h));
    }
};

template <int C, int V> struct alignas(128) shared {
    matrix<C, D> q, k;
    matrix<C, C> p, inverse;
    matrix<C, V> delta;

    union {
        matrix<D, V> state;
        matrix<C, V> solved;
    } scratch;

    float lower[C * C], prefix[C], beta[C], decay[C];

    static __device__ __forceinline__ int lower_index(int r, int c) { return r * C + c; }
};

#if defined(AMPERE_MMA_AVAILABLE) || (defined(AMD_WMMA_AVAILABLE) && defined(RDNA3))
using namespace ggml_cuda_mma;
#ifdef GGML_USE_HIP
using acc    = tile<16, 16, float, DATA_LAYOUT_J_MAJOR>;
using a_tile = tile<16, 8, nv_bfloat162, DATA_LAYOUT_I_MAJOR_MIRRORED>;
using b_tile = a_tile;
#else
using acc    = tile<16, 8, float>;
using a_tile = tile<16, 8, nv_bfloat162>;
using b_tile = tile<8, 8, nv_bfloat162>;
#endif // GGML_USE_HIP

template <class tile_t, bool TRANS, int R, int C>
__device__ __forceinline__ tile_t load(const matrix<R, C> & m, int row, int col, bool low) {
    tile_t       t;
    const bf16 * data = low ? m.lo : m.hi;
#ifdef GGML_USE_HIP
#pragma unroll
    for (int i = 0; i < tile_t::ne; ++i) {
        const int  r = row + tile_t::get_i(i), c = col + 2 * tile_t::get_j(i);
        const bf16 x = data[TRANS ? m.index(c, r) : m.index(r, c)];
        const bf16 y = data[TRANS ? m.index(c + 1, r) : m.index(r, c + 1)];
        t.x[i]       = __float22bfloat162_rn(make_float2(__bfloat162float(x), __bfloat162float(y)));
    }
#else
    const auto * packed = reinterpret_cast<const nv_bfloat162 *>(data);
    if constexpr (TRANS) {
        load_ldmatrix_trans_swizzled<C / 2>(t, packed, col * (C / 2) + row / 2);
    } else {
        load_ldmatrix_swizzled<C / 2>(t, packed, row * (C / 2) + col / 2);
    }
#endif // GGML_USE_HIP
    return t;
}

// B is addressed as B-transpose. Preserve both BF16 first-order residual terms.
template <int K, bool TA = false, bool TB = false, int AR, int AC, int BR, int BC>
__device__ __forceinline__ void product(acc & c, const matrix<AR, AC> & a, const matrix<BR, BC> & b, int row, int col) {
#pragma unroll
    for (int k = 0; k < K; k += 16) {
        const auto ah = load<a_tile, TA>(a, row, k, false);
        const auto al = load<a_tile, TA>(a, row, k, true);
        const auto bh = load<b_tile, TB>(b, col, k, false);
        const auto bl = load<b_tile, TB>(b, col, k, true);
        mma(c, al, bh);
        mma(c, ah, bl);
        mma(c, ah, bh);
    }
}

template <int R, int C> __device__ __forceinline__ void store(matrix<R, C> & m, const acc & x, int r, int c) {
#pragma unroll
#ifdef GGML_USE_HIP
    for (int i = 0; i < acc::ne; ++i) {
        m.store(r + acc::get_i(i), c + acc::get_j(i), x.x[i]);
    }
#else
    for (int i = 0; i < acc::ne; i += 2) {
        m.store2(r + acc::get_i(i), c + acc::get_j(i), make_float2(x.x[i], x.x[i + 1]));
    }
#endif // GGML_USE_HIP
}
#endif // defined(AMPERE_MMA_AVAILABLE) || (defined(AMD_WMMA_AVAILABLE) && defined(RDNA3))

// Each block owns V independent value columns and advances by C tokens.
// Keep recurrent state in FP32; pack high/residual BF16 operands inside the block.
template <int C, int V, int BLOCKS = 1>
static __global__ __launch_bounds__(256, BLOCKS) void gdn_single(ggml_cuda_gdn_mma_args a) {
#if defined(AMPERE_MMA_AVAILABLE) || (defined(AMD_WMMA_AVAILABLE) && defined(RDNA3))
    static_assert(ggml_cuda_get_physical_warp_size() == 32, "GDN MMA requires 32-lane warps");
    constexpr int WARPS = 8;
#ifdef GGML_USE_HIP
    constexpr int N = 16;
#else
    constexpr int N = 8;
#endif // GGML_USE_HIP
    static_assert(C >= 16 && (C & (C - 1)) == 0 && V % N == 0 && D % V == 0, "invalid GDN tile");
    extern __shared__ __align__(128) unsigned char bytes[];
    auto &                                         s    = *reinterpret_cast<shared<C, V> *>(bytes);
    const int                                      lane = threadIdx.x, warp = threadIdx.y, tid = warp * 32 + lane;
    const int                                      slice = blockIdx.x % (D / V);
    const int                                      h     = (blockIdx.x / (D / V)) % a.H;
    const int                                      seq   = blockIdx.x / ((D / V) * a.H);
    const int                                      v0    = slice * V;
    const int64_t                                  soff  = ((int64_t) seq * a.H + h) * D * D;
    const int64_t                                  qoff  = (seq / a.rq3) * a.sq3 + (h % a.H_k) * a.sq1;
    const int64_t                                  voff  = seq * a.sv3 + h * a.sv1 + v0;
    const int64_t                                  goff  = seq * a.sb3 + h * a.sb1;
    float *                                        out   = a.dst + ((int64_t) seq * a.n_tokens * a.H + h) * D + v0;
    constexpr int                                  STATE_TILES = (D / 16) * (V / N);
    constexpr int                                  PER_WARP    = STATE_TILES / WARPS;
    static_assert(STATE_TILES % WARPS == 0, "state must divide across warps");
    acc state[PER_WARP];
#pragma unroll
    for (int j = 0; j < PER_WARP; ++j) {
        const int tile_index = warp + j * WARPS, r = (tile_index / (V / N)) * 16, c = (tile_index % (V / N)) * N;
#pragma unroll
        for (int i = 0; i < acc::ne; ++i) {
            state[j].x[i] = a.state[soff + (v0 + c + acc::get_j(i)) * D + r + acc::get_i(i)];
        }
    }
    auto read_qk = [&](int64_t t, float4 & q, float4 & k) {
        t += warp;
        q = k = make_float4(0, 0, 0, 0);
        if (t < a.n_tokens) {
            const int64_t off = qoff + t * a.sq2 + 4 * lane;
            if ((((uintptr_t) (a.q + off) | (uintptr_t) (a.k + off)) & 15) == 0) {
                q = *reinterpret_cast<const float4 *>(a.q + off);
                k = *reinterpret_cast<const float4 *>(a.k + off);
            } else {
                q = make_float4(a.q[off], a.q[off + 1], a.q[off + 2], a.q[off + 3]);
                k = make_float4(a.k[off], a.k[off + 1], a.k[off + 2], a.k[off + 3]);
            }
        }
    };
    // Carry the first eight input rows and the next chunk's gates across chunks to hide global-load latency: every
    // operand a chunk needs is requested one chunk ahead, so the serial chain never waits on a fresh global load.
    constexpr bool PREFETCH = C == 16;
    float4         next_q, next_k;
    float          next_g = 0.f, next_beta = 0.f;
    if constexpr (PREFETCH) {
        read_qk(0, next_q, next_k);
        if (warp == 0 && lane < a.n_tokens) {
            next_g    = a.g[goff + lane * a.sb2];
            next_beta = a.beta[goff + lane * a.sb2];
        }
    }
    for (int64_t t0 = 0; t0 < a.n_tokens; t0 += C) {
        const int valid = min((int64_t) C, a.n_tokens - t0);
#pragma unroll
        for (int j = 0; j < C / 8; ++j) {
            const int t = warp + 8 * j, d = 4 * lane;
            float4    q, k;
            if (PREFETCH && j == 0) {
                q = next_q;
                k = next_k;
            } else {
                read_qk(t0 + 8 * j, q, k);
            }
            s.q.store2(t, d, make_float2(q.x * a.scale, q.y * a.scale));
            s.q.store2(t, d + 2, make_float2(q.z * a.scale, q.w * a.scale));
            s.k.store2(t, d, make_float2(k.x, k.y));
            s.k.store2(t, d + 2, make_float2(k.z, k.w));
        }
        // The FP32 prefix scan of the gates, and the per-token decay the projections read from a table.
        if (warp == 0) {
            float carry = 0.f;
            for (int base = 0; base < C; base += 32) {
                const int   t    = base + lane;
                float       g, beta;
                if constexpr (PREFETCH) {
                    g    = next_g;
                    beta = next_beta;
                    const int tn = t0 + C + t;
                    next_g    = tn < a.n_tokens ? a.g[goff + tn * a.sb2]    : 0.f;
                    next_beta = tn < a.n_tokens ? a.beta[goff + tn * a.sb2] : 0.f;
                } else {
                    g    = t < valid ? a.g[goff + (t0 + t) * a.sb2] : 0.f;
                    beta = t < valid ? a.beta[goff + (t0 + t) * a.sb2] : 0.f;
                }
#pragma unroll
                for (int offset = 1; offset < 32; offset *= 2) {
                    const float previous = __shfl_up_sync(0xffffffff, g, offset, 32);
                    if (lane >= offset) {
                        g += previous;
                    }
                }
                g += carry;
                if (t < C) {
                    s.prefix[t] = g;
                    s.beta[t]   = beta;
                    s.decay[t]  = expf(g);
                }
                carry = __shfl_sync(0xffffffff, g, 31, 32);
            }
        }
#pragma unroll
        for (int j = 0; j < PER_WARP; ++j) {
            const int ti = warp + j * WARPS;
            store(s.scratch.state, state[j], (ti / (V / N)) * 16, (ti % (V / N)) * N);
        }
        __syncthreads();
        if constexpr (PREFETCH) {
            if (t0 + C < a.n_tokens) {
                read_qk(t0 + C, next_q, next_k);
            }
        }
        // V32 overlaps Gram work with one output tile per projection warp.
        constexpr bool SPLIT_PROJECTION = C == 16 && V == 32;
        acc            base_output[((C / 16) * (V / N) + WARPS - 1) / WARPS];
        // delta = beta * (v - decay * K S), base output = decay * Q S
        // the shared A-tile form needs the projection tiles to divide across the warps; the RDNA3 V=64 form has 4 tiles for 8 warps
        constexpr bool SHARED_A = (C / 16) * (V / N) % WARPS == 0 && (C / 16) * (V / N) >= WARPS;
        auto           project = [&]() {
            if constexpr (V == 32 || !SHARED_A) {
                if constexpr (V == 32) {
                    if (warp < 4) {
                        return;
                    }
                }
                for (int ti = (V == 32 ? warp - 4 : warp); ti < (C / 16) * (V / N); ti += (V == 32 ? 4 : WARPS)) {
                    const int r = (ti / (V / N)) * 16, c = (ti % (V / N)) * N;
                    acc       qs, ks;
                    product<D, false, true>(qs, s.q, s.scratch.state, r, c);
                    product<D, false, true>(ks, s.k, s.scratch.state, r, c);
#pragma unroll
                    for (int i = 0; i < acc::ne; ++i) {
                        const int   t = r + acc::get_i(i), v = c + acc::get_j(i);
                        const float decay = s.decay[t];
                        const float vv    = t < valid ? a.v[voff + (t0 + t) * a.sv2 + v] : 0.f;
                        s.delta.store(t, v, s.beta[t] * (vv - decay * ks.x[i]));
                        base_output[ti / WARPS].x[i] = decay * qs.x[i];
                    }
                }
            } else {
                // The warp's tiles share the one 16-row group (C = 16), so each k-step loads the Q and K A-tiles once for
                // all of them and the tiles' MMA chains advance together; the chunk's v values are requested BEFORE the
                // products so their global latency hides behind the MMA work instead of following it in the epilogue.
                // Per tile the accumulation order is the one above, so the result is unchanged.
                static_assert(C == 16, "the shared A-tile assumes one 16-row group per chunk");
                constexpr int T = (C / 16) * (V / N) / WARPS;
                static_assert(T >= 1 && (C / 16) * (V / N) % WARPS == 0, "projection tiles divide across the warps");
                acc   qs[T], ks[T];
                float vpre[T][acc::ne];
#pragma unroll
                for (int j = 0; j < T; ++j) {
                    const int c = ((warp + j * WARPS) % (V / N)) * N;
#pragma unroll
                    for (int i = 0; i < acc::ne; ++i) {
                        const int t = acc::get_i(i), v = c + acc::get_j(i);
                        vpre[j][i] = t < valid ? a.v[voff + (t0 + t) * a.sv2 + v] : 0.f;
                    }
                }
#pragma unroll
                for (int k = 0; k < D; k += 16) {
                    const auto qh = load<a_tile, false>(s.q, 0, k, false);
                    const auto ql = load<a_tile, false>(s.q, 0, k, true);
                    const auto kh = load<a_tile, false>(s.k, 0, k, false);
                    const auto kl = load<a_tile, false>(s.k, 0, k, true);
#pragma unroll
                    for (int j = 0; j < T; ++j) {
                        const int  c  = ((warp + j * WARPS) % (V / N)) * N;
                        const auto bh = load<b_tile, true>(s.scratch.state, c, k, false);
                        const auto bl = load<b_tile, true>(s.scratch.state, c, k, true);
                        mma(qs[j], ql, bh);
                        mma(qs[j], qh, bl);
                        mma(qs[j], qh, bh);
                        mma(ks[j], kl, bh);
                        mma(ks[j], kh, bl);
                        mma(ks[j], kh, bh);
                    }
                }
#pragma unroll
                for (int j = 0; j < T; ++j) {
                    const int c = ((warp + j * WARPS) % (V / N)) * N;
#pragma unroll
                    for (int i = 0; i < acc::ne; ++i) {
                        const int   t = acc::get_i(i), v = c + acc::get_j(i);
                        const float decay = s.decay[t];
                        s.delta.store(t, v, s.beta[t] * (vpre[j][i] - decay * ks[j].x[i]));
                        base_output[j].x[i] = decay * qs[j].x[i];
                    }
                }
            }
        };
        // A = beta * strict_tril(K K^T * decay), P = tril(Q K^T * decay)
        for (int ti = warp; ti < 2 * (C / 16) * (C / N) && (V != 32 || warp < 4); ti += (V == 32 ? 4 : WARPS)) {
            const bool gram = ti < (C / 16) * (C / N);
            const int  t = ti % ((C / 16) * (C / N)), r = (t / (C / N)) * 16, c = (t % (C / N)) * N;
            acc        x;
            if (gram) {
                product<D>(x, s.k, s.k, r, c);
            } else {
                product<D>(x, s.q, s.k, r, c);
            }
#pragma unroll
            for (int i = 0; i < acc::ne; ++i) {
                const int   rr = r + acc::get_i(i), cc = c + acc::get_j(i);
                const float v = rr < valid && cc <= rr ? x.x[i] * expf(s.prefix[rr] - s.prefix[cc]) : 0.f;
                if (gram) {
                    if (cc < rr) {
                        s.lower[s.lower_index(rr, cc)] = v * s.beta[rr];
                    }
                } else {
                    s.p.store(rr, cc, v);
                }
            }
        }
        if constexpr (V == 32) {
            project();
        }
        __syncthreads();
        // inverse = (I + A)^-1 by forward substitution
        for (int row = warp; row < C; row += WARPS) {
            float x[(C + 31) / 32];
#pragma unroll
            for (int j = 0; j < (C + 31) / 32; ++j) {
                const int col = lane + 32 * j;
                x[j]          = col == row ? 1.f : (col < row ? -s.lower[s.lower_index(row, col)] : 0.f);
            }
            for (int pivot = row - 1; pivot > 0; --pivot) {
                const float xp = __shfl_sync(0xffffffff, x[pivot / 32], pivot % 32, 32);
#pragma unroll
                for (int j = 0; j < (C + 31) / 32; ++j) {
                    const int col = lane + 32 * j;
                    if (col < pivot) {
                        x[j] -= xp * s.lower[s.lower_index(pivot, col)];
                    }
                }
            }
#pragma unroll
            for (int j = 0; j < (C + 31) / 32; ++j) {
                if (lane + 32 * j < C) {
                    s.inverse.store(row, lane + 32 * j, x[j]);
                }
            }
        }
        if constexpr (V != 32) {
            project();
        }
        __syncthreads();
        // solved = inverse * delta
        for (int ti = warp; ti < (C / 16) * (V / N); ti += WARPS) {
            const int r = (ti / (V / N)) * 16, c = (ti % (V / N)) * N;
            acc       x;
            product<C, false, true>(x, s.inverse, s.delta, r, c);
            store(s.scratch.solved, x, r, c);
        }
        __syncthreads();
        // out = base output + P * solved
        for (int ti = (SPLIT_PROJECTION ? warp - 4 : warp); ti < (C / 16) * (V / N) && (!SPLIT_PROJECTION || warp >= 4);
             ti += WARPS) {
            const int r = (ti / (V / N)) * 16, c = (ti % (V / N)) * N;
            acc       x;
            product<C, false, true>(x, s.p, s.scratch.solved, r, c);
#pragma unroll
            for (int i = 0; i < acc::ne; ++i) {
                const int t = r + acc::get_i(i), v = c + acc::get_j(i);
                if (t < valid) {
                    out[(t0 + t) * a.H * D + v] = base_output[ti / WARPS].x[i] + x.x[i];
                }
            }
        }
        // S = decay(chunk) * S + K^T * (solved * decay to chunk end)
        for (int i = tid; i < C * V; i += 256) {
            const int   t = i / V, v = i % V, j = s.scratch.solved.index(t, v);
            const float x = __bfloat162float(s.scratch.solved.hi[j]) + __bfloat162float(s.scratch.solved.lo[j]);
            s.delta.store(t, v, x * expf(s.prefix[valid - 1] - s.prefix[t]));
        }
        __syncthreads();
        const float decay = expf(s.prefix[valid - 1]);
#pragma unroll
        for (int j = 0; j < PER_WARP; ++j) {
            const int ti = warp + j * WARPS, r = (ti / (V / N)) * 16, c = (ti % (V / N)) * N;
#pragma unroll
            for (int i = 0; i < acc::ne; ++i) {
                state[j].x[i] *= decay;
            }
            product<C, true, true>(state[j], s.k, s.delta, r, c);
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < PER_WARP; ++j) {
        const int ti = warp + j * WARPS, r = (ti / (V / N)) * 16, c = (ti % (V / N)) * N;
#pragma unroll
        for (int i = 0; i < acc::ne; ++i) {
            a.state_out[soff + (v0 + c + acc::get_j(i)) * D + r + acc::get_i(i)] = state[j].x[i];
        }
    }
#else
    GGML_UNUSED_VARS(a);
    NO_DEVICE_CODE;
#endif // defined(AMPERE_MMA_AVAILABLE) || (defined(AMD_WMMA_AVAILABLE) && defined(RDNA3))
}

// The chain form of the whole-head path. Consumer warp w of eight owns the state columns [16w, 16w+16) as S^T in
// sixteen accumulator tiles, so every state-dependent product of a chunk runs from registers with no barrier inside
// the chunk. Four producer warps prepare the next chunk's state-independent operands into the other half of a double
// buffer meanwhile. Named barriers hand each buffer over, so neither role waits on the other's chain.
constexpr int CHAIN_CONSUMERS = D / 16;
constexpr int CHAIN_PRODUCERS = 4;
constexpr int CHAIN_BUFS      = 2;
constexpr int CHAIN_THREADS   = 32 * (CHAIN_CONSUMERS + CHAIN_PRODUCERS);

// The v row stride keeps a consumer tile's eight rows and four token columns on distinct banks.
template <int C> struct alignas(128) chain_buf {
    matrix<C, D> q, k;
    matrix<C, C> p, inverse;
    float        v[C][D + 4];
    float        prefix[C], beta[C], decay[C], to_end[C];
    float        gamma;
};

template <int C> struct alignas(128) chain_shared {
    chain_buf<C> buf[CHAIN_BUFS];
    float        lower[C * C];

    static __device__ __forceinline__ int lower_index(int r, int c) { return r * C + c; }
};

#if defined(AMPERE_MMA_AVAILABLE) && !defined(GGML_USE_HIP)
// Barrier 0 stays __syncthreads. FULL[b] and EMPTY[b] pair the producers' arrive with the consumers' sync and the
// reverse, and PROD orders the producer warps among themselves.
constexpr int BAR_FULL = 1, BAR_EMPTY = BAR_FULL + CHAIN_BUFS, BAR_PROD = BAR_EMPTY + CHAIN_BUFS;

__device__ __forceinline__ void bar_sync(int id, int count) {
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(count) : "memory");
}

__device__ __forceinline__ void bar_arrive(int id, int count) {
    asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(count) : "memory");
}

__device__ __forceinline__ void split2(float x, float y, nv_bfloat162 & hi, nv_bfloat162 & lo) {
    const float2 v = make_float2(x, y);
    hi             = __float22bfloat162_rn(v);
    const float2 r = __bfloat1622float2(hi);
    lo             = __float22bfloat162_rn(make_float2(v.x - r.x, v.y - r.y));
}

// Two accumulator tiles over adjacent 8-column groups form one 16-wide A operand, split as matrix::store2 splits.
__device__ __forceinline__ void split_a(const acc & t0, const acc & t1, a_tile & hi, a_tile & lo) {
    split2(t0.x[0], t0.x[1], hi.x[0], lo.x[0]);
    split2(t0.x[2], t0.x[3], hi.x[1], lo.x[1]);
    split2(t1.x[0], t1.x[1], hi.x[2], lo.x[2]);
    split2(t1.x[2], t1.x[3], hi.x[3], lo.x[3]);
}

// The three BF16 terms of product() with the operand roles swapped, so the same products accumulate in the same order.
template <bool TB, int R, int CC>
__device__ __forceinline__ void product_t(acc & c, const a_tile & ah, const a_tile & al, const matrix<R, CC> & b,
                                          int n, int k) {
    const auto bh = load<b_tile, TB>(b, n, k, false);
    const auto bl = load<b_tile, TB>(b, n, k, true);
    mma(c, ah, bl);
    mma(c, al, bh);
    mma(c, ah, bh);
}
#endif // defined(AMPERE_MMA_AVAILABLE) && !defined(GGML_USE_HIP)

template <int C>
static __global__ __launch_bounds__(CHAIN_THREADS, 1) void gdn_chain(ggml_cuda_gdn_mma_args a) {
#if defined(AMPERE_MMA_AVAILABLE) && !defined(GGML_USE_HIP)
    static_assert(C == 16, "the chain kernel assumes one 16-row token group per chunk");
    constexpr int WARPS = CHAIN_CONSUMERS, NP = CHAIN_PRODUCERS, NB = CHAIN_BUFS, N = 8, VW = D / WARPS, TILES = D / N;
    static_assert(VW == 16, "each consumer warp owns one 16-row group of state columns");
    static_assert(C % NP == 0 && (2 * (C / 16) * (C / N)) % NP == 0, "the producer rows and tiles divide");
    extern __shared__ __align__(128) unsigned char bytes[];
    auto &        s        = *reinterpret_cast<chain_shared<C> *>(bytes);
    const int     lane     = threadIdx.x, warp = threadIdx.y;
    const int     h        = blockIdx.x % a.H;
    const int     seq      = blockIdx.x / a.H;
    const int64_t soff     = ((int64_t) seq * a.H + h) * D * D;
    const int64_t qoff     = (seq / a.rq3) * a.sq3 + (h % a.H_k) * a.sq1;
    const int64_t voff     = seq * a.sv3 + h * a.sv1;
    const int64_t goff     = seq * a.sb3 + h * a.sb1;
    float *       out      = a.dst + ((int64_t) seq * a.n_tokens * a.H + h) * D;
    const int64_t n_chunks = (a.n_tokens + C - 1) / C;
    if (warp < WARPS) {
        const int v0 = warp * VW;
        acc       st[TILES];
#pragma unroll
        for (int j = 0; j < TILES; ++j) {
#pragma unroll
            for (int i = 0; i < acc::ne; i += 2) {
                const float2 x = *reinterpret_cast<const float2 *>(
                    a.state + soff + (int64_t) (v0 + acc::get_i(i)) * D + j * N + acc::get_j(i));
                st[j].x[i]     = x.x;
                st[j].x[i + 1] = x.y;
            }
        }
        for (int64_t c = 0; c < n_chunks; ++c) {
            const int      slot  = (int) (c % NB);
            chain_buf<C> & cur   = s.buf[slot];
            const int64_t  t0    = c * C;
            const int      valid = min((int64_t) C, a.n_tokens - t0);
            bar_sync(BAR_FULL + slot, CHAIN_THREADS);
            // X^T = S^T K^T and Y^T = S^T Q^T over the warp's columns, the token groups as the two n-tiles
            acc xt[2], yt[2];
#pragma unroll
            for (int jj = 0; jj < D / 16; ++jj) {
                a_tile sh, sl;
                split_a(st[2 * jj], st[2 * jj + 1], sh, sl);
#pragma unroll
                for (int n = 0; n < 2; ++n) {
                    product_t<false>(xt[n], sh, sl, cur.k, n * N, 16 * jj);
                    product_t<false>(yt[n], sh, sl, cur.q, n * N, 16 * jj);
                }
            }
            // delta^T = (beta (v - decay K S))^T, solved^T = delta^T inverse^T, out^T = decay Y^T + solved^T P^T
#pragma unroll
            for (int n = 0; n < 2; ++n) {
#pragma unroll
                for (int i = 0; i < acc::ne; ++i) {
                    const int t = n * N + acc::get_j(i), v = v0 + acc::get_i(i);
                    xt[n].x[i]  = cur.beta[t] * (cur.v[t][v] - cur.decay[t] * xt[n].x[i]);
                }
            }
            a_tile dh, dl;
            split_a(xt[0], xt[1], dh, dl);
            acc so[2];
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                product_t<false>(so[n], dh, dl, cur.inverse, n * N, 0);
            }
            a_tile oh, ol;
            split_a(so[0], so[1], oh, ol);
            acc ps[2];
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                product_t<false>(ps[n], oh, ol, cur.p, n * N, 0);
            }
#pragma unroll
            for (int n = 0; n < 2; ++n) {
#pragma unroll
                for (int i = 0; i < acc::ne; ++i) {
                    const int t = n * N + acc::get_j(i), v = v0 + acc::get_i(i);
                    if (t < valid) {
                        out[(t0 + t) * a.H * D + v] = __fadd_rn(__fmul_rn(cur.decay[t], yt[n].x[i]), ps[n].x[i]);
                    }
                }
            }
            // the update operand is the BF16 copy of solved, decayed to the chunk end
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                const float2 h0 = __bfloat1622float2(oh.x[2 * n]),     l0 = __bfloat1622float2(ol.x[2 * n]);
                const float2 h1 = __bfloat1622float2(oh.x[2 * n + 1]), l1 = __bfloat1622float2(ol.x[2 * n + 1]);
                so[n].x[0] = (h0.x + l0.x) * cur.to_end[n * N + acc::get_j(0)];
                so[n].x[1] = (h0.y + l0.y) * cur.to_end[n * N + acc::get_j(1)];
                so[n].x[2] = (h1.x + l1.x) * cur.to_end[n * N + acc::get_j(2)];
                so[n].x[3] = (h1.y + l1.y) * cur.to_end[n * N + acc::get_j(3)];
            }
            a_tile uh, ul;
            split_a(so[0], so[1], uh, ul);
            // S^T = gamma S^T + (solved decayed)^T K
            const float gamma = cur.gamma;
#pragma unroll
            for (int j = 0; j < TILES; ++j) {
#pragma unroll
                for (int i = 0; i < acc::ne; ++i) {
                    st[j].x[i] *= gamma;
                }
                product_t<true>(st[j], uh, ul, cur.k, j * N, 0);
            }
            bar_arrive(BAR_EMPTY + slot, CHAIN_THREADS);
        }
#pragma unroll
        for (int j = 0; j < TILES; ++j) {
#pragma unroll
            for (int i = 0; i < acc::ne; i += 2) {
                *reinterpret_cast<float2 *>(a.state_out + soff + (int64_t) (v0 + acc::get_i(i)) * D + j * N +
                                            acc::get_j(i)) = make_float2(st[j].x[i], st[j].x[i + 1]);
            }
        }
    } else {
        const int pw = warp - WARPS;
        // The chunk's q and k rows into its buffer as BF16 hi and lo, its v rows as FP32, and the FP32 prefix scan
        // of the gates on the first producer warp.
        auto stage = [&](int64_t c) {
            chain_buf<C> & b     = s.buf[c % NB];
            const int64_t  t0    = c * C;
            const int      valid = min((int64_t) C, a.n_tokens - t0);
            constexpr int  ROWS  = C / NP;
            float4         q[ROWS], k[ROWS], v[ROWS];
#pragma unroll
            for (int j = 0; j < ROWS; ++j) {
                const int t = pw + NP * j;
                q[j] = k[j] = v[j] = make_float4(0, 0, 0, 0);
                if (t < valid) {
                    const int64_t off = qoff + (t0 + t) * a.sq2 + 4 * lane;
                    const int64_t vo  = voff + (t0 + t) * a.sv2 + 4 * lane;
                    if ((((uintptr_t) (a.q + off) | (uintptr_t) (a.k + off)) & 15) == 0) {
                        q[j] = *reinterpret_cast<const float4 *>(a.q + off);
                        k[j] = *reinterpret_cast<const float4 *>(a.k + off);
                    } else {
                        q[j] = make_float4(a.q[off], a.q[off + 1], a.q[off + 2], a.q[off + 3]);
                        k[j] = make_float4(a.k[off], a.k[off + 1], a.k[off + 2], a.k[off + 3]);
                    }
                    if (((uintptr_t) (a.v + vo) & 15) == 0) {
                        v[j] = *reinterpret_cast<const float4 *>(a.v + vo);
                    } else {
                        v[j] = make_float4(a.v[vo], a.v[vo + 1], a.v[vo + 2], a.v[vo + 3]);
                    }
                }
            }
            if (pw == 0) {
                const int64_t tn = t0 + lane;
                const bool    in = lane < C && tn < a.n_tokens;
                float         g = in ? a.g[goff + tn * a.sb2] : 0.f, beta = in ? a.beta[goff + tn * a.sb2] : 0.f;
#pragma unroll
                for (int offset = 1; offset < 32; offset *= 2) {
                    const float previous = __shfl_up_sync(0xffffffff, g, offset, 32);
                    if (lane >= offset) {
                        g += previous;
                    }
                }
                const float end = __shfl_sync(0xffffffff, g, valid - 1, 32);
                if (lane < C) {
                    b.prefix[lane] = g;
                    b.beta[lane]   = beta;
                    b.decay[lane]  = expf(g);
                    b.to_end[lane] = expf(end - g);
                }
                if (lane == 0) {
                    b.gamma = expf(end);
                }
            }
#pragma unroll
            for (int j = 0; j < ROWS; ++j) {
                const int t = pw + NP * j, d = 4 * lane;
                b.q.store2(t, d, make_float2(q[j].x * a.scale, q[j].y * a.scale));
                b.q.store2(t, d + 2, make_float2(q[j].z * a.scale, q[j].w * a.scale));
                b.k.store2(t, d, make_float2(k[j].x, k[j].y));
                b.k.store2(t, d + 2, make_float2(k[j].z, k[j].w));
                *reinterpret_cast<float4 *>(&b.v[t][d]) = v[j];
            }
        };
        // A = beta * strict_tril(K K^T * decay), P = tril(Q K^T * decay)
        auto gram_p = [&](int64_t c) {
            chain_buf<C> & b     = s.buf[c % NB];
            const int      valid = min((int64_t) C, a.n_tokens - c * C);
            constexpr int  GP    = (C / 16) * (C / N);
#pragma unroll
            for (int j = 0; j < 2 * GP / NP; ++j) {
                const int  ti   = pw + NP * j;
                const bool gram = ti < GP;
                const int  t = ti % GP, r = (t / (C / N)) * 16, c0 = (t % (C / N)) * N;
                acc        x;
                if (gram) {
                    product<D>(x, b.k, b.k, r, c0);
                } else {
                    product<D>(x, b.q, b.k, r, c0);
                }
#pragma unroll
                for (int i = 0; i < acc::ne; ++i) {
                    const int   rr = r + acc::get_i(i), cc = c0 + acc::get_j(i);
                    const float v = rr < valid && cc <= rr ? x.x[i] * expf(b.prefix[rr] - b.prefix[cc]) : 0.f;
                    if (gram) {
                        if (cc < rr) {
                            s.lower[s.lower_index(rr, cc)] = v * b.beta[rr];
                        }
                    } else {
                        b.p.store(rr, cc, v);
                    }
                }
            }
        };
        // inverse = (I + A)^-1 by forward substitution, the warp's rows as independent chains of one pivot loop
        auto invert = [&](int64_t c) {
            chain_buf<C> & b    = s.buf[c % NB];
            constexpr int  ROWS = C / NP;
            float          x[ROWS];
#pragma unroll
            for (int j = 0; j < ROWS; ++j) {
                const int r = pw + NP * j;
                x[j]        = lane == r ? 1.f : (lane < r ? -s.lower[s.lower_index(r, lane)] : 0.f);
            }
            for (int pivot = C - 2; pivot > 0; --pivot) {
#pragma unroll
                for (int j = 0; j < ROWS; ++j) {
                    const int   r  = pw + NP * j;
                    const float xp = __shfl_sync(0xffffffff, x[j], pivot, 32);
                    if (pivot < r && lane < pivot) {
                        x[j] -= xp * s.lower[s.lower_index(pivot, lane)];
                    }
                }
            }
#pragma unroll
            for (int j = 0; j < ROWS; ++j) {
                if (lane < C) {
                    b.inverse.store(pw + NP * j, lane, x[j]);
                }
            }
        };
        for (int64_t c = 0; c < n_chunks; ++c) {
            const int slot = (int) (c % NB);
            if (c >= NB) {
                bar_sync(BAR_EMPTY + slot, CHAIN_THREADS);
            }
            stage(c);
            bar_sync(BAR_PROD, 32 * NP);
            gram_p(c);
            bar_sync(BAR_PROD, 32 * NP);
            invert(c);
            __threadfence_block();
            bar_arrive(BAR_FULL + slot, CHAIN_THREADS);
        }
    }
#else
    GGML_UNUSED_VARS(a);
    NO_DEVICE_CODE;
#endif // defined(AMPERE_MMA_AVAILABLE) && !defined(GGML_USE_HIP)
}

static bool opt_in(int device, const void * kernel, size_t smem) {
    ggml_cuda_set_device(device);
    const bool        fits   = smem <= ggml_cuda_info().devices[device].smpbo;
    const cudaError_t status = fits ? cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem) :
                                      cudaErrorInvalidValue;
    if (status != cudaSuccess) {
        (void) cudaGetLastError();
    }
    return status == cudaSuccess;
}

template <int C, int V, int BLOCKS = 1> bool init(int device) {
    static std::once_flag once[GGML_CUDA_MAX_DEVICES];
    static bool           available[GGML_CUDA_MAX_DEVICES] = {};
    std::call_once(once[device], [device] {
        available[device] = opt_in(device, (const void *) gdn_single<C, V, BLOCKS>, sizeof(shared<C, V>));
    });
    return available[device];
}

template <int C> bool init_chain(int device) {
    static std::once_flag once[GGML_CUDA_MAX_DEVICES];
    static bool           available[GGML_CUDA_MAX_DEVICES] = {};
    std::call_once(once[device], [device] {
        available[device] = opt_in(device, (const void *) gdn_chain<C>, sizeof(chain_shared<C>));
    });
    return available[device];
}

enum class gdn_path { ar, slice32_one_block, slice32_two_blocks, slice64, whole_head };

static gdn_path select_gdn_mma_path(int device, const ggml_cuda_gdn_mma_args & a) {
    GGML_ASSERT(device >= 0 && device < GGML_CUDA_MAX_DEVICES);
    const auto & info  = ggml_cuda_info().devices[device];
    const bool   shape = a.eligible && a.H_k == 16 && (a.H == 16 || a.H == 32 || a.H == 48 || a.H == 64) &&
                         a.n_tokens >= 64 && a.n_tokens <= INT64_MAX - 32 && a.rq3 > 0 && a.n_seqs > 0 &&
                         a.n_seqs <= INT_MAX / (a.H * 4);
    gdn_path     path  = gdn_path::ar;
#ifdef GGML_USE_HIP
    if (shape && GGML_CUDA_CC_IS_RDNA3_5(info.cc) && a.n_tokens >= 2048 && init<16, 64>(device)) {
        path = gdn_path::slice64;
    }
#elif defined(GGML_USE_MUSA)
    GGML_UNUSED_VARS(info, shape);
#else
    // Enable only architectures benchmarked for this schedule. MMA support alone does not imply a speedup.
    const bool arch = ampere_mma_available(info.cc) &&
                      (info.cc == GGML_CUDA_CC_AMPERE_GA10X || info.cc == GGML_CUDA_CC_ADA_LOVELACE ||
                       info.cc == GGML_CUDA_CC_BLACKWELL || info.cc == GGML_CUDA_CC_DGX_SPARK);
    // Value splitting pays off when whole heads leave at least half the SMs idle; a grid that already fits in
    // one wave benefits from a larger register budget.
    if (shape && arch) {
        if (a.H * a.n_seqs * 4 <= info.nsm) {
            path = init<16, 32, 1>(device) ? gdn_path::slice32_one_block : gdn_path::ar;
        } else if (a.H * a.n_seqs * 2 <= info.nsm) {
            path = init<16, 32, 2>(device) ? gdn_path::slice32_two_blocks : gdn_path::ar;
        } else {
            path = init_chain<16>(device) ? gdn_path::whole_head : gdn_path::ar;
        }
    }
#endif // GGML_USE_HIP
    return path;
}
}  // namespace
using namespace ggml_cuda_gated_delta_net_mma;

bool ggml_cuda_gdn_mma_launch(int device, const ggml_cuda_gdn_mma_args & a, cudaStream_t stream) {
    const gdn_path path = a.state_out != nullptr ? select_gdn_mma_path(device, a) : gdn_path::ar;
    const dim3     block(32, 8);
#ifdef GGML_USE_HIP
    if (path == gdn_path::slice64) {
        gdn_single<16, 64><<<a.H * a.n_seqs * 2, block, sizeof(shared<16, 64>), stream>>>(a);
    }
#else
    const dim3 chain_block(32, CHAIN_CONSUMERS + CHAIN_PRODUCERS);
    if (path == gdn_path::slice32_two_blocks) {
        gdn_single<16, 32, 2><<<a.H * a.n_seqs * 4, block, sizeof(shared<16, 32>), stream>>>(a);
    } else if (path == gdn_path::slice32_one_block) {
        gdn_single<16, 32, 1><<<a.H * a.n_seqs * 4, block, sizeof(shared<16, 32>), stream>>>(a);
    } else if (path == gdn_path::whole_head) {
        gdn_chain<16><<<a.H * a.n_seqs, chain_block, sizeof(chain_shared<16>), stream>>>(a);
    }
#endif // GGML_USE_HIP
    if (path != gdn_path::ar) {
        CUDA_CHECK(cudaGetLastError());
    }
    return path != gdn_path::ar;
}
