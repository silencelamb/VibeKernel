// v23: v19 + surgical in-kernel K-split of the final partial wave.
// 4096^3 -> 512 tiles over 108 SMs = 4 full waves + 80 leftover tiles.
// The 28 blocks that would idle in round 5 become helpers: each computes
// the kt-tail [KS, NT) of 2-3 leftover tiles into an f32 workspace and
// raises a generation-tagged flag; the owner computes kt [0, KS), spins
// for the flag, merges the partial in registers and runs the normal
// epilogue. No extra kernel launches (that leaked ~100us/launch in
// v15/v22); flags use a generation counter so no per-call memset.
#include <cstdint>
#include <cuda_runtime.h>

#include "playground/matmul.hpp"

namespace playground
{
namespace v28
{

constexpr int BM = 256, BN = 128, BK = 64;
constexpr int STAGES = 3;
constexpr uint32_t A_STAGE = BM * BK * sizeof(half);           // 32 KB
constexpr uint32_t B_STAGE = BK * BN * sizeof(half);           // 16 KB
constexpr uint32_t SMEM_BYTES = STAGES * (A_STAGE + B_STAGE);  // 144 KB
constexpr uint32_t EPI_STRIDE = BN + 8;                        // halves
constexpr int KS_BIAS = 2;       // owner k-tiles = lastWave*NT/G + bias
constexpr int MAX_TAILS = 128;   // workspace slots (>= max lastWave)
constexpr int FLAG_STRIDE = 32;  // one 128B line per flag (avoid L2 storms)

__device__ __forceinline__ uint32_t cvta(const void* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cpAsync16(uint32_t dst, const void* src)
{
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(
                     dst),
                 "l"(src));
}
__device__ __forceinline__ void cpCommit()
{
    asm volatile("cp.async.commit_group;\n");
}
template <int N>
__device__ __forceinline__ void cpWait()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void ldsmX4(uint32_t (&r)[4], uint32_t addr)
{
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, "
                 "[%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}
__device__ __forceinline__ void ldsmX4T(uint32_t (&r)[4], uint32_t addr)
{
    asm volatile(
        "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0,%1,%2,%3}, "
        "[%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(addr));
}
__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4],
                                         const uint32_t* b)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void stsHalf2(uint32_t addr, half2 v)
{
    asm volatile("st.shared.b32 [%0], %1;\n" ::"r"(addr),
                 "r"(*reinterpret_cast<uint32_t*>(&v)));
}
__device__ __forceinline__ void ldsV4(uint32_t (&r)[4], uint32_t addr)
{
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}
__device__ __forceinline__ unsigned ldAcquire(const unsigned* p)
{
    unsigned v;
    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n"
                 : "=r"(v)
                 : "l"(p));
    return v;
}
__device__ __forceinline__ void stRelease(unsigned* p, unsigned v)
{
    asm volatile("st.release.gpu.global.b32 [%0], %1;\n" ::"l"(p), "r"(v));
}

__global__ __launch_bounds__(256, 1) void hgemmV28(
    const half* __restrict__ A, const half* __restrict__ B,
    half* __restrict__ C, int M, int N, int K, float* __restrict__ ws,
    unsigned* __restrict__ flags, unsigned gen)
{
    extern __shared__ __align__(128) uint8_t smemRaw[];
    const uint32_t sA = cvta(smemRaw);
    const uint32_t sB = sA + STAGES * A_STAGE;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm = warp >> 1;  // 0..3
    const int wn = warp & 1;   // 0..1

    constexpr int GW = 8;  // panel width in m-tiles here (mirror)
    const int tM = M / BM, tN = N / BN;
    const int numTiles = tM * tN;
    const int G = gridDim.x;

    // A tile 256x64 (128B rows, 8 chunks): thread t: rows {t>>3 + 32k},
    // chunk t&7; 8 rows per thread. Swizzle c ^ (row&7).
    const int aRow = tid >> 3;  // 0..31
    const int aChunk = tid & 7;
    const uint32_t aDst = aRow * 128 + ((aChunk ^ (aRow & 7)) << 4);

    // B tile 64x128 (256B rows, 16 chunks): thread t: rows {t>>4 + 16k},
    // chunk t&15; 4 rows per thread. Swizzle (c&8) | (low3 ^ (k&7)).
    const int bRow = tid >> 4;  // 0..15
    const int bChunk = tid & 15;
    const uint32_t bDst =
        bRow * 256 + (((bChunk & 8) | ((bChunk & 7) ^ (bRow & 7))) << 4);

    auto tileSrc = [&](int t, const half*& aS, const half*& bS, int& bmO,
                       int& bnO) {
        int tileM, tileN;
        if (tM % GW == 0) {
            const int panelSize = GW * tN;
            const int p = t / panelSize;
            const int r = t % panelSize;
            tileM = p * GW + (r % GW);
            tileN = r / GW;
        } else {
            tileM = t / tN;
            tileN = t % tN;
        }
        bmO = tileM * BM;
        bnO = tileN * BN;
        aS = A + (uint32_t)(bmO + aRow) * K + aChunk * 8;
        bS = B + (uint32_t)bRow * N + bnO + bChunk * 8;
    };

    uint32_t ldA[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = wm * 64 + i * 16 + (lane & 15);
        const int c = lane >> 4;
        ldA[i] = row * 128 + ((c ^ (row & 7)) << 4);  // kk: ^(kk<<5)
    }
    uint32_t ldB[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int kr = lane & 15;
        const int c = wn * 8 + j * 2 + (lane >> 4);
        ldB[j] = kr * 256 +
                 (((c & 8) | ((c & 7) ^ (kr & 7))) << 4);  // kk: +4096*kk
    }

    float acc[4][8][4];
    uint32_t fragA[2][4][4];
    uint32_t fragB[2][4][4];

    const int NT = K / BK;

    // final-partial-wave split parameters
    const int lastWave = numTiles % G;
    const int H = G - lastWave;  // helper blocks
    const int KS = (lastWave * NT) / G + KS_BIAS;
    const bool splitOn = (numTiles > G) && (lastWave > 0) && (H > 0) &&
                         (KS >= 4) && (NT - KS >= 4) &&
                         (lastWave <= MAX_TAILS);
    const int splitBase = numTiles - lastWave;

#define ISSUE_A(SRC_, S_, KT_)                                                \
    do {                                                                      \
        const half* ag = (SRC_) + (KT_) * BK;                                 \
        _Pragma("unroll") for (int r = 0; r < 8; ++r)                         \
        {                                                                     \
            cpAsync16(sA + (S_)*A_STAGE + aDst + r * 4096,                    \
                      ag + (uint32_t)(r * 32) * K);                           \
        }                                                                     \
    } while (0)
#define ISSUE_B(SRC_, S_, KT_)                                                \
    do {                                                                      \
        const half* bg = (SRC_) + (uint32_t)((KT_) * BK) * N;                 \
        _Pragma("unroll") for (int r = 0; r < 4; ++r)                         \
        {                                                                     \
            cpAsync16(sB + (S_)*B_STAGE + bDst + r * 4096,                    \
                      bg + (uint32_t)(r * 16) * N);                           \
        }                                                                     \
    } while (0)
#define LOAD_FRAGS(BUF_, S_, KK_)                                            \
    do {                                                                      \
        _Pragma("unroll") for (int i = 0; i < 4; ++i)                         \
        {                                                                     \
            ldsmX4(fragA[BUF_][i],                                            \
                   sA + (S_)*A_STAGE + (ldA[i] ^ ((KK_) << 5)));              \
        }                                                                     \
        _Pragma("unroll") for (int j = 0; j < 4; ++j)                         \
        {                                                                     \
            ldsmX4T(fragB[BUF_][j],                                           \
                    sB + (S_)*B_STAGE + ldB[j] + (KK_)*4096);                 \
        }                                                                     \
    } while (0)
#define MMA_STEP(BUF_)                                                        \
    do {                                                                      \
        _Pragma("unroll") for (int i = 0; i < 4; ++i)                         \
        {                                                                     \
            _Pragma("unroll") for (int j = 0; j < 4; ++j)                     \
            {                                                                 \
                mma16816(acc[i][2 * j], fragA[BUF_][i], &fragB[BUF_][j][0]);  \
                mma16816(acc[i][2 * j + 1], fragA[BUF_][i],                   \
                         &fragB[BUF_][j][2]);                                 \
            }                                                                 \
        }                                                                     \
    } while (0)
#define TILE_ITER(KT_, RS_, NS_, WS_)                                         \
    do {                                                                      \
        const bool fetch = (KT_) + 2 < k1;                                    \
        LOAD_FRAGS(1, RS_, 1);                                                \
        if (fetch) {                                                          \
            ISSUE_A(aSrc, WS_, (KT_) + 2);                                    \
        }                                                                     \
        MMA_STEP(0);                                                          \
        LOAD_FRAGS(0, RS_, 2);                                                \
        if (fetch) {                                                          \
            ISSUE_B(bSrc, WS_, (KT_) + 2);                                    \
        }                                                                     \
        MMA_STEP(1);                                                          \
        LOAD_FRAGS(1, RS_, 3);                                                \
        MMA_STEP(0);                                                          \
        cpWait<0>();                                                          \
        __syncthreads();                                                      \
        LOAD_FRAGS(0, NS_, 0);                                                \
        cpCommit();                                                           \
        MMA_STEP(1);                                                          \
    } while (0)
// like TILE_ITER but the (idle) issue slots prefetch stage XS_ of the
// next helper tail (warm handoff across tail boundaries)
#define TILE_ITER_X(KT_, RS_, NS_, WS_, XA_, XB_, XS_, XKT_)                  \
    do {                                                                      \
        LOAD_FRAGS(1, RS_, 1);                                                \
        ISSUE_A((XA_), XS_, XKT_);                                            \
        MMA_STEP(0);                                                          \
        LOAD_FRAGS(0, RS_, 2);                                                \
        ISSUE_B((XB_), XS_, XKT_);                                            \
        MMA_STEP(1);                                                          \
        LOAD_FRAGS(1, RS_, 3);                                                \
        MMA_STEP(0);                                                          \
        cpWait<0>();                                                          \
        __syncthreads();                                                      \
        LOAD_FRAGS(0, NS_, 0);                                                \
        cpCommit();                                                           \
        MMA_STEP(1);                                                          \
    } while (0)

    if ((int)blockIdx.x >= numTiles) {
        return;
    }
    const half *aSrc, *bSrc;
    int bm, bn;
    int t = blockIdx.x;
    bool helper = false;
    int tailJ = -1;
    int k0 = 0;
    int k1 = (splitOn && t >= splitBase) ? KS : NT;

    tileSrc(t, aSrc, bSrc, bm, bn);
    ISSUE_A(aSrc, 0, k0);
    ISSUE_B(bSrc, 0, k0);
    cpCommit();
    ISSUE_A(aSrc, 1, k0 + 1);
    ISSUE_B(bSrc, 1, k0 + 1);
    cpCommit();

    for (;;) {
        cpWait<1>();
        __syncthreads();
        LOAD_FRAGS(0, 0, 0);

#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
            for (int jj = 0; jj < 8; ++jj) {
#pragma unroll
                for (int x = 0; x < 4; ++x) {
                    acc[i][jj][x] = 0.0F;
                }
            }
        }

        // warm handoff: a helper tail with a successor and 3|kn prefetches
        // the successor's stages 0/1 from its last two (idle) issue slots
        const bool xferTile =
            helper && (tailJ + H < lastWave) && ((k1 - k0) % 3 == 0);

        const int kn = k1 - k0;
        const int knMain = xferTile ? kn - 3 : kn;  // peel handoff triple
        int kr = 0;
        for (; kr + 3 <= knMain; kr += 3) {
            TILE_ITER(k0 + kr + 0, 0, 1, 2);
            TILE_ITER(k0 + kr + 1, 1, 2, 0);
            TILE_ITER(k0 + kr + 2, 2, 0, 1);
        }
        if (xferTile) {
            const half *aN, *bN;
            int bmN, bnN;
            tileSrc(splitBase + tailJ + H, aN, bN, bmN, bnN);
            TILE_ITER(k0 + kr + 0, 0, 1, 2);
            TILE_ITER_X(k0 + kr + 1, 1, 2, 0, aN, bN, 0, k0);
            TILE_ITER_X(k0 + kr + 2, 2, 0, 1, aN, bN, 1, k0 + 1);
            kr += 3;
        }
        if (kr < kn) {
            TILE_ITER(k0 + kr, 0, 1, 2);
            if (kr + 1 < kn) {
                TILE_ITER(k0 + kr + 1, 1, 2, 0);
            }
        }

        if (!xferTile) {
            cpWait<0>();
        }
        __syncthreads();

        const int bmCur = bm, bnCur = bn;
        const bool curHelper = helper;
        const bool curOwner = !helper && splitOn && (t >= splitBase);
        const int curTail = curHelper ? tailJ : (t - splitBase);

        // pick next work item and prefetch its first two stages
        bool more = false;
        if (!helper) {
            const int tNext = t + G;
            if (tNext < numTiles) {
                t = tNext;
                k0 = 0;
                k1 = (splitOn && t >= splitBase) ? KS : NT;
                more = true;
            } else if (splitOn && (int)blockIdx.x >= lastWave) {
                helper = true;
                tailJ = (int)blockIdx.x - lastWave;
                if (tailJ < lastWave) {
                    t = splitBase + tailJ;
                    k0 = KS;
                    k1 = NT;
                    more = true;
                }
            }
        } else {
            tailJ += H;
            if (tailJ < lastWave) {
                t = splitBase + tailJ;
                k0 = KS;
                k1 = NT;
                more = true;
            }
        }
        if (more) {
            tileSrc(t, aSrc, bSrc, bm, bn);
            if (!xferTile) {  // handoff already prefetched stages 0/1
                ISSUE_A(aSrc, 0, k0);
                ISSUE_B(bSrc, 0, k0);
                cpCommit();
                ISSUE_A(aSrc, 1, k0 + 1);
                ISSUE_B(bSrc, 1, k0 + 1);
                cpCommit();
            }
        }

        if (curHelper) {
            // dump f16 partial (halves the L2 drain vs f32), lane-fastest
            // so each warp store is one contiguous 256B transaction group
            uint2* wp2 = reinterpret_cast<uint2*>(ws) +
                         (size_t)curTail * (BM * BN / 4) + warp * 1024 + lane;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int jj = 0; jj < 8; ++jj) {
                    const half2 lo =
                        __floats2half2_rn(acc[i][jj][0], acc[i][jj][1]);
                    const half2 hi =
                        __floats2half2_rn(acc[i][jj][2], acc[i][jj][3]);
                    uint2 p;
                    p.x = *reinterpret_cast<const uint32_t*>(&lo);
                    p.y = *reinterpret_cast<const uint32_t*>(&hi);
                    __stcs(wp2 + (i * 8 + jj) * 32, p);
                }
            }
            // bar.sync + st.release publish all threads' ws stores
            // gpu-wide; no explicit membar needed
            __syncthreads();
            if (tid == 0) {
                stRelease(flags + curTail * FLAG_STRIDE, gen);
            }
        } else {
            if (curOwner) {
                if (tid == 0) {
                    unsigned ns = 64;
                    while (ldAcquire(flags + curTail * FLAG_STRIDE) != gen) {
                        __nanosleep(ns);
                        ns = min(ns * 2, 2048U);
                    }
                }
                __syncthreads();
                const uint2* rp2 = reinterpret_cast<const uint2*>(ws) +
                                   (size_t)curTail * (BM * BN / 4) +
                                   warp * 1024 + lane;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
#pragma unroll
                    for (int jj = 0; jj < 8; ++jj) {
                        const uint2 p = __ldcs(rp2 + (i * 8 + jj) * 32);
                        const float2 lo = __half22float2(
                            *reinterpret_cast<const half2*>(&p.x));
                        const float2 hi = __half22float2(
                            *reinterpret_cast<const half2*>(&p.y));
                        acc[i][jj][0] += lo.x;
                        acc[i][jj][1] += lo.y;
                        acc[i][jj][2] += hi.x;
                        acc[i][jj][3] += hi.y;
                    }
                }
            }
            // epilogue: stage C (256 rows x 128 cols) through the B2 slot
            // (16KB), eight chunks of 32 rows.
            const uint32_t epiBase = sB + 2 * B_STAGE;
            const int r0e = wm * 64 + (lane >> 2);
            const int c0e = wn * 64 + (lane & 3) * 2;
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                if (wm == (q >> 1)) {
#pragma unroll
                    for (int i2 = 0; i2 < 2; ++i2) {
                        const int i = (q & 1) * 2 + i2;
#pragma unroll
                        for (int jj = 0; jj < 8; ++jj) {
                            const float* c4 = acc[i][jj];
                            const uint32_t a0 =
                                epiBase + ((((r0e & 63) + i * 16) & 31) *
                                               EPI_STRIDE +
                                           c0e + jj * 8) *
                                              2;
                            stsHalf2(a0, __floats2half2_rn(c4[0], c4[1]));
                            stsHalf2(a0 + 8 * EPI_STRIDE * 2,
                                     __floats2half2_rn(c4[2], c4[3]));
                        }
                    }
                }
                __syncthreads();
#pragma unroll
                for (int it = 0; it < 2; ++it) {
                    const int row = it * 16 + (tid >> 4);
                    uint32_t v[4];
                    ldsV4(v,
                          epiBase + (row * EPI_STRIDE + (tid & 15) * 8) * 2);
                    *reinterpret_cast<uint4*>(
                        &C[(uint32_t)(bmCur + q * 32 + row) * N + bnCur +
                           (tid & 15) * 8]) = *reinterpret_cast<uint4*>(v);
                }
                __syncthreads();
            }
        }

        if (!more) {
            break;
        }
    }
#undef TILE_ITER
#undef ISSUE_A
#undef ISSUE_B
#undef LOAD_FRAGS
#undef MMA_STEP
}

}  // namespace v28

PLAYGROUND_MATMUL_DEC(float16_t, 28, m, n, k, A, B, C)
{
    static int G = 0;
    static float* ws = nullptr;
    static unsigned* flags = nullptr;
    static unsigned gen = 0;
    if (G == 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        G = prop.multiProcessorCount;
        cudaFuncSetAttribute(v28::hgemmV28,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             v28::SMEM_BYTES);
        cudaMalloc(&ws, (size_t)v28::MAX_TAILS * v28::BM * v28::BN *
                            sizeof(float));
        cudaMalloc(&flags,
                   (size_t)v28::MAX_TAILS * v28::FLAG_STRIDE *
                       sizeof(unsigned));
        cudaMemset(flags, 0,
                   (size_t)v28::MAX_TAILS * v28::FLAG_STRIDE *
                       sizeof(unsigned));
    }
    ++gen;
    const int numTiles = int((m / v28::BM) * (n / v28::BN));
    v28::hgemmV28<<<min(G, numTiles), 256, v28::SMEM_BYTES>>>(
        A, B, C, int(m), int(n), int(k), ws, flags, gen);
}

}  // namespace playground
