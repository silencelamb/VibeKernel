// v13: v12 + prefetch BOTH stages (0,1) of the next tile during the
// epilogue; C staging therefore only uses the B2 slot (4 chunks x 32 rows).
#include <cstdint>
#include <cuda_runtime.h>

#include "playground/matmul.hpp"

namespace playground
{
namespace v13
{

constexpr int BM = 128, BN = 256, BK = 64;
constexpr int STAGES = 3;
constexpr uint32_t A_STAGE = BM * BK * sizeof(half);           // 16 KB
constexpr uint32_t B_STAGE = BK * BN * sizeof(half);           // 32 KB
constexpr uint32_t SMEM_BYTES = STAGES * (A_STAGE + B_STAGE);  // 144 KB
constexpr uint32_t EPI_STRIDE = BN + 8;
static_assert(128 * EPI_STRIDE * 2 <= SMEM_BYTES);

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

__global__ __launch_bounds__(256, 1) void hgemmV13(const half* __restrict__ A,
                                                   const half* __restrict__ B,
                                                   half* __restrict__ C, int M,
                                                   int N, int K)
{
    extern __shared__ __align__(128) uint8_t smemRaw[];
    const uint32_t sA = cvta(smemRaw);
    const uint32_t sB = sA + STAGES * A_STAGE;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm = warp >> 2;
    const int wn = warp & 3;

    constexpr int GW = 8;
    const int tM = M / BM, tN = N / BN;
    const int numTiles = tM * tN;
    const int G = gridDim.x;

    // A tile 128x64: rows of 128B = 8 chunks; swizzle c ^ (row&7).
    const int aRow = tid >> 3;
    const int aChunk = tid & 7;
    const uint32_t aDst = aRow * 128 + ((aChunk ^ (aRow & 7)) << 4);
    // B tile 64x256: rows of 512B = 32 chunks; swizzle low3 ^ (k&7).
    const int bRow = warp;
    const int bChunk = lane;
    const uint32_t bDst =
        bRow * 512 + (((bChunk & 24) | ((bChunk & 7) ^ (bRow & 7))) << 4);

    // tile id -> source pointers (panel-swizzled order)
    auto tileSrc = [&](int t, const half*& aS, const half*& bS, int& bmO,
                       int& bnO) {
        int tileM, tileN;
        if (tN % GW == 0) {
            const int panelSize = GW * tM;
            const int p = t / panelSize;
            const int r = t % panelSize;
            tileN = p * GW + (r % GW);
            tileM = r / GW;
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
        ldA[i] = row * 128 + ((c ^ (row & 7)) << 4);  // kk: ^ (kk<<5)
    }
    uint32_t ldB[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int kr = lane & 15;
        const int c = wn * 8 + j * 2 + (lane >> 4);
        ldB[j] = kr * 512 +
                 (((c & 24) | ((c & 7) ^ (kr & 7))) << 4);  // kk: +8192*kk
    }

    float acc[4][8][4] = {};
    uint32_t fragA[2][4][4];
    uint32_t fragB[2][4][4];

    const int NT = K / BK;  // 64 for 4096

#define ISSUE_A(SRC_, S_, KT_)                                                \
    do {                                                                      \
        const half* ag = (SRC_) + (KT_) * BK;                                 \
        cpAsync16(sA + (S_)*A_STAGE + aDst, ag);                              \
        cpAsync16(sA + (S_)*A_STAGE + aDst + 4096, ag + (uint32_t)32 * K);    \
        cpAsync16(sA + (S_)*A_STAGE + aDst + 8192, ag + (uint32_t)64 * K);    \
        cpAsync16(sA + (S_)*A_STAGE + aDst + 12288, ag + (uint32_t)96 * K);   \
    } while (0)
#define ISSUE_B(SRC_, S_, KT_, I0_)                                           \
    do {                                                                      \
        const half* bg = (SRC_) + (uint32_t)((KT_) * BK + (I0_)*8) * N;       \
        cpAsync16(sB + (S_)*B_STAGE + bDst + (I0_)*4096, bg);                 \
        cpAsync16(sB + (S_)*B_STAGE + bDst + ((I0_) + 1) * 4096,              \
                  bg + (uint32_t)8 * N);                                      \
        cpAsync16(sB + (S_)*B_STAGE + bDst + ((I0_) + 2) * 4096,              \
                  bg + (uint32_t)16 * N);                                     \
        cpAsync16(sB + (S_)*B_STAGE + bDst + ((I0_) + 3) * 4096,              \
                  bg + (uint32_t)24 * N);                                     \
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
                    sB + (S_)*B_STAGE + ldB[j] + (KK_)*8192);                 \
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
        const bool fetch = (KT_) + 2 < NT;                                    \
        LOAD_FRAGS(1, RS_, 1);                                                \
        if (fetch) {                                                          \
            ISSUE_A(aSrc, WS_, (KT_) + 2);                                    \
        }                                                                     \
        MMA_STEP(0);                                                          \
        LOAD_FRAGS(0, RS_, 2);                                                \
        if (fetch) {                                                          \
            ISSUE_B(bSrc, WS_, (KT_) + 2, 0);                                 \
        }                                                                     \
        MMA_STEP(1);                                                          \
        LOAD_FRAGS(1, RS_, 3);                                                \
        if (fetch) {                                                          \
            ISSUE_B(bSrc, WS_, (KT_) + 2, 4);                                 \
        }                                                                     \
        MMA_STEP(0);                                                          \
        cpWait<0>();                                                          \
        __syncthreads();                                                      \
        LOAD_FRAGS(0, NS_, 0);                                                \
        cpCommit();                                                           \
        MMA_STEP(1);                                                          \
    } while (0)

    const half *aSrc, *bSrc;
    int bm, bn;
    tileSrc(blockIdx.x, aSrc, bSrc, bm, bn);

    // first tile: prologue issues stages 0,1 (later tiles: both were
    // prefetched during the previous epilogue)
    ISSUE_A(aSrc, 0, 0);
    ISSUE_B(bSrc, 0, 0, 0);
    ISSUE_B(bSrc, 0, 0, 4);
    cpCommit();
    ISSUE_A(aSrc, 1, 1);
    ISSUE_B(bSrc, 1, 1, 0);
    ISSUE_B(bSrc, 1, 1, 4);
    cpCommit();

    for (int t = blockIdx.x;;) {
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
        cpWait<1>();
        __syncthreads();
        LOAD_FRAGS(0, 0, 0);

        int kt = 0;
        for (; kt + 3 <= NT; kt += 3) {
            TILE_ITER(kt + 0, 0, 1, 2);
            TILE_ITER(kt + 1, 1, 2, 0);
            TILE_ITER(kt + 2, 2, 0, 1);
        }
        if (kt < NT) {
            TILE_ITER(kt, 0, 1, 2);
            if (kt + 1 < NT) {
                TILE_ITER(kt + 1, 1, 2, 0);
            }
        }

        // ---- epilogue (+ prefetch next tile's stage 0 behind it)
        cpWait<0>();
        __syncthreads();

        const int tNext = t + G;
        const bool more = tNext < numTiles;
        const int bmCur = bm, bnCur = bn;
        if (more) {
            tileSrc(tNext, aSrc, bSrc, bm, bn);
            ISSUE_A(aSrc, 0, 0);
            ISSUE_B(bSrc, 0, 0, 0);
            ISSUE_B(bSrc, 0, 0, 4);
            cpCommit();
            ISSUE_A(aSrc, 1, 1);
            ISSUE_B(bSrc, 1, 1, 0);
            ISSUE_B(bSrc, 1, 1, 4);
            cpCommit();
        }

        // stage C through the B2 slot only (stage-0/1 prefetch uses A0,A1,
        // B0,B1); four chunks of 32 rows.
        const uint32_t epiBase = sB + 2 * B_STAGE;  // 32KB: B slot 2
        const int r0e = wm * 64 + (lane >> 2);
        const int c0e = wn * 64 + (lane & 3) * 2;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            // chunk q covers rows [32q, 32q+32): wm == q>>1, i in {2(q&1),+1}
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
            for (int it = 0; it < 4; ++it) {
                const int row = it * 8 + warp;  // 0..31 within chunk
                uint32_t v[4];
                ldsV4(v, epiBase + (row * EPI_STRIDE + lane * 8) * 2);
                *reinterpret_cast<uint4*>(
                    &C[(uint32_t)(bmCur + q * 32 + row) * N + bnCur +
                       lane * 8]) = *reinterpret_cast<uint4*>(v);
            }
            __syncthreads();
        }

        if (!more) {
            break;
        }
        t = tNext;
    }
#undef TILE_ITER
#undef ISSUE_A
#undef ISSUE_B
#undef LOAD_FRAGS
#undef MMA_STEP
}

}  // namespace v13

PLAYGROUND_MATMUL_DEC(float16_t, 13, m, n, k, A, B, C)
{
    static bool smemConfigured = false;
    if (!smemConfigured) {
        cudaFuncSetAttribute(v13::hgemmV13,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             v13::SMEM_BYTES);
        smemConfigured = true;
    }
    static int G = 0;
    if (G == 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        G = prop.multiProcessorCount;
    }
    const int numTiles = int((m / v13::BM) * (n / v13::BN));
    v13::hgemmV13<<<min(G, numTiles), 256, v13::SMEM_BYTES>>>(
        A, B, C, int(m), int(n), int(k));
}

}  // namespace playground
