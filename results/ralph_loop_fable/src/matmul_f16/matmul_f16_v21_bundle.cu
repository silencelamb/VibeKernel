// v18: BM=256, BN=128 mirror of v13 (same BK=64, 3 stages, 8 warps 64x64
// in a 4x2 grid, f32 acc, persistent + dual-stage prefetch in epilogue).
// Probes whether the transposed tile shape behaves better for L2 / stores.
#include <cstdint>
#include <cuda_runtime.h>

#include "playground/matmul.hpp"

namespace playground
{
namespace v21
{

constexpr int BM = 256, BN = 128, BK = 64;
constexpr int STAGES = 3;
constexpr uint32_t A_STAGE = BM * BK * sizeof(half);           // 32 KB
constexpr uint32_t B_STAGE = BK * BN * sizeof(half);           // 16 KB
constexpr uint32_t SMEM_BYTES = STAGES * (A_STAGE + B_STAGE);  // 144 KB
constexpr uint32_t EPI_STRIDE = BN + 8;                        // halves

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

__global__ __launch_bounds__(256, 1) void hgemmV21(const half* __restrict__ A,
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
        // panel order: GW wide in N? mirror: walk m fastest inside GW-tall
        // n panels -> use GW n-tiles per panel like v13 (tN=32 here).
        int tileM, tileN;
        if (tM % GW == 0) {
            // panel walks m fastest (GW m-tiles wide), n advances inside
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
        const bool fetch = (KT_) + 2 < NT;                                    \
        LOAD_FRAGS(1, RS_, 1);                                                \
        if (fetch) {                                                          \
            ISSUE_A(aSrc, WS_, (KT_) + 2);                                    \
        }                                                                     \
        MMA_STEP(0);                                                          \
        LOAD_FRAGS(0, RS_, 2);                                                \
        MMA_STEP(1);                                                          \
        LOAD_FRAGS(1, RS_, 3);                                                \
        if (fetch) {                                                          \
            ISSUE_B(bSrc, WS_, (KT_) + 2);                                    \
        }                                                                     \
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
    tileSrc(blockIdx.x, aSrc, bSrc, bm, bn);

    ISSUE_A(aSrc, 0, 0);
    ISSUE_B(bSrc, 0, 0);
    cpCommit();
    ISSUE_A(aSrc, 1, 1);
    ISSUE_B(bSrc, 1, 1);
    cpCommit();

    for (int t = blockIdx.x;;) {
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

        cpWait<0>();
        __syncthreads();

        const int tNext = t + G;
        const bool more = tNext < numTiles;
        const int bmCur = bm, bnCur = bn;
        if (more) {
            tileSrc(tNext, aSrc, bSrc, bm, bn);
            ISSUE_A(aSrc, 0, 0);
            ISSUE_B(bSrc, 0, 0);
            cpCommit();
            ISSUE_A(aSrc, 1, 1);
            ISSUE_B(bSrc, 1, 1);
            cpCommit();
        }

        // epilogue: stage C (256x128) through the free A2 slot (32KB),
        // four chunks of 64 rows (chunk q owned by warp-row wm==q).
        const uint32_t epiBase = sA + 2 * A_STAGE;
        const int r0e = wm * 64 + (lane >> 2);
        const int c0e = wn * 64 + (lane & 3) * 2;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            if (wm == q) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
#pragma unroll
                    for (int jj = 0; jj < 8; ++jj) {
                        const float* c4 = acc[i][jj];
                        const uint32_t a0 =
                            epiBase + (((r0e & 63) + i * 16) * EPI_STRIDE +
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
                const int row = it * 16 + (tid >> 4);
                uint32_t v[4];
                ldsV4(v, epiBase + (row * EPI_STRIDE + (tid & 15) * 8) * 2);
                *reinterpret_cast<uint4*>(
                    &C[(uint32_t)(bmCur + q * 64 + row) * N + bnCur +
                       (tid & 15) * 8]) = *reinterpret_cast<uint4*>(v);
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

}  // namespace v21

PLAYGROUND_MATMUL_DEC(float16_t, 21, m, n, k, A, B, C)
{
    static int G = 0;
    if (G == 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        G = prop.multiProcessorCount;
        cudaFuncSetAttribute(v21::hgemmV21,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             v21::SMEM_BYTES);
    }
    const int numTiles = int((m / v21::BM) * (n / v21::BN));
    v21::hgemmV21<<<min(G, numTiles), 256, v21::SMEM_BYTES>>>(
        A, B, C, int(m), int(n), int(k));
}

}  // namespace playground
