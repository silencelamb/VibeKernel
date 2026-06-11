// v17: 2 CTAs/SM with desynchronized barriers. 128 threads/block (4 warps,
// warp tile 64x64), BM=BN=128, BK=64, 2-stage cp.async (64KB smem/block, so
// two blocks co-reside; when one block sits at a barrier the other feeds the
// tensor cores). Persistent strided tiles + stage-0 prefetch in the epilogue.
#include <cstdint>
#include <cuda_runtime.h>

#include "playground/matmul.hpp"

namespace playground
{
namespace v17
{

constexpr int BM = 128, BN = 128, BK = 64;
constexpr int STAGES = 2;
constexpr uint32_t A_STAGE = BM * BK * sizeof(half);           // 16 KB
constexpr uint32_t B_STAGE = BK * BN * sizeof(half);           // 16 KB
constexpr uint32_t SMEM_BYTES = STAGES * (A_STAGE + B_STAGE);  // 64 KB
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

__global__ __launch_bounds__(128, 2) void hgemmV17(const half* __restrict__ A,
                                                   const half* __restrict__ B,
                                                   half* __restrict__ C, int M,
                                                   int N, int K)
{
    extern __shared__ __align__(128) uint8_t smemRaw[];
    const uint32_t sA = cvta(smemRaw);
    const uint32_t sB = sA + STAGES * A_STAGE;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;  // 0..3
    const int wm = warp >> 1;   // 0..1
    const int wn = warp & 1;    // 0..1

    constexpr int GW = 8;
    const int tM = M / BM, tN = N / BN;
    const int numTiles = tM * tN;
    const int G = gridDim.x;

    // A tile 128x64 (128B rows, 8 chunks): thread t -> rows {t>>3 + 16k},
    // chunk t&7. Swizzle c ^ (row&7).
    const int aRow = tid >> 3;  // 0..15
    const int aChunk = tid & 7;
    const uint32_t aDst = aRow * 128 + ((aChunk ^ (aRow & 7)) << 4);

    // B tile 64x128 (256B rows, 16 chunks): thread t -> rows {t>>4 + 8k},
    // chunk t&15. Swizzle (c&8) | ((c&7) ^ (k&7)).
    const int bRow = tid >> 4;  // 0..7
    const int bChunk = tid & 15;
    const uint32_t bDst =
        bRow * 256 + (((bChunk & 8) | ((bChunk & 7) ^ (bRow & 7))) << 4);

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

    // ldmatrix lane offsets (kk=0)
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

// 8 A-copies + 8 B-copies per stage, one thread each.
#define ISSUE_STAGE(SRC_A, SRC_B, S_, KT_)                                    \
    do {                                                                      \
        const half* ag = (SRC_A) + (KT_) * BK;                                \
        _Pragma("unroll") for (int r = 0; r < 8; ++r)                         \
        {                                                                     \
            cpAsync16(sA + (S_)*A_STAGE + aDst + r * 2048,                    \
                      ag + (uint32_t)(r * 16) * K);                           \
        }                                                                     \
        const half* bg = (SRC_B) + (uint32_t)((KT_) * BK) * N;                \
        _Pragma("unroll") for (int r = 0; r < 8; ++r)                         \
        {                                                                     \
            cpAsync16(sB + (S_)*B_STAGE + bDst + r * 2048,                    \
                      bg + (uint32_t)(r * 8) * N);                            \
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

// One BK=64 tile: 4 k16-steps. Stage to write = the slot being read (2-stage
// ping-pong), so the issue happens after the barrier at step 3.
#define TILE_ITER(KT_, RS_)                                                   \
    do {                                                                      \
        LOAD_FRAGS(1, RS_, 1);                                                \
        MMA_STEP(0);                                                          \
        LOAD_FRAGS(0, RS_, 2);                                                \
        MMA_STEP(1);                                                          \
        LOAD_FRAGS(1, RS_, 3);                                                \
        MMA_STEP(0);                                                          \
        cpWait<0>();                                                          \
        __syncthreads();                                                      \
        LOAD_FRAGS(0, (RS_) ^ 1, 0);                                          \
        if ((KT_) + 2 < NT) {                                                 \
            ISSUE_STAGE(aSrc, bSrc, RS_, (KT_) + 2);                          \
        }                                                                     \
        cpCommit();                                                           \
        MMA_STEP(1);                                                          \
    } while (0)

    const half *aSrc, *bSrc;
    int bm, bn;
    if ((int)blockIdx.x >= numTiles) {
        return;
    }
    tileSrc(blockIdx.x, aSrc, bSrc, bm, bn);

    // first tile: issue stages 0,1 here (later tiles: stage 0 prefetched
    // during the previous epilogue)
    ISSUE_STAGE(aSrc, bSrc, 0, 0);
    cpCommit();

    for (int t = blockIdx.x;;) {
        ISSUE_STAGE(aSrc, bSrc, 1, 1);
        cpCommit();
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
        for (; kt + 2 <= NT; kt += 2) {
            TILE_ITER(kt + 0, 0);
            TILE_ITER(kt + 1, 1);
        }
        if (kt < NT) {
            TILE_ITER(kt, 0);
        }

        cpWait<0>();
        __syncthreads();

        const int tNext = t + G;
        const bool more = tNext < numTiles;
        const int bmCur = bm, bnCur = bn;
        if (more) {
            tileSrc(tNext, aSrc, bSrc, bm, bn);
            ISSUE_STAGE(aSrc, bSrc, 0, 0);
            cpCommit();
        }

        // epilogue: stage C through the B1 slot (16KB; untouched by the
        // stage-0 prefetch) in four 32-row chunks.
        const uint32_t epiBase = sB + B_STAGE;
        const int r0e = wm * 64 + (lane >> 2);
        const int c0e = wn * 64 + (lane & 3) * 2;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
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
            // 32 rows x 256B; warp covers 2 rows per iter (16 lanes/row)
#pragma unroll
            for (int it = 0; it < 4; ++it) {
                const int row = it * 8 + (tid >> 4);
                uint32_t v[4];
                ldsV4(v, epiBase + (row * EPI_STRIDE + (tid & 15) * 8) * 2);
                *reinterpret_cast<uint4*>(
                    &C[(uint32_t)(bmCur + q * 32 + row) * N + bnCur +
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
#undef ISSUE_STAGE
#undef LOAD_FRAGS
#undef MMA_STEP
}

}  // namespace v17

PLAYGROUND_MATMUL_DEC(float16_t, 17, m, n, k, A, B, C)
{
    static int G = 0;
    if (G == 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        G = 2 * prop.multiProcessorCount;
        cudaFuncSetAttribute(v17::hgemmV17,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             v17::SMEM_BYTES);
    }
    const int numTiles = int((m / v17::BM) * (n / v17::BN));
    v17::hgemmV17<<<min(G, numTiles), 128, v17::SMEM_BYTES>>>(
        A, B, C, int(m), int(n), int(k));
}

}  // namespace playground
