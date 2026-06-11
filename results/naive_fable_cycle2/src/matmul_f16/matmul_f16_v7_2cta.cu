// v7: 2 CTAs/SM to overlap barriers (4 issue streams per SMSP).
// BM=BN=128 BK=32, 4-stage cp.async (64KB smem/block), 8 warps,
// warp tile 64x32, f16 accumulators (regs <= 128 so 2 blocks co-reside).
#include <cstdint>
#include <cuda_runtime.h>

#include "playground/matmul.hpp"

namespace playground
{
namespace v7
{

constexpr int BM = 128, BN = 128, BK = 32;
constexpr int STAGES = 4;
constexpr uint32_t A_STAGE = BM * BK * sizeof(half);           // 8 KB
constexpr uint32_t B_STAGE = BK * BN * sizeof(half);           // 8 KB
constexpr uint32_t SMEM_BYTES = STAGES * (A_STAGE + B_STAGE);  // 64 KB

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

// f16-accumulator mma: D(2x b32 = 4 half) = A*B + C
__device__ __forceinline__ void mma16816f16(uint32_t (&d)[2],
                                            const uint32_t (&a)[4],
                                            const uint32_t* b)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(d[0]), "+r"(d[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__global__ __launch_bounds__(256, 2) void hgemmV7(const half* __restrict__ A,
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
    const int wm = warp >> 2;  // 0..1
    const int wn = warp & 3;   // 0..3

    const int bm = blockIdx.y * BM;
    const int bn = blockIdx.x * BN;

    // ---- A tile 128x32 (rows = 64B, 4 chunks): thread t -> rows
    // (t>>2, +64), chunk t&3. Swizzle chunk ^ ((row>>1)&3).
    const int aRow = tid >> 2;
    const int aChunk = tid & 3;
    const half* aSrc = A + (uint32_t)(bm + aRow) * K + aChunk * 8;
    const uint32_t aDst = aRow * 64 + ((aChunk ^ ((aRow >> 1) & 3)) << 4);

    // ---- B tile 32x128 (rows = 256B, 16 chunks): thread t -> rows
    // (t>>4, +16), chunk t&15. Swizzle low3 ^ (k&7), keep bit3.
    const int bRow = tid >> 4;
    const int bChunk = tid & 15;
    const half* bSrc = B + (uint32_t)bRow * N + bn + bChunk * 8;
    const uint32_t bDst =
        bRow * 256 + (((bChunk & 8) | ((bChunk & 7) ^ (bRow & 7))) << 4);

    // ---- ldmatrix lane offsets (kk=0)
    uint32_t ldA[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = wm * 64 + i * 16 + (lane & 15);
        const int c = lane >> 4;
        ldA[i] = row * 64 + ((c ^ ((row >> 1) & 3)) << 4);  // kk=1: ^32
    }
    uint32_t ldB[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int kr = lane & 15;
        const int c = wn * 4 + j * 2 + (lane >> 4);
        ldB[j] = kr * 256 +
                 (((c & 8) | ((c & 7) ^ (kr & 7))) << 4);  // kk=1: +4096
    }

    uint32_t acc[4][4][2] = {};  // [m16 i][n8 jj][2x b32]
    uint32_t fragA[2][4][4];
    uint32_t fragB[2][2][4];

    const int NT = K / BK;

#define ISSUE_STAGE(S_, KT_)                                                  \
    do {                                                                      \
        const half* ag = aSrc + (KT_) * BK;                                   \
        cpAsync16(sA + (S_)*A_STAGE + aDst, ag);                              \
        cpAsync16(sA + (S_)*A_STAGE + aDst + 4096, ag + (uint32_t)64 * K);    \
        const half* bg = bSrc + (uint32_t)((KT_) * BK) * N;                   \
        cpAsync16(sB + (S_)*B_STAGE + bDst, bg);                              \
        cpAsync16(sB + (S_)*B_STAGE + bDst + 16 * 256,                        \
                  bg + (uint32_t)16 * N);                                     \
    } while (0)

#define LOAD_FRAGS(BUF_, S_, KK_)                                            \
    do {                                                                      \
        _Pragma("unroll") for (int i = 0; i < 4; ++i)                         \
        {                                                                     \
            ldsmX4(fragA[BUF_][i],                                            \
                   sA + (S_)*A_STAGE + (ldA[i] ^ ((KK_) << 5)));              \
        }                                                                     \
        _Pragma("unroll") for (int j = 0; j < 2; ++j)                         \
        {                                                                     \
            ldsmX4T(fragB[BUF_][j],                                           \
                    sB + (S_)*B_STAGE + ldB[j] + (KK_)*4096);                 \
        }                                                                     \
    } while (0)

#define MMA_STEP(BUF_)                                                        \
    do {                                                                      \
        _Pragma("unroll") for (int i = 0; i < 4; ++i)                         \
        {                                                                     \
            _Pragma("unroll") for (int j = 0; j < 2; ++j)                     \
            {                                                                 \
                mma16816f16(acc[i][2 * j], fragA[BUF_][i],                    \
                            &fragB[BUF_][j][0]);                              \
                mma16816f16(acc[i][2 * j + 1], fragA[BUF_][i],                \
                            &fragB[BUF_][j][2]);                              \
            }                                                                 \
        }                                                                     \
    } while (0)

    // ---- prologue
    ISSUE_STAGE(0, 0);
    cpCommit();
    ISSUE_STAGE(1, 1);
    cpCommit();
    ISSUE_STAGE(2, 2);
    cpCommit();
    cpWait<STAGES - 2>();
    __syncthreads();
    LOAD_FRAGS(0, 0, 0);

#define TILE_ITER(KT_, RS_, NS_, WS_)                                         \
    do {                                                                      \
        if ((KT_) + 3 < NT) {                                                 \
            ISSUE_STAGE(WS_, (KT_) + 3);                                      \
        }                                                                     \
        cpCommit();                                                           \
        LOAD_FRAGS(1, RS_, 1);                                                \
        MMA_STEP(0);                                                          \
        cpWait<STAGES - 2>();                                                 \
        __syncthreads();                                                      \
        LOAD_FRAGS(0, NS_, 0);                                                \
        MMA_STEP(1);                                                          \
    } while (0)

    // NT divisible by 4 for the 4096^3 target.
    for (int kt = 0; kt < NT; kt += 4) {
        TILE_ITER(kt + 0, 0, 1, 3);
        TILE_ITER(kt + 1, 1, 2, 0);
        TILE_ITER(kt + 2, 2, 3, 1);
        TILE_ITER(kt + 3, 3, 0, 2);
    }
#undef TILE_ITER
#undef ISSUE_STAGE
#undef LOAD_FRAGS
#undef MMA_STEP

    // ---- epilogue: acc already packed half2
    const int cRow = bm + wm * 64 + (lane >> 2);
    const int cCol = bn + wn * 32 + (lane & 3) * 2;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int jj = 0; jj < 4; ++jj) {
            const uint32_t r0 = (uint32_t)(cRow + i * 16) * N + cCol + jj * 8;
            *reinterpret_cast<uint32_t*>(&C[r0]) = acc[i][jj][0];
            *reinterpret_cast<uint32_t*>(&C[r0 + 8 * N]) = acc[i][jj][1];
        }
    }
}

}  // namespace v7

PLAYGROUND_MATMUL_DEC(float16_t, 7, m, n, k, A, B, C)
{
    static bool smemConfigured = false;
    if (!smemConfigured) {
        cudaFuncSetAttribute(v7::hgemmV7,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             v7::SMEM_BYTES);
        smemConfigured = true;
    }
    dim3 grid(n / v7::BN, m / v7::BM);
    v7::hgemmV7<<<grid, 256, v7::SMEM_BYTES>>>(A, B, C, int(m), int(n),
                                               int(k));
}

}  // namespace playground
