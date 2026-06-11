// @file: task-1/src/matmul_f16/matmul_f16_v11_split.cu
//
// v11: clean fp32-accumulate primary kernel — the converged best of the
//      v8/v9 design-space sweeps. ~158 TFLOPS, error ~3e-5 (stable).
//   Block 256x128, BK=64, 8 warps (4x2) -> warp tile 64x64.
//   mma.m16n8k16.f32 (fp32 accum), ldmatrix.x4 (A) / x4.trans (B), 3-stage
//   cp.async pipeline, interleaved inner-K mainloop (ldmatrix(ki+1) overlaps
//   mma(ki)), padded conflict-free smem, next-tile prefetch issued on the last
//   inner step so cp.async overlaps the final mma group.
//   Profiled limit: latency-bound at 8 warps/SM (1 block/SM; 64x64 warp tile's
//   128 fp32 accum regs + 159KB smem pin occupancy). loadRange()'s a0/a1/b0/b1
//   range args are kept for experimentation though the whole tile is prefetched
//   at once here (splitting it across ki measured ~2% slower).

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "playground/matmul.hpp"

namespace playground
{
namespace
{
constexpr int BM = 256;
constexpr int BN = 128;
constexpr int BK = 64;
constexpr int WM_CNT = 4;   // warp grid M
constexpr int WN_CNT = 2;   // warp grid N
constexpr int NSTAGE = 3;
constexpr int NWARPS = WM_CNT * WN_CNT;  // 8
constexpr int NTHREADS = NWARPS * 32;    // 256
constexpr int WM = BM / WM_CNT;          // 64
constexpr int WN = BN / WN_CNT;          // 64
constexpr int WMITER = WM / 16;          // 4
constexpr int WNITER = WN / 8;           // 8
constexpr int K16 = BK / 16;             // 4
constexpr int AP = 8;
constexpr int ALDM = BK + AP;            // 72
constexpr int BLDM = BN + AP;            // 136
constexpr int A_STAGE = BM * ALDM;
constexpr int B_STAGE = BK * BLDM;
constexpr int SMEM_HALFS = NSTAGE * (A_STAGE + B_STAGE);
constexpr int VA = BM * BK / 8;          // 2048 vectors
constexpr int VB = BK * BN / 8;          // 1024 vectors
constexpr int A_ITERS = VA / NTHREADS;   // 8
constexpr int B_ITERS = VB / NTHREADS;   // 4
constexpr int AVPR = BK / 8;             // 8
constexpr int BVPR = BN / 8;             // 16
constexpr int APP = A_ITERS / K16;       // 2 A-loads per ki
constexpr int BPP = B_ITERS / K16;       // 1 B-load per ki
static_assert(APP * K16 == A_ITERS && BPP * K16 == B_ITERS, "split divisible");

__device__ __forceinline__ uint32_t smem_u32(const void* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem)
{
    uint32_t s = smem_u32(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s),
                 "l"(gmem));
}
__device__ __forceinline__ void cp_commit()
{
    asm volatile("cp.async.commit_group;\n");
}
template <int N>
__device__ __forceinline__ void cp_wait()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* smem)
{
    uint32_t s = smem_u32(smem);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(s));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4],
                                                  const void* smem)
{
    uint32_t s = smem_u32(smem);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(s));
}
__device__ __forceinline__ void mma_m16n8k16(float (&acc)[4],
                                             const uint32_t (&a)[4],
                                             const uint32_t (&b)[2])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__global__ void __launch_bounds__(NTHREADS)
matmul_v11_kernel(int M, int N, int K, const half* __restrict__ A,
                  const half* __restrict__ B, half* __restrict__ C)
{
    extern __shared__ half smem[];
    half* As = smem;
    half* Bs = smem + NSTAGE * A_STAGE;

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int warpId = tid / 32;
    const int lane = tid % 32;
    const int warpM = warpId / WN_CNT;
    const int warpN = warpId % WN_CNT;

    float acc[WMITER][WNITER][4];
#pragma unroll
    for (int i = 0; i < WMITER; ++i)
#pragma unroll
        for (int j = 0; j < WNITER; ++j)
#pragma unroll
            for (int t = 0; t < 4; ++t) acc[i][j][t] = 0.0f;

    auto AsP = [&](int s, int m, int k) -> half* {
        return &As[s * A_STAGE + m * ALDM + k];
    };
    auto BsP = [&](int s, int k, int n) -> half* {
        return &Bs[s * B_STAGE + k * BLDM + n];
    };
    // Issue the A-load range [a0,a1) and B-load range [b0,b1) of one tile.
    auto loadRange = [&](int s, int k0, int a0, int a1, int b0, int b1) {
#pragma unroll
        for (int i = a0; i < a1; ++i) {
            const int v = tid + i * NTHREADS;
            const int row = v / AVPR;
            const int col = (v % AVPR) * 8;
            cp_async16(AsP(s, row, col),
                       &A[(blockRow + row) * K + (k0 + col)]);
        }
#pragma unroll
        for (int i = b0; i < b1; ++i) {
            const int v = tid + i * NTHREADS;
            const int row = v / BVPR;
            const int col = (v % BVPR) * 8;
            cp_async16(BsP(s, row, col),
                       &B[(k0 + row) * N + (blockCol + col)]);
        }
    };

    const int numTiles = K / BK;
#pragma unroll
    for (int s = 0; s < NSTAGE - 1; ++s) {
        loadRange(s, s * BK, 0, A_ITERS, 0, B_ITERS);
        cp_commit();
    }

    int readStage = 0;
    int writeStage = NSTAGE - 1;

    uint32_t aF[2][WMITER][4];
    uint32_t bF[2][WNITER][2];
    auto loadKi = [&](int buf, int stage, int ki) {
        const int kB = ki * 16;
#pragma unroll
        for (int mi = 0; mi < WMITER; ++mi)
            ldmatrix_x4(aF[buf][mi],
                        AsP(stage, warpM * WM + mi * 16 + (lane % 16),
                            kB + (lane / 16) * 8));
#pragma unroll
        for (int ni = 0; ni < WNITER; ni += 2) {
            uint32_t r[4];
            ldmatrix_x4_trans(r, BsP(stage, kB + (lane % 16),
                                     warpN * WN + ni * 8 + (lane / 16) * 8));
            bF[buf][ni][0] = r[0];
            bF[buf][ni][1] = r[1];
            bF[buf][ni + 1][0] = r[2];
            bF[buf][ni + 1][1] = r[3];
        }
    };

    for (int tile = 0; tile < numTiles; ++tile) {
        cp_wait<NSTAGE - 2>();
        __syncthreads();

        const int nextTile = tile + (NSTAGE - 1);
        const bool doPrefetch = nextTile < numTiles;

        loadKi(0, readStage, 0);
#pragma unroll
        for (int ki = 0; ki < K16; ++ki) {
            const int cur = ki & 1;
            if (ki + 1 < K16) loadKi((ki + 1) & 1, readStage, ki + 1);
            // Issue the whole next-tile prefetch on the last inner step so the
            // cp.async issue overlaps this final mma group (splitting it across
            // ki measured slightly slower).
            if (ki == K16 - 1 && doPrefetch)
                loadRange(writeStage, nextTile * BK, 0, A_ITERS, 0, B_ITERS);
#pragma unroll
            for (int ni = 0; ni < WNITER; ++ni)
#pragma unroll
                for (int mi = 0; mi < WMITER; ++mi)
                    mma_m16n8k16(acc[mi][ni], aF[cur][mi], bF[cur][ni]);
        }
        cp_commit();
        readStage = (readStage + 1) % NSTAGE;
        writeStage = (writeStage + 1) % NSTAGE;
    }

#pragma unroll
    for (int mi = 0; mi < WMITER; ++mi) {
#pragma unroll
        for (int ni = 0; ni < WNITER; ++ni) {
            const int baseM = blockRow + warpM * WM + mi * 16;
            const int baseN = blockCol + warpN * WN + ni * 8;
            const int row0 = baseM + lane / 4;
            const int col0 = baseN + (lane % 4) * 2;
            C[row0 * N + col0] = __float2half(acc[mi][ni][0]);
            C[row0 * N + col0 + 1] = __float2half(acc[mi][ni][1]);
            C[(row0 + 8) * N + col0] = __float2half(acc[mi][ni][2]);
            C[(row0 + 8) * N + col0 + 1] = __float2half(acc[mi][ni][3]);
        }
    }
}
}  // namespace

PLAYGROUND_MATMUL_DEC(float16_t, 11, m, n, k, A, B, C)
{
    static bool attrSet = false;
    const int smemBytes = SMEM_HALFS * int(sizeof(half));
    if (!attrSet) {
        cudaFuncSetAttribute(matmul_v11_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smemBytes);
        attrSet = true;
    }
    dim3 block(NTHREADS);
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    matmul_v11_kernel<<<grid, block, smemBytes>>>(int(m), int(n), int(k), A, B,
                                                  C);
}

}  // namespace playground
