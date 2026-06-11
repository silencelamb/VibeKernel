// @file: task-1/src/matmul_f16/matmul_f16_v4_pipeline.cu
//
// v4: v3 + deep multi-stage cp.async software pipeline (dynamic smem).
//   At 12.5% occupancy a 2-stage pipeline can't hide global-load latency, so
//   the Tensor Cores starve (40% util). A NSTAGE-deep pipeline prefetches
//   NSTAGE-1 tiles ahead, hiding latency the way CUTLASS does at low occupancy.
//
// Tiling: block 128x128, BK=32, 8 warps (2x4), warp tile 64x32.
//   mma.m16n8k16 / ldmatrix.x4 / .x2.trans, fp32 accum, padded smem.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "playground/matmul.hpp"

namespace playground
{
namespace
{
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int NTHREADS = 256;
constexpr int WN_CNT = 4;
constexpr int WM_CNT = 2;
constexpr int WM = BM / WM_CNT;   // 64
constexpr int WN = BN / WN_CNT;   // 32
constexpr int WMITER = WM / 16;   // 4
constexpr int WNITER = WN / 8;    // 4
constexpr int K16 = BK / 16;      // 2
constexpr int AP = 8;             // smem pad (fp16)
constexpr int ALDM = BK + AP;     // 40
constexpr int BLDM = BN + AP;     // 136
constexpr int NSTAGE = 4;         // pipeline depth

constexpr int A_STAGE = BM * ALDM;  // halfs per A stage
constexpr int B_STAGE = BK * BLDM;  // halfs per B stage
constexpr int SMEM_HALFS = NSTAGE * (A_STAGE + B_STAGE);

// ---- PTX helpers -----------------------------------------------------------
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
__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t (&r)[2],
                                                  const void* smem)
{
    uint32_t s = smem_u32(smem);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1])
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
matmul_v4_kernel(int M, int N, int K, const half* __restrict__ A,
                 const half* __restrict__ B, half* __restrict__ C)
{
    extern __shared__ half smem[];
    half* As = smem;                       // NSTAGE * A_STAGE
    half* Bs = smem + NSTAGE * A_STAGE;    // NSTAGE * B_STAGE

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

    auto loadStage = [&](int s, int k0) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int v = tid + i * NTHREADS;
            const int row = v / 4;
            const int col = (v % 4) * 8;
            cp_async16(AsP(s, row, col),
                       &A[(blockRow + row) * K + (k0 + col)]);
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int v = tid + i * NTHREADS;
            const int row = v / 16;
            const int col = (v % 16) * 8;
            cp_async16(BsP(s, row, col),
                       &B[(k0 + row) * N + (blockCol + col)]);
        }
    };

    const int numTiles = K / BK;

    // Prologue: issue NSTAGE-1 prefetches.
#pragma unroll
    for (int s = 0; s < NSTAGE - 1; ++s) {
        loadStage(s, s * BK);
        cp_commit();
    }

    int readStage = 0;
    int writeStage = NSTAGE - 1;

    for (int tile = 0; tile < numTiles; ++tile) {
        cp_wait<NSTAGE - 2>();  // current readStage ready
        __syncthreads();

        uint32_t aFrag[K16][WMITER][4];
        uint32_t bFrag[K16][WNITER][2];
#pragma unroll
        for (int ki = 0; ki < K16; ++ki) {
            const int kB = ki * 16;
#pragma unroll
            for (int mi = 0; mi < WMITER; ++mi)
                ldmatrix_x4(aFrag[ki][mi],
                            AsP(readStage, warpM * WM + mi * 16 + (lane % 16),
                                kB + (lane / 16) * 8));
#pragma unroll
            for (int ni = 0; ni < WNITER; ++ni)
                ldmatrix_x2_trans(bFrag[ki][ni],
                                  BsP(readStage, kB + (lane % 16),
                                      warpN * WN + ni * 8));
        }
#pragma unroll
        for (int ki = 0; ki < K16; ++ki)
#pragma unroll
            for (int mi = 0; mi < WMITER; ++mi)
#pragma unroll
                for (int ni = 0; ni < WNITER; ++ni)
                    mma_m16n8k16(acc[mi][ni], aFrag[ki][mi], bFrag[ki][ni]);

        // Prefetch a future tile into the freed stage.
        const int nextTile = tile + (NSTAGE - 1);
        __syncthreads();  // all warps done reading readStage region reuse
        if (nextTile < numTiles) {
            loadStage(writeStage, nextTile * BK);
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

PLAYGROUND_MATMUL_DEC(float16_t, 4, m, n, k, A, B, C)
{
    static bool attrSet = false;
    const int smemBytes = SMEM_HALFS * int(sizeof(half));
    if (!attrSet) {
        cudaFuncSetAttribute(matmul_v4_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smemBytes);
        attrSet = true;
    }
    dim3 block(NTHREADS);
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    matmul_v4_kernel<<<grid, block, smemBytes>>>(int(m), int(n), int(k), A, B,
                                                 C);
}

}  // namespace playground
