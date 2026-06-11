// @file: task-1/src/matmul_f16/matmul_f16_v8_sweep.cu
//
// v8: One configurable mma+cp.async GEMM used to sweep the tile/occupancy
//     design space. Pick a config with -DCFG=<id> (default below). All the
//     proven pieces from v3/v4/v7: padded smem (conflict-free ldmatrix),
//     NSTAGE cp.async pipeline, ldmatrix.x4 (A) + x4.trans (B), fp32 accum.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "playground/matmul.hpp"

#ifndef CFG
#define CFG 0
#endif

namespace playground
{
namespace
{
struct Config {
    int BM, BN, BK, WM_CNT, WN_CNT, NSTAGE;
};
// id:        BM   BN   BK  WMc WNc  ST
constexpr Config CFGS[] = {
    {128, 128, 32, 2, 4, 4},  // 0: = v4 (133), warp 64x32, 2 blk/SM
    {128, 128, 32, 4, 4, 4},  // 1: warp 32x32, 16 warps -> more occupancy
    {128, 128, 64, 2, 4, 4},  // 2: BK=64, K16=4, fewer barriers
    {256, 128, 32, 4, 2, 4},  // 3: warp 64x64, 16 warps (512 thr)
    {128, 128, 32, 4, 2, 4},  // 4: warp 32x64, 16 warps
    {128, 128, 32, 2, 4, 3},  // 5: 3 stages (more smem headroom)
    {256, 128, 32, 4, 4, 3},  // 6: big block, warp 64x32, 16 warps
    {128, 64, 32, 2, 2, 4},   // 7: warp 64x32, 4 warps small block
    {128, 128, 64, 4, 4, 3},  // 8: BK=64, warp 32x32, 16 warps
    {256, 64, 32, 4, 2, 4},   // 9: warp 64x32, 16 warps tall
};
constexpr Config CF = CFGS[CFG];
constexpr int BM = CF.BM;
constexpr int BN = CF.BN;
constexpr int BK = CF.BK;
constexpr int WM_CNT = CF.WM_CNT;
constexpr int WN_CNT = CF.WN_CNT;
constexpr int NSTAGE = CF.NSTAGE;
constexpr int NWARPS = WM_CNT * WN_CNT;
constexpr int NTHREADS = NWARPS * 32;
constexpr int WM = BM / WM_CNT;
constexpr int WN = BN / WN_CNT;
constexpr int WMITER = WM / 16;
constexpr int WNITER = WN / 8;
constexpr int K16 = BK / 16;
constexpr int AP = 8;
constexpr int ALDM = BK + AP;
constexpr int BLDM = BN + AP;
constexpr int A_STAGE = BM * ALDM;
constexpr int B_STAGE = BK * BLDM;
constexpr int SMEM_HALFS = NSTAGE * (A_STAGE + B_STAGE);
constexpr int VA = BM * BK / 8;  // 16B vectors for A tile
constexpr int VB = BK * BN / 8;  // 16B vectors for B tile

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
matmul_v8_kernel(int M, int N, int K, const half* __restrict__ A,
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
    constexpr int AVPR = BK / 8;  // A vectors per row
    constexpr int BVPR = BN / 8;  // B vectors per row
    constexpr int A_ITERS = VA / NTHREADS;  // compile-time -> full unroll
    constexpr int B_ITERS = VB / NTHREADS;
    static_assert(A_ITERS * NTHREADS == VA && B_ITERS * NTHREADS == VB,
                  "tile must divide evenly across threads");
    auto loadStage = [&](int s, int k0) {
#pragma unroll
        for (int i = 0; i < A_ITERS; ++i) {
            const int v = tid + i * NTHREADS;
            const int row = v / AVPR;
            const int col = (v % AVPR) * 8;
            cp_async16(AsP(s, row, col),
                       &A[(blockRow + row) * K + (k0 + col)]);
        }
#pragma unroll
        for (int i = 0; i < B_ITERS; ++i) {
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
        loadStage(s, s * BK);
        cp_commit();
    }

    int readStage = 0;
    int writeStage = NSTAGE - 1;

    for (int tile = 0; tile < numTiles; ++tile) {
        cp_wait<NSTAGE - 2>();
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
            for (int ni = 0; ni < WNITER; ni += 2) {
                uint32_t r[4];
                ldmatrix_x4_trans(
                    r, BsP(readStage, kB + (lane % 16),
                           warpN * WN + ni * 8 + (lane / 16) * 8));
                bFrag[ki][ni][0] = r[0];
                bFrag[ki][ni][1] = r[1];
                bFrag[ki][ni + 1][0] = r[2];
                bFrag[ki][ni + 1][1] = r[3];
            }
        }
#pragma unroll
        for (int ki = 0; ki < K16; ++ki)
#pragma unroll
            for (int mi = 0; mi < WMITER; ++mi)
#pragma unroll
                for (int ni = 0; ni < WNITER; ++ni)
                    mma_m16n8k16(acc[mi][ni], aFrag[ki][mi], bFrag[ki][ni]);

        const int nextTile = tile + (NSTAGE - 1);
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

PLAYGROUND_MATMUL_DEC(float16_t, 8, m, n, k, A, B, C)
{
    static bool attrSet = false;
    const int smemBytes = SMEM_HALFS * int(sizeof(half));
    if (!attrSet) {
        cudaFuncSetAttribute(matmul_v8_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smemBytes);
        attrSet = true;
    }
    dim3 block(NTHREADS);
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    matmul_v8_kernel<<<grid, block, smemBytes>>>(int(m), int(n), int(k), A, B,
                                                 C);
}

}  // namespace playground
