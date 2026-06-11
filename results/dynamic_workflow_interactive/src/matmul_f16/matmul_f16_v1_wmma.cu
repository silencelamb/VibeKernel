// @file: task-1/src/matmul_f16/matmul_f16_v1_wmma.cu
//
// v1: Baseline Tensor-Core GEMM using the WMMA API.
//   - Block tile  BM x BN = 64 x 64, K-step BK = 16.
//   - 4 warps / block, each warp owns a 32x32 (2x2 WMMA-16x16) output tile.
//   - fp32 accumulation (same TC rate as fp16 accum on A100, better accuracy),
//     converted to fp16 on store.
// Goal: establish a correct, working baseline; later versions move to
// hand-written mma.sync + cp.async pipelines.

#include <cuda_fp16.h>
#include <mma.h>

#include "playground/matmul.hpp"

namespace playground
{
using namespace nvcuda;

namespace
{
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;

constexpr int WARP_M = 32;  // each warp: 2x2 WMMA fragments
constexpr int WARP_N = 32;
constexpr int WARPS_M = BM / WARP_M;            // 2
constexpr int WARPS_N = BN / WARP_N;            // 2
constexpr int NWARPS = WARPS_M * WARPS_N;       // 4
constexpr int NTHREADS = NWARPS * 32;           // 128

constexpr int APAD = 8;  // pad to reduce shared-memory bank conflicts
constexpr int BPAD = 8;

__global__ void __launch_bounds__(NTHREADS)
matmul_v1_kernel(int M, int N, int K, const half* __restrict__ A,
                 const half* __restrict__ B, half* __restrict__ C)
{
    __shared__ half As[BM][BK + APAD];
    __shared__ half Bs[BK][BN + BPAD];

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    const int tid = threadIdx.x;
    const int warpId = tid / 32;
    const int warpRow = warpId / WARPS_N;
    const int warpCol = warpId % WARPS_N;
    const int rowBase = warpRow * WARP_M;
    const int colBase = warpCol * WARP_N;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        cFrag[2][2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
            wmma::fill_fragment(cFrag[i][j], 0.0f);

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Load A[blockRow .. +BM][k0 .. +BK] -> As   (BM*BK = 1024 elems)
#pragma unroll
        for (int idx = tid; idx < BM * BK; idx += NTHREADS) {
            const int r = idx / BK;
            const int c = idx % BK;
            As[r][c] = A[(blockRow + r) * K + (k0 + c)];
        }
        // Load B[k0 .. +BK][blockCol .. +BN] -> Bs   (BK*BN = 1024 elems)
#pragma unroll
        for (int idx = tid; idx < BK * BN; idx += NTHREADS) {
            const int r = idx / BN;
            const int c = idx % BN;
            Bs[r][c] = B[(k0 + r) * N + (blockCol + c)];
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                       wmma::row_major>
            aFrag[2];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                       wmma::row_major>
            bFrag[2];
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
            wmma::load_matrix_sync(aFrag[mi], &As[rowBase + mi * WMMA_M][0],
                                   BK + APAD);
#pragma unroll
        for (int ni = 0; ni < 2; ++ni)
            wmma::load_matrix_sync(bFrag[ni], &Bs[0][colBase + ni * WMMA_N],
                                   BN + BPAD);
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
            for (int ni = 0; ni < 2; ++ni)
                wmma::mma_sync(cFrag[mi][ni], aFrag[mi], bFrag[ni],
                               cFrag[mi][ni]);
        __syncthreads();
    }

#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
        for (int ni = 0; ni < 2; ++ni) {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half>
                cHalf;
#pragma unroll
            for (int t = 0; t < cFrag[mi][ni].num_elements; ++t)
                cHalf.x[t] = __float2half(cFrag[mi][ni].x[t]);
            const int cRow = blockRow + rowBase + mi * WMMA_M;
            const int cCol = blockCol + colBase + ni * WMMA_N;
            wmma::store_matrix_sync(&C[cRow * N + cCol], cHalf, N,
                                    wmma::mem_row_major);
        }
    }
}
}  // namespace

PLAYGROUND_MATMUL_DEC(float16_t, 1, m, n, k, A, B, C)
{
    dim3 block(NTHREADS);
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    matmul_v1_kernel<<<grid, block>>>(int(m), int(n), int(k), A, B, C);
}

}  // namespace playground
