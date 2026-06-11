// fp16 GEMM v1 — WMMA tiled baseline (Tensor Core, f32 accumulate).
// Row-major A(m,k) * B(k,n) = C(m,n). Correctness anchor for later versions.
#include <cuda_fp16.h>
#include <mma.h>

#include "playground/matmul.hpp"

namespace playground
{
namespace v1
{
using namespace nvcuda;

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 16;
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 4;
constexpr int NWARPS = WARPS_M * WARPS_N;  // 8
constexpr int NTHREADS = NWARPS * 32;      // 256
constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
constexpr int WM = BM / WARPS_M;  // 64
constexpr int WN = BN / WARPS_N;  // 32
constexpr int FM = WM / WMMA_M;   // 4
constexpr int FN = WN / WMMA_N;   // 2

__global__ void __launch_bounds__(NTHREADS)
    gemm(const half* __restrict__ A, const half* __restrict__ B,
         half* __restrict__ C, int M, int N, int K)
{
    __shared__ half sA[BM][BK];
    __shared__ half sB[BK][BN];

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    const int warpId = threadIdx.x / 32;
    const int warpRow = warpId / WARPS_N;  // [0,WARPS_M)
    const int warpCol = warpId % WARPS_N;  // [0,WARPS_N)
    const int tid = threadIdx.x;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[FM][FN];
#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j)
            wmma::fill_fragment(acc[i][j], 0.0f);

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Load A tile: 128x16 halves = 2048, 256 threads * float4(8) = 2048.
        {
            const int row = tid / 2;
            const int col = (tid % 2) * 8;
            *(float4*) &sA[row][col] =
                *(const float4*) &A[(blockRow + row) * K + (k0 + col)];
        }
        // Load B tile: 16x128 halves = 2048.
        {
            const int row = tid / 16;
            const int col = (tid % 16) * 8;
            *(float4*) &sB[row][col] =
                *(const float4*) &B[(k0 + row) * N + (blockCol + col)];
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                       wmma::row_major>
            fragA[FM];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                       wmma::row_major>
            fragB[FN];
#pragma unroll
        for (int i = 0; i < FM; ++i)
            wmma::load_matrix_sync(fragA[i], &sA[warpRow * WM + i * WMMA_M][0],
                                   BK);
#pragma unroll
        for (int j = 0; j < FN; ++j)
            wmma::load_matrix_sync(fragB[j], &sB[0][warpCol * WN + j * WMMA_N],
                                   BN);
#pragma unroll
        for (int i = 0; i < FM; ++i)
#pragma unroll
            for (int j = 0; j < FN; ++j)
                wmma::mma_sync(acc[i][j], fragA[i], fragB[j], acc[i][j]);
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j) {
            const int cRow = blockRow + warpRow * WM + i * WMMA_M;
            const int cCol = blockCol + warpCol * WN + j * WMMA_N;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> accH;
#pragma unroll
            for (int t = 0; t < acc[i][j].num_elements; ++t)
                accH.x[t] = __float2half(acc[i][j].x[t]);
            wmma::store_matrix_sync(&C[cRow * N + cCol], accH, N,
                                    wmma::mem_row_major);
        }
}
}  // namespace v1

PLAYGROUND_MATMUL_DEC(float16_t, 1, m, n, k, A, B, C)
{
    dim3 block(v1::NTHREADS);
    dim3 grid(n / v1::BN, m / v1::BM);
    v1::gemm<<<grid, block>>>(A, B, C, int(m), int(n), int(k));
}

}  // namespace playground
