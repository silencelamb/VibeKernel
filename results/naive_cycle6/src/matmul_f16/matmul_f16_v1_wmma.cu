#include "playground/matmul.hpp"

#include <cuda_fp16.h>
#include <mma.h>

namespace playground
{
using namespace nvcuda;

namespace
{

// Block tile / warp tile configuration for the WMMA baseline.
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 4;
constexpr int NUM_WARPS = WARPS_M * WARPS_N;   // 8
constexpr int NUM_THREADS = NUM_WARPS * 32;    // 256
constexpr int WM = BM / WARPS_M;               // 64
constexpr int WN = BN / WARPS_N;               // 32
constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
constexpr int WTILES_M = WM / WMMA_M;          // 4
constexpr int WTILES_N = WN / WMMA_N;          // 2

__global__ void __launch_bounds__(NUM_THREADS)
kernel_v1(int M, int N, int K, const half* __restrict__ A,
          const half* __restrict__ B, half* __restrict__ C)
{
    const int blockRow = blockIdx.y;   // along M
    const int blockCol = blockIdx.x;   // along N
    const int warpId = threadIdx.x / 32;
    const int warpRow = warpId / WARPS_N;   // 0..1
    const int warpCol = warpId % WARPS_N;   // 0..3
    const int tid = threadIdx.x;

    __shared__ half sA[BM * BK];   // row-major [BM][BK]
    __shared__ half sB[BK * BN];   // row-major [BK][BN]

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        acc[WTILES_M][WTILES_N];
#pragma unroll
    for (int i = 0; i < WTILES_M; i++)
#pragma unroll
        for (int j = 0; j < WTILES_N; j++)
            wmma::fill_fragment(acc[i][j], 0.0f);

    for (int kk = 0; kk < K; kk += BK) {
        // Load A[BM][BK] tile, vectorized as float4 (8 halfs each).
#pragma unroll
        for (int i = 0; i < (BM * BK / 8) / NUM_THREADS; i++) {
            int vec = tid + i * NUM_THREADS;       // 0..511
            int row = vec / (BK / 8);              // 0..127
            int col8 = (vec % (BK / 8)) * 8;       // 0,8,16,24
            const half* gptr = A + (blockRow * BM + row) * K + (kk + col8);
            *reinterpret_cast<float4*>(&sA[row * BK + col8]) =
                *reinterpret_cast<const float4*>(gptr);
        }
        // Load B[BK][BN] tile, vectorized as float4.
#pragma unroll
        for (int i = 0; i < (BK * BN / 8) / NUM_THREADS; i++) {
            int vec = tid + i * NUM_THREADS;       // 0..511
            int row = vec / (BN / 8);              // 0..31
            int col8 = (vec % (BN / 8)) * 8;       // 0..120
            const half* gptr = B + (kk + row) * N + (blockCol * BN + col8);
            *reinterpret_cast<float4*>(&sB[row * BN + col8]) =
                *reinterpret_cast<const float4*>(gptr);
        }
        __syncthreads();

#pragma unroll
        for (int kf = 0; kf < BK / WMMA_K; kf++) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                           wmma::row_major>
                a_frag[WTILES_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                           wmma::row_major>
                b_frag[WTILES_N];
#pragma unroll
            for (int i = 0; i < WTILES_M; i++) {
                int row = warpRow * WM + i * WMMA_M;
                wmma::load_matrix_sync(a_frag[i], &sA[row * BK + kf * WMMA_K],
                                       BK);
            }
#pragma unroll
            for (int j = 0; j < WTILES_N; j++) {
                int col = warpCol * WN + j * WMMA_N;
                wmma::load_matrix_sync(b_frag[j],
                                       &sB[(kf * WMMA_K) * BN + col], BN);
            }
#pragma unroll
            for (int i = 0; i < WTILES_M; i++)
#pragma unroll
                for (int j = 0; j < WTILES_N; j++)
                    wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }
        __syncthreads();
    }

    // Convert fp32 accumulators to half and store row-major to C.
#pragma unroll
    for (int i = 0; i < WTILES_M; i++) {
#pragma unroll
        for (int j = 0; j < WTILES_N; j++) {
            int row = blockRow * BM + warpRow * WM + i * WMMA_M;
            int col = blockCol * BN + warpCol * WN + j * WMMA_N;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half>
                c_half;
#pragma unroll
            for (int t = 0; t < acc[i][j].num_elements; t++)
                c_half.x[t] = __float2half(acc[i][j].x[t]);
            wmma::store_matrix_sync(&C[row * N + col], c_half, N,
                                    wmma::mem_row_major);
        }
    }
}

}   // namespace

PLAYGROUND_MATMUL_DEC(float16_t, 1, m, n, k, A, B, C)
{
    dim3 block(NUM_THREADS);
    dim3 grid(n / BN, m / BM);
    kernel_v1<<<grid, block>>>((int) m, (int) n, (int) k, A, B, C);
}

}   // namespace playground
