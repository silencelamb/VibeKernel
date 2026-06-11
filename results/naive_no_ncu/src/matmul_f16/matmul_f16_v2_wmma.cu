#include <cuda_runtime.h>
#include <mma.h>

#include "playground/matmul.hpp"

namespace playground
{
using namespace nvcuda;

namespace v2
{
// Block tile
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
// WMMA fragment shape
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
// Warp grid over the block tile: 2 (row) x 4 (col) = 8 warps = 256 threads.
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 4;
constexpr int WARP_TILE_M = BM / WARPS_M;  // 64
constexpr int WARP_TILE_N = BN / WARPS_N;  // 32
constexpr int WT_M = WARP_TILE_M / WMMA_M;  // 4
constexpr int WT_N = WARP_TILE_N / WMMA_N;  // 2
constexpr int THREADS = WARPS_M * WARPS_N * 32;  // 256

__global__ __launch_bounds__(THREADS) void wmmaKernel(int M, int N, int K,
                                                      const half* __restrict__ A,
                                                      const half* __restrict__ B,
                                                      half* __restrict__ C)
{
    __shared__ half As[BM][BK];
    __shared__ half Bs[BK][BN];

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    const int warpId = threadIdx.x / 32;
    const int warpRow = warpId / WARPS_N;  // 0..1
    const int warpCol = warpId % WARPS_N;  // 0..3

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        acc[WT_M][WT_N];
#pragma unroll
    for (int i = 0; i < WT_M; ++i)
#pragma unroll
        for (int j = 0; j < WT_N; ++j)
            wmma::fill_fragment(acc[i][j], 0.0f);

    const int tid = threadIdx.x;

    // Each thread loads via float4 (8 halfs). A tile = 128*32 = 4096 halfs =
    // 512 float4; B tile likewise. 256 threads -> 2 float4 each.
    const float4* Agl = reinterpret_cast<const float4*>(A);
    const float4* Bgl = reinterpret_cast<const float4*>(B);

    for (int k0 = 0; k0 < K; k0 += BK) {
        // ---- Load A tile (BM x BK), row-major in global, ld = K ----
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            int idx = tid + r * THREADS;       // 0..511
            int row = idx / (BK / 8);          // BK/8 = 4 float4 per row
            int col8 = (idx % (BK / 8)) * 8;   // 0,8,16,24
            int gRow = blockRow + row;
            int gCol = k0 + col8;
            reinterpret_cast<float4*>(&As[row][col8])[0] =
                Agl[(size_t(gRow) * K + gCol) / 8];
        }
        // ---- Load B tile (BK x BN), row-major in global, ld = N ----
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            int idx = tid + r * THREADS;       // 0..511
            int row = idx / (BN / 8);          // BN/8 = 16 float4 per row
            int col8 = (idx % (BN / 8)) * 8;
            int gRow = k0 + row;
            int gCol = blockCol + col8;
            reinterpret_cast<float4*>(&Bs[row][col8])[0] =
                Bgl[(size_t(gRow) * N + gCol) / 8];
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                           wmma::row_major>
                aFrag[WT_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                           wmma::row_major>
                bFrag[WT_N];
#pragma unroll
            for (int i = 0; i < WT_M; ++i) {
                int sRow = warpRow * WARP_TILE_M + i * WMMA_M;
                wmma::load_matrix_sync(aFrag[i], &As[sRow][kk], BK);
            }
#pragma unroll
            for (int j = 0; j < WT_N; ++j) {
                int sCol = warpCol * WARP_TILE_N + j * WMMA_N;
                wmma::load_matrix_sync(bFrag[j], &Bs[kk][sCol], BN);
            }
#pragma unroll
            for (int i = 0; i < WT_M; ++i)
#pragma unroll
                for (int j = 0; j < WT_N; ++j)
                    wmma::mma_sync(acc[i][j], aFrag[i], bFrag[j], acc[i][j]);
        }
        __syncthreads();
    }

    // ---- Epilogue: convert to half and store ----
#pragma unroll
    for (int i = 0; i < WT_M; ++i) {
#pragma unroll
        for (int j = 0; j < WT_N; ++j) {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half>
                cFrag;
#pragma unroll
            for (int t = 0; t < acc[i][j].num_elements; ++t)
                cFrag.x[t] = __float2half(acc[i][j].x[t]);
            int cRow = blockRow + warpRow * WARP_TILE_M + i * WMMA_M;
            int cCol = blockCol + warpCol * WARP_TILE_N + j * WMMA_N;
            wmma::store_matrix_sync(&C[size_t(cRow) * N + cCol], cFrag, N,
                                    wmma::mem_row_major);
        }
    }
}
}  // namespace v2

PLAYGROUND_MATMUL_DEC(float16_t, 2, m, n, k, A, B, C)
{
    using namespace v2;
    dim3 block(THREADS);
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    wmmaKernel<<<grid, block>>>(int(m), int(n), int(k), A, B, C);
}

}  // namespace playground
