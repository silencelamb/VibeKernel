#include <cuda_runtime.h>
#include <mma.h>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{
using namespace nvcuda;

namespace v1
{
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int WARPS_M = 4;
constexpr int WARPS_N = 2;
constexpr int WM = BM / WARPS_M;       // 32
constexpr int WN = BN / WARPS_N;       // 64
constexpr int WM_TILES = WM / 16;      // 2
constexpr int WN_TILES = WN / 16;      // 4
constexpr int BK_TILES = BK / 16;      // 2
constexpr int PAD = 8;                 // bank-conflict padding
constexpr int NTHREADS = WARPS_M * WARPS_N * 32;  // 256

__global__ void __launch_bounds__(NTHREADS)
matmul_kernel(int M, int N, int K, const half* __restrict__ A,
              const half* __restrict__ B, half* __restrict__ C)
{
    __shared__ half As[BM][BK + PAD];
    __shared__ half Bs[BK][BN + PAD];

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    const int warpId = threadIdx.x >> 5;
    const int warpRow = warpId / WARPS_N;  // 0..3
    const int warpCol = warpId % WARPS_N;  // 0..1
    const int tid = threadIdx.x;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[WM_TILES][WN_TILES];
#pragma unroll
    for (int i = 0; i < WM_TILES; i++)
#pragma unroll
        for (int j = 0; j < WN_TILES; j++)
            wmma::fill_fragment(acc[i][j], 0.0f);

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Load A tile: 128x32 halfs = 4096, 256 threads * 8(float4) => 2 each
#pragma unroll
        for (int i = 0; i < (BM * BK) / (NTHREADS * 8); i++) {
            int idx = (i * NTHREADS + tid) * 8;
            int r = idx / BK;
            int c = idx % BK;
            *reinterpret_cast<float4*>(&As[r][c]) =
                *reinterpret_cast<const float4*>(&A[(blockRow + r) * K + k0 + c]);
        }
#pragma unroll
        for (int i = 0; i < (BK * BN) / (NTHREADS * 8); i++) {
            int idx = (i * NTHREADS + tid) * 8;
            int r = idx / BN;
            int c = idx % BN;
            *reinterpret_cast<float4*>(&Bs[r][c]) =
                *reinterpret_cast<const float4*>(&B[(k0 + r) * N + blockCol + c]);
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK_TILES; kk++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[WM_TILES];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[WN_TILES];
#pragma unroll
            for (int i = 0; i < WM_TILES; i++)
                wmma::load_matrix_sync(a_frag[i], &As[warpRow * WM + i * 16][kk * 16], BK + PAD);
#pragma unroll
            for (int j = 0; j < WN_TILES; j++)
                wmma::load_matrix_sync(b_frag[j], &Bs[kk * 16][warpCol * WN + j * 16], BN + PAD);
#pragma unroll
            for (int i = 0; i < WM_TILES; i++)
#pragma unroll
                for (int j = 0; j < WN_TILES; j++)
                    wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < WM_TILES; i++)
#pragma unroll
        for (int j = 0; j < WN_TILES; j++) {
            int row = blockRow + warpRow * WM + i * 16;
            int col = blockCol + warpCol * WN + j * 16;
            wmma::fragment<wmma::accumulator, 16, 16, 16, half> acc_h;
#pragma unroll
            for (int t = 0; t < acc[i][j].num_elements; t++)
                acc_h.x[t] = __float2half(acc[i][j].x[t]);
            wmma::store_matrix_sync(&C[row * N + col], acc_h, N, wmma::mem_row_major);
        }
}
}  // namespace v1

PLAYGROUND_MATMUL_DEC(float16_t, 1, m, n, k, A, B, C)
{
    dim3 block(v1::NTHREADS);
    dim3 grid(n / v1::BN, m / v1::BM);
    v1::matmul_kernel<<<grid, block>>>(int(m), int(n), int(k), A, B, C);
}

}  // namespace playground
