#include <cuda_runtime.h>
#include <mma.h>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{
using namespace nvcuda;

namespace
{
// Block tile: BM x BN = 128 x 128, BK = 32.
// 256 threads = 8 warps, warp grid 2 (M) x 4 (N) -> each warp owns 64 x 32.
// WMMA 16x16x16: per warp 4 (M) x 2 (N) = 8 accumulator fragments.
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int WARP_M = 2;
constexpr int WARP_N = 4;
constexpr int WM = BM / WARP_M;  // 64
constexpr int WN = BN / WARP_N;  // 32
constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
constexpr int WT_M = WM / WMMA_M;  // 4
constexpr int WT_N = WN / WMMA_N;  // 2
constexpr int APAD = 8;
constexpr int BPAD = 8;

__global__ void wmma_kernel(const half* __restrict__ A,
                            const half* __restrict__ B, half* __restrict__ C,
                            int M, int N, int K)
{
    __shared__ half As[BM][BK + APAD];
    __shared__ half Bs[BK][BN + BPAD];

    const int tid = threadIdx.x;
    const int warpId = tid >> 5;
    const int warpM = warpId / WARP_N;
    const int warpN = warpId % WARP_N;

    const int rowBase = blockIdx.y * BM;
    const int colBase = blockIdx.x * BN;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        c_frag[WT_M][WT_N];
#pragma unroll
    for (int i = 0; i < WT_M; ++i)
#pragma unroll
        for (int j = 0; j < WT_N; ++j)
            wmma::fill_fragment(c_frag[i][j], 0.0f);

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Load A tile (128x32) via float4.
#pragma unroll
        for (int i = tid; i < BM * BK / 8; i += blockDim.x) {
            int r = (i * 8) / BK;
            int c = (i * 8) % BK;
            *reinterpret_cast<float4*>(&As[r][c]) =
                *reinterpret_cast<const float4*>(&A[(rowBase + r) * K + k0 + c]);
        }
        // Load B tile (32x128) via float4.
#pragma unroll
        for (int i = tid; i < BK * BN / 8; i += blockDim.x) {
            int r = (i * 8) / BN;
            int c = (i * 8) % BN;
            *reinterpret_cast<float4*>(&Bs[r][c]) =
                *reinterpret_cast<const float4*>(&B[(k0 + r) * N + colBase + c]);
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                           wmma::row_major>
                a_frag[WT_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                           wmma::row_major>
                b_frag[WT_N];
#pragma unroll
            for (int i = 0; i < WT_M; ++i) {
                int aRow = warpM * WM + i * WMMA_M;
                wmma::load_matrix_sync(a_frag[i], &As[aRow][kk], BK + APAD);
            }
#pragma unroll
            for (int j = 0; j < WT_N; ++j) {
                int bCol = warpN * WN + j * WMMA_N;
                wmma::load_matrix_sync(b_frag[j], &Bs[kk][bCol], BN + BPAD);
            }
#pragma unroll
            for (int i = 0; i < WT_M; ++i)
#pragma unroll
                for (int j = 0; j < WT_N; ++j)
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j],
                                   c_frag[i][j]);
        }
        __syncthreads();
    }

    // Convert to half and store.
#pragma unroll
    for (int i = 0; i < WT_M; ++i) {
#pragma unroll
        for (int j = 0; j < WT_N; ++j) {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half>
                c_half;
#pragma unroll
            for (int t = 0; t < 8; ++t)
                c_half.x[t] = __float2half(c_frag[i][j].x[t]);
            int cRow = rowBase + warpM * WM + i * WMMA_M;
            int cCol = colBase + warpN * WN + j * WMMA_N;
            wmma::store_matrix_sync(&C[cRow * N + cCol], c_half, N,
                                    wmma::mem_row_major);
        }
    }
}

}  // namespace

PLAYGROUND_MATMUL_DEC(float16_t, 1, m, n, k, A, B, C)
{
    dim3 block(256);
    dim3 grid(n / BN, m / BM);
    wmma_kernel<<<grid, block>>>(A, B, C, int(m), int(n), int(k));
}

}  // namespace playground
