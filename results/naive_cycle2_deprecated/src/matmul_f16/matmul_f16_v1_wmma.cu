#include <cuda_fp16.h>
#include <mma.h>

#include "playground/matmul.hpp"

namespace playground
{
using namespace nvcuda;

namespace v1impl
{
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int WM = 64;  // warp tile rows (M)
constexpr int WN = 32;  // warp tile cols (N)
constexpr int WARPS_M = BM / WM;        // 2
constexpr int WARPS_N = BN / WN;        // 4
constexpr int NWARPS = WARPS_M * WARPS_N;  // 8
constexpr int NTHREADS = NWARPS * 32;   // 256

__global__ void __launch_bounds__(NTHREADS)
    kernel_v1(int M, int N, int K, const half* __restrict__ A,
              const half* __restrict__ B, half* __restrict__ C)
{
    __shared__ half As[BM][BK];
    __shared__ half Bs[BK][BN];

    const int blockRow = blockIdx.y;  // along M
    const int blockCol = blockIdx.x;  // along N
    const int warpId = threadIdx.x / 32;
    const int warpRow = warpId / WARPS_N;  // 0..1
    const int warpCol = warpId % WARPS_N;  // 0..3
    const int tid = threadIdx.x;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4][2];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 2; j++)
            wmma::fill_fragment(acc[i][j], 0.0f);

    const half* Ablock = A + blockRow * BM * K;
    const half* Bblock = B + blockCol * BN;

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Load A tile (BM x BK), row-major, vectorized by float4 (8 halfs)
#pragma unroll
        for (int idx = tid; idx < BM * BK / 8; idx += NTHREADS) {
            int r = (idx * 8) / BK;
            int c = (idx * 8) % BK;
            *reinterpret_cast<float4*>(&As[r][c]) =
                *reinterpret_cast<const float4*>(&Ablock[r * K + k0 + c]);
        }
        // Load B tile (BK x BN), row-major
#pragma unroll
        for (int idx = tid; idx < BK * BN / 8; idx += NTHREADS) {
            int r = (idx * 8) / BN;
            int c = (idx * 8) % BN;
            *reinterpret_cast<float4*>(&Bs[r][c]) =
                *reinterpret_cast<const float4*>(&Bblock[(k0 + r) * N + c]);
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>
                a_frag[4];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>
                b_frag[2];
#pragma unroll
            for (int i = 0; i < 4; i++)
                wmma::load_matrix_sync(a_frag[i], &As[warpRow * WM + i * 16][kk],
                                       BK);
#pragma unroll
            for (int j = 0; j < 2; j++)
                wmma::load_matrix_sync(b_frag[j], &Bs[kk][warpCol * WN + j * 16],
                                       BN);
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 2; j++)
                    wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }
        __syncthreads();
    }

    half* Cblock = C + (blockRow * BM) * N + blockCol * BN;
#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < 2; j++) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, half> hc;
#pragma unroll
            for (int t = 0; t < acc[i][j].num_elements; t++)
                hc.x[t] = __float2half(acc[i][j].x[t]);
            wmma::store_matrix_sync(
                &Cblock[(warpRow * WM + i * 16) * N + warpCol * WN + j * 16], hc,
                N, wmma::mem_row_major);
        }
    }
}
}  // namespace v1impl

PLAYGROUND_MATMUL_DEC(float16_t, 1, m, n, k, A, B, C)
{
    using namespace v1impl;
    dim3 block(NTHREADS);
    dim3 grid(n / BN, m / BM);
    kernel_v1<<<grid, block>>>((int) m, (int) n, (int) k, A, B, C);
}

}  // namespace playground
