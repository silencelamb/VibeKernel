#include <cuda_fp16.h>
#include <mma.h>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{
namespace v1
{
using namespace nvcuda;

// Block tile computed per CTA.
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;

// WMMA fragment shape (fp16 in, fp32 accumulate).
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Warp grid: 8 warps arranged 2 (M) x 4 (N).
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 4;
constexpr int NUM_WARPS = WARPS_M * WARPS_N;  // 8
constexpr int NUM_THREADS = NUM_WARPS * 32;   // 256

// Per-warp output tile.
constexpr int WM = BM / WARPS_M;  // 64
constexpr int WN = BN / WARPS_N;  // 32
constexpr int WMITER = WM / WMMA_M;  // 4
constexpr int WNITER = WN / WMMA_N;  // 2

// Shared-memory padding (halfs) to dodge bank conflicts; keeps 16B alignment.
constexpr int SKEW = 8;

__global__ __launch_bounds__(NUM_THREADS) void wmma_kernel(
    int M, int N, int K, const half* __restrict__ A, const half* __restrict__ B,
    half* __restrict__ C)
{
    __shared__ half As[BM][BK + SKEW];
    __shared__ half Bs[BK][BN + SKEW];

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    const int warpId = threadIdx.x / 32;
    const int warpRow = warpId / WARPS_N;  // 0..1
    const int warpCol = warpId % WARPS_N;  // 0..3

    // Accumulator fragments.
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        acc[WMITER][WNITER];
    for (int i = 0; i < WMITER; ++i) {
        for (int j = 0; j < WNITER; ++j) {
            wmma::fill_fragment(acc[i][j], 0.0f);
        }
    }

    const int tid = threadIdx.x;

    for (int kk = 0; kk < K; kk += BK) {
        // Load A tile (BM x BK) into shared, vectorized as float4 (8 halfs).
        // 128*32/8 = 512 float4 loads, 256 threads -> 2 each.
        for (int l = tid; l < (BM * BK) / 8; l += NUM_THREADS) {
            int row = l / (BK / 8);
            int col = (l % (BK / 8)) * 8;
            const float4* src = reinterpret_cast<const float4*>(
                &A[(blockRow + row) * K + kk + col]);
            *reinterpret_cast<float4*>(&As[row][col]) = *src;
        }
        // Load B tile (BK x BN) into shared.
        // 32*128/8 = 512 float4 loads.
        for (int l = tid; l < (BK * BN) / 8; l += NUM_THREADS) {
            int row = l / (BN / 8);
            int col = (l % (BN / 8)) * 8;
            const float4* src = reinterpret_cast<const float4*>(
                &B[(kk + row) * N + blockCol + col]);
            *reinterpret_cast<float4*>(&Bs[row][col]) = *src;
        }

        __syncthreads();

        // Compute over BK in steps of WMMA_K.
#pragma unroll
        for (int ks = 0; ks < BK; ks += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                           wmma::row_major>
                aFrag[WMITER];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                           wmma::row_major>
                bFrag[WNITER];
#pragma unroll
            for (int i = 0; i < WMITER; ++i) {
                int aRow = warpRow * WM + i * WMMA_M;
                wmma::load_matrix_sync(aFrag[i], &As[aRow][ks], BK + SKEW);
            }
#pragma unroll
            for (int j = 0; j < WNITER; ++j) {
                int bCol = warpCol * WN + j * WMMA_N;
                wmma::load_matrix_sync(bFrag[j], &Bs[ks][bCol], BN + SKEW);
            }
#pragma unroll
            for (int i = 0; i < WMITER; ++i) {
#pragma unroll
                for (int j = 0; j < WNITER; ++j) {
                    wmma::mma_sync(acc[i][j], aFrag[i], bFrag[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }

    // Store accumulators to C (convert fp32 -> fp16 via a half fragment).
#pragma unroll
    for (int i = 0; i < WMITER; ++i) {
#pragma unroll
        for (int j = 0; j < WNITER; ++j) {
            int cRow = blockRow + warpRow * WM + i * WMMA_M;
            int cCol = blockCol + warpCol * WN + j * WMMA_N;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half>
                cFrag;
#pragma unroll
            for (int t = 0; t < cFrag.num_elements; ++t) {
                cFrag.x[t] = __float2half(acc[i][j].x[t]);
            }
            wmma::store_matrix_sync(&C[cRow * N + cCol], cFrag, N,
                                    wmma::mem_row_major);
        }
    }
}
}  // namespace v1

PLAYGROUND_MATMUL_DEC(float16_t, 1, m, n, k, A, B, C)
{
    dim3 block(v1::NUM_THREADS);
    dim3 grid(n / v1::BN, m / v1::BM);
    v1::wmma_kernel<<<grid, block>>>(int(m), int(n), int(k), A, B, C);
}

}  // namespace playground
