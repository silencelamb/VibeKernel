#include <cuda_fp16.h>
#include <mma.h>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{
using namespace nvcuda;

namespace v1
{
// Threadblock computes a BM x BN tile of C.
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;

// Warp layout: 2 warp-rows x 4 warp-cols = 8 warps (256 threads).
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 4;
constexpr int WM = BM / WARPS_M;  // 64
constexpr int WN = BN / WARPS_N;  // 32

// WMMA tile shape.
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int WTILES_M = WM / WMMA_M;  // 4
constexpr int WTILES_N = WN / WMMA_N;  // 2

constexpr int THREADS = WARPS_M * WARPS_N * 32;  // 256

__global__ __launch_bounds__(THREADS) void kernel(int m, int n, int k,
                                                  const half* __restrict__ A,
                                                  const half* __restrict__ B,
                                                  half* __restrict__ C)
{
    __shared__ half As[BM * BK];
    __shared__ half Bs[BK * BN];

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    const int tid = threadIdx.x;
    const int warpId = tid / 32;
    const int warpRow = warpId / WARPS_N;  // 0..1
    const int warpCol = warpId % WARPS_N;  // 0..3

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        acc[WTILES_M][WTILES_N];
#pragma unroll
    for (int i = 0; i < WTILES_M; ++i)
#pragma unroll
        for (int j = 0; j < WTILES_N; ++j)
            wmma::fill_fragment(acc[i][j], 0.0f);

    // Vectorized global->shared copy bookkeeping (float4 = 8 halfs).
    // As: BM x BK = 128 x 32 = 4096 halfs -> 512 float4 -> 2 per thread.
    // Bs: BK x BN = 32 x 128 = 4096 halfs -> 512 float4 -> 2 per thread.
    const float4* Av = reinterpret_cast<const float4*>(A);
    const float4* Bv = reinterpret_cast<const float4*>(B);

    for (int k0 = 0; k0 < k; k0 += BK) {
        // Load A tile: As[r][c], r in [0,128), c in [0,32)
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            int e = (p * THREADS + tid);          // float4 index in [0,512)
            int r = e / (BK / 8);                 // BK/8 = 4 float4 per row
            int c8 = (e % (BK / 8)) * 8;          // col base
            int gRow = blockRow + r;
            int gColV = (k0 + c8) / 8;
            reinterpret_cast<float4*>(As)[(r * BK + c8) / 8] =
                Av[(size_t(gRow) * k) / 8 + gColV];
        }
        // Load B tile: Bs[r][c], r in [0,32), c in [0,128)
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            int e = (p * THREADS + tid);          // float4 index in [0,512)
            int r = e / (BN / 8);                 // BN/8 = 16 float4 per row
            int c8 = (e % (BN / 8)) * 8;
            int gRow = k0 + r;
            int gColV = (blockCol + c8) / 8;
            reinterpret_cast<float4*>(Bs)[(r * BN + c8) / 8] =
                Bv[(size_t(gRow) * n) / 8 + gColV];
        }
        __syncthreads();

        // Compute: iterate over BK in WMMA_K steps.
#pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                           wmma::row_major>
                aFrag[WTILES_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                           wmma::row_major>
                bFrag[WTILES_N];
#pragma unroll
            for (int i = 0; i < WTILES_M; ++i) {
                int aRow = warpRow * WM + i * WMMA_M;
                wmma::load_matrix_sync(aFrag[i], &As[aRow * BK + kk], BK);
            }
#pragma unroll
            for (int j = 0; j < WTILES_N; ++j) {
                int bCol = warpCol * WN + j * WMMA_N;
                wmma::load_matrix_sync(bFrag[j], &Bs[kk * BN + bCol], BN);
            }
#pragma unroll
            for (int i = 0; i < WTILES_M; ++i)
#pragma unroll
                for (int j = 0; j < WTILES_N; ++j)
                    wmma::mma_sync(acc[i][j], aFrag[i], bFrag[j], acc[i][j]);
        }
        __syncthreads();
    }

    // Store accumulators to C (row-major), converting fp32->fp16.
#pragma unroll
    for (int i = 0; i < WTILES_M; ++i) {
#pragma unroll
        for (int j = 0; j < WTILES_N; ++j) {
            int cRow = blockRow + warpRow * WM + i * WMMA_M;
            int cCol = blockCol + warpCol * WN + j * WMMA_N;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half>
                cFrag;
#pragma unroll
            for (int t = 0; t < cFrag.num_elements; ++t)
                cFrag.x[t] = __float2half(acc[i][j].x[t]);
            wmma::store_matrix_sync(&C[cRow * n + cCol], cFrag, n,
                                    wmma::mem_row_major);
        }
    }
}
}  // namespace v1

PLAYGROUND_MATMUL_DEC(float16_t, 1, m, n, k, A, B, C)
{
    dim3 block(v1::THREADS);
    dim3 grid(n / v1::BN, m / v1::BM);
    v1::kernel<<<grid, block>>>(int(m), int(n), int(k), A, B, C);
}

}  // namespace playground
