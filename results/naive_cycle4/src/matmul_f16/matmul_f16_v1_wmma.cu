#include <cuda_fp16.h>
#include <mma.h>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{
using namespace nvcuda;

namespace v1
{
// Block tile
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
// Warp tile: 8 warps arranged 2 (rows) x 4 (cols)
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 4;
constexpr int WM = BM / WARPS_M;  // 64
constexpr int WN = BN / WARPS_N;  // 32
// WMMA fragment shape
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int FRAG_M = WM / WMMA_M;  // 4
constexpr int FRAG_N = WN / WMMA_N;  // 2

constexpr int NTHREADS = WARPS_M * WARPS_N * 32;  // 256
constexpr int PAD = 8;                            // skew to reduce bank conflicts
constexpr int LDA_S = BK + PAD;                   // shared A leading dim
constexpr int LDB_S = BN + PAD;                   // shared B leading dim

__global__ void __launch_bounds__(NTHREADS)
    matmul_kernel(int M, int N, int K, const half* __restrict__ A,
                  const half* __restrict__ B, half* __restrict__ C)
{
    __shared__ half As[BM * LDA_S];
    __shared__ half Bs[BK * LDB_S];

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    const int warpId = threadIdx.x / 32;
    const int warpRow = (warpId / WARPS_N) * WM;  // row offset within block tile
    const int warpCol = (warpId % WARPS_N) * WN;  // col offset within block tile

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half>
        acc[FRAG_M][FRAG_N];
#pragma unroll
    for (int i = 0; i < FRAG_M; ++i)
#pragma unroll
        for (int j = 0; j < FRAG_N; ++j)
            wmma::fill_fragment(acc[i][j], __float2half(0.0f));

    // Vectorized global -> shared loads (float4 = 8 halves)
    const int tid = threadIdx.x;
    // A tile: BM x BK ; row-major in global (contiguous in K)
    // each row has BK/8 = 4 float4; BM rows -> 512 float4 / 256 threads = 2 each
    const int a_row0 = tid / (BK / 8);        // which row
    const int a_col0 = (tid % (BK / 8)) * 8;  // which col (in halves)
    const int A_ROW_STRIDE = NTHREADS / (BK / 8);  // 64 rows per pass
    // B tile: BK x BN ; row-major in global (contiguous in N)
    const int b_row0 = tid / (BN / 8);        // which row
    const int b_col0 = (tid % (BN / 8)) * 8;  // which col
    const int B_ROW_STRIDE = NTHREADS / (BN / 8);  // 16 rows per pass

    for (int kt = 0; kt < K; kt += BK) {
        // Load A tile
#pragma unroll
        for (int r = 0; r < BM; r += A_ROW_STRIDE) {
            int gr = blockRow + a_row0 + r;
            int gc = kt + a_col0;
            float4 v = *reinterpret_cast<const float4*>(&A[gr * K + gc]);
            *reinterpret_cast<float4*>(&As[(a_row0 + r) * LDA_S + a_col0]) = v;
        }
        // Load B tile
#pragma unroll
        for (int r = 0; r < BK; r += B_ROW_STRIDE) {
            int gr = kt + b_row0 + r;
            int gc = blockCol + b_col0;
            float4 v = *reinterpret_cast<const float4*>(&B[gr * N + gc]);
            *reinterpret_cast<float4*>(&Bs[(b_row0 + r) * LDB_S + b_col0]) = v;
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                           wmma::row_major>
                a_frag[FRAG_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                           wmma::row_major>
                b_frag[FRAG_N];
#pragma unroll
            for (int i = 0; i < FRAG_M; ++i)
                wmma::load_matrix_sync(
                    a_frag[i],
                    &As[(warpRow + i * WMMA_M) * LDA_S + kk], LDA_S);
#pragma unroll
            for (int j = 0; j < FRAG_N; ++j)
                wmma::load_matrix_sync(
                    b_frag[j], &Bs[kk * LDB_S + warpCol + j * WMMA_N], LDB_S);
#pragma unroll
            for (int i = 0; i < FRAG_M; ++i)
#pragma unroll
                for (int j = 0; j < FRAG_N; ++j)
                    wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }
        __syncthreads();
    }

    // Store accumulators to C
#pragma unroll
    for (int i = 0; i < FRAG_M; ++i)
#pragma unroll
        for (int j = 0; j < FRAG_N; ++j) {
            int cr = blockRow + warpRow + i * WMMA_M;
            int cc = blockCol + warpCol + j * WMMA_N;
            wmma::store_matrix_sync(&C[cr * N + cc], acc[i][j], N,
                                    wmma::mem_row_major);
        }
}

}  // namespace v1

PLAYGROUND_MATMUL_DEC(float16_t, 1, m, n, k, A, B, C)
{
    dim3 block(v1::NTHREADS);
    dim3 grid((n + v1::BN - 1) / v1::BN, (m + v1::BM - 1) / v1::BM);
    v1::matmul_kernel<<<grid, block>>>(int(m), int(n), int(k), A, B, C);
}

}  // namespace playground
