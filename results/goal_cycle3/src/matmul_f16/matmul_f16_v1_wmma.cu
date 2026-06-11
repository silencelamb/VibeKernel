#include <cuda_fp16.h>
#include <mma.h>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{
namespace v1
{
using namespace nvcuda;

// Block tile: 128 x 128, K-step 32.
// 8 warps (256 threads). Warp tile 64(M) x 32(N): warps arranged 2(M) x 4(N).
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int WARP_M = 64;
constexpr int WARP_N = 32;
constexpr int WARPS_M = BM / WARP_M;  // 2
constexpr int WARPS_N = BN / WARP_N;  // 4
constexpr int N_WARPS = WARPS_M * WARPS_N;  // 8
constexpr int N_THREADS = N_WARPS * 32;     // 256

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int PAD = 8;  // smem padding (halfs) to avoid bank conflicts, keeps 16B align

__global__ void __launch_bounds__(N_THREADS)
    gemm_wmma(const half* __restrict__ A, const half* __restrict__ B,
              half* __restrict__ C, int M, int N, int K)
{
    const int blockRow = blockIdx.y;  // along M
    const int blockCol = blockIdx.x;  // along N

    const int tid = threadIdx.x;
    const int warpId = tid / 32;
    const int warpRow = warpId / WARPS_N;  // 0..1
    const int warpCol = warpId % WARPS_N;  // 0..3

    __shared__ half As[BM][BK + PAD];
    __shared__ half Bs[BK][BN + PAD];

    // Accumulators: warp tile 64x32 -> 4x2 fragments of 16x16
    constexpr int FM = WARP_M / WMMA_M;  // 4
    constexpr int FN = WARP_N / WMMA_N;  // 2
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[FM][FN];
#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j)
            wmma::fill_fragment(acc[i][j], 0.0f);

    // Global tile base
    const half* Atile = A + blockRow * BM * K;          // [BM][K] rows
    const half* Btile = B + blockCol * BN;              // [K][BN] cols

    for (int kk = 0; kk < K; kk += BK) {
        // ---- Load A tile [BM][BK] from global (row-major, row stride K) ----
        // 128*32 = 4096 halfs = 512 float4; 256 threads -> 2 float4/thread
#pragma unroll
        for (int i = 0; i < (BM * BK) / (N_THREADS * 8); ++i) {
            int idx = (i * N_THREADS + tid) * 8;  // element index in tile
            int r = idx / BK;
            int c = idx % BK;
            const float4* src =
                reinterpret_cast<const float4*>(Atile + r * K + kk + c);
            *reinterpret_cast<float4*>(&As[r][c]) = *src;
        }
        // ---- Load B tile [BK][BN] from global (row-major, row stride N) ----
#pragma unroll
        for (int i = 0; i < (BK * BN) / (N_THREADS * 8); ++i) {
            int idx = (i * N_THREADS + tid) * 8;
            int r = idx / BN;
            int c = idx % BN;
            const float4* src =
                reinterpret_cast<const float4*>(Btile + (kk + r) * N + c);
            *reinterpret_cast<float4*>(&Bs[r][c]) = *src;
        }
        __syncthreads();

        // ---- Compute ----
#pragma unroll
        for (int ks = 0; ks < BK; ks += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                           wmma::row_major>
                aFrag[FM];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                           wmma::row_major>
                bFrag[FN];
#pragma unroll
            for (int i = 0; i < FM; ++i) {
                int row = warpRow * WARP_M + i * WMMA_M;
                wmma::load_matrix_sync(aFrag[i], &As[row][ks], BK + PAD);
            }
#pragma unroll
            for (int j = 0; j < FN; ++j) {
                int col = warpCol * WARP_N + j * WMMA_N;
                // Bs is row-major [BK][BN]; for col_major B fragment we read the
                // [WMMA_K][WMMA_N] block starting at [ks][col] with ld = BN+PAD.
                wmma::load_matrix_sync(bFrag[j], &Bs[ks][col], BN + PAD);
            }
#pragma unroll
            for (int i = 0; i < FM; ++i)
#pragma unroll
                for (int j = 0; j < FN; ++j)
                    wmma::mma_sync(acc[i][j], aFrag[i], bFrag[j], acc[i][j]);
        }
        __syncthreads();
    }

    // ---- Store: convert fp32 acc -> fp16 fragment, write to C ----
    half* Ctile = C + blockRow * BM * N + blockCol * BN;
#pragma unroll
    for (int i = 0; i < FM; ++i) {
#pragma unroll
        for (int j = 0; j < FN; ++j) {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half>
                hFrag;
#pragma unroll
            for (int t = 0; t < acc[i][j].num_elements; ++t)
                hFrag.x[t] = __float2half(acc[i][j].x[t]);
            int row = warpRow * WARP_M + i * WMMA_M;
            int col = warpCol * WARP_N + j * WMMA_N;
            wmma::store_matrix_sync(Ctile + row * N + col, hFrag, N,
                                    wmma::mem_row_major);
        }
    }
}

}  // namespace v1

PLAYGROUND_MATMUL_DEC(float16_t, 1, m, n, k, A, B, C)
{
    dim3 block(v1::N_THREADS);
    dim3 grid(n / v1::BN, m / v1::BM);
    v1::gemm_wmma<<<grid, block>>>(A, B, C, int(m), int(n), int(k));
}

}  // namespace playground
