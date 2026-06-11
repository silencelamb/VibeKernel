#include <cuda_fp16.h>
#include <mma.h>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{
namespace
{
using namespace nvcuda;

// -----------------------------------------------------------------------------
// v1: Baseline WMMA tiled GEMM.
//   Block computes a 128x128 output tile. 256 threads = 8 warps arranged 2x4.
//   Each warp owns a 64x32 sub-tile = 4x2 of the 16x16x16 WMMA fragments.
//   BK = 16, single wmma k-step per main-loop iteration.
//   Shared mem holds one A(128x16) + B(16x128) stage (no double buffering yet).
// -----------------------------------------------------------------------------
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 16;
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// warp grid within the block
constexpr int WARP_ROWS = 2;  // warps along M
constexpr int WARP_COLS = 4;  // warps along N
// each warp's fragment grid
constexpr int FRAG_M = (BM / WARP_ROWS) / WMMA_M;  // 64/16 = 4
constexpr int FRAG_N = (BN / WARP_COLS) / WMMA_N;  // 32/16 = 2

__global__ void __launch_bounds__(256)
    hgemm_v1_wmma(const half* __restrict__ A, const half* __restrict__ B,
                  half* __restrict__ C, int M, int N, int K)
{
    __shared__ half As[BM][BK];
    __shared__ half Bs[BK][BN];

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    const int tid = threadIdx.x;
    const int warpId = tid >> 5;
    const int warpRow = warpId / WARP_COLS;  // 0..1
    const int warpCol = warpId % WARP_COLS;  // 0..3

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        acc[FRAG_M][FRAG_N];
#pragma unroll
    for (int i = 0; i < FRAG_M; ++i)
#pragma unroll
        for (int j = 0; j < FRAG_N; ++j)
            wmma::fill_fragment(acc[i][j], 0.0f);

    // Each thread loads 8 halves (one float4) of As and 8 of Bs per K-step.
    // As: 128x16 = 2048 halves / 256 threads = 8 each.
    const int aRow = tid / 2;          // 0..127
    const int aCol = (tid % 2) * 8;    // 0 or 8
    // Bs: 16x128 = 2048 halves / 256 threads = 8 each.
    const int bRow = tid / 16;         // 0..15
    const int bCol = (tid % 16) * 8;   // 0..120

    for (int k0 = 0; k0 < K; k0 += BK) {
        // ---- global -> shared ----
        const half* aGlobal = A + (blockRow + aRow) * K + (k0 + aCol);
        *reinterpret_cast<float4*>(&As[aRow][aCol]) =
            *reinterpret_cast<const float4*>(aGlobal);

        const half* bGlobal = B + (k0 + bRow) * N + (blockCol + bCol);
        *reinterpret_cast<float4*>(&Bs[bRow][bCol]) =
            *reinterpret_cast<const float4*>(bGlobal);

        __syncthreads();

        // ---- compute one BK step ----
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                       wmma::row_major>
            aFrag[FRAG_M];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                       wmma::row_major>
            bFrag[FRAG_N];

#pragma unroll
        for (int i = 0; i < FRAG_M; ++i) {
            const int r = warpRow * (BM / WARP_ROWS) + i * WMMA_M;
            wmma::load_matrix_sync(aFrag[i], &As[r][0], BK);
        }
#pragma unroll
        for (int j = 0; j < FRAG_N; ++j) {
            const int c = warpCol * (BN / WARP_COLS) + j * WMMA_N;
            wmma::load_matrix_sync(bFrag[j], &Bs[0][c], BN);
        }
#pragma unroll
        for (int i = 0; i < FRAG_M; ++i)
#pragma unroll
            for (int j = 0; j < FRAG_N; ++j)
                wmma::mma_sync(acc[i][j], aFrag[i], bFrag[j], acc[i][j]);

        __syncthreads();
    }

    // ---- store back (convert fp32 acc -> fp16) ----
#pragma unroll
    for (int i = 0; i < FRAG_M; ++i) {
#pragma unroll
        for (int j = 0; j < FRAG_N; ++j) {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half>
                cFrag;
#pragma unroll
            for (int t = 0; t < acc[i][j].num_elements; ++t)
                cFrag.x[t] = __float2half(acc[i][j].x[t]);

            const int r =
                blockRow + warpRow * (BM / WARP_ROWS) + i * WMMA_M;
            const int c =
                blockCol + warpCol * (BN / WARP_COLS) + j * WMMA_N;
            wmma::store_matrix_sync(C + r * N + c, cFrag, N,
                                    wmma::mem_row_major);
        }
    }
}

}  // namespace

PLAYGROUND_MATMUL_DEC(float16_t, 1, m, n, k, A, B, C)
{
    dim3 block(256);
    dim3 grid(n / BN, m / BM);
    hgemm_v1_wmma<<<grid, block>>>(A, B, C, int(m), int(n), int(k));
}

}  // namespace playground
