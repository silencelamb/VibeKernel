// @file: task-1/src/matmul_f16/matmul_f16_v1_wmma.cu
//
// v1: Tensor-Core baseline using the WMMA API (nvcuda::wmma).
//   - Block tile 128x128, BK=32, 256 threads (8 warps, 2x4 warp grid).
//   - Each warp owns a 64x32 output region -> 4x2 = 8 WMMA 16x16x16 tiles.
//   - Single-buffered shared memory (load -> sync -> mma -> sync) for clarity;
//     fp32 accumulate to match the cBLAS fp32 ground truth, fp16 epilogue.
//   - Vectorized (float4 / 128-bit) global->shared loads, +8 half skew pad to
//     cut shared-memory bank conflicts on the WMMA reads.
// This establishes correctness + a working bench loop; later versions move to
// mma.sync + cp.async multistage + ldmatrix + swizzle for the real push.

#include <cuda_fp16.h>
#include <mma.h>

#include "playground/matmul.hpp"

namespace playground
{
namespace v1impl
{
using namespace nvcuda;

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;

constexpr int WM = 64;  // warp tile rows
constexpr int WN = 32;  // warp tile cols
constexpr int WARPS_M = BM / WM;            // 2
constexpr int WARPS_N = BN / WN;            // 4
constexpr int NWARPS = WARPS_M * WARPS_N;   // 8
constexpr int NTHREADS = NWARPS * 32;       // 256

constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
constexpr int FM = WM / WMMA_M;  // 4 acc tiles along M
constexpr int FN = WN / WMMA_N;  // 2 acc tiles along N

constexpr int SKEW = 8;  // half elements of skew padding (16B) to ease conflicts
constexpr int AS_LD = BK + SKEW;   // 40
constexpr int BS_LD = BN + SKEW;   // 136

__global__ void __launch_bounds__(NTHREADS) gemm_v1(int M, int N, int K,
                                                    const half* __restrict__ A,
                                                    const half* __restrict__ B,
                                                    half* __restrict__ C)
{
    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    __shared__ half As[BM][AS_LD];
    __shared__ half Bs[BK][BS_LD];

    const int warpId = threadIdx.x >> 5;
    const int warpRow = warpId / WARPS_N;  // 0..1
    const int warpCol = warpId % WARPS_N;  // 0..3

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[FM][FN];
#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j)
            wmma::fill_fragment(acc[i][j], 0.0f);

    const int tid = threadIdx.x;

    // Global load mapping (float4 = 8 halfs per transaction):
    //   A tile 128x32 -> 512 float4; 256 threads -> 2 each.
    //   B tile  32x128 -> 512 float4; 256 threads -> 2 each.
    const half* Aptr = A + blockRow * K;        // top-left of block's A panel
    const half* Bptr = B + blockCol;            // top-left of block's B panel

    for (int k0 = 0; k0 < K; k0 += BK) {
        // ---- load A tile (128 rows x 32 cols) ----
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            int f = tid + i * NTHREADS;       // float4 index 0..511
            int row = f >> 2;                 // /4 (4 float4 per 32-wide row)
            int col = (f & 3) << 3;           // *8
            const half* g = Aptr + row * K + (k0 + col);
            *reinterpret_cast<float4*>(&As[row][col]) =
                *reinterpret_cast<const float4*>(g);
        }
        // ---- load B tile (32 rows x 128 cols) ----
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            int f = tid + i * NTHREADS;       // 0..511
            int row = f >> 4;                 // /16 (16 float4 per 128-wide row)
            int col = (f & 15) << 3;          // *8
            const half* g = Bptr + (k0 + row) * N + col;
            *reinterpret_cast<float4*>(&Bs[row][col]) =
                *reinterpret_cast<const float4*>(g);
        }

        __syncthreads();

        // ---- mma over BK in steps of WMMA_K (16) ----
#pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                           wmma::row_major>
                aFrag[FM];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                           wmma::row_major>
                bFrag[FN];
#pragma unroll
            for (int i = 0; i < FM; ++i) {
                int r = warpRow * WM + i * WMMA_M;
                wmma::load_matrix_sync(aFrag[i], &As[r][kk], AS_LD);
            }
#pragma unroll
            for (int j = 0; j < FN; ++j) {
                int c = warpCol * WN + j * WMMA_N;
                wmma::load_matrix_sync(bFrag[j], &Bs[kk][c], BS_LD);
            }
#pragma unroll
            for (int i = 0; i < FM; ++i)
#pragma unroll
                for (int j = 0; j < FN; ++j)
                    wmma::mma_sync(acc[i][j], aFrag[i], bFrag[j], acc[i][j]);
        }

        __syncthreads();
    }

    // ---- epilogue: convert fp32 acc -> fp16, store row-major to C ----
#pragma unroll
    for (int i = 0; i < FM; ++i) {
#pragma unroll
        for (int j = 0; j < FN; ++j) {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half>
                cFrag;
#pragma unroll
            for (int t = 0; t < acc[i][j].num_elements; ++t)
                cFrag.x[t] = __float2half(acc[i][j].x[t]);

            int r = blockRow + warpRow * WM + i * WMMA_M;
            int c = blockCol + warpCol * WN + j * WMMA_N;
            wmma::store_matrix_sync(C + r * N + c, cFrag, N,
                                    wmma::mem_row_major);
        }
    }
}
}  // namespace v1impl

PLAYGROUND_MATMUL_DEC(float16_t, 1, m, n, k, A, B, C)
{
    using namespace v1impl;
    dim3 block(NTHREADS);
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    gemm_v1<<<grid, block>>>(int(m), int(n), int(k), A, B, C);
}

}  // namespace playground
