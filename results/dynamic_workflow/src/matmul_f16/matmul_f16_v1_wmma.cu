// @file: task-1/src/matmul_f16/matmul_f16_v1_wmma.cu
// v1: WMMA Tensor Core baseline. 128x128 block tile, BK=16, 8 warps (2x4),
// each warp computes a 64x32 sub-tile via 4x2 wmma 16x16x16 fragments.
// fp32 accumulate (full TC throughput on A100), convert to half on store.

#include <cuda_fp16.h>
#include <mma.h>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{
using namespace nvcuda;

namespace
{
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 16;

constexpr int WM = 16;  // wmma tile M
constexpr int WN = 16;  // wmma tile N
constexpr int WK = 16;  // wmma tile K

constexpr int WARP_M = 2;  // warps along M
constexpr int WARP_N = 4;  // warps along N
constexpr int WARP_TILE_M = BM / WARP_M;  // 64
constexpr int WARP_TILE_N = BN / WARP_N;  // 32
constexpr int FRAG_M = WARP_TILE_M / WM;  // 4
constexpr int FRAG_N = WARP_TILE_N / WN;  // 2
constexpr int THREADS = WARP_M * WARP_N * 32;  // 256

constexpr int APAD = 8;  // pad shared As stride -> avoid bank conflicts
constexpr int BPAD = 8;

__global__ void __launch_bounds__(THREADS)
    wmma_v1(const half* __restrict__ A, const half* __restrict__ B,
            half* __restrict__ C, int M, int N, int K)
{
    __shared__ half As[BM][BK + APAD];  // 128 x 24
    __shared__ half Bs[BK][BN + BPAD];  // 16  x 136

    const int blockRow = blockIdx.y;  // along M
    const int blockCol = blockIdx.x;  // along N
    const int tid = threadIdx.x;
    const int warpId = tid >> 5;
    const int warpRow = warpId / WARP_N;  // [0,2)
    const int warpCol = warpId % WARP_N;  // [0,4)

    const int aRowBase = blockRow * BM;
    const int bColBase = blockCol * BN;

    wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc[FRAG_M][FRAG_N];
#pragma unroll
    for (int i = 0; i < FRAG_M; ++i)
#pragma unroll
        for (int j = 0; j < FRAG_N; ++j)
            wmma::fill_fragment(acc[i][j], 0.0f);

    // ---- global->shared load mapping (one float4 = 8 halves per thread) ----
    // As: 128x16 = 2048 halves = 256 float4. row = tid/2, colHalf=(tid%2)*8.
    const int aRow = tid >> 1;
    const int aColH = (tid & 1) * 8;
    // Bs: 16x128 = 2048 halves = 256 float4. row = tid/16, col=(tid%16)*8.
    const int bRow = tid >> 4;
    const int bColH = (tid & 15) * 8;

    const int kSteps = K / BK;
    for (int kt = 0; kt < kSteps; ++kt) {
        const int kBase = kt * BK;

        // load A tile: A[aRowBase + aRow][kBase + aColH .. +8]
        {
            const half* gptr = A + (size_t)(aRowBase + aRow) * K + kBase + aColH;
            *reinterpret_cast<float4*>(&As[aRow][aColH]) =
                *reinterpret_cast<const float4*>(gptr);
        }
        // load B tile: B[kBase + bRow][bColBase + bColH .. +8]
        {
            const half* gptr = B + (size_t)(kBase + bRow) * N + bColBase + bColH;
            *reinterpret_cast<float4*>(&Bs[bRow][bColH]) =
                *reinterpret_cast<const float4*>(gptr);
        }
        __syncthreads();

        // load fragments and accumulate
        wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major>
            aFrag[FRAG_M];
        wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::row_major>
            bFrag[FRAG_N];
#pragma unroll
        for (int m = 0; m < FRAG_M; ++m)
            wmma::load_matrix_sync(
                aFrag[m], &As[warpRow * WARP_TILE_M + m * WM][0], BK + APAD);
#pragma unroll
        for (int n = 0; n < FRAG_N; ++n)
            wmma::load_matrix_sync(
                bFrag[n], &Bs[0][warpCol * WARP_TILE_N + n * WN], BN + BPAD);
#pragma unroll
        for (int m = 0; m < FRAG_M; ++m)
#pragma unroll
            for (int n = 0; n < FRAG_N; ++n)
                wmma::mma_sync(acc[m][n], aFrag[m], bFrag[n], acc[m][n]);

        __syncthreads();
    }

    // ---- store: convert fp32 acc -> half fragment -> C ----
#pragma unroll
    for (int m = 0; m < FRAG_M; ++m) {
#pragma unroll
        for (int n = 0; n < FRAG_N; ++n) {
            wmma::fragment<wmma::accumulator, WM, WN, WK, half> cFrag;
#pragma unroll
            for (int t = 0; t < acc[m][n].num_elements; ++t)
                cFrag.x[t] = __float2half(acc[m][n].x[t]);

            const int cRow = aRowBase + warpRow * WARP_TILE_M + m * WM;
            const int cCol = bColBase + warpCol * WARP_TILE_N + n * WN;
            wmma::store_matrix_sync(C + (size_t)cRow * N + cCol, cFrag, N,
                                    wmma::mem_row_major);
        }
    }
}
}  // namespace

PLAYGROUND_MATMUL_DEC(float16_t, 1, m, n, k, A, B, C)
{
    dim3 block(THREADS);
    dim3 grid(n / BN, m / BM);
    wmma_v1<<<grid, block>>>(A, B, C, (int) m, (int) n, (int) k);
}

}  // namespace playground
