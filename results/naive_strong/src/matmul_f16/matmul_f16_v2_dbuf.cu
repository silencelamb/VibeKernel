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
// v2: WMMA + cp.async double-buffered pipeline.
//   Block: 128x128 output, 256 threads = 8 warps (2x4). Warp tile 64x32.
//   BK = 32. Two shared-mem stages, prefetch next K-tile via cp.async while
//   computing the current one.
// -----------------------------------------------------------------------------
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int WARP_ROWS = 2;
constexpr int WARP_COLS = 4;
constexpr int FRAG_M = (BM / WARP_ROWS) / WMMA_M;  // 4
constexpr int FRAG_N = (BN / WARP_COLS) / WMMA_N;  // 2

__device__ __forceinline__ void cp_async_cg(void* smem, const void* gmem)
{
    unsigned s = __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s),
                 "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit()
{
    asm volatile("cp.async.commit_group;\n");
}
template <int N>
__device__ __forceinline__ void cp_async_wait()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__global__ void __launch_bounds__(256)
    hgemm_v2(const half* __restrict__ A, const half* __restrict__ B,
             half* __restrict__ C, int M, int N, int K)
{
    __shared__ half As[2][BM][BK];
    __shared__ half Bs[2][BK][BN];

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    const int tid = threadIdx.x;
    const int warpId = tid >> 5;
    const int warpRow = warpId / WARP_COLS;
    const int warpCol = warpId % WARP_COLS;

    // load index mapping (float4 granularity)
    const int aRow = tid / 4;         // 0..63 (covers 0..127 with +64)
    const int aCol = (tid % 4) * 8;   // 0,8,16,24
    const int bRow = tid / 16;        // 0..15 (covers 0..31 with +16)
    const int bCol = (tid % 16) * 8;  // 0..120

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        acc[FRAG_M][FRAG_N];
#pragma unroll
    for (int i = 0; i < FRAG_M; ++i)
#pragma unroll
        for (int j = 0; j < FRAG_N; ++j)
            wmma::fill_fragment(acc[i][j], 0.0f);

    const int numTiles = K / BK;

    auto loadStage = [&](int stage, int k0) {
        // As: two float4 rows per thread (aRow, aRow+64)
        cp_async_cg(&As[stage][aRow][aCol],
                    A + (blockRow + aRow) * K + (k0 + aCol));
        cp_async_cg(&As[stage][aRow + 64][aCol],
                    A + (blockRow + aRow + 64) * K + (k0 + aCol));
        // Bs: two float4 rows per thread (bRow, bRow+16)
        cp_async_cg(&Bs[stage][bRow][bCol],
                    B + (k0 + bRow) * N + (blockCol + bCol));
        cp_async_cg(&Bs[stage][bRow + 16][bCol],
                    B + (k0 + bRow + 16) * N + (blockCol + bCol));
        cp_async_commit();
    };

    // prologue
    loadStage(0, 0);

    for (int ki = 0; ki < numTiles; ++ki) {
        const int cur = ki & 1;
        if (ki + 1 < numTiles) {
            loadStage((ki + 1) & 1, (ki + 1) * BK);
        }
        cp_async_wait<0>();  // wait current (and next, conservative) ready
        __syncthreads();

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                       wmma::row_major>
            aFrag[FRAG_M];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                       wmma::row_major>
            bFrag[FRAG_N];

#pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) {
#pragma unroll
            for (int i = 0; i < FRAG_M; ++i) {
                const int r = warpRow * (BM / WARP_ROWS) + i * WMMA_M;
                wmma::load_matrix_sync(aFrag[i], &As[cur][r][kk], BK);
            }
#pragma unroll
            for (int j = 0; j < FRAG_N; ++j) {
                const int c = warpCol * (BN / WARP_COLS) + j * WMMA_N;
                wmma::load_matrix_sync(bFrag[j], &Bs[cur][kk][c], BN);
            }
#pragma unroll
            for (int i = 0; i < FRAG_M; ++i)
#pragma unroll
                for (int j = 0; j < FRAG_N; ++j)
                    wmma::mma_sync(acc[i][j], aFrag[i], bFrag[j], acc[i][j]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < FRAG_M; ++i) {
#pragma unroll
        for (int j = 0; j < FRAG_N; ++j) {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half>
                cFrag;
#pragma unroll
            for (int t = 0; t < acc[i][j].num_elements; ++t)
                cFrag.x[t] = __float2half(acc[i][j].x[t]);
            const int r = blockRow + warpRow * (BM / WARP_ROWS) + i * WMMA_M;
            const int c = blockCol + warpCol * (BN / WARP_COLS) + j * WMMA_N;
            wmma::store_matrix_sync(C + r * N + c, cFrag, N,
                                    wmma::mem_row_major);
        }
    }
}

}  // namespace

PLAYGROUND_MATMUL_DEC(float16_t, 2, m, n, k, A, B, C)
{
    dim3 block(256);
    dim3 grid(n / BN, m / BM);
    hgemm_v2<<<grid, block>>>(A, B, C, int(m), int(n), int(k));
}

}  // namespace playground
