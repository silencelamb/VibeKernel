// @file: task-1/src/matmul_f16/matmul_f16_v2_cpasync.cu
//
// v2: v1 (WMMA) + cp.async double-buffered shared memory pipeline.
//   - Same tiling as v1 (128x128 block, BK=32, 8 warps 2x4, warp 64x32).
//   - Global->shared via cp.async.cg (16B) into a 2-stage ring; the async copy
//     of the next K-tile overlaps the WMMA compute of the current one, hiding
//     global-memory latency that stalled the single-buffered v1.
//   - fp32 accumulate, fp16 epilogue (matches cBLAS GT).

#include <cuda_fp16.h>
#include <mma.h>

#include "playground/matmul.hpp"

namespace playground
{
namespace v2impl
{
using namespace nvcuda;

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;

constexpr int WM = 64;
constexpr int WN = 32;
constexpr int WARPS_M = BM / WM;            // 2
constexpr int WARPS_N = BN / WN;            // 4
constexpr int NWARPS = WARPS_M * WARPS_N;   // 8
constexpr int NTHREADS = NWARPS * 32;       // 256

constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
constexpr int FM = WM / WMMA_M;  // 4
constexpr int FN = WN / WMMA_N;  // 2

constexpr int SKEW = 8;
constexpr int AS_LD = BK + SKEW;   // 40
constexpr int BS_LD = BN + SKEW;   // 136
constexpr int STAGES = 2;

__device__ __forceinline__ uint32_t smem_u32(const void* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async_cg(void* dst, const void* src)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(
                     smem_u32(dst)),
                 "l"(src));
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

__global__ void __launch_bounds__(NTHREADS) gemm_v2(int M, int N, int K,
                                                    const half* __restrict__ A,
                                                    const half* __restrict__ B,
                                                    half* __restrict__ C)
{
    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    __shared__ half As[STAGES][BM][AS_LD];
    __shared__ half Bs[STAGES][BK][BS_LD];

    const int warpId = threadIdx.x >> 5;
    const int warpRow = warpId / WARPS_N;
    const int warpCol = warpId % WARPS_N;
    const int tid = threadIdx.x;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[FM][FN];
#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j)
            wmma::fill_fragment(acc[i][j], 0.0f);

    const half* Aptr = A + blockRow * K;
    const half* Bptr = B + blockCol;
    const int NT = K / BK;

    // load K-tile `t` into shared buffer `buf` via cp.async
    auto loadTile = [&](int t, int buf) {
        int k0 = t * BK;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            int f = tid + i * NTHREADS;     // 0..511 (A: 512 float4)
            int row = f >> 2;
            int col = (f & 3) << 3;
            cp_async_cg(&As[buf][row][col], Aptr + row * K + (k0 + col));
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            int f = tid + i * NTHREADS;     // 0..511 (B: 512 float4)
            int row = f >> 4;
            int col = (f & 15) << 3;
            cp_async_cg(&Bs[buf][row][col], Bptr + (k0 + row) * N + col);
        }
        cp_async_commit();
    };

    // prologue: kick off tile 0
    loadTile(0, 0);
    cp_async_wait<0>();
    __syncthreads();

    for (int t = 0; t < NT; ++t) {
        int cur = t & 1;
        if (t + 1 < NT)
            loadTile(t + 1, (t + 1) & 1);

#pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                           wmma::row_major>
                aFrag[FM];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                           wmma::row_major>
                bFrag[FN];
#pragma unroll
            for (int i = 0; i < FM; ++i)
                wmma::load_matrix_sync(aFrag[i],
                                       &As[cur][warpRow * WM + i * WMMA_M][kk],
                                       AS_LD);
#pragma unroll
            for (int j = 0; j < FN; ++j)
                wmma::load_matrix_sync(bFrag[j],
                                       &Bs[cur][kk][warpCol * WN + j * WMMA_N],
                                       BS_LD);
#pragma unroll
            for (int i = 0; i < FM; ++i)
#pragma unroll
                for (int j = 0; j < FN; ++j)
                    wmma::mma_sync(acc[i][j], aFrag[i], bFrag[j], acc[i][j]);
        }

        cp_async_wait<0>();
        __syncthreads();
    }

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
            wmma::store_matrix_sync(C + r * N + c, cFrag, N, wmma::mem_row_major);
        }
    }
}
}  // namespace v2impl

PLAYGROUND_MATMUL_DEC(float16_t, 2, m, n, k, A, B, C)
{
    using namespace v2impl;
    dim3 block(NTHREADS);
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    gemm_v2<<<grid, block>>>(int(m), int(n), int(k), A, B, C);
}

}  // namespace playground
