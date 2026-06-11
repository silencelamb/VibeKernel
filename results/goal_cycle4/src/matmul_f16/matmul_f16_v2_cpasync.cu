#include <cuda_fp16.h>
#include <mma.h>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{
namespace v2
{
using namespace nvcuda;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;

constexpr int WARPS_M = 4;
constexpr int WARPS_N = 2;
constexpr int WARP_M = BM / WARPS_M;  // 32
constexpr int WARP_N = BN / WARPS_N;  // 64
constexpr int FRAG_M = WARP_M / WMMA_M;  // 2
constexpr int FRAG_N = WARP_N / WMMA_N;  // 4

constexpr int THREADS = WARPS_M * WARPS_N * 32;  // 256

constexpr int APAD = 8;
constexpr int BPAD = 8;
constexpr int STAGES = 2;  // double buffer

__device__ __forceinline__ void cp_async_cg(void* smem, const void* gmem)
{
    unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
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

__global__ void __launch_bounds__(THREADS)
    kernel(int M, int N, int K, const half* __restrict__ A,
           const half* __restrict__ B, half* __restrict__ C)
{
    __shared__ half As[STAGES][BM][BK + APAD];
    __shared__ half Bs[STAGES][BK][BN + BPAD];

    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;
    const int warpId = threadIdx.x / 32;
    const int warpRow = warpId / WARPS_N;
    const int warpCol = warpId % WARPS_N;
    const int tid = threadIdx.x;

    const int rowBase = blockRow * BM;
    const int colBase = blockCol * BN;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        acc[FRAG_M][FRAG_N];
#pragma unroll
    for (int i = 0; i < FRAG_M; ++i)
#pragma unroll
        for (int j = 0; j < FRAG_N; ++j)
            wmma::fill_fragment(acc[i][j], 0.0f);

    // Precompute load coordinates (constant across k0).
    // A: 2 float4 per thread. row = f/4, col=(f%4)*8.
    // B: 2 float4 per thread. row = f/16, col=(f%16)*8.
    auto load_stage = [&](int stage, int k0) {
#pragma unroll
        for (int it = 0; it < 2; ++it) {
            int f = tid + it * THREADS;
            int row = f / (BK / 8);
            int col = (f % (BK / 8)) * 8;
            cp_async_cg(&As[stage][row][col],
                        &A[(rowBase + row) * K + k0 + col]);
        }
#pragma unroll
        for (int it = 0; it < 2; ++it) {
            int f = tid + it * THREADS;
            int row = f / (BN / 8);
            int col = (f % (BN / 8)) * 8;
            cp_async_cg(&Bs[stage][row][col],
                        &B[(k0 + row) * N + colBase + col]);
        }
        cp_async_commit();
    };

    int nK = K / BK;
    // Prologue: issue stage 0.
    load_stage(0, 0);

    int cur = 0;
    for (int kt = 0; kt < nK; ++kt) {
        // Issue next stage before computing on current.
        if (kt + 1 < nK) {
            load_stage((cur + 1) % STAGES, (kt + 1) * BK);
        }
        // Wait until the current stage's group is ready. With STAGES=2 and one
        // outstanding prefetch, wait_group<1> keeps 1 in flight (the next).
        if (kt + 1 < nK)
            cp_async_wait<1>();
        else
            cp_async_wait<0>();
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
            for (int i = 0; i < FRAG_M; ++i) {
                int ar = warpRow * WARP_M + i * WMMA_M;
                wmma::load_matrix_sync(a_frag[i], &As[cur][ar][kk], BK + APAD);
            }
#pragma unroll
            for (int j = 0; j < FRAG_N; ++j) {
                int bc = warpCol * WARP_N + j * WMMA_N;
                wmma::load_matrix_sync(b_frag[j], &Bs[cur][kk][bc], BN + BPAD);
            }
#pragma unroll
            for (int i = 0; i < FRAG_M; ++i)
#pragma unroll
                for (int j = 0; j < FRAG_N; ++j)
                    wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }
        __syncthreads();
        cur = (cur + 1) % STAGES;
    }

#pragma unroll
    for (int i = 0; i < FRAG_M; ++i) {
#pragma unroll
        for (int j = 0; j < FRAG_N; ++j) {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> ch;
#pragma unroll
            for (int t = 0; t < acc[i][j].num_elements; ++t)
                ch.x[t] = __float2half(acc[i][j].x[t]);
            int cr = rowBase + warpRow * WARP_M + i * WMMA_M;
            int cc = colBase + warpCol * WARP_N + j * WMMA_N;
            wmma::store_matrix_sync(&C[cr * N + cc], ch, N, wmma::mem_row_major);
        }
    }
}
}  // namespace v2

PLAYGROUND_MATMUL_DEC(float16_t, 2, m, n, k, A, B, C)
{
    dim3 block(v2::THREADS);
    dim3 grid(n / v2::BN, m / v2::BM);
    v2::kernel<<<grid, block>>>((int) m, (int) n, (int) k, A, B, C);
}

}  // namespace playground
