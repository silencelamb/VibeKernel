#include "playground/matmul.hpp"

#include <cuda_fp16.h>
#include <mma.h>

namespace playground
{
using namespace nvcuda;

namespace
{

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 4;
constexpr int NUM_WARPS = WARPS_M * WARPS_N;   // 8
constexpr int NUM_THREADS = NUM_WARPS * 32;    // 256
constexpr int WM = BM / WARPS_M;               // 64
constexpr int WN = BN / WARPS_N;               // 32
constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
constexpr int WTILES_M = WM / WMMA_M;          // 4
constexpr int WTILES_N = WN / WMMA_N;          // 2

__device__ __forceinline__ void cp_async_cg16(void* smem, const void* gmem)
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

__global__ void __launch_bounds__(NUM_THREADS)
kernel_v2(int M, int N, int K, const half* __restrict__ A,
          const half* __restrict__ B, half* __restrict__ C)
{
    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;
    const int warpId = threadIdx.x / 32;
    const int warpRow = warpId / WARPS_N;
    const int warpCol = warpId % WARPS_N;
    const int tid = threadIdx.x;

    __shared__ half sA[2][BM * BK];
    __shared__ half sB[2][BK * BN];

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        acc[WTILES_M][WTILES_N];
#pragma unroll
    for (int i = 0; i < WTILES_M; i++)
#pragma unroll
        for (int j = 0; j < WTILES_N; j++)
            wmma::fill_fragment(acc[i][j], 0.0f);

    const int nK = K / BK;

    // Per-thread load coordinates (constant across K iterations).
    // A: BM*BK/8 = 512 float4 over 256 threads -> 2 each.
    // B: BK*BN/8 = 512 float4 over 256 threads -> 2 each.
    auto load_stage = [&](int stage, int kk) {
#pragma unroll
        for (int i = 0; i < (BM * BK / 8) / NUM_THREADS; i++) {
            int vec = tid + i * NUM_THREADS;
            int row = vec / (BK / 8);
            int col8 = (vec % (BK / 8)) * 8;
            const half* gptr = A + (blockRow * BM + row) * K + (kk + col8);
            cp_async_cg16(&sA[stage][row * BK + col8], gptr);
        }
#pragma unroll
        for (int i = 0; i < (BK * BN / 8) / NUM_THREADS; i++) {
            int vec = tid + i * NUM_THREADS;
            int row = vec / (BN / 8);
            int col8 = (vec % (BN / 8)) * 8;
            const half* gptr = B + (kk + row) * N + (blockCol * BN + col8);
            cp_async_cg16(&sB[stage][row * BN + col8], gptr);
        }
    };

    // Prologue: issue load of stage 0.
    load_stage(0, 0);
    cp_async_commit();

    int stage = 0;
    for (int i = 0; i < nK; i++) {
        // Prefetch next tile into the other buffer.
        if (i + 1 < nK) {
            load_stage(stage ^ 1, (i + 1) * BK);
        }
        cp_async_commit();
        cp_async_wait<1>();   // keep at most 1 group pending -> stage ready
        __syncthreads();

#pragma unroll
        for (int kf = 0; kf < BK / WMMA_K; kf++) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                           wmma::row_major>
                a_frag[WTILES_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                           wmma::row_major>
                b_frag[WTILES_N];
#pragma unroll
            for (int ii = 0; ii < WTILES_M; ii++) {
                int row = warpRow * WM + ii * WMMA_M;
                wmma::load_matrix_sync(
                    a_frag[ii], &sA[stage][row * BK + kf * WMMA_K], BK);
            }
#pragma unroll
            for (int j = 0; j < WTILES_N; j++) {
                int col = warpCol * WN + j * WMMA_N;
                wmma::load_matrix_sync(
                    b_frag[j], &sB[stage][(kf * WMMA_K) * BN + col], BN);
            }
#pragma unroll
            for (int ii = 0; ii < WTILES_M; ii++)
#pragma unroll
                for (int j = 0; j < WTILES_N; j++)
                    wmma::mma_sync(acc[ii][j], a_frag[ii], b_frag[j],
                                   acc[ii][j]);
        }
        __syncthreads();
        stage ^= 1;
    }

#pragma unroll
    for (int i = 0; i < WTILES_M; i++) {
#pragma unroll
        for (int j = 0; j < WTILES_N; j++) {
            int row = blockRow * BM + warpRow * WM + i * WMMA_M;
            int col = blockCol * BN + warpCol * WN + j * WMMA_N;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half>
                c_half;
#pragma unroll
            for (int t = 0; t < acc[i][j].num_elements; t++)
                c_half.x[t] = __float2half(acc[i][j].x[t]);
            wmma::store_matrix_sync(&C[row * N + col], c_half, N,
                                    wmma::mem_row_major);
        }
    }
}

}   // namespace

PLAYGROUND_MATMUL_DEC(float16_t, 2, m, n, k, A, B, C)
{
    dim3 block(NUM_THREADS);
    dim3 grid(n / BN, m / BM);
    kernel_v2<<<grid, block>>>((int) m, (int) n, (int) k, A, B, C);
}

}   // namespace playground
