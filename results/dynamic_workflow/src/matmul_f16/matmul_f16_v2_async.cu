// @file: task-1/src/matmul_f16/matmul_f16_v2_async.cu
// v2: WMMA + cp.async 3-stage software pipeline. 128x128 tile, BK=32.
// Overlaps global->shared loads with Tensor Core compute. fp32 accumulate.

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
constexpr int BK = 32;
constexpr int STAGES = 3;

constexpr int WM = 16, WN = 16, WK = 16;
constexpr int WARP_M = 2, WARP_N = 4;
constexpr int WARP_TILE_M = BM / WARP_M;  // 64
constexpr int WARP_TILE_N = BN / WARP_N;  // 32
constexpr int FRAG_M = WARP_TILE_M / WM;  // 4
constexpr int FRAG_N = WARP_TILE_N / WN;  // 2
constexpr int FRAG_K = BK / WK;           // 2
constexpr int THREADS = 256;

constexpr int APAD = 8, BPAD = 8;
constexpr int AS_STRIDE = BK + APAD;   // 40
constexpr int BS_STRIDE = BN + BPAD;   // 136
constexpr int AS_SIZE = BM * AS_STRIDE;  // 5120 halves
constexpr int BS_SIZE = BK * BS_STRIDE;  // 4352 halves
constexpr int STAGE_HALFS = AS_SIZE + BS_SIZE;

__device__ __forceinline__ void cp_async_cg16(void* smem, const void* gmem)
{
    unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s),
                 "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit()
{
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void cp_async_wait()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__global__ void __launch_bounds__(THREADS)
    wmma_v2(const half* __restrict__ A, const half* __restrict__ B,
            half* __restrict__ C, int M, int N, int K)
{
    extern __shared__ half smem[];
    half* As[STAGES];
    half* Bs[STAGES];
#pragma unroll
    for (int s = 0; s < STAGES; ++s) {
        As[s] = smem + s * STAGE_HALFS;
        Bs[s] = As[s] + AS_SIZE;
    }

    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;
    const int tid = threadIdx.x;
    const int warpId = tid >> 5;
    const int warpRow = warpId / WARP_N;
    const int warpCol = warpId % WARP_N;

    const int aRowBase = blockRow * BM;
    const int bColBase = blockCol * BN;

    wmma::fragment<wmma::accumulator, WM, WN, WK, float> acc[FRAG_M][FRAG_N];
#pragma unroll
    for (int i = 0; i < FRAG_M; ++i)
#pragma unroll
        for (int j = 0; j < FRAG_N; ++j)
            wmma::fill_fragment(acc[i][j], 0.0f);

    const int kSteps = K / BK;

    auto load_stage = [&](int st, int kt) {
        const int kBase = kt * BK;
#pragma unroll
        for (int i = 0; i < 2; ++i) {  // As: 512 float4 / 256 threads = 2 each
            const int f = tid + i * THREADS;
            const int row = f >> 2;       // 4 float4 per row (32 cols)
            const int colH = (f & 3) * 8;
            const half* g = A + (size_t)(aRowBase + row) * K + kBase + colH;
            cp_async_cg16(&As[st][row * AS_STRIDE + colH], g);
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {  // Bs: 512 float4 / 256 threads = 2 each
            const int f = tid + i * THREADS;
            const int row = f >> 4;       // 16 float4 per row (128 cols)
            const int col = (f & 15) * 8;
            const half* g = B + (size_t)(kBase + row) * N + bColBase + col;
            cp_async_cg16(&Bs[st][row * BS_STRIDE + col], g);
        }
        cp_async_commit();
    };

    // prologue: issue first STAGES-1 stage loads
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s)
        load_stage(s, s);

    for (int kt = 0; kt < kSteps; ++kt) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();

        const int r = kt % STAGES;
        wmma::fragment<wmma::matrix_a, WM, WN, WK, half, wmma::row_major>
            aFrag[FRAG_M];
        wmma::fragment<wmma::matrix_b, WM, WN, WK, half, wmma::row_major>
            bFrag[FRAG_N];
#pragma unroll
        for (int kk = 0; kk < FRAG_K; ++kk) {
#pragma unroll
            for (int m = 0; m < FRAG_M; ++m)
                wmma::load_matrix_sync(
                    aFrag[m],
                    &As[r][(warpRow * WARP_TILE_M + m * WM) * AS_STRIDE +
                           kk * WK],
                    AS_STRIDE);
#pragma unroll
            for (int n = 0; n < FRAG_N; ++n)
                wmma::load_matrix_sync(
                    bFrag[n],
                    &Bs[r][(kk * WK) * BS_STRIDE + warpCol * WARP_TILE_N +
                           n * WN],
                    BS_STRIDE);
#pragma unroll
            for (int m = 0; m < FRAG_M; ++m)
#pragma unroll
                for (int n = 0; n < FRAG_N; ++n)
                    wmma::mma_sync(acc[m][n], aFrag[m], bFrag[n], acc[m][n]);
        }

        // prefetch the stage that frees up
        const int nextK = kt + STAGES - 1;
        if (nextK < kSteps)
            load_stage(nextK % STAGES, nextK);
    }

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

PLAYGROUND_MATMUL_DEC(float16_t, 2, m, n, k, A, B, C)
{
    static bool configured = false;
    constexpr int smemBytes = STAGES * STAGE_HALFS * sizeof(half);
    if (!configured) {
        cudaFuncSetAttribute(wmma_v2,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smemBytes);
        configured = true;
    }
    dim3 block(THREADS);
    dim3 grid(n / BN, m / BM);
    wmma_v2<<<grid, block, smemBytes>>>(A, B, C, (int) m, (int) n, (int) k);
}

}  // namespace playground
