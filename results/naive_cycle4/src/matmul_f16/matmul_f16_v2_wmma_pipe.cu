#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <mma.h>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{
using namespace nvcuda;

namespace v2
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
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int FRAG_M = WM / WMMA_M;  // 4
constexpr int FRAG_N = WN / WMMA_N;  // 2

constexpr int NTHREADS = WARPS_M * WARPS_N * 32;  // 256
constexpr int PAD = 8;
constexpr int LDA_S = BK + PAD;  // 40
constexpr int LDB_S = BN + PAD;  // 136
constexpr int NSTAGES = 2;

__device__ __forceinline__ void cp_async16(half* dst, const half* src)
{
    unsigned smem = __cvta_generic_to_shared(dst);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem), "l"(src));
}

__global__ void __launch_bounds__(NTHREADS)
    matmul_kernel(int M, int N, int K, const half* __restrict__ A,
                  const half* __restrict__ B, half* __restrict__ C)
{
    __shared__ half As[NSTAGES][BM * LDA_S];
    __shared__ half Bs[NSTAGES][BK * LDB_S];

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    const int warpId = threadIdx.x / 32;
    const int warpRow = (warpId / WARPS_N) * WM;
    const int warpCol = (warpId % WARPS_N) * WN;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half>
        acc[FRAG_M][FRAG_N];
#pragma unroll
    for (int i = 0; i < FRAG_M; ++i)
#pragma unroll
        for (int j = 0; j < FRAG_N; ++j)
            wmma::fill_fragment(acc[i][j], __float2half(0.0f));

    const int tid = threadIdx.x;
    const int a_row0 = tid / (BK / 8);
    const int a_col0 = (tid % (BK / 8)) * 8;
    constexpr int A_ROW_STRIDE = NTHREADS / (BK / 8);  // 64
    const int b_row0 = tid / (BN / 8);
    const int b_col0 = (tid % (BN / 8)) * 8;
    constexpr int B_ROW_STRIDE = NTHREADS / (BN / 8);  // 16

    const int nKTiles = K / BK;

    auto load_tile = [&](int kt, int stage) {
#pragma unroll
        for (int r = 0; r < BM; r += A_ROW_STRIDE) {
            int gr = blockRow + a_row0 + r;
            int gc = kt + a_col0;
            cp_async16(&As[stage][(a_row0 + r) * LDA_S + a_col0],
                       &A[gr * K + gc]);
        }
#pragma unroll
        for (int r = 0; r < BK; r += B_ROW_STRIDE) {
            int gr = kt + b_row0 + r;
            int gc = blockCol + b_col0;
            cp_async16(&Bs[stage][(b_row0 + r) * LDB_S + b_col0],
                       &B[gr * N + gc]);
        }
        __pipeline_commit();
    };

    // Prologue: kick off first NSTAGES-1 loads
#pragma unroll
    for (int s = 0; s < NSTAGES - 1; ++s) {
        load_tile(s * BK, s);
    }

    int writeStage = NSTAGES - 1;
    int readStage = 0;

    for (int kt = 0; kt < nKTiles; ++kt) {
        // Issue load for the tile NSTAGES-1 ahead
        int loadKt = kt + (NSTAGES - 1);
        if (loadKt < nKTiles) {
            load_tile(loadKt * BK, writeStage);
        } else {
            __pipeline_commit();  // keep group count consistent
        }
        // Wait until the tile we want to read is ready
        __pipeline_wait_prior(NSTAGES - 1);
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
                    a_frag[i], &As[readStage][(warpRow + i * WMMA_M) * LDA_S + kk],
                    LDA_S);
#pragma unroll
            for (int j = 0; j < FRAG_N; ++j)
                wmma::load_matrix_sync(
                    b_frag[j],
                    &Bs[readStage][kk * LDB_S + warpCol + j * WMMA_N], LDB_S);
#pragma unroll
            for (int i = 0; i < FRAG_M; ++i)
#pragma unroll
                for (int j = 0; j < FRAG_N; ++j)
                    wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }
        __syncthreads();

        writeStage = (writeStage + 1) % NSTAGES;
        readStage = (readStage + 1) % NSTAGES;
    }

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

}  // namespace v2

PLAYGROUND_MATMUL_DEC(float16_t, 2, m, n, k, A, B, C)
{
    dim3 block(v2::NTHREADS);
    dim3 grid((n + v2::BN - 1) / v2::BN, (m + v2::BM - 1) / v2::BM);
    v2::matmul_kernel<<<grid, block>>>(int(m), int(n), int(k), A, B, C);
}

}  // namespace playground
