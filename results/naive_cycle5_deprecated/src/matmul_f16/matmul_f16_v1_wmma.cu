#include <cuda_fp16.h>
#include <mma.h>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{
namespace v1
{
using namespace nvcuda;

// ---- Block / warp tiling configuration --------------------------------------
constexpr int BM = 128;   // block tile rows (M)
constexpr int BN = 128;   // block tile cols (N)
constexpr int BK = 32;    // block tile depth (K)
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 4;
constexpr int NWARPS = WARPS_M * WARPS_N;          // 8 warps
constexpr int NTHREADS = NWARPS * 32;              // 256 threads
constexpr int WARP_M = BM / WARPS_M;               // 64
constexpr int WARP_N = BN / WARPS_N;               // 32
constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
constexpr int MFRAG = WARP_M / WMMA_M;             // 4
constexpr int NFRAG = WARP_N / WMMA_N;             // 2
// Shared-memory skew (in halfs) to reduce bank conflicts.
constexpr int SKEW = 8;
constexpr int LDA = BK + SKEW;                     // 40
constexpr int LDB = BN + SKEW;                     // 136

__global__ void __launch_bounds__(NTHREADS)
    wmma_kernel(int M, int N, int K, const half* __restrict__ A,
                const half* __restrict__ B, half* __restrict__ C)
{
    __shared__ half As[BM * LDA];
    __shared__ half Bs[BK * LDB];

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    const int warpId = threadIdx.x >> 5;
    const int warpM = warpId / WARPS_N;   // 0..1
    const int warpN = warpId % WARPS_N;   // 0..3
    const int warpRow = warpM * WARP_M;   // row offset within block tile
    const int warpCol = warpN * WARP_N;   // col offset within block tile

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        acc[MFRAG][NFRAG];
#pragma unroll
    for (int i = 0; i < MFRAG; ++i)
#pragma unroll
        for (int j = 0; j < NFRAG; ++j)
            wmma::fill_fragment(acc[i][j], 0.0f);

    const int tid = threadIdx.x;

    for (int k0 = 0; k0 < K; k0 += BK) {
        // ---- Load A tile [BM x BK] (row-major) into As --------------------
        // 128*32 = 4096 halfs = 512 float4; 256 threads -> 2 float4 each.
#pragma unroll
        for (int i = 0; i < (BM * BK / 8) / NTHREADS; ++i) {
            int idx = tid + i * NTHREADS;          // float4 index
            int row = idx / (BK / 8);              // BK/8 = 4 float4 per row
            int col8 = (idx % (BK / 8)) * 8;       // half-column within tile
            const float4* gptr = reinterpret_cast<const float4*>(
                &A[(blockRow + row) * K + k0 + col8]);
            *reinterpret_cast<float4*>(&As[row * LDA + col8]) = *gptr;
        }
        // ---- Load B tile [BK x BN] (row-major) into Bs --------------------
#pragma unroll
        for (int i = 0; i < (BK * BN / 8) / NTHREADS; ++i) {
            int idx = tid + i * NTHREADS;
            int row = idx / (BN / 8);              // BN/8 = 16 float4 per row
            int col8 = (idx % (BN / 8)) * 8;
            const float4* gptr = reinterpret_cast<const float4*>(
                &B[(k0 + row) * N + blockCol + col8]);
            *reinterpret_cast<float4*>(&Bs[row * LDB + col8]) = *gptr;
        }
        __syncthreads();

        // ---- Compute -----------------------------------------------------
#pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                           wmma::row_major>
                aFrag[MFRAG];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                           wmma::row_major>
                bFrag[NFRAG];
#pragma unroll
            for (int mi = 0; mi < MFRAG; ++mi)
                wmma::load_matrix_sync(
                    aFrag[mi], &As[(warpRow + mi * WMMA_M) * LDA + kk], LDA);
#pragma unroll
            for (int ni = 0; ni < NFRAG; ++ni)
                wmma::load_matrix_sync(
                    bFrag[ni], &Bs[kk * LDB + warpCol + ni * WMMA_N], LDB);
#pragma unroll
            for (int mi = 0; mi < MFRAG; ++mi)
#pragma unroll
                for (int ni = 0; ni < NFRAG; ++ni)
                    wmma::mma_sync(acc[mi][ni], aFrag[mi], bFrag[ni],
                                   acc[mi][ni]);
        }
        __syncthreads();
    }

    // ---- Epilogue: convert f32 acc -> half and store to C ----------------
#pragma unroll
    for (int mi = 0; mi < MFRAG; ++mi) {
#pragma unroll
        for (int ni = 0; ni < NFRAG; ++ni) {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half>
                cFrag;
#pragma unroll
            for (int t = 0; t < acc[mi][ni].num_elements; ++t)
                cFrag.x[t] = __float2half(acc[mi][ni].x[t]);
            int cRow = blockRow + warpRow + mi * WMMA_M;
            int cCol = blockCol + warpCol + ni * WMMA_N;
            wmma::store_matrix_sync(&C[cRow * N + cCol], cFrag, N,
                                    wmma::mem_row_major);
        }
    }
}

}  // namespace v1

PLAYGROUND_MATMUL_DEC(float16_t, 1, m, n, k, A, B, C)
{
    dim3 block(v1::NTHREADS);
    dim3 grid(n / v1::BN, m / v1::BM);
    v1::wmma_kernel<<<grid, block>>>(int(m), int(n), int(k), A, B, C);
}

}  // namespace playground
