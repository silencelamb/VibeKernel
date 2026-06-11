#include <cuda_fp16.h>
#include <mma.h>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{
namespace v1
{
using namespace nvcuda;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Block tile.
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;

// Warp tile: 8 warps (256 threads) laid out 4 (M) x 2 (N).
constexpr int WARPS_M = 4;
constexpr int WARPS_N = 2;
constexpr int WARP_M = BM / WARPS_M;  // 32
constexpr int WARP_N = BN / WARPS_N;  // 64
constexpr int FRAG_M = WARP_M / WMMA_M;  // 2
constexpr int FRAG_N = WARP_N / WMMA_N;  // 4

constexpr int THREADS = WARPS_M * WARPS_N * 32;  // 256

// Shared-memory padding (in halfs) to reduce bank conflicts.
constexpr int APAD = 8;
constexpr int BPAD = 8;

__global__ void __launch_bounds__(THREADS)
    kernel(int M, int N, int K, const half* __restrict__ A,
           const half* __restrict__ B, half* __restrict__ C)
{
    __shared__ half As[BM][BK + APAD];
    __shared__ half Bs[BK][BN + BPAD];

    const int blockRow = blockIdx.y;  // along M
    const int blockCol = blockIdx.x;  // along N

    const int warpId = threadIdx.x / 32;
    const int warpRow = warpId / WARPS_N;  // 0..3
    const int warpCol = warpId % WARPS_N;  // 0..1

    // Accumulator fragments (float).
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        acc[FRAG_M][FRAG_N];
#pragma unroll
    for (int i = 0; i < FRAG_M; ++i)
#pragma unroll
        for (int j = 0; j < FRAG_N; ++j)
            wmma::fill_fragment(acc[i][j], 0.0f);

    const int rowBase = blockRow * BM;
    const int colBase = blockCol * BN;

    // For loading A tile (BM x BK) with float4 (8 halfs) vectorized loads.
    // 128*32/8 = 512 float4 / 256 threads = 2 per thread.
    const int tid = threadIdx.x;

    for (int k0 = 0; k0 < K; k0 += BK) {
        // --- load A tile: As[row][col] = A[(rowBase+row)*K + k0+col] ---
#pragma unroll
        for (int it = 0; it < 2; ++it) {
            int f = tid + it * THREADS;           // float4 index, 0..511
            int row = f / (BK / 8);               // BK/8 = 4
            int col = (f % (BK / 8)) * 8;
            const float4* gptr = reinterpret_cast<const float4*>(
                &A[(rowBase + row) * K + k0 + col]);
            *reinterpret_cast<float4*>(&As[row][col]) = *gptr;
        }
        // --- load B tile: Bs[row][col] = B[(k0+row)*N + colBase+col] ---
#pragma unroll
        for (int it = 0; it < 2; ++it) {
            int f = tid + it * THREADS;           // 0..511
            int row = f / (BN / 8);               // BN/8 = 16
            int col = (f % (BN / 8)) * 8;
            const float4* gptr = reinterpret_cast<const float4*>(
                &B[(k0 + row) * N + colBase + col]);
            *reinterpret_cast<float4*>(&Bs[row][col]) = *gptr;
        }
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
                wmma::load_matrix_sync(a_frag[i], &As[ar][kk], BK + APAD);
            }
#pragma unroll
            for (int j = 0; j < FRAG_N; ++j) {
                int bc = warpCol * WARP_N + j * WMMA_N;
                wmma::load_matrix_sync(b_frag[j], &Bs[kk][bc], BN + BPAD);
            }
#pragma unroll
            for (int i = 0; i < FRAG_M; ++i)
#pragma unroll
                for (int j = 0; j < FRAG_N; ++j)
                    wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }
        __syncthreads();
    }

    // --- store: convert f32 acc -> half fragment, store to C ---
#pragma unroll
    for (int i = 0; i < FRAG_M; ++i) {
#pragma unroll
        for (int j = 0; j < FRAG_N; ++j) {
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half>
                ch;
#pragma unroll
            for (int t = 0; t < acc[i][j].num_elements; ++t)
                ch.x[t] = __float2half(acc[i][j].x[t]);
            int cr = rowBase + warpRow * WARP_M + i * WMMA_M;
            int cc = colBase + warpCol * WARP_N + j * WMMA_N;
            wmma::store_matrix_sync(&C[cr * N + cc], ch, N,
                                    wmma::mem_row_major);
        }
    }
}
}  // namespace v1

PLAYGROUND_MATMUL_DEC(float16_t, 1, m, n, k, A, B, C)
{
    dim3 block(v1::THREADS);
    dim3 grid(n / v1::BN, m / v1::BM);
    v1::kernel<<<grid, block>>>((int) m, (int) n, (int) k, A, B, C);
}

}  // namespace playground
