#include "playground/matmul.hpp"
#include "playground/system.hpp"

#include <cuda_fp16.h>
#include <mma.h>

namespace playground
{
namespace v1
{
using namespace nvcuda;

// Block tile / warp tile configuration.
constexpr int BM = 128, BN = 128, BK = 32;
constexpr int WM = 32, WN = 64;            // per-warp output tile
constexpr int WARPS_M = BM / WM;           // 4
constexpr int WARPS_N = BN / WN;           // 2
constexpr int NWARPS = WARPS_M * WARPS_N;  // 8
constexpr int NTHREADS = NWARPS * 32;      // 256

constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
constexpr int FM = WM / WMMA_M;  // 2 fragments along M
constexpr int FN = WN / WMMA_N;  // 4 fragments along N

constexpr int APAD = 8, BPAD = 8;  // shared-memory padding (avoid bank conflicts)

__global__ __launch_bounds__(NTHREADS) void kernel(const half* __restrict__ A,
                                                    const half* __restrict__ B,
                                                    half* __restrict__ C, int M,
                                                    int N, int K)
{
    const int bx = blockIdx.x;  // along N
    const int by = blockIdx.y;  // along M
    const int tid = threadIdx.x;
    const int warpId = tid >> 5;
    const int warpM = warpId / WARPS_N;  // 0..3
    const int warpN = warpId % WARPS_N;  // 0..1

    __shared__ __align__(16) half As[BM][BK + APAD];
    __shared__ __align__(16) half Bs[BK][BN + BPAD];

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[FM][FN];
#pragma unroll
    for (int i = 0; i < FM; i++)
#pragma unroll
        for (int j = 0; j < FN; j++)
            wmma::fill_fragment(acc[i][j], 0.0f);

    const int rowBase = by * BM;
    const int colBase = bx * BN;

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Global -> shared, vectorized 128-bit (8 halfs) loads.
#pragma unroll
        for (int i = tid; i < BM * BK / 8; i += NTHREADS) {
            int r = (i * 8) / BK;
            int c = (i * 8) % BK;
            *reinterpret_cast<float4*>(&As[r][c]) =
                *reinterpret_cast<const float4*>(&A[(rowBase + r) * K + k0 + c]);
        }
#pragma unroll
        for (int i = tid; i < BK * BN / 8; i += NTHREADS) {
            int r = (i * 8) / BN;
            int c = (i * 8) % BN;
            *reinterpret_cast<float4*>(&Bs[r][c]) =
                *reinterpret_cast<const float4*>(&B[(k0 + r) * N + colBase + c]);
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                           wmma::row_major>
                a_frag[FM];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                           wmma::row_major>
                b_frag[FN];
#pragma unroll
            for (int i = 0; i < FM; i++)
                wmma::load_matrix_sync(a_frag[i],
                                       &As[warpM * WM + i * WMMA_M][kk],
                                       BK + APAD);
#pragma unroll
            for (int j = 0; j < FN; j++)
                wmma::load_matrix_sync(b_frag[j],
                                       &Bs[kk][warpN * WN + j * WMMA_N],
                                       BN + BPAD);
#pragma unroll
            for (int i = 0; i < FM; i++)
#pragma unroll
                for (int j = 0; j < FN; j++)
                    wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }
        __syncthreads();
    }

    // Epilogue: fp32 accumulator -> fp16 via per-warp shared scratch.
    // NOTE: store_matrix_sync requires an alignment-friendly ldm; an odd pad
    // (e.g. WMMA_N+1) silently produces shifted results, so keep ldm == WMMA_N.
    __shared__ float Cs[NWARPS][WMMA_M][WMMA_N];
    const int lane = tid & 31;
#pragma unroll
    for (int i = 0; i < FM; i++) {
#pragma unroll
        for (int j = 0; j < FN; j++) {
            __syncwarp();
            wmma::store_matrix_sync(&Cs[warpId][0][0], acc[i][j], WMMA_N,
                                    wmma::mem_row_major);
            __syncwarp();
#pragma unroll
            for (int e = 0; e < (WMMA_M * WMMA_N) / 32; e++) {
                int idx = lane + e * 32;
                int rr = idx / WMMA_N;
                int cc = idx % WMMA_N;
                int gr = rowBase + warpM * WM + i * WMMA_M + rr;
                int gc = colBase + warpN * WN + j * WMMA_N + cc;
                C[gr * N + gc] = __float2half(Cs[warpId][rr][cc]);
            }
        }
    }
}

}  // namespace v1

PLAYGROUND_MATMUL_DEC(float16_t, 1, m, n, k, A, B, C)
{
    dim3 block(v1::NTHREADS);
    dim3 grid(n / v1::BN, m / v1::BM);
    v1::kernel<<<grid, block>>>(A, B, C, int(m), int(n), int(k));
}

}  // namespace playground
