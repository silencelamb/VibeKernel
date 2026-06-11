#include "playground/matmul.hpp"
#include "playground/system.hpp"

#include <cuda_fp16.h>
#include <mma.h>

namespace playground
{
namespace v2
{
using namespace nvcuda;

constexpr int BM = 128, BN = 128, BK = 32;
constexpr int WM = 32, WN = 64;
constexpr int WARPS_M = BM / WM, WARPS_N = BN / WN;
constexpr int NWARPS = WARPS_M * WARPS_N, NTHREADS = NWARPS * 32;
constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
constexpr int FM = WM / WMMA_M, FN = WN / WMMA_N;
constexpr int APAD = 8, BPAD = 8;
constexpr int NSTAGES = 3;
constexpr int AS_STRIDE = BK + APAD, AS_TILE = BM * AS_STRIDE;
constexpr int BS_STRIDE = BN + BPAD, BS_TILE = BK * BS_STRIDE;

__device__ __forceinline__ void cp_async_cg(void* smem, const void* gmem)
{
    unsigned s = __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s),
                 "l"(gmem));
}
__device__ __forceinline__ void cp_commit()
{
    asm volatile("cp.async.commit_group;\n");
}
template <int N>
__device__ __forceinline__ void cp_wait()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__global__ __launch_bounds__(NTHREADS) void kernel(const half* __restrict__ A,
                                                    const half* __restrict__ B,
                                                    half* __restrict__ C, int M,
                                                    int N, int K)
{
    extern __shared__ half smem[];
    half* As = smem;
    half* Bs = smem + NSTAGES * AS_TILE;

    const int bx = blockIdx.x, by = blockIdx.y, tid = threadIdx.x;
    const int warpId = tid >> 5, warpM = warpId / WARPS_N,
              warpN = warpId % WARPS_N;
    const int rowBase = by * BM, colBase = bx * BN;
    const int numTiles = K / BK;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[FM][FN];
#pragma unroll
    for (int i = 0; i < FM; i++)
#pragma unroll
        for (int j = 0; j < FN; j++)
            wmma::fill_fragment(acc[i][j], 0.0f);

    auto load_stage = [&](int stage, int k0) {
#pragma unroll
        for (int i = tid; i < BM * BK / 8; i += NTHREADS) {
            int r = (i * 8) / BK, c = (i * 8) % BK;
            cp_async_cg(&As[stage * AS_TILE + r * AS_STRIDE + c],
                        &A[(rowBase + r) * K + k0 + c]);
        }
#pragma unroll
        for (int i = tid; i < BK * BN / 8; i += NTHREADS) {
            int r = (i * 8) / BN, c = (i * 8) % BN;
            cp_async_cg(&Bs[stage * BS_TILE + r * BS_STRIDE + c],
                        &B[(k0 + r) * N + colBase + c]);
        }
    };

#pragma unroll
    for (int s = 0; s < NSTAGES - 1; s++) {
        load_stage(s, s * BK);
        cp_commit();
    }

    for (int t = 0; t < numTiles; t++) {
        cp_wait<NSTAGES - 2>();
        __syncthreads();
        int cur = t % NSTAGES;
        int pf = t + NSTAGES - 1;
        if (pf < numTiles)
            load_stage(pf % NSTAGES, pf * BK);
        cp_commit();

        half* Acur = As + cur * AS_TILE;
        half* Bcur = Bs + cur * BS_TILE;
#pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>
                a_frag[FM];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>
                b_frag[FN];
#pragma unroll
            for (int i = 0; i < FM; i++)
                wmma::load_matrix_sync(
                    a_frag[i],
                    &Acur[(warpM * WM + i * WMMA_M) * AS_STRIDE + kk],
                    AS_STRIDE);
#pragma unroll
            for (int j = 0; j < FN; j++)
                wmma::load_matrix_sync(
                    b_frag[j], &Bcur[kk * BS_STRIDE + warpN * WN + j * WMMA_N],
                    BS_STRIDE);
#pragma unroll
            for (int i = 0; i < FM; i++)
#pragma unroll
                for (int j = 0; j < FN; j++)
                    wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }
    }

    // Epilogue: reuse smem as fp32 scratch (As/Bs no longer needed).
    cp_wait<0>();
    __syncthreads();
    float* Cs = reinterpret_cast<float*>(smem);
    const int lane = tid & 31;
#pragma unroll
    for (int i = 0; i < FM; i++)
#pragma unroll
        for (int j = 0; j < FN; j++) {
            __syncwarp();
            wmma::store_matrix_sync(&Cs[warpId * 256], acc[i][j], 16,
                                    wmma::mem_row_major);
            __syncwarp();
#pragma unroll
            for (int e = 0; e < 8; e++) {
                int idx = lane + e * 32, rr = idx / 16, cc = idx % 16;
                int gr = rowBase + warpM * WM + i * WMMA_M + rr;
                int gc = colBase + warpN * WN + j * WMMA_N + cc;
                C[gr * N + gc] = __float2half(Cs[warpId * 256 + rr * 16 + cc]);
            }
        }
}

}  // namespace v2

PLAYGROUND_MATMUL_DEC(float16_t, 2, m, n, k, A, B, C)
{
    static bool inited = false;
    constexpr int smemBytes =
        (v2::NSTAGES * v2::AS_TILE + v2::NSTAGES * v2::BS_TILE) * sizeof(half);
    if (!inited) {
        cudaFuncSetAttribute((const void*) v2::kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smemBytes);
        inited = true;
    }
    dim3 block(v2::NTHREADS);
    dim3 grid(n / v2::BN, m / v2::BM);
    v2::kernel<<<grid, block, smemBytes>>>(A, B, C, int(m), int(n), int(k));
}

}  // namespace playground
