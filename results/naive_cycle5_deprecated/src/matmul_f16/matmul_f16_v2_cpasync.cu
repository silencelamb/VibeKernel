#include <cuda_fp16.h>
#include <mma.h>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{
namespace v2
{
using namespace nvcuda;

constexpr int BM = 128, BN = 128, BK = 32;
constexpr int WARPS_M = 2, WARPS_N = 4;
constexpr int NWARPS = WARPS_M * WARPS_N;
constexpr int NTHREADS = NWARPS * 32;          // 256
constexpr int WARP_M = BM / WARPS_M;           // 64
constexpr int WARP_N = BN / WARPS_N;           // 32
constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
constexpr int MFRAG = WARP_M / WMMA_M;         // 4
constexpr int NFRAG = WARP_N / WMMA_N;         // 2
constexpr int SKEW = 8;
constexpr int LDA = BK + SKEW;                 // 40
constexpr int LDB = BN + SKEW;                 // 136
constexpr int STAGES = 3;
constexpr int AS = BM * LDA;                    // halfs per A stage
constexpr int BS = BK * LDB;                    // halfs per B stage

__device__ __forceinline__ uint32_t smem_addr(const void* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(void* d, const void* s)
{
    asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n" ::"r"(smem_addr(d)),
                 "l"(s));
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

__global__ void __launch_bounds__(NTHREADS)
    kernel(int M, int N, int K, const half* __restrict__ A,
           const half* __restrict__ B, half* __restrict__ C)
{
    extern __shared__ half smem[];
    half* As = smem;                 // STAGES * AS
    half* Bs = smem + STAGES * AS;   // STAGES * BS

    const int blockRow = blockIdx.y * BM, blockCol = blockIdx.x * BN;
    const int warpId = threadIdx.x >> 5;
    const int warpRow = (warpId / WARPS_N) * WARP_M;
    const int warpCol = (warpId % WARPS_N) * WARP_N;
    const int tid = threadIdx.x;
    const int numK = K / BK;

    auto load_tile = [&](int kt, int st) {
        int k0 = kt * BK;
        half* Asb = As + st * AS;
        half* Bsb = Bs + st * BS;
#pragma unroll
        for (int i = 0; i < (BM * BK / 8) / NTHREADS; ++i) {
            int idx = tid + i * NTHREADS;
            int row = idx / (BK / 8);
            int col8 = (idx % (BK / 8)) * 8;
            cp_async16(&Asb[row * LDA + col8],
                       &A[(blockRow + row) * K + k0 + col8]);
        }
#pragma unroll
        for (int i = 0; i < (BK * BN / 8) / NTHREADS; ++i) {
            int idx = tid + i * NTHREADS;
            int row = idx / (BN / 8);
            int col8 = (idx % (BN / 8)) * 8;
            cp_async16(&Bsb[row * LDB + col8],
                       &B[(k0 + row) * N + blockCol + col8]);
        }
        cp_commit();
    };

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[MFRAG][NFRAG];
#pragma unroll
    for (int i = 0; i < MFRAG; ++i)
#pragma unroll
        for (int j = 0; j < NFRAG; ++j) wmma::fill_fragment(acc[i][j], 0.0f);

    // Prologue: prefetch STAGES-1 tiles.
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) load_tile(s, s);

    for (int kt = 0; kt < numK; ++kt) {
        cp_wait<STAGES - 2>();
        __syncthreads();

        int cur = kt % STAGES;
        half* Asb = As + cur * AS;
        half* Bsb = Bs + cur * BS;
#pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>
                aF[MFRAG];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>
                bF[NFRAG];
#pragma unroll
            for (int mi = 0; mi < MFRAG; ++mi)
                wmma::load_matrix_sync(aF[mi],
                                       &Asb[(warpRow + mi * 16) * LDA + kk], LDA);
#pragma unroll
            for (int ni = 0; ni < NFRAG; ++ni)
                wmma::load_matrix_sync(bF[ni],
                                       &Bsb[kk * LDB + warpCol + ni * 16], LDB);
#pragma unroll
            for (int mi = 0; mi < MFRAG; ++mi)
#pragma unroll
                for (int ni = 0; ni < NFRAG; ++ni)
                    wmma::mma_sync(acc[mi][ni], aF[mi], bF[ni], acc[mi][ni]);
        }

        int nxt = kt + STAGES - 1;
        if (nxt < numK) load_tile(nxt, nxt % STAGES);
    }

#pragma unroll
    for (int mi = 0; mi < MFRAG; ++mi) {
#pragma unroll
        for (int ni = 0; ni < NFRAG; ++ni) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, half> cF;
#pragma unroll
            for (int t = 0; t < acc[mi][ni].num_elements; ++t)
                cF.x[t] = __float2half(acc[mi][ni].x[t]);
            int cRow = blockRow + warpRow + mi * 16;
            int cCol = blockCol + warpCol + ni * 16;
            wmma::store_matrix_sync(&C[cRow * N + cCol], cF, N,
                                    wmma::mem_row_major);
        }
    }
}

}  // namespace v2

PLAYGROUND_MATMUL_DEC(float16_t, 2, m, n, k, A, B, C)
{
    constexpr int smemBytes =
        v2::STAGES * (v2::AS + v2::BS) * int(sizeof(half));
    static bool attrSet = false;
    if (!attrSet) {
        cudaFuncSetAttribute(v2::kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smemBytes);
        attrSet = true;
    }
    dim3 block(v2::NTHREADS);
    dim3 grid(n / v2::BN, m / v2::BM);
    v2::kernel<<<grid, block, smemBytes>>>(int(m), int(n), int(k), A, B, C);
}

}  // namespace playground
