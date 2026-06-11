#include <cuda_fp16.h>
#include <mma.h>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{
using namespace nvcuda;

namespace v2
{
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int NSTAGE = 3;

constexpr int WARPS_M = 2;
constexpr int WARPS_N = 4;
constexpr int WM = BM / WARPS_M;  // 64
constexpr int WN = BN / WARPS_N;  // 32

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int WTILES_M = WM / WMMA_M;  // 4
constexpr int WTILES_N = WN / WMMA_N;  // 2

constexpr int THREADS = WARPS_M * WARPS_N * 32;  // 256

constexpr int STILE_A = BM * BK;  // halfs per stage for A
constexpr int STILE_B = BK * BN;  // halfs per stage for B
constexpr int SMEM_HALFS = NSTAGE * (STILE_A + STILE_B);

__device__ __forceinline__ unsigned smem_u32(const void* p)
{
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async_cg16(void* dst, const void* src)
{
    unsigned s = smem_u32(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s),
                 "l"(src));
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

__global__ __launch_bounds__(THREADS) void kernel(int m, int n, int k,
                                                  const half* __restrict__ A,
                                                  const half* __restrict__ B,
                                                  half* __restrict__ C)
{
    extern __shared__ half smem[];
    half* As = smem;                          // NSTAGE * STILE_A
    half* Bs = smem + NSTAGE * STILE_A;       // NSTAGE * STILE_B

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    const int tid = threadIdx.x;
    const int warpId = tid / 32;
    const int warpRow = warpId / WARPS_N;
    const int warpCol = warpId % WARPS_N;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        acc[WTILES_M][WTILES_N];
#pragma unroll
    for (int i = 0; i < WTILES_M; ++i)
#pragma unroll
        for (int j = 0; j < WTILES_N; ++j)
            wmma::fill_fragment(acc[i][j], 0.0f);

    const int numK = k / BK;

    // Each thread issues 2 float4 loads for A and 2 for B per stage.
    // A tile row r in [0,128), col c8 in {0,8,16,24}; 512 float4 / 256 thr = 2
    auto loadStage = [&](int stage, int k0) {
        half* Ad = As + stage * STILE_A;
        half* Bd = Bs + stage * STILE_B;
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            int e = p * THREADS + tid;            // 0..511
            int r = e / (BK / 8);                 // 0..127
            int c8 = (e % (BK / 8)) * 8;          // 0,8,16,24
            const half* gptr = A + (size_t(blockRow + r) * k) + (k0 + c8);
            cp_async_cg16(&Ad[r * BK + c8], gptr);
        }
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            int e = p * THREADS + tid;            // 0..511
            int r = e / (BN / 8);                 // 0..31
            int c8 = (e % (BN / 8)) * 8;          // 0..120
            const half* gptr = B + (size_t(k0 + r) * n) + (blockCol + c8);
            cp_async_cg16(&Bd[r * BN + c8], gptr);
        }
        cp_async_commit();
    };

    // Prologue: issue first NSTAGE-1 stages.
#pragma unroll
    for (int s = 0; s < NSTAGE - 1; ++s) {
        loadStage(s, s * BK);
    }

    int writeStage = NSTAGE - 1;
    int readStage = 0;

    for (int kt = 0; kt < numK; ++kt) {
        // Issue load for the stage NSTAGE-1 ahead (if exists).
        int loadKt = kt + (NSTAGE - 1);
        if (loadKt < numK) {
            loadStage(writeStage, loadKt * BK);
        }
        // Wait until the tile we are about to read is ready: we have at most
        // NSTAGE-1 outstanding groups after issuing; wait so that <= NSTAGE-2
        // remain (i.e. readStage's group completed).
        cp_async_wait<NSTAGE - 2>();
        __syncthreads();

        half* Ar = As + readStage * STILE_A;
        half* Br = Bs + readStage * STILE_B;
#pragma unroll
        for (int kk = 0; kk < BK; kk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half,
                           wmma::row_major>
                aFrag[WTILES_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half,
                           wmma::row_major>
                bFrag[WTILES_N];
#pragma unroll
            for (int i = 0; i < WTILES_M; ++i) {
                int aRow = warpRow * WM + i * WMMA_M;
                wmma::load_matrix_sync(aFrag[i], &Ar[aRow * BK + kk], BK);
            }
#pragma unroll
            for (int j = 0; j < WTILES_N; ++j) {
                int bCol = warpCol * WN + j * WMMA_N;
                wmma::load_matrix_sync(bFrag[j], &Br[kk * BN + bCol], BN);
            }
#pragma unroll
            for (int i = 0; i < WTILES_M; ++i)
#pragma unroll
                for (int j = 0; j < WTILES_N; ++j)
                    wmma::mma_sync(acc[i][j], aFrag[i], bFrag[j], acc[i][j]);
        }
        __syncthreads();

        writeStage = (writeStage + 1) % NSTAGE;
        readStage = (readStage + 1) % NSTAGE;
    }

#pragma unroll
    for (int i = 0; i < WTILES_M; ++i) {
#pragma unroll
        for (int j = 0; j < WTILES_N; ++j) {
            int cRow = blockRow + warpRow * WM + i * WMMA_M;
            int cCol = blockCol + warpCol * WN + j * WMMA_N;
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half>
                cFrag;
#pragma unroll
            for (int t = 0; t < cFrag.num_elements; ++t)
                cFrag.x[t] = __float2half(acc[i][j].x[t]);
            wmma::store_matrix_sync(&C[cRow * n + cCol], cFrag, n,
                                    wmma::mem_row_major);
        }
    }
}
}  // namespace v2

PLAYGROUND_MATMUL_DEC(float16_t, 2, m, n, k, A, B, C)
{
    static bool inited = false;
    int smemBytes = v2::SMEM_HALFS * int(sizeof(half));
    if (!inited) {
        cudaFuncSetAttribute(v2::kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smemBytes);
        inited = true;
    }
    dim3 block(v2::THREADS);
    dim3 grid(n / v2::BN, m / v2::BM);
    v2::kernel<<<grid, block, smemBytes>>>(int(m), int(n), int(k), A, B, C);
}

}  // namespace playground
