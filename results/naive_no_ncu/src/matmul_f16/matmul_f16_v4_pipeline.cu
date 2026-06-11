#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "playground/matmul.hpp"

namespace playground
{
namespace v4
{
// Block tile / warp tiling identical to v3, but global->shared uses cp.async
// with an NSTAGE-deep software pipeline to overlap memory with compute.
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 4;
constexpr int THREADS = WARPS_M * WARPS_N * 32;  // 256
constexpr int WM = BM / WARPS_M;                 // 64
constexpr int WN = BN / WARPS_N;                 // 32
constexpr int WT_M = WM / 16;                    // 4
constexpr int WT_N = WN / 8;                     // 4
constexpr int APAD = 8;
constexpr int BPAD = 8;
constexpr int NSTAGE = 3;
constexpr int AS_STRIDE = BK + APAD;
constexpr int AS_TILE = BM * AS_STRIDE;
constexpr int BS_STRIDE = BN + BPAD;
constexpr int BS_TILE = BK * BS_STRIDE;
constexpr int SMEM_BYTES = NSTAGE * (AS_TILE + BS_TILE) * 2;

__device__ __forceinline__ uint32_t saddr(const void* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void ldm_x4(uint32_t& a, uint32_t& b, uint32_t& c,
                                        uint32_t& d, uint32_t s)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
        : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
        : "r"(s));
}
__device__ __forceinline__ void ldm_x4_t(uint32_t& a, uint32_t& b, uint32_t& c,
                                          uint32_t& d, uint32_t s)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];\n"
        : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
        : "r"(s));
}
__device__ __forceinline__ void mma1688(float* d, const uint32_t* a,
                                         const uint32_t* b)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3]));
}
__device__ __forceinline__ void cp_async16(uint32_t s, const void* g)
{
    asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n" ::"r"(s), "l"(g));
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

__global__ __launch_bounds__(THREADS) void kernel(int M, int N, int K,
                                                   const half* __restrict__ A,
                                                   const half* __restrict__ B,
                                                   half* __restrict__ C)
{
    extern __shared__ half smem[];
    half* As = smem;
    half* Bs = smem + NSTAGE * AS_TILE;
    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;
    const int warpId = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int warpRowBase = (warpId / WARPS_N) * WM;
    const int warpColBase = (warpId % WARPS_N) * WN;
    const int tid = threadIdx.x;

    float acc[WT_M][WT_N][4];
#pragma unroll
    for (int i = 0; i < WT_M; ++i)
#pragma unroll
        for (int j = 0; j < WT_N; ++j)
#pragma unroll
            for (int t = 0; t < 4; ++t)
                acc[i][j][t] = 0.0f;

    auto load = [&](int stage, int k0) {
        half* Asb = As + stage * AS_TILE;
        half* Bsb = Bs + stage * BS_TILE;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            int idx = tid + r * THREADS;
            int row = idx / (BK / 8);
            int c8 = (idx % (BK / 8)) * 8;
            cp_async16(saddr(&Asb[row * AS_STRIDE + c8]),
                       &A[size_t(blockRow + row) * K + k0 + c8]);
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            int idx = tid + r * THREADS;
            int row = idx / (BN / 8);
            int c8 = (idx % (BN / 8)) * 8;
            cp_async16(saddr(&Bsb[row * BS_STRIDE + c8]),
                       &B[size_t(k0 + row) * N + blockCol + c8]);
        }
    };

    const int numK = K / BK;
#pragma unroll
    for (int s = 0; s < NSTAGE - 1; ++s) {
        load(s, s * BK);
        cp_commit();
    }

    for (int kIter = 0; kIter < numK; ++kIter) {
        cp_wait<NSTAGE - 2>();
        __syncthreads();
        int readStage = kIter % NSTAGE;
        half* Asb = As + readStage * AS_TILE;
        half* Bsb = Bs + readStage * BS_TILE;
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            uint32_t Aop[WT_M][4];
            uint32_t Bop[WT_N][2];
#pragma unroll
            for (int i = 0; i < WT_M; ++i) {
                int row = warpRowBase + i * 16 + (lane % 16);
                int col = kk + (lane / 16) * 8;
                ldm_x4(Aop[i][0], Aop[i][1], Aop[i][2], Aop[i][3],
                       saddr(&Asb[row * AS_STRIDE + col]));
            }
#pragma unroll
            for (int j = 0; j < WN / 16; ++j) {
                uint32_t r0, r1, r2, r3;
                int row = kk + (lane % 16);
                int col = warpColBase + j * 16 + (lane / 16) * 8;
                ldm_x4_t(r0, r1, r2, r3, saddr(&Bsb[row * BS_STRIDE + col]));
                Bop[2 * j + 0][0] = r0;
                Bop[2 * j + 0][1] = r1;
                Bop[2 * j + 1][0] = r2;
                Bop[2 * j + 1][1] = r3;
            }
#pragma unroll
            for (int i = 0; i < WT_M; ++i)
#pragma unroll
                for (int nn = 0; nn < WT_N; ++nn)
                    mma1688(acc[i][nn], Aop[i], Bop[nn]);
        }
        int prefetch = kIter + (NSTAGE - 1);
        if (prefetch < numK) {
            load(prefetch % NSTAGE, prefetch * BK);
        }
        cp_commit();
    }

    const int group = lane / 4;
    const int tig = lane % 4;
#pragma unroll
    for (int i = 0; i < WT_M; ++i)
#pragma unroll
        for (int nn = 0; nn < WT_N; ++nn) {
            int rowBase = blockRow + warpRowBase + i * 16;
            int colBase = blockCol + warpColBase + nn * 8;
            half2 lo = __floats2half2_rn(acc[i][nn][0], acc[i][nn][1]);
            half2 hi = __floats2half2_rn(acc[i][nn][2], acc[i][nn][3]);
            *reinterpret_cast<half2*>(
                &C[size_t(rowBase + group) * N + colBase + 2 * tig]) = lo;
            *reinterpret_cast<half2*>(
                &C[size_t(rowBase + group + 8) * N + colBase + 2 * tig]) = hi;
        }
}
}  // namespace v4

PLAYGROUND_MATMUL_DEC(float16_t, 4, m, n, k, A, B, C)
{
    using namespace v4;
    static bool init = [] {
        cudaFuncSetAttribute(kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             SMEM_BYTES);
        return true;
    }();
    (void) init;
    dim3 block(THREADS);
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    kernel<<<grid, block, SMEM_BYTES>>>(int(m), int(n), int(k), A, B, C);
}

}  // namespace playground
