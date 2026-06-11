#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "playground/matmul.hpp"

namespace playground
{
namespace v6
{
// FP16-accumulate GEMM (matches cuBLAS compute model: ~1.7e-2 error, halves
// accumulator register pressure vs FP32 accumulate). Swizzled smem + cp.async
// multistage pipeline + mma.m16n8k16.f16.f16.f16.f16 + ldmatrix.
// ---- config ----
constexpr int BM = 256;
constexpr int BN = 128;
constexpr int BK = 64;
constexpr int WARPS_M = 8;
constexpr int WARPS_N = 2;
constexpr int NSTAGE = 3;
// -----------------
constexpr int THREADS = WARPS_M * WARPS_N * 32;
constexpr int WM = BM / WARPS_M;
constexpr int WN = BN / WARPS_N;
constexpr int WT_M = WM / 16;
constexpr int WT_N = WN / 8;
constexpr int AS_TILE = BM * BK;
constexpr int BS_TILE = BK * BN;
constexpr int A_ITERS = (BM * BK / 8) / THREADS;
constexpr int B_ITERS = (BK * BN / 8) / THREADS;
constexpr int AMASK = (BK / 8) - 1;
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
// fp16-accumulate mma; C/D are 2 regs (4 halfs).
__device__ __forceinline__ void mma_f16(uint32_t* d, const uint32_t* a,
                                         const uint32_t* b)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%8,%9};\n"
        : "=r"(d[0]), "=r"(d[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "r"(d[0]), "r"(d[1]));
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
__device__ __forceinline__ int offA(int row, int c8)
{
    return row * BK + (((c8 >> 3) ^ (row & AMASK)) << 3);
}
__device__ __forceinline__ int offB(int row, int c8)
{
    return row * BN + (((c8 >> 3) ^ (row & 7)) << 3);
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

    uint32_t acc[WT_M][WT_N][2];
#pragma unroll
    for (int i = 0; i < WT_M; ++i)
#pragma unroll
        for (int j = 0; j < WT_N; ++j) {
            acc[i][j][0] = 0;
            acc[i][j][1] = 0;
        }

    auto load = [&](int stage, int k0) {
        half* Asb = As + stage * AS_TILE;
        half* Bsb = Bs + stage * BS_TILE;
#pragma unroll
        for (int it = 0; it < A_ITERS; ++it) {
            int idx = tid + it * THREADS;
            int row = idx / (BK / 8);
            int c8 = (idx % (BK / 8)) * 8;
            cp_async16(saddr(&Asb[offA(row, c8)]),
                       &A[size_t(blockRow + row) * K + k0 + c8]);
        }
#pragma unroll
        for (int it = 0; it < B_ITERS; ++it) {
            int idx = tid + it * THREADS;
            int row = idx / (BN / 8);
            int c8 = (idx % (BN / 8)) * 8;
            cp_async16(saddr(&Bsb[offB(row, c8)]),
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
        int rs = kIter % NSTAGE;
        half* Asb = As + rs * AS_TILE;
        half* Bsb = Bs + rs * BS_TILE;
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            uint32_t Aop[WT_M][4];
            uint32_t Bop[WT_N][2];
#pragma unroll
            for (int i = 0; i < WT_M; ++i) {
                int row = warpRowBase + i * 16 + (lane % 16);
                int col = kk + (lane / 16) * 8;
                ldm_x4(Aop[i][0], Aop[i][1], Aop[i][2], Aop[i][3],
                       saddr(&Asb[offA(row, col)]));
            }
#pragma unroll
            for (int j = 0; j < WN / 16; ++j) {
                uint32_t r0, r1, r2, r3;
                int row = kk + (lane % 16);
                int col = warpColBase + j * 16 + (lane / 16) * 8;
                ldm_x4_t(r0, r1, r2, r3, saddr(&Bsb[offB(row, col)]));
                Bop[2 * j + 0][0] = r0;
                Bop[2 * j + 0][1] = r1;
                Bop[2 * j + 1][0] = r2;
                Bop[2 * j + 1][1] = r3;
            }
#pragma unroll
            for (int i = 0; i < WT_M; ++i)
#pragma unroll
                for (int nn = 0; nn < WT_N; ++nn)
                    mma_f16(acc[i][nn], Aop[i], Bop[nn]);
        }
        int pf = kIter + (NSTAGE - 1);
        if (pf < numK) {
            load(pf % NSTAGE, pf * BK);
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
            *reinterpret_cast<uint32_t*>(
                &C[size_t(rowBase + group) * N + colBase + 2 * tig]) =
                acc[i][nn][0];
            *reinterpret_cast<uint32_t*>(
                &C[size_t(rowBase + group + 8) * N + colBase + 2 * tig]) =
                acc[i][nn][1];
        }
}
}  // namespace v6

PLAYGROUND_MATMUL_DEC(float16_t, 6, m, n, k, A, B, C)
{
    using namespace v6;
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
