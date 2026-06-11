#include <cuda_fp16.h>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{
namespace v6
{
constexpr int BM = 128;
constexpr int BN = 256;
constexpr int BK = 32;

constexpr int WARPS_M = 2;
constexpr int WARPS_N = 4;
constexpr int WARP_M = BM / WARPS_M;  // 64
constexpr int WARP_N = BN / WARPS_N;  // 64

constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;
constexpr int MT = WARP_M / MMA_M;  // 4
constexpr int NT = WARP_N / MMA_N;  // 8

constexpr int THREADS = WARPS_M * WARPS_N * 32;  // 256
constexpr int APAD = 8;
constexpr int BPAD = 8;
constexpr int STAGES = 3;

constexpr int AS_STRIDE = BK + APAD;
constexpr int BS_STRIDE = BN + BPAD;
constexpr int AS_TILE = BM * AS_STRIDE;
constexpr int BS_TILE = BK * BS_STRIDE;
constexpr int SMEM_HALFS = STAGES * (AS_TILE + BS_TILE);

constexpr int A_PER = (BM * BK / 8) / THREADS;  // float4 per thread for A
constexpr int B_PER = (BK * BN / 8) / THREADS;  // float4 per thread for B

__device__ __forceinline__ unsigned smem_addr(const void* p)
{
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async_cg(void* smem, const void* gmem)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(
                     smem_addr(smem)),
                 "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit()
{
    asm volatile("cp.async.commit_group;\n");
}
template <int N>
__device__ __forceinline__ void cp_async_wait()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void ldm_x4(uint32_t (&r)[4], const half* p)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(smem_addr(p)));
}
__device__ __forceinline__ void ldm_x4_trans(uint32_t (&r)[4], const half* p)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(smem_addr(p)));
}
__device__ __forceinline__ void mma(uint32_t (&d)[2], const uint32_t (&a)[4],
                                    const uint32_t (&b)[2])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(d[0]), "+r"(d[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__global__ void __launch_bounds__(THREADS, 2)
    kernel(int M, int N, int K, const half* __restrict__ A,
           const half* __restrict__ B, half* __restrict__ C)
{
    extern __shared__ half smem[];
    half* Asb = smem;
    half* Bsb = smem + STAGES * AS_TILE;

    auto As = [&](int s, int r, int c) -> half& {
        return Asb[s * AS_TILE + r * AS_STRIDE + c];
    };
    auto Bs = [&](int s, int r, int c) -> half& {
        return Bsb[s * BS_TILE + r * BS_STRIDE + c];
    };

    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;
    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;
    const int warpRow = warpId / WARPS_N;
    const int warpCol = warpId % WARPS_N;
    const int tid = threadIdx.x;

    const int rowBase = blockRow * BM;
    const int colBase = blockCol * BN;

    uint32_t acc[MT][NT][2];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            acc[i][j][0] = 0;
            acc[i][j][1] = 0;
        }

    const int lr = laneId & 15;
    const int lc = (laneId & 16) >> 1;

    auto load_stage = [&](int stage, int k0) {
#pragma unroll
        for (int it = 0; it < A_PER; ++it) {
            int f = tid + it * THREADS;
            int row = f / (BK / 8);
            int col = (f % (BK / 8)) * 8;
            cp_async_cg(&As(stage, row, col),
                        &A[(rowBase + row) * K + k0 + col]);
        }
#pragma unroll
        for (int it = 0; it < B_PER; ++it) {
            int f = tid + it * THREADS;
            int row = f / (BN / 8);
            int col = (f % (BN / 8)) * 8;
            cp_async_cg(&Bs(stage, row, col),
                        &B[(k0 + row) * N + colBase + col]);
        }
        cp_async_commit();
    };

    const int nK = K / BK;
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s)
        load_stage(s, s * BK);
    cp_async_wait<STAGES - 2>();
    __syncthreads();

    int read = 0;
    int write = STAGES - 1;
    for (int kt = 0; kt < nK; ++kt) {
        int next = kt + STAGES - 1;
        if (next < nK)
            load_stage(write, next * BK);

#pragma unroll
        for (int kk = 0; kk < BK; kk += MMA_K) {
            uint32_t a[MT][4];
            uint32_t b[NT][2];
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                int ar = warpRow * WARP_M + i * MMA_M;
                ldm_x4(a[i], &As(read, ar + lr, kk + lc));
            }
#pragma unroll
            for (int p = 0; p < NT / 2; ++p) {
                int nc0 = warpCol * WARP_N + p * 16;
                uint32_t r[4];
                ldm_x4_trans(r, &Bs(read, kk + lr, nc0 + lc));
                b[2 * p][0] = r[0];
                b[2 * p][1] = r[1];
                b[2 * p + 1][0] = r[2];
                b[2 * p + 1][1] = r[3];
            }
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int j = 0; j < NT; ++j)
                    mma(acc[i][j], a[i], b[j]);
        }

        read = (read + 1) % STAGES;
        write = (write + 1) % STAGES;
        if (kt + 1 < nK) {
            cp_async_wait<STAGES - 2>();
            __syncthreads();
        }
    }

    const int gid = laneId >> 2;
    const int tig = laneId & 3;
#pragma unroll
    for (int i = 0; i < MT; ++i) {
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            int base_r = rowBase + warpRow * WARP_M + i * MMA_M;
            int base_c = colBase + warpCol * WARP_N + j * MMA_N;
            *reinterpret_cast<uint32_t*>(
                &C[(base_r + gid) * N + base_c + tig * 2]) = acc[i][j][0];
            *reinterpret_cast<uint32_t*>(
                &C[(base_r + gid + 8) * N + base_c + tig * 2]) = acc[i][j][1];
        }
    }
}
}  // namespace v6

PLAYGROUND_MATMUL_DEC(float16_t, 6, m, n, k, A, B, C)
{
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(v6::kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             v6::SMEM_HALFS * (int) sizeof(half));
        configured = true;
    }
    dim3 block(v6::THREADS);
    dim3 grid(n / v6::BN, m / v6::BM);
    v6::kernel<<<grid, block, v6::SMEM_HALFS * sizeof(half)>>>(
        (int) m, (int) n, (int) k, A, B, C);
}

}  // namespace playground
