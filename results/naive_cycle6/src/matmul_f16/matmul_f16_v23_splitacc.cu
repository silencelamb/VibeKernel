#include "playground/matmul.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace playground
{
namespace
{

constexpr int BM = 128;
constexpr int BN = 256;
constexpr int BK = 64;
constexpr int NSTAGE = 3;
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 4;
constexpr int NUM_WARPS = WARPS_M * WARPS_N;   // 8
constexpr int NUM_THREADS = NUM_WARPS * 32;    // 256
constexpr int WM = BM / WARPS_M;               // 64
constexpr int WN = BN / WARPS_N;               // 64
constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;
constexpr int WTILES_M = WM / MMA_M;           // 4
constexpr int WTILES_N = WN / MMA_N;           // 8
constexpr int N16_GROUPS = WN / 16;            // 4
constexpr int KF = BK / MMA_K;                 // 4
constexpr int STAGE_A = BM * BK;
constexpr int STAGE_B = BK * BN;

// XOR swizzle (LD>=64 -> conflict-free ldmatrix).
__device__ __forceinline__ int swz(int row, int col, int LD)
{
    return row * LD + (((col >> 3) ^ (row & 7)) << 3) + (col & 7);
}
__device__ __forceinline__ unsigned smem_u32(const void* p)
{
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async_cg16(unsigned smem, const void* gmem)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem),
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
__device__ __forceinline__ void ldmatrix_x4(unsigned addr, uint32_t& r0,
                                            uint32_t& r1, uint32_t& r2,
                                            uint32_t& r3)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x4_trans(unsigned addr, uint32_t& r0,
                                                  uint32_t& r1, uint32_t& r2,
                                                  uint32_t& r3)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr));
}
__device__ __forceinline__ void mma_m16n8k16(uint32_t& d0, uint32_t& d1,
                                             uint32_t a0, uint32_t a1,
                                             uint32_t a2, uint32_t a3,
                                             uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(d0), "+r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__global__ void __launch_bounds__(NUM_THREADS)
kernel_v23(int M, int N, int K, const half* __restrict__ A,
           const half* __restrict__ B, half* __restrict__ C)
{
    extern __shared__ half smem[];
    half* sA = smem;
    half* sB = smem + NSTAGE * STAGE_A;

    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;
    const int warpId = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int warpRow = warpId / WARPS_N;
    const int warpCol = warpId % WARPS_N;
    const int tid = threadIdx.x;

    uint32_t acc[2][WTILES_M][WTILES_N][2];  // even/odd-kf banks
#pragma unroll
    for (int b = 0; b < 2; b++)
#pragma unroll
        for (int i = 0; i < WTILES_M; i++)
#pragma unroll
            for (int j = 0; j < WTILES_N; j++)
#pragma unroll
                for (int t = 0; t < 2; t++)
                    acc[b][i][j][t] = 0u;

    const int nK = K / BK;

    auto load_stage = [&](int stage, int kk) {
        half* dA = sA + stage * STAGE_A;
        half* dB = sB + stage * STAGE_B;
#pragma unroll
        for (int i = 0; i < (BM * BK / 8) / NUM_THREADS; i++) {
            int vec = tid + i * NUM_THREADS;
            int row = vec / (BK / 8);
            int col8 = (vec % (BK / 8)) * 8;
            const half* gptr = A + (blockRow * BM + row) * K + (kk + col8);
            cp_async_cg16(smem_u32(&dA[swz(row, col8, BK)]), gptr);
        }
#pragma unroll
        for (int i = 0; i < (BK * BN / 8) / NUM_THREADS; i++) {
            int vec = tid + i * NUM_THREADS;
            int row = vec / (BN / 8);
            int col8 = (vec % (BN / 8)) * 8;
            const half* gptr = B + (kk + row) * N + (blockCol * BN + col8);
            cp_async_cg16(smem_u32(&dB[swz(row, col8, BN)]), gptr);
        }
    };

    uint32_t a_frag[2][WTILES_M][4];
    uint32_t b_frag[2][WTILES_N][2];

    auto load_frag = [&](int buf, int rstage, int kf) {
        half* cA = sA + rstage * STAGE_A;
        half* cB = sB + rstage * STAGE_B;
#pragma unroll
        for (int i = 0; i < WTILES_M; i++) {
            int rowA = warpRow * WM + i * MMA_M + (lane % 16);
            int colA = kf * MMA_K + (lane / 16) * 8;
            ldmatrix_x4(smem_u32(&cA[swz(rowA, colA, BK)]), a_frag[buf][i][0],
                        a_frag[buf][i][1], a_frag[buf][i][2],
                        a_frag[buf][i][3]);
        }
#pragma unroll
        for (int g = 0; g < N16_GROUPS; g++) {
            int rowB = kf * MMA_K + (lane & 8) + (lane % 8);
            int colB = warpCol * WN + g * 16 + ((lane & 16) ? 8 : 0);
            ldmatrix_x4_trans(smem_u32(&cB[swz(rowB, colB, BN)]),
                              b_frag[buf][g * 2 + 0][0],
                              b_frag[buf][g * 2 + 0][1],
                              b_frag[buf][g * 2 + 1][0],
                              b_frag[buf][g * 2 + 1][1]);
        }
    };

    auto do_mma = [&](int buf) {
#pragma unroll
        for (int i = 0; i < WTILES_M; i++)
#pragma unroll
            for (int j = 0; j < WTILES_N; j++)
                mma_m16n8k16(acc[buf][i][j][0], acc[buf][i][j][1],
                             a_frag[buf][i][0], a_frag[buf][i][1],
                             a_frag[buf][i][2], a_frag[buf][i][3],
                             b_frag[buf][j][0], b_frag[buf][j][1]);
    };

#pragma unroll
    for (int s = 0; s < NSTAGE - 1; s++) {
        load_stage(s, s * BK);
        cp_async_commit();
    }
    cp_async_wait<NSTAGE - 2>();
    __syncthreads();
    load_frag(0, 0, 0);

    int read_stage = 0;
    int write_stage = NSTAGE - 1;
    int load_k = (NSTAGE - 1) * BK;

    for (int ki = 0; ki < nK; ki++) {
#pragma unroll
        for (int kf = 0; kf < KF; kf++) {
            int cur = kf & 1;
            int nxt = cur ^ 1;
            if (kf == KF - 1) {
                cp_async_wait<NSTAGE - 2>();
                __syncthreads();
                int next_read = (read_stage + 1) % NSTAGE;
                load_frag(nxt, next_read, 0);
            } else {
                load_frag(nxt, read_stage, kf + 1);
            }
            if (kf == 0) {
                if (load_k < K) {
                    load_stage(write_stage, load_k);
                }
                cp_async_commit();
                write_stage = (write_stage + 1) % NSTAGE;
                load_k += BK;
            }
            do_mma(cur);
        }
        read_stage = (read_stage + 1) % NSTAGE;
    }

    const int g = lane / 4;
    const int tg = lane % 4;
#pragma unroll
    for (int i = 0; i < WTILES_M; i++) {
#pragma unroll
        for (int j = 0; j < WTILES_N; j++) {
            int row0 = blockRow * BM + warpRow * WM + i * MMA_M + g;
            int col0 = blockCol * BN + warpCol * WN + j * MMA_N + tg * 2;
            half2 s0 = __hadd2(*reinterpret_cast<half2*>(&acc[0][i][j][0]),
                              *reinterpret_cast<half2*>(&acc[1][i][j][0]));
            half2 s1 = __hadd2(*reinterpret_cast<half2*>(&acc[0][i][j][1]),
                              *reinterpret_cast<half2*>(&acc[1][i][j][1]));
            *reinterpret_cast<half2*>(&C[row0 * N + col0]) = s0;
            *reinterpret_cast<half2*>(&C[(row0 + 8) * N + col0]) = s1;
        }
    }
}

}   // namespace

PLAYGROUND_MATMUL_DEC(float16_t, 23, m, n, k, A, B, C)
{
    static bool configured = false;
    int smemBytes = NSTAGE * (STAGE_A + STAGE_B) * sizeof(half);
    if (!configured) {
        cudaFuncSetAttribute(kernel_v23,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smemBytes);
        configured = true;
    }
    dim3 block(NUM_THREADS);
    dim3 grid(n / BN, m / BM);
    kernel_v23<<<grid, block, smemBytes>>>((int) m, (int) n, (int) k, A, B, C);
}

}   // namespace playground
