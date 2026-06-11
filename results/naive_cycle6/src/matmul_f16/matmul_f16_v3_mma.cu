#include "playground/matmul.hpp"

#include <cuda_fp16.h>

namespace playground
{
namespace
{

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 4;
constexpr int NUM_WARPS = WARPS_M * WARPS_N;   // 8
constexpr int NUM_THREADS = NUM_WARPS * 32;    // 256
constexpr int WM = BM / WARPS_M;               // 64
constexpr int WN = BN / WARPS_N;               // 32
constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;
constexpr int WTILES_M = WM / MMA_M;           // 4  (A frags per kf)
constexpr int WTILES_N = WN / MMA_N;           // 4  (B frags / acc cols)
constexpr int N16_GROUPS = WN / 16;            // 2  (trans loads per kf)

__device__ __forceinline__ unsigned smem_u32(const void* p)
{
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async_cg16(void* smem, const void* gmem)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(
                     smem_u32(smem)),
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
__device__ __forceinline__ void mma_m16n8k16(float& c0, float& c1, float& c2,
                                             float& c3, uint32_t a0,
                                             uint32_t a1, uint32_t a2,
                                             uint32_t a3, uint32_t b0,
                                             uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__global__ void __launch_bounds__(NUM_THREADS)
kernel_v3(int M, int N, int K, const half* __restrict__ A,
          const half* __restrict__ B, half* __restrict__ C)
{
    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;
    const int warpId = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int warpRow = warpId / WARPS_N;
    const int warpCol = warpId % WARPS_N;
    const int tid = threadIdx.x;

    __shared__ half sA[2][BM * BK];
    __shared__ half sB[2][BK * BN];

    float acc[WTILES_M][WTILES_N][4];
#pragma unroll
    for (int i = 0; i < WTILES_M; i++)
#pragma unroll
        for (int j = 0; j < WTILES_N; j++)
#pragma unroll
            for (int t = 0; t < 4; t++)
                acc[i][j][t] = 0.0f;

    const int nK = K / BK;

    auto load_stage = [&](int stage, int kk) {
#pragma unroll
        for (int i = 0; i < (BM * BK / 8) / NUM_THREADS; i++) {
            int vec = tid + i * NUM_THREADS;
            int row = vec / (BK / 8);
            int col8 = (vec % (BK / 8)) * 8;
            const half* gptr = A + (blockRow * BM + row) * K + (kk + col8);
            cp_async_cg16(&sA[stage][row * BK + col8], gptr);
        }
#pragma unroll
        for (int i = 0; i < (BK * BN / 8) / NUM_THREADS; i++) {
            int vec = tid + i * NUM_THREADS;
            int row = vec / (BN / 8);
            int col8 = (vec % (BN / 8)) * 8;
            const half* gptr = B + (kk + row) * N + (blockCol * BN + col8);
            cp_async_cg16(&sB[stage][row * BN + col8], gptr);
        }
    };

    load_stage(0, 0);
    cp_async_commit();

    int stage = 0;
    for (int ki = 0; ki < nK; ki++) {
        if (ki + 1 < nK) {
            load_stage(stage ^ 1, (ki + 1) * BK);
        }
        cp_async_commit();
        cp_async_wait<1>();
        __syncthreads();

#pragma unroll
        for (int kf = 0; kf < BK / MMA_K; kf++) {
            uint32_t a_frag[WTILES_M][4];
            uint32_t b_frag[WTILES_N][2];
#pragma unroll
            for (int i = 0; i < WTILES_M; i++) {
                int rowA = warpRow * WM + i * MMA_M + (lane % 16);
                int colA = kf * MMA_K + (lane / 16) * 8;
                unsigned addr = smem_u32(&sA[stage][rowA * BK + colA]);
                ldmatrix_x4(addr, a_frag[i][0], a_frag[i][1], a_frag[i][2],
                            a_frag[i][3]);
            }
#pragma unroll
            for (int g = 0; g < N16_GROUPS; g++) {
                int rowB = kf * MMA_K + (lane & 8) + (lane % 8);
                int colB = warpCol * WN + g * 16 + ((lane & 16) ? 8 : 0);
                unsigned addr = smem_u32(&sB[stage][rowB * BN + colB]);
                uint32_t b0, b1, b2, b3;
                ldmatrix_x4_trans(addr, b0, b1, b2, b3);
                b_frag[g * 2 + 0][0] = b0;
                b_frag[g * 2 + 0][1] = b1;
                b_frag[g * 2 + 1][0] = b2;
                b_frag[g * 2 + 1][1] = b3;
            }
#pragma unroll
            for (int i = 0; i < WTILES_M; i++)
#pragma unroll
                for (int j = 0; j < WTILES_N; j++)
                    mma_m16n8k16(acc[i][j][0], acc[i][j][1], acc[i][j][2],
                                 acc[i][j][3], a_frag[i][0], a_frag[i][1],
                                 a_frag[i][2], a_frag[i][3], b_frag[j][0],
                                 b_frag[j][1]);
        }
        __syncthreads();
        stage ^= 1;
    }

    // Store. Each mma tile is 16x8; lane holds 4 fp32 at:
    //   (g, tg*2), (g, tg*2+1), (g+8, tg*2), (g+8, tg*2+1)
    // where g = lane/4, tg = lane%4.
    const int g = lane / 4;
    const int tg = lane % 4;
#pragma unroll
    for (int i = 0; i < WTILES_M; i++) {
#pragma unroll
        for (int j = 0; j < WTILES_N; j++) {
            int row0 = blockRow * BM + warpRow * WM + i * MMA_M + g;
            int col0 = blockCol * BN + warpCol * WN + j * MMA_N + tg * 2;
            half2 lo = __floats2half2_rn(acc[i][j][0], acc[i][j][1]);
            half2 hi = __floats2half2_rn(acc[i][j][2], acc[i][j][3]);
            *reinterpret_cast<half2*>(&C[row0 * N + col0]) = lo;
            *reinterpret_cast<half2*>(&C[(row0 + 8) * N + col0]) = hi;
        }
    }
}

}   // namespace

PLAYGROUND_MATMUL_DEC(float16_t, 3, m, n, k, A, B, C)
{
    dim3 block(NUM_THREADS);
    dim3 grid(n / BN, m / BM);
    kernel_v3<<<grid, block>>>((int) m, (int) n, (int) k, A, B, C);
}

}   // namespace playground
