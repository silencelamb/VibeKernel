// @file: task-1/src/matmul_f16/matmul_f16_v4_occ.cu
// v4: v3 + occupancy push. launch_bounds(256,2) caps regs<=128 so 2 blocks/SM
// fit (16 warps) -> hides __syncthreads stalls. Fragment loads hoisted (all
// ldmatrix issued before the mma burst) for max ILP. STAGES=3, BK=32.

#include <cuda_fp16.h>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{
namespace
{
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int STAGES = 3;

constexpr int WARP_M = 2, WARP_N = 4;
constexpr int WARP_TILE_M = BM / WARP_M;   // 64
constexpr int WARP_TILE_N = BN / WARP_N;   // 32
constexpr int M_TILES = WARP_TILE_M / 16;  // 4
constexpr int N_TILES = WARP_TILE_N / 8;   // 4
constexpr int K_TILES = BK / 16;           // 2
constexpr int THREADS = WARP_M * WARP_N * 32;  // 256
constexpr int MIN_BLOCKS = 2;

constexpr int APAD = 8, BPAD = 8;
constexpr int AS_STRIDE = BK + APAD;    // 40
constexpr int BS_STRIDE = BN + BPAD;    // 136
constexpr int AS_SIZE = BM * AS_STRIDE;
constexpr int BS_SIZE = BK * BS_STRIDE;
constexpr int STAGE_HALFS = AS_SIZE + BS_SIZE;

__device__ __forceinline__ uint32_t smem_u32(const void* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(
                     smem_u32(smem)),
                 "l"(gmem));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1,
                                            uint32_t& r2, uint32_t& r3,
                                            const void* smem)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(smem_u32(smem)));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t& r0, uint32_t& r1,
                                                  uint32_t& r2, uint32_t& r3,
                                                  const void* smem)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(smem_u32(smem)));
}
__device__ __forceinline__ void mma16816(float* d, const uint32_t* a,
                                         const uint32_t* b)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__global__ void __launch_bounds__(THREADS, MIN_BLOCKS)
    mma_v4(const half* __restrict__ A, const half* __restrict__ B,
           half* __restrict__ C, int M, int N, int K)
{
    extern __shared__ half smem[];
    half* As[STAGES];
    half* Bs[STAGES];
#pragma unroll
    for (int s = 0; s < STAGES; ++s) {
        As[s] = smem + s * STAGE_HALFS;
        Bs[s] = As[s] + AS_SIZE;
    }

    const int tid = threadIdx.x;
    const int warpId = tid >> 5;
    const int lane = tid & 31;
    const int warpRow = warpId / WARP_N;
    const int warpCol = warpId % WARP_N;

    const int aRowBase = blockIdx.y * BM;
    const int bColBase = blockIdx.x * BN;
    const int warpMBase = warpRow * WARP_TILE_M;
    const int warpNBase = warpCol * WARP_TILE_N;

    float acc[M_TILES][N_TILES][4];
#pragma unroll
    for (int i = 0; i < M_TILES; ++i)
#pragma unroll
        for (int j = 0; j < N_TILES; ++j)
#pragma unroll
            for (int t = 0; t < 4; ++t)
                acc[i][j][t] = 0.0f;

    const int kSteps = K / BK;

    auto load_stage = [&](int st, int kt) {
        const int kBase = kt * BK;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int f = tid + i * THREADS;
            const int row = f >> 2;
            const int colH = (f & 3) * 8;
            const half* g = A + (size_t)(aRowBase + row) * K + kBase + colH;
            cp_async16(&As[st][row * AS_STRIDE + colH], g);
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int f = tid + i * THREADS;
            const int row = f >> 4;
            const int col = (f & 15) * 8;
            const half* g = B + (size_t)(kBase + row) * N + bColBase + col;
            cp_async16(&Bs[st][row * BS_STRIDE + col], g);
        }
        cp_commit();
    };

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s)
        load_stage(s, s);
    cp_wait<STAGES - 2>();
    __syncthreads();

    for (int kt = 0; kt < kSteps; ++kt) {
        const int r = kt % STAGES;

        // per-kk loads + mma over stage r
#pragma unroll
        for (int kk = 0; kk < K_TILES; ++kk) {
            const int k0 = kk * 16;
            uint32_t RA[M_TILES][4];
            uint32_t RB[N_TILES][2];
#pragma unroll
            for (int mi = 0; mi < M_TILES; ++mi) {
                const half* p =
                    &As[r][(warpMBase + mi * 16 + (lane & 15)) * AS_STRIDE + k0 +
                           (lane >> 4) * 8];
                ldmatrix_x4(RA[mi][0], RA[mi][1], RA[mi][2], RA[mi][3], p);
            }
#pragma unroll
            for (int nj = 0; nj < N_TILES / 2; ++nj) {
                const half* p =
                    &Bs[r][(k0 + (lane & 15)) * BS_STRIDE + warpNBase +
                           nj * 16 + (lane >> 4) * 8];
                uint32_t t0, t1, t2, t3;
                ldmatrix_x4_trans(t0, t1, t2, t3, p);
                RB[nj * 2 + 0][0] = t0;
                RB[nj * 2 + 0][1] = t1;
                RB[nj * 2 + 1][0] = t2;
                RB[nj * 2 + 1][1] = t3;
            }
#pragma unroll
            for (int mi = 0; mi < M_TILES; ++mi)
#pragma unroll
                for (int ni = 0; ni < N_TILES; ++ni)
                    mma16816(acc[mi][ni], RA[mi], RB[ni]);
        }

        const int nextK = kt + STAGES - 1;
        if (nextK < kSteps)
            load_stage(nextK % STAGES, nextK);
        else
            cp_commit();  // empty commit: keep wait_group accounting consistent
        cp_wait<STAGES - 2>();
        __syncthreads();
    }

    const int gid = lane >> 2;
    const int tIg = lane & 3;
#pragma unroll
    for (int mi = 0; mi < M_TILES; ++mi) {
#pragma unroll
        for (int ni = 0; ni < N_TILES; ++ni) {
            const int rowB = aRowBase + warpMBase + mi * 16;
            const int colB = bColBase + warpNBase + ni * 8;
            half2 lo = __floats2half2_rn(acc[mi][ni][0], acc[mi][ni][1]);
            half2 hi = __floats2half2_rn(acc[mi][ni][2], acc[mi][ni][3]);
            *reinterpret_cast<half2*>(&C[(size_t)(rowB + gid) * N + colB +
                                         2 * tIg]) = lo;
            *reinterpret_cast<half2*>(&C[(size_t)(rowB + gid + 8) * N + colB +
                                         2 * tIg]) = hi;
        }
    }
}
}  // namespace

PLAYGROUND_MATMUL_DEC(float16_t, 4, m, n, k, A, B, C)
{
    static bool configured = false;
    constexpr int smemBytes = STAGES * STAGE_HALFS * sizeof(half);
    if (!configured) {
        cudaFuncSetAttribute(mma_v4,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smemBytes);
        configured = true;
    }
    dim3 block(THREADS);
    dim3 grid(n / BN, m / BM);
    mma_v4<<<grid, block, smemBytes>>>(A, B, C, (int) m, (int) n, (int) k);
}

}  // namespace playground
