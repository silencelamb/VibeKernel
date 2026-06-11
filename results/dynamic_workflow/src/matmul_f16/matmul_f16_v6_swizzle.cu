// @file: task-1/src/matmul_f16/matmul_f16_v6_swizzle.cu
// v6: v5 + XOR-swizzled shared memory (no padding) to remove ldmatrix bank
// conflicts and shrink shared usage. The swizzle permutes the 8-half (128-bit)
// chunk index by (row & mask); the SAME map is used for cp.async stores and
// ldmatrix loads, so data correctness is preserved while banks spread out.
// Configurable via the same PG_* macros as v5.

#include <cuda_fp16.h>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

#ifndef PG_BM
    #define PG_BM 128
#endif
#ifndef PG_BN
    #define PG_BN 128
#endif
#ifndef PG_BK
    #define PG_BK 32
#endif
#ifndef PG_STAGES
    #define PG_STAGES 3
#endif
#ifndef PG_WARP_M
    #define PG_WARP_M 2
#endif
#ifndef PG_WARP_N
    #define PG_WARP_N 4
#endif
#ifndef PG_MIN_BLOCKS
    #define PG_MIN_BLOCKS 2
#endif

namespace playground
{
namespace
{
constexpr int BM = PG_BM;
constexpr int BN = PG_BN;
constexpr int BK = PG_BK;
constexpr int STAGES = PG_STAGES;
constexpr int WARP_M = PG_WARP_M;
constexpr int WARP_N = PG_WARP_N;
constexpr int MIN_BLOCKS = PG_MIN_BLOCKS;

constexpr int WARP_TILE_M = BM / WARP_M;
constexpr int WARP_TILE_N = BN / WARP_N;
constexpr int M_TILES = WARP_TILE_M / 16;
constexpr int N_TILES = WARP_TILE_N / 8;
constexpr int K_TILES = BK / 16;
constexpr int THREADS = WARP_M * WARP_N * 32;

// No padding: swizzle handles bank conflicts. Strides are the raw widths.
constexpr int AS_SIZE = BM * BK;
constexpr int BS_SIZE = BK * BN;
constexpr int STAGE_HALFS = AS_SIZE + BS_SIZE;

constexpr int A_ROW_CHUNKS = BK / 8;
constexpr int B_ROW_CHUNKS = BN / 8;
constexpr int A_CHUNKS = BM * A_ROW_CHUNKS;
constexpr int B_CHUNKS = BK * B_ROW_CHUNKS;
constexpr int A_PER_THREAD = A_CHUNKS / THREADS;
constexpr int B_PER_THREAD = B_CHUNKS / THREADS;

constexpr int clog2(int x)
{
    int r = 0;
    while ((1 << r) < x)
        ++r;
    return r;
}
// 8 distinct 128-bit lanes. A row of RC chunks advances the base lane by RC%8.
// shift de-aliases consecutive rows; mask spreads up to 8 ways.
constexpr int A_SHIFT = (3 - clog2(A_ROW_CHUNKS)) > 0 ? 3 - clog2(A_ROW_CHUNKS) : 0;
constexpr int B_SHIFT = (3 - clog2(B_ROW_CHUNKS)) > 0 ? 3 - clog2(B_ROW_CHUNKS) : 0;
constexpr int A_SMASK = (A_ROW_CHUNKS < 8 ? A_ROW_CHUNKS : 8) - 1;
constexpr int B_SMASK = (B_ROW_CHUNKS < 8 ? B_ROW_CHUNKS : 8) - 1;

static_assert(WARP_TILE_M % 16 == 0 && WARP_TILE_N % 8 == 0, "warp tile");
static_assert(BK % 16 == 0, "BK");
static_assert(A_CHUNKS % THREADS == 0 && B_CHUNKS % THREADS == 0, "load even");
static_assert(N_TILES % 2 == 0, "N_TILES even");
static_assert(WARP_TILE_N % 8 == 0, "warpNBase chunk-aligned");

// swizzled half-offset of a chunk-aligned position (row, col) in a [*][W] buffer
__device__ __forceinline__ int swzA(int row, int col)
{
    const int chunk = col >> 3;
    return row * BK + ((chunk ^ ((row >> A_SHIFT) & A_SMASK)) << 3) + (col & 7);
}
__device__ __forceinline__ int swzB(int row, int col)
{
    const int chunk = col >> 3;
    return row * BN + ((chunk ^ ((row >> B_SHIFT) & B_SMASK)) << 3) + (col & 7);
}

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
    mma_v6(const half* __restrict__ A, const half* __restrict__ B,
           half* __restrict__ C, int M, int N, int K)
{
    extern __shared__ half smem[];
    auto As = [&](int s) -> half* { return smem + s * STAGE_HALFS; };
    auto Bs = [&](int s) -> half* { return smem + s * STAGE_HALFS + AS_SIZE; };

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
        for (int i = 0; i < A_PER_THREAD; ++i) {
            const int c = tid + i * THREADS;
            const int row = c / A_ROW_CHUNKS;
            const int colH = (c % A_ROW_CHUNKS) * 8;
            const half* g = A + (size_t)(aRowBase + row) * K + kBase + colH;
            cp_async16(&As(st)[swzA(row, colH)], g);
        }
#pragma unroll
        for (int i = 0; i < B_PER_THREAD; ++i) {
            const int c = tid + i * THREADS;
            const int row = c / B_ROW_CHUNKS;
            const int col = (c % B_ROW_CHUNKS) * 8;
            const half* g = B + (size_t)(kBase + row) * N + bColBase + col;
            cp_async16(&Bs(st)[swzB(row, col)], g);
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
#pragma unroll
        for (int kk = 0; kk < K_TILES; ++kk) {
            const int k0 = kk * 16;
            uint32_t RA[M_TILES][4];
            uint32_t RB[N_TILES][2];
#pragma unroll
            for (int mi = 0; mi < M_TILES; ++mi) {
                const int row = warpMBase + mi * 16 + (lane & 15);
                const int col = k0 + (lane >> 4) * 8;
                ldmatrix_x4(RA[mi][0], RA[mi][1], RA[mi][2], RA[mi][3],
                            &As(r)[swzA(row, col)]);
            }
#pragma unroll
            for (int nj = 0; nj < N_TILES / 2; ++nj) {
                const int row = k0 + (lane & 15);
                const int col = warpNBase + nj * 16 + (lane >> 4) * 8;
                uint32_t t0, t1, t2, t3;
                ldmatrix_x4_trans(t0, t1, t2, t3, &Bs(r)[swzB(row, col)]);
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
            cp_commit();
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

PLAYGROUND_MATMUL_DEC(float16_t, 6, m, n, k, A, B, C)
{
    static bool configured = false;
    constexpr int smemBytes = STAGES * STAGE_HALFS * sizeof(half);
    if (!configured) {
        cudaFuncSetAttribute(mma_v6,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smemBytes);
        configured = true;
    }
    dim3 block(THREADS);
    dim3 grid(n / BN, m / BM);
    mma_v6<<<grid, block, smemBytes>>>(A, B, C, (int) m, (int) n, (int) k);
}

}  // namespace playground
