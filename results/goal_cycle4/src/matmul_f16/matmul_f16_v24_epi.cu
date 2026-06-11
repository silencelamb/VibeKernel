#include <cuda_fp16.h>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{
namespace v24
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
constexpr int MMA_K = 8;
constexpr int MT = WARP_M / MMA_M;  // 4
constexpr int NT = WARP_N / MMA_N;  // 8
constexpr int KK = BK / MMA_K;      // 4

constexpr int THREADS = WARPS_M * WARPS_N * 32;  // 256
constexpr int APAD = 8;
constexpr int BPAD = 8;
constexpr int STAGES = 3;

constexpr int AS_STRIDE = BK + APAD;
constexpr int BS_STRIDE = BN + BPAD;
constexpr int AS_TILE = BM * AS_STRIDE;
constexpr int BS_TILE = BK * BS_STRIDE;
constexpr int SMEM_HALFS = STAGES * (AS_TILE + BS_TILE);

constexpr unsigned AS_TILE_B = AS_TILE * 2;
constexpr unsigned BS_TILE_B = BS_TILE * 2;
constexpr unsigned BS_STRIDE_B = BS_STRIDE * 2;

constexpr int A_PER = (BM * BK / 8) / THREADS;
constexpr int B_PER = (BK * BN / 8) / THREADS;

__device__ __forceinline__ unsigned smem_addr(const void* p)
{
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async_cg(unsigned s, const void* gmem)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s),
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
__device__ __forceinline__ void ldm_x4(uint32_t (&r)[4], unsigned a)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(a));
}
__device__ __forceinline__ void ldm_x4_trans(uint32_t (&r)[4], unsigned a)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(a));
}
__device__ __forceinline__ void mma(uint32_t (&d)[2], const uint32_t (&a)[2],
                                    uint32_t b)
{
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3}, {%4}, {%0,%1};\n"
        : "+r"(d[0]), "+r"(d[1])
        : "r"(a[0]), "r"(a[1]), "r"(b));
}

__global__ void __launch_bounds__(THREADS, 2)
    kernel(int M, int N, int K, const half* __restrict__ A,
           const half* __restrict__ B, half* __restrict__ C)
{
    extern __shared__ half smem[];
    half* Asb = smem;
    half* Bsb = smem + STAGES * AS_TILE;
    const unsigned AsbU = smem_addr(Asb);
    const unsigned BsbU = smem_addr(Bsb);

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

    const int warpArow = warpRow * WARP_M;
    const int warpBcol = warpCol * WARP_N;
    const int aRow = laneId;
    const int bRow = laneId & 7;
    const int bColG = (laneId >> 3) * 8;

    // Precompute ldmatrix base shared addresses (stage 0, kk8 0).
    unsigned aBase[2];
#pragma unroll
    for (int g = 0; g < 2; ++g)
        aBase[g] = AsbU + ((warpArow + g * 32 + aRow) * AS_STRIDE) * 2u;
    unsigned bBase[2];
#pragma unroll
    for (int g = 0; g < 2; ++g)
        bBase[g] = BsbU + (bRow * BS_STRIDE + warpBcol + g * 32 + bColG) * 2u;

    unsigned aDst[A_PER];
    int aSrcRow[A_PER], aSrcCol[A_PER];
#pragma unroll
    for (int it = 0; it < A_PER; ++it) {
        int f = tid + it * THREADS;
        int row = f / (BK / 8);
        int col = (f % (BK / 8)) * 8;
        aDst[it] = AsbU + (row * AS_STRIDE + col) * 2u;
        aSrcRow[it] = rowBase + row;
        aSrcCol[it] = col;
    }
    unsigned bDst[B_PER];
    int bSrcRow[B_PER], bSrcCol[B_PER];
#pragma unroll
    for (int it = 0; it < B_PER; ++it) {
        int f = tid + it * THREADS;
        int row = f / (BN / 8);
        int col = (f % (BN / 8)) * 8;
        bDst[it] = BsbU + (row * BS_STRIDE + col) * 2u;
        bSrcRow[it] = row;
        bSrcCol[it] = colBase + col;
    }

    auto load_stage = [&](int stage, int k0) {
        unsigned aoff = (unsigned) stage * AS_TILE_B;
        unsigned boff = (unsigned) stage * BS_TILE_B;
#pragma unroll
        for (int it = 0; it < A_PER; ++it)
            cp_async_cg(aDst[it] + aoff,
                        &A[(size_t) aSrcRow[it] * K + k0 + aSrcCol[it]]);
#pragma unroll
        for (int it = 0; it < B_PER; ++it)
            cp_async_cg(bDst[it] + boff,
                        &B[(size_t) (k0 + bSrcRow[it]) * N + bSrcCol[it]]);
        cp_async_commit();
    };

    auto load_frag = [&](int read, int kk8, uint32_t (&a)[MT][2],
                         uint32_t (&b)[NT][1]) {
        unsigned akk = (unsigned) read * AS_TILE_B + (unsigned) kk8 * 2u;
        unsigned bkk =
            (unsigned) read * BS_TILE_B + (unsigned) kk8 * BS_STRIDE_B;
        uint32_t rA[2][4];
#pragma unroll
        for (int g = 0; g < 2; ++g)
            ldm_x4(rA[g], aBase[g] + akk);
#pragma unroll
        for (int g = 0; g < 2; ++g) {
            a[g * 2 + 0][0] = rA[g][0];
            a[g * 2 + 0][1] = rA[g][1];
            a[g * 2 + 1][0] = rA[g][2];
            a[g * 2 + 1][1] = rA[g][3];
        }
        uint32_t rB[2][4];
#pragma unroll
        for (int g = 0; g < 2; ++g)
            ldm_x4_trans(rB[g], bBase[g] + bkk);
#pragma unroll
        for (int g = 0; g < 2; ++g) {
            b[g * 4 + 0][0] = rB[g][0];
            b[g * 4 + 1][0] = rB[g][1];
            b[g * 4 + 2][0] = rB[g][2];
            b[g * 4 + 3][0] = rB[g][3];
        }
    };

    const int nK = K / BK;
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s)
        load_stage(s, s * BK);
    cp_async_wait<STAGES - 2>();
    __syncthreads();

    // Operand pipeline: prefetch 2 k8-steps ahead (2 buffers, loaded early).
    // Hides ~2 mma worth of ldmatrix latency; the cross-stage mma also issues
    // before the barrier (overlapping the sync wait).
    uint32_t af[2][MT][2];
    uint32_t bf[2][NT][1];
    load_frag(0, 0, af[0], bf[0]);
    load_frag(0, MMA_K, af[1], bf[1]);

    int buf = 0;
    int read = 0;
    int write = STAGES - 1;
    for (int kt = 0; kt < nK; ++kt) {
        if (kt + STAGES - 1 < nK)
            load_stage(write, (kt + STAGES - 1) * BK);

#pragma unroll
        for (int ki = 0; ki < KK; ++ki) {
            // mma current step (operands preloaded 2 steps ago).
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int j = 0; j < NT; ++j)
                    mma(acc[i][j], af[buf][i], bf[buf][j][0]);

            // Prefetch the step 2 ahead into the buffer this mma just freed.
            int gp2 = ki + 2;
            int p_read = (gp2 < KK) ? read : (read + 1) % STAGES;
            int p_kk = (gp2 < KK ? gp2 : gp2 - KK) * MMA_K;
            bool p_valid = (kt * KK + ki + 2) < nK * KK;

            // Sync before the first prefetch that reads the next stage.
            if (ki == KK - 2 && kt + 1 < nK) {
                cp_async_wait<STAGES - 2>();
                __syncthreads();
            }
            if (p_valid)
                load_frag(p_read, p_kk, af[buf], bf[buf]);
            buf ^= 1;
        }
        read = (read + 1) % STAGES;
        write = (write + 1) % STAGES;
    }

    const int gid = laneId >> 2;
    const int tig = laneId & 3;
    // Coalesced epilogue: stage the block's BM x BN tile in shared memory
    // (reuse the now-free As/Bs smem), then write to global C as fully
    // coalesced 128-bit (float4) rows. The direct per-thread half2 stores were
    // only 50% sector-efficient (scattered); staging makes them 100%.
    half* Cs = smem;  // BM*BN halfs, fits within SMEM_HALFS
    __syncthreads();  // ensure all warps done reading As/Bs before reuse
#pragma unroll
    for (int i = 0; i < MT; ++i) {
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            int r = warpArow + i * MMA_M + gid;
            int c = warpBcol + j * MMA_N + tig * 2;
            *reinterpret_cast<uint32_t*>(&Cs[r * BN + c]) = acc[i][j][0];
            *reinterpret_cast<uint32_t*>(&Cs[(r + 8) * BN + c]) = acc[i][j][1];
        }
    }
    __syncthreads();
    constexpr int C_F4 = BM * BN / 8;  // float4 count for the tile
#pragma unroll
    for (int it = 0; it < C_F4 / THREADS; ++it) {
        int f = tid + it * THREADS;
        int r = f / (BN / 8);
        int c = (f % (BN / 8)) * 8;
        *reinterpret_cast<float4*>(&C[(rowBase + r) * N + colBase + c]) =
            *reinterpret_cast<float4*>(&Cs[r * BN + c]);
    }
}
}  // namespace v24

PLAYGROUND_MATMUL_DEC(float16_t, 24, m, n, k, A, B, C)
{
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(v24::kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             v24::SMEM_HALFS * (int) sizeof(half));
        configured = true;
    }
    dim3 block(v24::THREADS);
    dim3 grid(n / v24::BN, m / v24::BM);
    v24::kernel<<<grid, block, v24::SMEM_HALFS * sizeof(half)>>>(
        (int) m, (int) n, (int) k, A, B, C);
}

}  // namespace playground
