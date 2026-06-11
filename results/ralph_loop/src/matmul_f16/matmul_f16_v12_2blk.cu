// @file: task-1/src/matmul_f16/matmul_f16_v12_2blk.cu
//
// v7: templated mma kernel with a true register-level software pipeline so the
//     ldmatrix of the *next* k-substep overlaps the mma of the current one,
//     keeping the tensor pipe fed (v6's "load-all-frags-then-mma" plateaued at
//     ~139 regardless of geometry). Larger warp tiles (64x64) cut MIO bytes/flop
//     (= 1/WM + 1/WN) and are now viable because the pipeline supplies the ILP
//     that low occupancy needs.

#include <cuda_fp16.h>

#include "playground/matmul.hpp"

namespace playground
{
namespace v12impl
{

__device__ __forceinline__ uint32_t smem_u32(const void* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async_cg(void* dst, const void* src)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(
                     smem_u32(dst)),
                 "l"(src));
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
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* p)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t (&r)[2],
                                                  const void* p)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1])
        : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma_m16n8k16(uint32_t (&d)[2],
                                             const uint32_t (&a)[4],
                                             const uint32_t (&b)[2])
{
    // fp16 accumulate: half the accumulator registers (2 vs 4 per tile) ->
    // lets the 64x64 tile potentially fit 2 blocks/SM. Same tensor throughput
    // as fp32-accumulate on A100; precision worse (verify Average Error).
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(d[0]), "+r"(d[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template <int BM, int BN, int BK, int WARPS_M, int WARPS_N, int STAGES,
          int MIN_BLK>
__global__ void __launch_bounds__(WARPS_M* WARPS_N * 32, MIN_BLK)
    gemm(int M, int N, int K, const half* __restrict__ A,
         const half* __restrict__ B, half* __restrict__ C)
{
    constexpr int NWARPS = WARPS_M * WARPS_N;
    constexpr int NTHREADS = NWARPS * 32;
    constexpr int WM = BM / WARPS_M;
    constexpr int WN = BN / WARPS_N;
    constexpr int MT = WM / 16;
    constexpr int NT = WN / 8;
    constexpr int KSTEP = BK / 16;
    constexpr int SKEW = 8;
    constexpr int AS_LD = BK + SKEW;
    constexpr int BS_LD = BN + SKEW;
    constexpr int A_STAGE = BM * AS_LD;
    constexpr int B_STAGE = BK * BS_LD;
    constexpr int A_PT = (BM * BK / 8) / NTHREADS;
    constexpr int B_PT = (BK * BN / 8) / NTHREADS;
    constexpr int A_F4PR = BK / 8;
    constexpr int B_F4PR = BN / 8;

    extern __shared__ half smem[];
    half* Asm = smem;
    half* Bsm = smem + STAGES * A_STAGE;

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;
    const int warpId = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warpRow = warpId / WARPS_N;
    const int warpCol = warpId % WARPS_N;
    const int tid = threadIdx.x;

    uint32_t acc[MT][NT][2];  // fp16 accumulate, 2 halfs packed per reg
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int t = 0; t < 2; ++t)
                acc[i][j][t] = 0u;

    const half* Aptr = A + blockRow * K;
    const half* Bptr = B + blockCol;
    const int NTILES = K / BK;

    auto As = [&](int s, int r, int c) -> half* {
        return Asm + s * A_STAGE + r * AS_LD + c;
    };
    auto Bs = [&](int s, int r, int c) -> half* {
        return Bsm + s * B_STAGE + r * BS_LD + c;
    };

    auto loadTile = [&](int t, int buf) {
        int k0 = t * BK;
#pragma unroll
        for (int i = 0; i < A_PT; ++i) {
            int f = tid + i * NTHREADS;
            int row = f / A_F4PR;
            int col = (f % A_F4PR) << 3;
            cp_async_cg(As(buf, row, col), Aptr + row * K + (k0 + col));
        }
#pragma unroll
        for (int i = 0; i < B_PT; ++i) {
            int f = tid + i * NTHREADS;
            int row = f / B_F4PR;
            int col = (f % B_F4PR) << 3;
            cp_async_cg(Bs(buf, row, col), Bptr + (k0 + row) * N + col);
        }
        cp_async_commit();
    };

    // Single-buffered fragments (no register prefetch) to cut register count so
    // the fp16-acc 64x64 tile fits 2 blocks/SM (16 warps) -> rely on occupancy
    // (not the reg pipeline) to hide ldmatrix/mma latency.
    uint32_t a_rf[MT][4];
    uint32_t b_rf[NT][2];

    auto loadFrag = [&](int cur, int kk) {
#pragma unroll
        for (int i = 0; i < MT; ++i) {
            int r = warpRow * WM + i * 16 + (lane & 15);
            int c = kk + (lane >> 4) * 8;
            ldmatrix_x4(a_rf[i], As(cur, r, c));
        }
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            int r = kk + (lane & 15);
            int c = warpCol * WN + j * 8;
            ldmatrix_x2_trans(b_rf[j], Bs(cur, r, c));
        }
    };

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s)
        loadTile(s, s);
    cp_async_wait<STAGES - 2>();
    __syncthreads();

    for (int t = 0; t < NTILES; ++t) {
        int cur = t % STAGES;
        int loadT = t + STAGES - 1;
        if (loadT < NTILES)
            loadTile(loadT, loadT % STAGES);
        else
            cp_async_commit();
#pragma unroll
        for (int ks = 0; ks < KSTEP; ++ks) {
            loadFrag(cur, ks * 16);
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int j = 0; j < NT; ++j)
                    mma_m16n8k16(acc[i][j], a_rf[i], b_rf[j]);
        }
        cp_async_wait<STAGES - 2>();
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < MT; ++i) {
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            int rowBase = blockRow + warpRow * WM + i * 16 + (lane >> 2);
            int colBase = blockCol + warpCol * WN + j * 8 + (lane & 3) * 2;
            // acc already fp16-packed (c0,c1)=(row,col..),(c2,c3)=(row+8,..)
            *reinterpret_cast<uint32_t*>(C + rowBase * N + colBase) =
                acc[i][j][0];
            *reinterpret_cast<uint32_t*>(C + (rowBase + 8) * N + colBase) =
                acc[i][j][1];
        }
    }
}

// ---- config knob (one per line so sweeps can sed each reliably) ----
constexpr int BM = 128;
constexpr int BN = 256;
constexpr int BK = 32;
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 4;
constexpr int STAGES = 4;
constexpr int MIN_BLK = 2;
constexpr int SMEM_BYTES =
    STAGES * (BM * (BK + 8) + BK * (BN + 8)) * (int) sizeof(half);
}  // namespace v12impl

PLAYGROUND_MATMUL_DEC(float16_t, 12, m, n, k, A, B, C)
{
    using namespace v12impl;
    auto kern = gemm<BM, BN, BK, WARPS_M, WARPS_N, STAGES, MIN_BLK>;
    static bool inited = false;
    if (!inited) {
        cudaFuncSetAttribute(
            kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
        inited = true;
    }
    dim3 block(WARPS_M * WARPS_N * 32);
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    kern<<<grid, block, SMEM_BYTES>>>(int(m), int(n), int(k), A, B, C);
}

}  // namespace playground
