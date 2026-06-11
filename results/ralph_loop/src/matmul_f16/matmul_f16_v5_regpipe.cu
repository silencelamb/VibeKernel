// @file: task-1/src/matmul_f16/matmul_f16_v5_regpipe.cu
//
// v5: v3 geometry (128x128, warp 64x32, 25% occ) + register double-buffering
//     of the A/B fragments so each k-substep's ldmatrix overlaps the previous
//     substep's mma (hides ldmatrix latency that left v3's tensor pipe at 58%).
//   - 3-stage cp.async ring (global) + 2-slot register ring (ldmatrix->mma).
//   - A: ldmatrix.x4 per m16 tile; B: ldmatrix.x4.trans per 2 n8 tiles.

#include <cuda_fp16.h>

#include "playground/matmul.hpp"

namespace playground
{
namespace v5impl
{

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;

constexpr int WARPS_M = 2;
constexpr int WARPS_N = 4;
constexpr int NWARPS = WARPS_M * WARPS_N;
constexpr int NTHREADS = NWARPS * 32;       // 256

constexpr int WM = BM / WARPS_M;  // 64
constexpr int WN = BN / WARPS_N;  // 32
constexpr int MT = WM / 16;       // 4
constexpr int NT = WN / 8;        // 4
constexpr int KSTEP = BK / 16;    // 2

constexpr int SKEW = 8;
constexpr int AS_LD = BK + SKEW;   // 40
constexpr int BS_LD = BN + SKEW;   // 136
constexpr int STAGES = 3;

constexpr int A_STAGE = BM * AS_LD;
constexpr int B_STAGE = BK * BS_LD;
constexpr int SMEM_HALFS = STAGES * (A_STAGE + B_STAGE);

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
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4],
                                                  const void* p)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma_m16n8k16(float (&d)[4],
                                             const uint32_t (&a)[4],
                                             const uint32_t (&b)[2])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__global__ void __launch_bounds__(NTHREADS) gemm_v5(int M, int N, int K,
                                                    const half* __restrict__ A,
                                                    const half* __restrict__ B,
                                                    half* __restrict__ C)
{
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

    float acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int t = 0; t < 4; ++t)
                acc[i][j][t] = 0.0f;

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
        for (int i = 0; i < 2; ++i) {
            int f = tid + i * NTHREADS;
            int row = f >> 2;
            int col = (f & 3) << 3;
            cp_async_cg(As(buf, row, col), Aptr + row * K + (k0 + col));
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            int f = tid + i * NTHREADS;
            int row = f >> 4;
            int col = (f & 15) << 3;
            cp_async_cg(Bs(buf, row, col), Bptr + (k0 + row) * N + col);
        }
        cp_async_commit();
    };

    // register fragment ring (2 slots)
    uint32_t a_rf[2][MT][4];
    uint32_t b_rf[2][NT][2];

    auto loadFrag = [&](int slot, int cur, int kk) {
#pragma unroll
        for (int i = 0; i < MT; ++i) {
            int r = warpRow * WM + i * 16 + (lane & 15);
            int c = kk + (lane >> 4) * 8;
            ldmatrix_x4(a_rf[slot][i], As(cur, r, c));
        }
#pragma unroll
        for (int j = 0; j < NT; j += 2) {
            int r = kk + (lane & 15);
            int c = warpCol * WN + j * 8 + (lane >> 4) * 8;
            uint32_t t4[4];
            ldmatrix_x4_trans(t4, Bs(cur, r, c));
            b_rf[slot][j][0] = t4[0];
            b_rf[slot][j][1] = t4[1];
            b_rf[slot][j + 1][0] = t4[2];
            b_rf[slot][j + 1][1] = t4[3];
        }
    };

    // prologue: global stages
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s)
        loadTile(s, s);
    cp_async_wait<STAGES - 2>();
    __syncthreads();

    // preload first register fragment (tile 0, kk 0)
    loadFrag(0, 0, 0);
    int slot = 0;

    for (int t = 0; t < NTILES; ++t) {
        int cur = t % STAGES;
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            int ks = kk >> 4;                       // 0..KSTEP-1
            bool lastKk = (ks == KSTEP - 1);

            // figure out the next (tile, kk) to prefetch fragments for
            int nTile = lastKk ? (t + 1) : t;
            int nKk = lastKk ? 0 : (kk + 16);
            int nCur = nTile % STAGES;

            if (lastKk) {
                // issue global prefetch for tile (t + STAGES-1) once per tile,
                // then advance the smem stage.
                int loadT = t + STAGES - 1;
                if (loadT < NTILES)
                    loadTile(loadT, loadT % STAGES);
                else
                    cp_async_commit();
                cp_async_wait<STAGES - 2>();
                __syncthreads();
            }

            if (nTile < NTILES)
                loadFrag(slot ^ 1, nCur, nKk);

#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int j = 0; j < NT; ++j)
                    mma_m16n8k16(acc[i][j], a_rf[slot][i], b_rf[slot][j]);

            slot ^= 1;
        }
    }

#pragma unroll
    for (int i = 0; i < MT; ++i) {
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            int rowBase = blockRow + warpRow * WM + i * 16 + (lane >> 2);
            int colBase = blockCol + warpCol * WN + j * 8 + (lane & 3) * 2;
            half2 lo = __floats2half2_rn(acc[i][j][0], acc[i][j][1]);
            half2 hi = __floats2half2_rn(acc[i][j][2], acc[i][j][3]);
            *reinterpret_cast<half2*>(C + rowBase * N + colBase) = lo;
            *reinterpret_cast<half2*>(C + (rowBase + 8) * N + colBase) = hi;
        }
    }
}
}  // namespace v5impl

PLAYGROUND_MATMUL_DEC(float16_t, 5, m, n, k, A, B, C)
{
    using namespace v5impl;
    static bool inited = false;
    if (!inited) {
        cudaFuncSetAttribute(gemm_v5,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             SMEM_HALFS * sizeof(half));
        inited = true;
    }
    dim3 block(NTHREADS);
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    gemm_v5<<<grid, block, SMEM_HALFS * sizeof(half)>>>(int(m), int(n), int(k),
                                                        A, B, C);
}

}  // namespace playground
