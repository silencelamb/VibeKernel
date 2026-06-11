// @file: task-1/src/matmul_f16/matmul_f16_v3_mma.cu
//
// v3: hand-written mma.sync + ldmatrix + 3-stage cp.async pipeline.
//   - Block 128x128, BK=32, 8 warps (warpM=2 x warpN=4) -> warp tile 64x32.
//   - Per warp: MT=4 (m16) x NT=4 (n8) = 16 mma.m16n8k16 per k16 substep.
//   - A loaded with ldmatrix.x4 (row-major), B with ldmatrix.x2.trans (so the
//     row-major Bs feeds mma's col-major B operand). fp32 accumulate.
//   - 3-stage async ring in dynamic shared memory hides global latency.

#include <cuda_fp16.h>

#include "playground/matmul.hpp"

namespace playground
{
namespace v3impl
{

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;

constexpr int WARPS_M = 2;
constexpr int WARPS_N = 4;
constexpr int NWARPS = WARPS_M * WARPS_N;   // 8
constexpr int NTHREADS = NWARPS * 32;       // 256

constexpr int WM = BM / WARPS_M;  // 64
constexpr int WN = BN / WARPS_N;  // 32
constexpr int MT = WM / 16;       // 4
constexpr int NT = WN / 8;        // 4

constexpr int SKEW = 8;
constexpr int AS_LD = BK + SKEW;   // 40
constexpr int BS_LD = BN + SKEW;   // 136
constexpr int STAGES = 3;

constexpr int A_STAGE = BM * AS_LD;  // halfs per stage (A)
constexpr int B_STAGE = BK * BS_LD;  // halfs per stage (B)
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
__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t (&r)[2],
                                                  const void* p)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1])
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

__global__ void __launch_bounds__(NTHREADS) gemm_v3(int M, int N, int K,
                                                    const half* __restrict__ A,
                                                    const half* __restrict__ B,
                                                    half* __restrict__ C)
{
    extern __shared__ half smem[];
    half* Asm = smem;                       // STAGES * A_STAGE
    half* Bsm = smem + STAGES * A_STAGE;     // STAGES * B_STAGE

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
            int f = tid + i * NTHREADS;       // A: 512 float4
            int row = f >> 2;
            int col = (f & 3) << 3;
            cp_async_cg(As(buf, row, col), Aptr + row * K + (k0 + col));
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            int f = tid + i * NTHREADS;       // B: 512 float4
            int row = f >> 4;
            int col = (f & 15) << 3;
            cp_async_cg(Bs(buf, row, col), Bptr + (k0 + row) * N + col);
        }
        cp_async_commit();
    };

    // prologue: fill first STAGES-1 buffers
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s)
        loadTile(s, s);
    cp_async_wait<STAGES - 2>();
    __syncthreads();

    for (int t = 0; t < NTILES; ++t) {
        int cur = t % STAGES;
        // prefetch the tile STAGES-1 ahead
        int loadT = t + STAGES - 1;
        if (loadT < NTILES)
            loadTile(loadT, loadT % STAGES);
        else
            cp_async_commit();  // keep group count consistent

#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            uint32_t a[MT][4];
            uint32_t b[NT][2];
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                int r = warpRow * WM + i * 16 + (lane & 15);
                int c = kk + (lane >> 4) * 8;
                ldmatrix_x4(a[i], As(cur, r, c));
            }
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                int r = kk + (lane & 15);
                int c = warpCol * WN + j * 8;
                ldmatrix_x2_trans(b[j], Bs(cur, r, c));
            }
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int j = 0; j < NT; ++j)
                    mma_m16n8k16(acc[i][j], a[i], b[j]);
        }

        cp_async_wait<STAGES - 2>();
        __syncthreads();
    }

    // epilogue: C fragment (m16n8 f32) -> half, row-major store
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
}  // namespace v3impl

PLAYGROUND_MATMUL_DEC(float16_t, 3, m, n, k, A, B, C)
{
    using namespace v3impl;
    static bool inited = false;
    if (!inited) {
        cudaFuncSetAttribute(gemm_v3,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             SMEM_HALFS * sizeof(half));
        inited = true;
    }
    dim3 block(NTHREADS);
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    gemm_v3<<<grid, block, SMEM_HALFS * sizeof(half)>>>(int(m), int(n), int(k),
                                                        A, B, C);
}

}  // namespace playground
