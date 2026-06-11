// @file: task-1/src/matmul_f16/matmul_f16_v4_warp64.cu
//
// v4: raise arithmetic intensity to relieve the shared-memory pipe that bound
//     v3 (73% mem throughput, only 18% DRAM -> ldmatrix-bound).
//   - Block 128x256, BK=32, 8 warps (2x4) -> warp tile 64x64 (MT=4, NT=8).
//   - A via ldmatrix.x4 (1 per m16 tile); B via ldmatrix.x4.trans (1 per TWO
//     n8 tiles) -> 4 A + 4 B = 8 ldmatrix for 32 mma => 4 mma / ldmatrix
//     (v3 was 2). Same 3-stage cp.async ring, fp32 accumulate.

#include <cuda_fp16.h>

#include "playground/matmul.hpp"

namespace playground
{
namespace v4impl
{

constexpr int BM = 128;
constexpr int BN = 256;
constexpr int BK = 32;

constexpr int WARPS_M = 2;
constexpr int WARPS_N = 4;
constexpr int NWARPS = WARPS_M * WARPS_N;   // 8
constexpr int NTHREADS = NWARPS * 32;       // 256

constexpr int WM = BM / WARPS_M;  // 64
constexpr int WN = BN / WARPS_N;  // 64
constexpr int MT = WM / 16;       // 4
constexpr int NT = WN / 8;        // 8

constexpr int SKEW = 8;
constexpr int AS_LD = BK + SKEW;   // 40
constexpr int BS_LD = BN + SKEW;   // 264
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

__global__ void __launch_bounds__(NTHREADS) gemm_v4(int M, int N, int K,
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
        for (int i = 0; i < 2; ++i) {            // A: 512 float4 / 256 thr
            int f = tid + i * NTHREADS;
            int row = f >> 2;
            int col = (f & 3) << 3;
            cp_async_cg(As(buf, row, col), Aptr + row * K + (k0 + col));
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {            // B: 1024 float4 / 256 thr
            int f = tid + i * NTHREADS;
            int row = f >> 5;                    // 32 float4 per 256-wide row
            int col = (f & 31) << 3;
            cp_async_cg(Bs(buf, row, col), Bptr + (k0 + row) * N + col);
        }
        cp_async_commit();
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
            for (int j = 0; j < NT; j += 2) {   // x4.trans loads 2 n8 tiles
                int r = kk + (lane & 15);
                int c = warpCol * WN + j * 8 + (lane >> 4) * 8;
                uint32_t t4[4];
                ldmatrix_x4_trans(t4, Bs(cur, r, c));
                b[j][0] = t4[0];
                b[j][1] = t4[1];
                b[j + 1][0] = t4[2];
                b[j + 1][1] = t4[3];
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
}  // namespace v4impl

PLAYGROUND_MATMUL_DEC(float16_t, 4, m, n, k, A, B, C)
{
    using namespace v4impl;
    static bool inited = false;
    if (!inited) {
        cudaFuncSetAttribute(gemm_v4,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             SMEM_HALFS * sizeof(half));
        inited = true;
    }
    dim3 block(NTHREADS);
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    gemm_v4<<<grid, block, SMEM_HALFS * sizeof(half)>>>(int(m), int(n), int(k),
                                                        A, B, C);
}

}  // namespace playground
