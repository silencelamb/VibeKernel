// matmul_f16_v2 — FULL STACK reference (Phase 1).
// Ingredients: cp.async (cg.16B) + 3-stage software pipeline (triple buffer)
// + shared-memory XOR swizzle (bank-conflict reduction) + 128-bit float4 loads
// + warp tiling / register blocking + ldmatrix->mma (m16n8k16, fp32 accumulate).
// Tile 128x128x32, 8 warps. m=n=k assumed multiples of 128.
#include <cstdint>
#include <cuda_fp16.h>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{
namespace v2cfg
{
constexpr int BM = 128, BN = 128, BK = 32;
constexpr int WARPS_M = 2, WARPS_N = 4;  // 8 warps = 256 threads
constexpr int WM = BM / WARPS_M;         // 64
constexpr int WN = BN / WARPS_N;         // 32
constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;
constexpr int WMITER = WM / MMA_M;  // 4
constexpr int WNITER = WN / MMA_N;  // 4
constexpr int KSTEPS = BK / MMA_K;  // 2
constexpr int THREADS = WARPS_M * WARPS_N * 32;  // 256
constexpr int STAGES = 3;
constexpr int ASIZE = BM * BK;  // halves per A stage
constexpr int BSIZE = BK * BN;  // halves per B stage
}  // namespace v2cfg

__device__ __forceinline__ uint32_t smem_u32(const void* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// XOR swizzle at 8-half (16-byte) segment granularity.  COLS = halves per row.
template <int COLS>
__device__ __forceinline__ int swz(int row, int col)
{
    constexpr int NSEG = COLS >> 3;
    int seg = (col >> 3) ^ (row & (NSEG - 1));
    return row * COLS + (seg << 3) + (col & 7);
}

__device__ __forceinline__ void cpasync_cg16(uint32_t dst, const void* src)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst),
                 "l"(src));
}

__device__ __forceinline__ void ldm_x4(uint32_t (&r)[4], uint32_t a)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(a));
}

__device__ __forceinline__ void ldm_x2_trans(uint32_t (&r)[2], uint32_t a)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1])
        : "r"(a));
}

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4],
                                          const uint32_t (&b)[2],
                                          const float (&c)[4])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

__global__ __launch_bounds__(v2cfg::THREADS) void gemm_v2(
    const half* __restrict__ A, const half* __restrict__ B,
    half* __restrict__ C, int M, int N, int K)
{
    using namespace v2cfg;
    extern __shared__ half smem[];
    half* sA = smem;                    // STAGES * ASIZE
    half* sB = smem + STAGES * ASIZE;   // STAGES * BSIZE

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int warp_m = (warp / WARPS_N) * WM;
    const int warp_n = (warp % WARPS_N) * WN;
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    float acc[WMITER][WNITER][4];
#pragma unroll
    for (int i = 0; i < WMITER; ++i)
#pragma unroll
        for (int j = 0; j < WNITER; ++j)
#pragma unroll
            for (int t = 0; t < 4; ++t)
                acc[i][j][t] = 0.0f;

    // global->shared maps (float4 = 8 halves)
    const int aRow0 = tid / 4, aColF4 = tid % 4;    // rows 0..63, 2 chunks/thread (+64)
    const int bRow0 = tid / 16, bColF4 = tid % 16;  // rows 0..15, 2 chunks/thread (+16)
    const int NTILES = K / BK;

    auto load_tile = [&](int buf, int k0) {
        half* dA = sA + buf * ASIZE;
        half* dB = sB + buf * BSIZE;
#pragma unroll
        for (int s = 0; s < 2; ++s) {
            int row = aRow0 + s * 64, col = aColF4 * 8;
            cpasync_cg16(smem_u32(&dA[swz<BK>(row, col)]),
                         &A[(block_row + row) * K + (k0 + col)]);
        }
#pragma unroll
        for (int s = 0; s < 2; ++s) {
            int row = bRow0 + s * 16, col = bColF4 * 8;
            cpasync_cg16(smem_u32(&dB[swz<BN>(row, col)]),
                         &B[(k0 + row) * N + (block_col + col)]);
        }
    };

    // prologue: issue STAGES-1 tiles
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        load_tile(s, s * BK);
        asm volatile("cp.async.commit_group;\n");
    }

    const int ar = (lane & 7) + ((lane >> 3) & 1) * 8;
    const int ac = ((lane >> 3) >> 1) * 8;
    const int br = lane & 15;

    for (int tile = 0; tile < NTILES; ++tile) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(STAGES - 2));
        __syncthreads();

        int read_buf = tile % STAGES;
        half* rA = sA + read_buf * ASIZE;
        half* rB = sB + read_buf * BSIZE;

        // prefetch tile (tile+STAGES-1)
        int pf = tile + STAGES - 1;
        if (pf < NTILES) {
            load_tile(pf % STAGES, pf * BK);
        }
        asm volatile("cp.async.commit_group;\n");

#pragma unroll
        for (int ks = 0; ks < KSTEPS; ++ks) {
            int kk = ks * MMA_K;
            uint32_t af[WMITER][4];
            uint32_t bf[WNITER][2];
#pragma unroll
            for (int mi = 0; mi < WMITER; ++mi)
                ldm_x4(af[mi],
                       smem_u32(&rA[swz<BK>(warp_m + mi * MMA_M + ar, kk + ac)]));
#pragma unroll
            for (int ni = 0; ni < WNITER; ++ni)
                ldm_x2_trans(
                    bf[ni], smem_u32(&rB[swz<BN>(kk + br, warp_n + ni * MMA_N)]));
#pragma unroll
            for (int mi = 0; mi < WMITER; ++mi)
#pragma unroll
                for (int ni = 0; ni < WNITER; ++ni)
                    mma16816(acc[mi][ni], af[mi], bf[ni], acc[mi][ni]);
        }
    }

    const int gid = lane >> 2;
    const int t4 = lane & 3;
#pragma unroll
    for (int mi = 0; mi < WMITER; ++mi)
#pragma unroll
        for (int ni = 0; ni < WNITER; ++ni) {
            int row = block_row + warp_m + mi * MMA_M + gid;
            int col = block_col + warp_n + ni * MMA_N + t4 * 2;
            __half2 lo = __floats2half2_rn(acc[mi][ni][0], acc[mi][ni][1]);
            __half2 hi = __floats2half2_rn(acc[mi][ni][2], acc[mi][ni][3]);
            *reinterpret_cast<__half2*>(&C[row * N + col]) = lo;
            *reinterpret_cast<__half2*>(&C[(row + 8) * N + col]) = hi;
        }
}

PLAYGROUND_MATMUL_DEC(float16_t, 2, m, n, k, A, B, C)
{
    using namespace v2cfg;
    static bool configured = false;
    constexpr int smem_bytes = STAGES * (ASIZE + BSIZE) * sizeof(half);
    if (!configured) {
        cudaFuncSetAttribute(gemm_v2,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smem_bytes);
        configured = true;
    }
    dim3 block(THREADS);
    dim3 grid(static_cast<unsigned>(n / BN), static_cast<unsigned>(m / BM));
    gemm_v2<<<grid, block, smem_bytes>>>(A, B, C, static_cast<int>(m),
                                         static_cast<int>(n),
                                         static_cast<int>(k));
}

}  // namespace playground
