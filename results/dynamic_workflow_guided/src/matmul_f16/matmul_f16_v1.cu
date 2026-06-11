// matmul_f16_v1 — first correct fp16 tensor-core GEMM (mma.sync m16n8k16 + ldmatrix).
// Single-stage shared staging, synchronous float4 global loads, fp32 accumulate.
// Purpose: validate the MMA / ldmatrix fragment layouts (the hardest correctness
// part) before building up the full optimization stack. m=n=k assumed multiples of 128.
#include <cstdint>
#include <cuda_fp16.h>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{
namespace v1cfg
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
}  // namespace v1cfg

__device__ __forceinline__ uint32_t smem_u32(const void* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
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

__global__ __launch_bounds__(v1cfg::THREADS) void gemm_v1(
    const half* __restrict__ A, const half* __restrict__ B,
    half* __restrict__ C, int M, int N, int K)
{
    using namespace v1cfg;
    __shared__ half As[BM][BK];  // 128 x 32
    __shared__ half Bs[BK][BN];  // 32 x 128

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int warp_m = (warp / WARPS_N) * WM;  // 0, 64
    const int warp_n = (warp % WARPS_N) * WN;  // 0,32,64,96

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

    // ---- global->shared load index maps (float4 = 8 halves) ----
    // A tile [128][32]: 512 float4, 2 per thread.  B tile [32][128]: 512 float4, 2 per thread.
    const int aRow0 = tid / 4, aColF4 = tid % 4;       // 256 threads cover rows 0..63
    const int bRow0 = tid / 16, bColF4 = tid % 16;     // 256 threads cover rows 0..15

    for (int k0 = 0; k0 < K; k0 += BK) {
        // ---- load A: each thread 2 float4 (rows aRow0 and aRow0+64) ----
#pragma unroll
        for (int s = 0; s < 2; ++s) {
            int row = aRow0 + s * 64;
            int col = aColF4 * 8;
            const float4* gp = reinterpret_cast<const float4*>(
                &A[(block_row + row) * K + (k0 + col)]);
            *reinterpret_cast<float4*>(&As[row][col]) = *gp;
        }
        // ---- load B: each thread 2 float4 (rows bRow0 and bRow0+16) ----
#pragma unroll
        for (int s = 0; s < 2; ++s) {
            int row = bRow0 + s * 16;
            int col = bColF4 * 8;
            const float4* gp = reinterpret_cast<const float4*>(
                &B[(k0 + row) * N + (block_col + col)]);
            *reinterpret_cast<float4*>(&Bs[row][col]) = *gp;
        }
        __syncthreads();

        // ---- compute ----
#pragma unroll
        for (int ks = 0; ks < KSTEPS; ++ks) {
            int kk = ks * MMA_K;
            uint32_t af[WMITER][4];
            uint32_t bf[WNITER][2];
            // A fragments: ldmatrix.x4, lane -> (row,col) within 16x16
            int ar = (lane & 7) + ((lane >> 3) & 1) * 8;
            int ac = ((lane >> 3) >> 1) * 8;
#pragma unroll
            for (int mi = 0; mi < WMITER; ++mi) {
                ldm_x4(af[mi],
                       smem_u32(&As[warp_m + mi * MMA_M + ar][kk + ac]));
            }
            // B fragments: ldmatrix.x2.trans, lane(0..15) -> row kk+lane
            int br = lane & 15;
#pragma unroll
            for (int ni = 0; ni < WNITER; ++ni) {
                ldm_x2_trans(bf[ni],
                             smem_u32(&Bs[kk + br][warp_n + ni * MMA_N]));
            }
#pragma unroll
            for (int mi = 0; mi < WMITER; ++mi)
#pragma unroll
                for (int ni = 0; ni < WNITER; ++ni)
                    mma16816(acc[mi][ni], af[mi], bf[ni], acc[mi][ni]);
        }
        __syncthreads();
    }

    // ---- store C (row-major). lane -> (gid=lane>>2, t=lane&3) ----
    const int gid = lane >> 2;
    const int t4 = lane & 3;
#pragma unroll
    for (int mi = 0; mi < WMITER; ++mi) {
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
}

PLAYGROUND_MATMUL_DEC(float16_t, 1, m, n, k, A, B, C)
{
    dim3 block(v1cfg::THREADS);
    dim3 grid(static_cast<unsigned>(n / v1cfg::BN),
              static_cast<unsigned>(m / v1cfg::BM));
    gemm_v1<<<grid, block>>>(A, B, C, static_cast<int>(m),
                             static_cast<int>(n), static_cast<int>(k));
}

}  // namespace playground
