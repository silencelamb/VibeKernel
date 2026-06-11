#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "playground/matmul.hpp"

namespace playground
{
namespace v3
{
// Block tile
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
// Warp grid over block: 2 rows x 4 cols = 8 warps = 256 threads.
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 4;
constexpr int THREADS = WARPS_M * WARPS_N * 32;  // 256
constexpr int WM = BM / WARPS_M;                 // 64
constexpr int WN = BN / WARPS_N;                 // 32
constexpr int WT_M = WM / 16;                    // 4  (m16 tiles per warp)
constexpr int WT_N = WN / 8;                     // 4  (n8 tiles per warp)
constexpr int APAD = 8;
constexpr int BPAD = 8;

__device__ __forceinline__ uint32_t saddr(const void* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void ldm_x4(uint32_t& a, uint32_t& b, uint32_t& c,
                                        uint32_t& d, uint32_t s)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
        : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
        : "r"(s));
}

__device__ __forceinline__ void ldm_x4_t(uint32_t& a, uint32_t& b, uint32_t& c,
                                          uint32_t& d, uint32_t s)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];\n"
        : "=r"(a), "=r"(b), "=r"(c), "=r"(d)
        : "r"(s));
}

__device__ __forceinline__ void mma1688(float* d, const uint32_t* a,
                                         const uint32_t* b)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3]));
}

__global__ __launch_bounds__(THREADS) void kernel(int M, int N, int K,
                                                   const half* __restrict__ A,
                                                   const half* __restrict__ B,
                                                   half* __restrict__ C)
{
    __shared__ half As[BM][BK + APAD];
    __shared__ half Bs[BK][BN + BPAD];

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;
    const int warpId = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int warpRow = warpId / WARPS_N;
    const int warpCol = warpId % WARPS_N;
    const int warpRowBase = warpRow * WM;
    const int warpColBase = warpCol * WN;

    float acc[WT_M][WT_N][4];
#pragma unroll
    for (int i = 0; i < WT_M; ++i)
#pragma unroll
        for (int j = 0; j < WT_N; ++j)
#pragma unroll
            for (int t = 0; t < 4; ++t)
                acc[i][j][t] = 0.0f;

    const float4* Ag = reinterpret_cast<const float4*>(A);
    const float4* Bg = reinterpret_cast<const float4*>(B);
    const int tid = threadIdx.x;

    for (int k0 = 0; k0 < K; k0 += BK) {
        // Load A tile (BM x BK): 512 float4, 2 per thread.
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            int idx = tid + r * THREADS;
            int row = idx / (BK / 8);
            int c8 = (idx % (BK / 8)) * 8;
            *reinterpret_cast<float4*>(&As[row][c8]) =
                Ag[(size_t(blockRow + row) * K + k0 + c8) / 8];
        }
        // Load B tile (BK x BN): 512 float4, 2 per thread.
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            int idx = tid + r * THREADS;
            int row = idx / (BN / 8);
            int c8 = (idx % (BN / 8)) * 8;
            *reinterpret_cast<float4*>(&Bs[row][c8]) =
                Bg[(size_t(k0 + row) * N + blockCol + c8) / 8];
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            uint32_t Aop[WT_M][4];
            uint32_t Bop[WT_N][2];
#pragma unroll
            for (int i = 0; i < WT_M; ++i) {
                int row = warpRowBase + i * 16 + (lane % 16);
                int col = kk + (lane / 16) * 8;
                ldm_x4(Aop[i][0], Aop[i][1], Aop[i][2], Aop[i][3],
                       saddr(&As[row][col]));
            }
#pragma unroll
            for (int j = 0; j < WN / 16; ++j) {
                uint32_t r0, r1, r2, r3;
                int row = kk + (lane % 16);
                int col = warpColBase + j * 16 + (lane / 16) * 8;
                ldm_x4_t(r0, r1, r2, r3, saddr(&Bs[row][col]));
                Bop[2 * j + 0][0] = r0;
                Bop[2 * j + 0][1] = r1;
                Bop[2 * j + 1][0] = r2;
                Bop[2 * j + 1][1] = r3;
            }
#pragma unroll
            for (int i = 0; i < WT_M; ++i)
#pragma unroll
                for (int nn = 0; nn < WT_N; ++nn)
                    mma1688(acc[i][nn], Aop[i], Bop[nn]);
        }
        __syncthreads();
    }

    // Epilogue: store. m16n8k16 .f32 accumulator layout:
    //   d0,d1 -> row=group,   col=2*tig + {0,1}
    //   d2,d3 -> row=group+8, col=2*tig + {0,1}
    const int group = lane / 4;
    const int tig = lane % 4;
#pragma unroll
    for (int i = 0; i < WT_M; ++i) {
#pragma unroll
        for (int nn = 0; nn < WT_N; ++nn) {
            int rowBase = blockRow + warpRowBase + i * 16;
            int colBase = blockCol + warpColBase + nn * 8;
            half2 lo = __floats2half2_rn(acc[i][nn][0], acc[i][nn][1]);
            half2 hi = __floats2half2_rn(acc[i][nn][2], acc[i][nn][3]);
            *reinterpret_cast<half2*>(&C[size_t(rowBase + group) * N + colBase +
                                         2 * tig]) = lo;
            *reinterpret_cast<half2*>(
                &C[size_t(rowBase + group + 8) * N + colBase + 2 * tig]) = hi;
        }
    }
}
}  // namespace v3

PLAYGROUND_MATMUL_DEC(float16_t, 3, m, n, k, A, B, C)
{
    using namespace v3;
    dim3 block(THREADS);
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    kernel<<<grid, block>>>(int(m), int(n), int(k), A, B, C);
}

}  // namespace playground
