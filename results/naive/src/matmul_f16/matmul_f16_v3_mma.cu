#include "playground/matmul.hpp"
#include "playground/system.hpp"

#include <cstdint>
#include <cuda_fp16.h>

namespace playground
{
namespace v3
{
constexpr int BM = 128, BN = 128, BK = 32;
constexpr int WARPS_M = 2, WARPS_N = 4, NWARPS = 8, NTHREADS = 256;
constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N;  // 64, 32
constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;
constexpr int M_ITER = WM / MMA_M, N_ITER = WN / MMA_N, K_ITER = BK / MMA_K;
constexpr int APAD = 8, BPAD = 8;
constexpr int AS_STRIDE = BK + APAD, AS_TILE = BM * AS_STRIDE;
constexpr int BS_STRIDE = BN + BPAD, BS_TILE = BK * BS_STRIDE;
constexpr int NSTAGES = 3;

__device__ __forceinline__ uint32_t cvta(const void* p)
{
    return (uint32_t) __cvta_generic_to_shared(p);
}
__device__ __forceinline__ void cp16(void* s, const void* g)
{
    asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n" ::"r"(cvta(s)),
                 "l"(g));
}
__device__ __forceinline__ void commit()
{
    asm volatile("cp.async.commit_group;\n");
}
template <int N>
__device__ __forceinline__ void waitg()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void ldm4(uint32_t r[4], uint32_t a)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(a));
}
__device__ __forceinline__ void ldm4t(uint32_t r[4], uint32_t a)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(a));
}
__device__ __forceinline__ void mma(float d[4], const uint32_t a[4],
                                    const uint32_t b[2])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__global__ __launch_bounds__(NTHREADS) void kernel(const half* __restrict__ A,
                                                    const half* __restrict__ B,
                                                    half* __restrict__ C, int M,
                                                    int N, int K)
{
    extern __shared__ half smem[];
    half* As = smem;
    half* Bs = smem + NSTAGES * AS_TILE;
    const int bx = blockIdx.x, by = blockIdx.y, tid = threadIdx.x;
    const int warpId = tid >> 5, laneId = tid & 31;
    const int warpM = warpId / WARPS_N, warpN = warpId % WARPS_N;
    const int rowBase = by * BM, colBase = bx * BN;
    const int numTiles = K / BK;

    float acc[M_ITER][N_ITER][4];
#pragma unroll
    for (int i = 0; i < M_ITER; i++)
#pragma unroll
        for (int j = 0; j < N_ITER; j++)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    auto load_stage = [&](int st, int k0) {
#pragma unroll
        for (int i = tid; i < BM * BK / 8; i += NTHREADS) {
            int r = (i * 8) / BK, c = (i * 8) % BK;
            cp16(&As[st * AS_TILE + r * AS_STRIDE + c],
                 &A[(rowBase + r) * K + k0 + c]);
        }
#pragma unroll
        for (int i = tid; i < BK * BN / 8; i += NTHREADS) {
            int r = (i * 8) / BN, c = (i * 8) % BN;
            cp16(&Bs[st * BS_TILE + r * BS_STRIDE + c],
                 &B[(k0 + r) * N + colBase + c]);
        }
    };

#pragma unroll
    for (int s = 0; s < NSTAGES - 1; s++) {
        load_stage(s, s * BK);
        commit();
    }

    for (int t = 0; t < numTiles; t++) {
        waitg<NSTAGES - 2>();
        __syncthreads();
        int cur = t % NSTAGES;
        int pf = t + NSTAGES - 1;
        if (pf < numTiles)
            load_stage(pf % NSTAGES, pf * BK);
        commit();

        half* Acur = As + cur * AS_TILE;
        half* Bcur = Bs + cur * BS_TILE;
#pragma unroll
        for (int kk = 0; kk < BK; kk += MMA_K) {
            uint32_t a[M_ITER][4];
            uint32_t b[N_ITER][2];
#pragma unroll
            for (int mi = 0; mi < M_ITER; mi++) {
                uint32_t addr = cvta(
                    &Acur[(warpM * WM + mi * MMA_M + (laneId % 16)) * AS_STRIDE +
                          kk + (laneId / 16) * 8]);
                ldm4(a[mi], addr);
            }
#pragma unroll
            for (int chunk = 0; chunk < WN / 16; chunk++) {
                uint32_t r[4];
                uint32_t addr = cvta(
                    &Bcur[(kk + (laneId % 16)) * BS_STRIDE + warpN * WN +
                          chunk * 16 + (laneId / 16) * 8]);
                ldm4t(r, addr);
                b[chunk * 2 + 0][0] = r[0];
                b[chunk * 2 + 0][1] = r[1];
                b[chunk * 2 + 1][0] = r[2];
                b[chunk * 2 + 1][1] = r[3];
            }
#pragma unroll
            for (int mi = 0; mi < M_ITER; mi++)
#pragma unroll
                for (int ni = 0; ni < N_ITER; ni++)
                    mma(acc[mi][ni], a[mi], b[ni]);
        }
    }

    // Epilogue: direct fp32 -> fp16, vectorized half2 writes.
    const int group = laneId / 4, base = (laneId % 4) * 2;
#pragma unroll
    for (int mi = 0; mi < M_ITER; mi++)
#pragma unroll
        for (int ni = 0; ni < N_ITER; ni++) {
            int r0 = rowBase + warpM * WM + mi * MMA_M + group;
            int cc = colBase + warpN * WN + ni * MMA_N + base;
            half2 lo = __floats2half2_rn(acc[mi][ni][0], acc[mi][ni][1]);
            half2 hi = __floats2half2_rn(acc[mi][ni][2], acc[mi][ni][3]);
            *reinterpret_cast<half2*>(&C[r0 * N + cc]) = lo;
            *reinterpret_cast<half2*>(&C[(r0 + 8) * N + cc]) = hi;
        }
}

}  // namespace v3

PLAYGROUND_MATMUL_DEC(float16_t, 3, m, n, k, A, B, C)
{
    static bool inited = false;
    constexpr int smemBytes =
        (v3::NSTAGES * (v3::AS_TILE + v3::BS_TILE)) * sizeof(half);
    if (!inited) {
        cudaFuncSetAttribute((const void*) v3::kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smemBytes);
        inited = true;
    }
    dim3 block(v3::NTHREADS), grid(n / v3::BN, m / v3::BM);
    v3::kernel<<<grid, block, smemBytes>>>(A, B, C, int(m), int(n), int(k));
}

}  // namespace playground
