#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "playground/matmul.hpp"

namespace playground
{
namespace v3impl
{
constexpr int BM = 128;
constexpr int BN = 256;
constexpr int BK = 32;
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 4;
constexpr int WM = BM / WARPS_M;  // 64
constexpr int WN = BN / WARPS_N;  // 64
constexpr int MMA_M = WM / 16;    // 4
constexpr int MMA_N = WN / 8;     // 8
constexpr int MMA_K = BK / 16;    // 2
constexpr int NWARPS = WARPS_M * WARPS_N;  // 8
constexpr int NTHREADS = NWARPS * 32;      // 256
constexpr int STAGES = 4;

constexpr int APAD = 8;
constexpr int BPAD = 8;
constexpr int ALEAD = BK + APAD;  // 40
constexpr int BLEAD = BN + BPAD;  // 264
constexpr int ASTAGE = BM * ALEAD;
constexpr int BSTAGE = BK * BLEAD;

__device__ __forceinline__ uint32_t smem_addr(const void* ptr)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}
__device__ __forceinline__ void cp_async_cg(uint32_t dst, const void* src)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst),
                 "l"(src));
}
__device__ __forceinline__ void cp_commit()
{
    asm volatile("cp.async.commit_group;\n");
}
template <int N>
__device__ __forceinline__ void cp_wait()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void ldm_x4(uint32_t addr, uint32_t& r0,
                                       uint32_t& r1, uint32_t& r2, uint32_t& r3)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr));
}
__device__ __forceinline__ void ldm_x4_trans(uint32_t addr, uint32_t& r0,
                                             uint32_t& r1, uint32_t& r2,
                                             uint32_t& r3)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr));
}
__device__ __forceinline__ void mma_m16n8k16(float* d, const uint32_t* a,
                                             const uint32_t* b)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__global__ void __launch_bounds__(NTHREADS)
    kernel_v3(int M, int N, int K, const half* __restrict__ A,
              const half* __restrict__ B, half* __restrict__ C)
{
    extern __shared__ half smem[];
    half* As = smem;
    half* Bs = smem + STAGES * ASTAGE;

    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;
    const int tid = threadIdx.x;
    const int warpId = tid >> 5;
    const int lane = tid & 31;
    const int warpRow = warpId / WARPS_N;
    const int warpCol = warpId % WARPS_N;

    const half* Atile = A + (size_t) blockRow * BM * K;
    const half* Btile = B + (size_t) blockCol * BN;

    auto load_A = [&](int stage, int k0) {
#pragma unroll
        for (int i = 0; i < (BM * BK / 8) / NTHREADS; i++) {
            int lin = tid + i * NTHREADS;
            int r = (lin * 8) / BK;
            int c = (lin * 8) % BK;
            cp_async_cg(smem_addr(&As[stage * ASTAGE + r * ALEAD + c]),
                        &Atile[(size_t) r * K + k0 + c]);
        }
    };
    auto load_B = [&](int stage, int k0) {
#pragma unroll
        for (int i = 0; i < (BK * BN / 8) / NTHREADS; i++) {
            int lin = tid + i * NTHREADS;
            int r = (lin * 8) / BN;
            int c = (lin * 8) % BN;
            cp_async_cg(smem_addr(&Bs[stage * BSTAGE + r * BLEAD + c]),
                        &Btile[(size_t)(k0 + r) * N + c]);
        }
    };

    const int aLaneRow = lane & 15;
    const int aLaneCol = (lane >> 4) * 8;
    // precompute base smem offsets (in halfs) for ldmatrix
    auto aOff = [&](int stage, int kk, int mm) {
        int row = warpRow * WM + mm * 16 + aLaneRow;
        int col = kk * 16 + aLaneCol;
        return stage * ASTAGE + row * ALEAD + col;
    };
    auto bOff = [&](int stage, int kk, int nn) {
        int row = kk * 16 + aLaneRow;
        int col = warpCol * WN + nn * 8 + aLaneCol;
        return stage * BSTAGE + row * BLEAD + col;
    };

    auto ld_A = [&](int stage, int kk, uint32_t af[][4]) {
#pragma unroll
        for (int mm = 0; mm < MMA_M; mm++)
            ldm_x4(smem_addr(&As[aOff(stage, kk, mm)]), af[mm][0], af[mm][1],
                   af[mm][2], af[mm][3]);
    };
    auto ld_B = [&](int stage, int kk, uint32_t bf[][2]) {
#pragma unroll
        for (int nn = 0; nn < MMA_N; nn += 2) {
            uint32_t r0, r1, r2, r3;
            ldm_x4_trans(smem_addr(&Bs[bOff(stage, kk, nn)]), r0, r1, r2, r3);
            bf[nn][0] = r0;
            bf[nn][1] = r1;
            bf[nn + 1][0] = r2;
            bf[nn + 1][1] = r3;
        }
    };

    float acc[MMA_M][MMA_N][4];
#pragma unroll
    for (int i = 0; i < MMA_M; i++)
#pragma unroll
        for (int j = 0; j < MMA_N; j++)
#pragma unroll
            for (int t = 0; t < 4; t++) acc[i][j][t] = 0.0f;

    const int numK = K / BK;

    // prologue
#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        load_A(s, s * BK);
        load_B(s, s * BK);
        cp_commit();
    }
    cp_wait<STAGES - 2>();
    __syncthreads();

    // register-level double buffer of fragments
    uint32_t af[2][MMA_M][4];
    uint32_t bf[2][MMA_N][2];
    ld_A(0, 0, af[0]);
    ld_B(0, 0, bf[0]);

    for (int kIter = 0; kIter < numK; kIter++) {
        int readStage = kIter % STAGES;
        int loadK = kIter + STAGES - 1;
        int wstage = loadK % STAGES;

#pragma unroll
        for (int kk = 0; kk < MMA_K; kk++) {
            int cur = kk & 1;
            if (kk + 1 < MMA_K) {
                ld_A(readStage, kk + 1, af[cur ^ 1]);
                ld_B(readStage, kk + 1, bf[cur ^ 1]);
            }
#pragma unroll
            for (int mm = 0; mm < MMA_M; mm++)
#pragma unroll
                for (int nn = 0; nn < MMA_N; nn++)
                    mma_m16n8k16(acc[mm][nn], af[cur][mm], bf[cur][nn]);
        }

        if (loadK < numK) {
            load_A(wstage, loadK * BK);
            load_B(wstage, loadK * BK);
        }
        cp_commit();
        cp_wait<STAGES - 2>();
        __syncthreads();

        if (kIter + 1 < numK) {
            int nextStage = (kIter + 1) % STAGES;
            ld_A(nextStage, 0, af[0]);
            ld_B(nextStage, 0, bf[0]);
        }
    }

    // scattered epilogue
    const int gid = lane >> 2;
    const int tig = lane & 3;
    const int cRow0 = blockRow * BM + warpRow * WM;
    const int cCol0 = blockCol * BN + warpCol * WN;
#pragma unroll
    for (int mm = 0; mm < MMA_M; mm++) {
#pragma unroll
        for (int nn = 0; nn < MMA_N; nn++) {
            float* d = acc[mm][nn];
            int r = cRow0 + mm * 16 + gid;
            int c = cCol0 + nn * 8 + tig * 2;
            *reinterpret_cast<__half2*>(&C[(size_t) r * N + c]) =
                __floats2half2_rn(d[0], d[1]);
            *reinterpret_cast<__half2*>(&C[(size_t)(r + 8) * N + c]) =
                __floats2half2_rn(d[2], d[3]);
        }
    }
}
}  // namespace v3impl

PLAYGROUND_MATMUL_DEC(float16_t, 3, m, n, k, A, B, C)
{
    using namespace v3impl;
    static bool configured = false;
    size_t smemBytes = STAGES * (ASTAGE + BSTAGE) * sizeof(half);
    if (!configured) {
        cudaFuncSetAttribute(kernel_v3,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int) smemBytes);
        configured = true;
    }
    dim3 block(NTHREADS);
    dim3 grid(n / BN, m / BM);
    kernel_v3<<<grid, block, smemBytes>>>((int) m, (int) n, (int) k, A, B, C);
}

}  // namespace playground
