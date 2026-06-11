#include <cuda_runtime.h>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{
namespace v6
{
constexpr int BM = 128;
constexpr int BN = 256;
constexpr int BK = 32;
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 4;
constexpr int NWARPS = WARPS_M * WARPS_N;  // 8
constexpr int NTHREADS = NWARPS * 32;      // 256
constexpr int WM = BM / WARPS_M;           // 64
constexpr int WN = BN / WARPS_N;           // 64
constexpr int MMA_M = WM / 16;             // 4
constexpr int MMA_N = WN / 8;              // 8
constexpr int MMA_K = BK / 16;             // 2
constexpr int PADA = 8;
constexpr int PADB = 8;
constexpr int SA = BK + PADA;
constexpr int SB = BN + PADB;
constexpr int STAGES = 4;
constexpr int A_STAGE = BM * SA;
constexpr int B_STAGE = BK * SB;
constexpr int A_ITERS = (BM * BK / 8) / NTHREADS;     // 2
constexpr int B_ITERS = (BK * BN / 8) / NTHREADS;     // 4
constexpr int A_ROWS_PER_ITER = NTHREADS / (BK / 8);  // 64
constexpr int B_ROWS_PER_ITER = NTHREADS / (BN / 8);  // 8

__device__ __forceinline__ uint32_t smem_addr(const void* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async_cg(uint32_t dst, const void* src)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
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
__device__ __forceinline__ void ldm_x4(uint32_t a, uint32_t& r0, uint32_t& r1,
                                        uint32_t& r2, uint32_t& r3)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(a));
}
__device__ __forceinline__ void ldm_x4_trans(uint32_t a, uint32_t& r0, uint32_t& r1,
                                              uint32_t& r2, uint32_t& r3)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(a));
}
__device__ __forceinline__ void mma_m16n8k16(float* acc, const uint32_t* a,
                                              const uint32_t* b)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__global__ void __launch_bounds__(NTHREADS, 1)
matmul_kernel(int M, int N, int K, const half* __restrict__ A,
              const half* __restrict__ B, half* __restrict__ C)
{
    extern __shared__ half smem[];
    half* Abuf = smem;
    half* Bbuf = smem + STAGES * A_STAGE;

    const int tid = threadIdx.x;
    const int warpId = tid >> 5;
    const int lane = tid & 31;
    const int warpRow = warpId / WARPS_N;
    const int warpCol = warpId % WARPS_N;
    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    const int aRowT = tid >> 2;
    const int aColT = (tid & 3) << 3;
    const int bRowT = tid / (BN / 8);
    const int bColT = (tid % (BN / 8)) << 3;

    const int aLdRow = warpRow * WM + (lane & 15);
    const int aLdColBase = (lane >> 4) << 3;
    const int bLdRow = lane & 15;
    const int bLdCol = warpCol * WN;

    float acc[MMA_M][MMA_N][4];
#pragma unroll
    for (int m = 0; m < MMA_M; ++m)
#pragma unroll
        for (int n = 0; n < MMA_N; ++n)
#pragma unroll
            for (int t = 0; t < 4; ++t)
                acc[m][n][t] = 0.0f;

    auto load_tile = [&](int stage, int k0) {
        half* As = Abuf + stage * A_STAGE;
        half* Bs = Bbuf + stage * B_STAGE;
#pragma unroll
        for (int i = 0; i < A_ITERS; ++i) {
            int r = aRowT + i * A_ROWS_PER_ITER;
            cp_async_cg(smem_addr(&As[r * SA + aColT]),
                        &A[(blockRow + r) * K + k0 + aColT]);
        }
#pragma unroll
        for (int i = 0; i < B_ITERS; ++i) {
            int r = bRowT + i * B_ROWS_PER_ITER;
            cp_async_cg(smem_addr(&Bs[r * SB + bColT]),
                        &B[(k0 + r) * N + blockCol + bColT]);
        }
    };

    uint32_t RA[2][MMA_M][4];
    uint32_t RB[2][MMA_N][2];

    auto load_frag = [&](int stage, int kt, int b) {
        half* As = Abuf + stage * A_STAGE;
        half* Bs = Bbuf + stage * B_STAGE;
#pragma unroll
        for (int m = 0; m < MMA_M; ++m) {
            int r = aLdRow + m * 16;
            int c = kt * 16 + aLdColBase;
            ldm_x4(smem_addr(&As[r * SA + c]), RA[b][m][0], RA[b][m][1],
                   RA[b][m][2], RA[b][m][3]);
        }
#pragma unroll
        for (int n2 = 0; n2 < MMA_N / 2; ++n2) {
            int r = kt * 16 + bLdRow;
            int c = bLdCol + n2 * 16 + ((lane >> 4) << 3);
            ldm_x4_trans(smem_addr(&Bs[r * SB + c]), RB[b][2 * n2][0],
                         RB[b][2 * n2][1], RB[b][2 * n2 + 1][0], RB[b][2 * n2 + 1][1]);
        }
    };

    const int numK = K / BK;

    // Prologue: issue STAGES-1 tile loads
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        load_tile(s, s * BK);
        cp_async_commit();
    }
    cp_async_wait<STAGES - 3>();  // publish 2 stages ahead
    __syncthreads();

    // Preload first fragment (tile 0, kt 0) into buffer 0
    load_frag(0, 0, 0);

    for (int T = 0; T < numK; ++T) {
        int readStage = T % STAGES;
        int writeStage = (T + STAGES - 1) % STAGES;
        int writeTile = T + STAGES - 1;
        if (writeTile < numK) {
            load_tile(writeStage, writeTile * BK);
        }
        cp_async_commit();

#pragma unroll
        for (int kt = 0; kt < MMA_K; ++kt) {
            int cur = kt & 1;
            int nxt = cur ^ 1;
            // prefetch next fragment
            if (kt + 1 < MMA_K) {
                load_frag(readStage, kt + 1, nxt);
            } else if (T + 1 < numK) {
                load_frag((T + 1) % STAGES, 0, nxt);
            }
#pragma unroll
            for (int m = 0; m < MMA_M; ++m)
#pragma unroll
                for (int n = 0; n < MMA_N; ++n)
                    mma_m16n8k16(acc[m][n], RA[cur][m], RB[cur][n]);
        }

        cp_async_wait<STAGES - 3>();
        __syncthreads();
    }

    const int gid = lane >> 2;
    const int tig = lane & 3;
#pragma unroll
    for (int m = 0; m < MMA_M; ++m)
#pragma unroll
        for (int n = 0; n < MMA_N; ++n) {
            int row = blockRow + warpRow * WM + m * 16 + gid;
            int col = blockCol + warpCol * WN + n * 8 + tig * 2;
            float* d = acc[m][n];
            *reinterpret_cast<half2*>(&C[row * N + col]) =
                __floats2half2_rn(d[0], d[1]);
            *reinterpret_cast<half2*>(&C[(row + 8) * N + col]) =
                __floats2half2_rn(d[2], d[3]);
        }
}
}  // namespace v6

PLAYGROUND_MATMUL_DEC(float16_t, 6, m, n, k, A, B, C)
{
    static bool configured = false;
    size_t smem = v6::STAGES * (v6::A_STAGE + v6::B_STAGE) * sizeof(half);
    if (!configured) {
        cudaFuncSetAttribute(v6::matmul_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        configured = true;
    }
    dim3 block(v6::NTHREADS);
    dim3 grid(n / v6::BN, m / v6::BM);
    v6::matmul_kernel<<<grid, block, smem>>>(int(m), int(n), int(k), A, B, C);
}

}  // namespace playground
