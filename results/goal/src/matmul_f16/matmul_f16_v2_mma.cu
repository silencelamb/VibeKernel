#include <cuda_runtime.h>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{
namespace v2
{
// Block tile
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
// Warp grid
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 4;
constexpr int NWARPS = WARPS_M * WARPS_N;  // 8
constexpr int NTHREADS = NWARPS * 32;      // 256
// Warp tile
constexpr int WM = BM / WARPS_M;  // 64
constexpr int WN = BN / WARPS_N;  // 32
// mma m16n8k16
constexpr int MMA_M = WM / 16;  // 4
constexpr int MMA_N = WN / 8;   // 4
constexpr int MMA_K = BK / 16;  // 2
// padding (halfs)
constexpr int PADA = 8;
constexpr int PADB = 8;
constexpr int SA = BK + PADA;  // A row stride
constexpr int SB = BN + PADB;  // B row stride
constexpr int STAGES = 3;

constexpr int A_STAGE = BM * SA;
constexpr int B_STAGE = BK * SB;

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
__device__ __forceinline__ void ldm_x2_trans(uint32_t a, uint32_t& r0, uint32_t& r1)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
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

__global__ void __launch_bounds__(NTHREADS)
matmul_kernel(int M, int N, int K, const half* __restrict__ A,
              const half* __restrict__ B, half* __restrict__ C)
{
    extern __shared__ half smem[];
    half* Abuf = smem;
    half* Bbuf = smem + STAGES * A_STAGE;

    const int tid = threadIdx.x;
    const int warpId = tid >> 5;
    const int lane = tid & 31;
    const int warpRow = warpId / WARPS_N;  // 0..1
    const int warpCol = warpId % WARPS_N;  // 0..3
    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    // A load mapping: 512 cp.async / 256 thr = 2 each ; r=t/4, c=(t%4)*8
    const int aRowT = tid >> 2;          // 0..63 (for i=0), +64 for i=1
    const int aColT = (tid & 3) << 3;    // 0,8,16,24
    // B load mapping: r=t/16, c=(t%16)*8
    const int bRowT = tid >> 4;          // 0..15 (i=0), +16 i=1
    const int bColT = (tid & 15) << 3;   // 0..120

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
        for (int i = 0; i < 2; ++i) {
            int r = aRowT + i * 64;
            cp_async_cg(smem_addr(&As[r * SA + aColT]),
                        &A[(blockRow + r) * K + k0 + aColT]);
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            int r = bRowT + i * 16;
            cp_async_cg(smem_addr(&Bs[r * SB + bColT]),
                        &B[(k0 + r) * N + blockCol + bColT]);
        }
    };

    // Prologue
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        load_tile(s, s * BK);
        cp_async_commit();
    }
    cp_async_wait<STAGES - 2>();
    __syncthreads();

    const int numK = K / BK;
    int smem_read = 0;
    int smem_write = STAGES - 1;
    int k0 = (STAGES - 1) * BK;

    for (int kIter = 0; kIter < numK; ++kIter) {
        if (k0 < K) {
            load_tile(smem_write, k0);
        }
        cp_async_commit();

        half* As = Abuf + smem_read * A_STAGE;
        half* Bs = Bbuf + smem_read * B_STAGE;

        uint32_t RA[MMA_M][MMA_K][4];
        uint32_t RB[MMA_N][MMA_K][2];
#pragma unroll
        for (int m = 0; m < MMA_M; ++m)
#pragma unroll
            for (int kt = 0; kt < MMA_K; ++kt) {
                int r = warpRow * WM + m * 16 + (lane & 15);
                int c = kt * 16 + ((lane >> 4) << 3);
                ldm_x4(smem_addr(&As[r * SA + c]), RA[m][kt][0], RA[m][kt][1],
                       RA[m][kt][2], RA[m][kt][3]);
            }
#pragma unroll
        for (int n = 0; n < MMA_N; ++n)
#pragma unroll
            for (int kt = 0; kt < MMA_K; ++kt) {
                int r = kt * 16 + (lane & 15);
                int c = warpCol * WN + n * 8;
                ldm_x2_trans(smem_addr(&Bs[r * SB + c]), RB[n][kt][0], RB[n][kt][1]);
            }
#pragma unroll
        for (int m = 0; m < MMA_M; ++m)
#pragma unroll
            for (int n = 0; n < MMA_N; ++n)
#pragma unroll
                for (int kt = 0; kt < MMA_K; ++kt)
                    mma_m16n8k16(acc[m][n], RA[m][kt], RB[n][kt]);

        cp_async_wait<STAGES - 2>();
        __syncthreads();
        smem_read = (smem_read + 1) % STAGES;
        smem_write = (smem_write + 1) % STAGES;
        k0 += BK;
    }

    // Epilogue: store C
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
}  // namespace v2

PLAYGROUND_MATMUL_DEC(float16_t, 2, m, n, k, A, B, C)
{
    static bool configured = false;
    if (!configured) {
        size_t smem = v2::STAGES * (v2::A_STAGE + v2::B_STAGE) * sizeof(half);
        cudaFuncSetAttribute(v2::matmul_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        configured = true;
    }
    size_t smem = v2::STAGES * (v2::A_STAGE + v2::B_STAGE) * sizeof(half);
    dim3 block(v2::NTHREADS);
    dim3 grid(n / v2::BN, m / v2::BM);
    v2::matmul_kernel<<<grid, block, smem>>>(int(m), int(n), int(k), A, B, C);
}

}  // namespace playground
