#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "playground/matmul.hpp"

namespace playground
{
namespace v2impl
{
// ---- Block / warp tiling ----
constexpr int BM = 128;
constexpr int BN = 256;
constexpr int BK = 32;
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 4;
constexpr int WM = BM / WARPS_M;  // 64
constexpr int WN = BN / WARPS_N;  // 64
constexpr int MMA_M = WM / 16;    // 4  (m16)
constexpr int MMA_N = WN / 8;     // 8  (n8)
constexpr int MMA_K = BK / 16;    // 2  (k16 substeps per BK)
constexpr int NWARPS = WARPS_M * WARPS_N;  // 8
constexpr int NTHREADS = NWARPS * 32;      // 256
constexpr int STAGES = 4;

// smem leading dims with padding to avoid bank conflicts (PAD=8 halfs = 16B)
constexpr int APAD = 8;
constexpr int BPAD = 8;
constexpr int ALEAD = BK + APAD;   // 40
constexpr int BLEAD = BN + BPAD;   // 264
// smem element counts per stage
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

__global__ void __launch_bounds__(NTHREADS)
    kernel_v2(int M, int N, int K, const half* __restrict__ A,
              const half* __restrict__ B, half* __restrict__ C)
{
    extern __shared__ half smem[];
    half* As = smem;                      // STAGES * ASTAGE
    half* Bs = smem + STAGES * ASTAGE;    // STAGES * BSTAGE

    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;
    const int tid = threadIdx.x;
    const int warpId = tid >> 5;
    const int lane = tid & 31;
    const int warpRow = warpId / WARPS_N;  // 0..1
    const int warpCol = warpId % WARPS_N;  // 0..3

    // global tile base
    const half* Atile = A + (size_t) blockRow * BM * K;
    const half* Btile = B + (size_t) blockCol * BN;

    // ---- load index maps (float4 = 8 halfs) ----
    // A: BM*BK/8 = 512 float4, 2 per thread
    // B: BK*BN/8 = 1024 float4, 4 per thread
    // For A: row in [0,BM), col in [0,BK)
    // For B: row in [0,BK), col in [0,BN)

    auto load_A = [&](int stage, int k0) {
#pragma unroll
        for (int i = 0; i < (BM * BK / 8) / NTHREADS; i++) {
            int lin = tid + i * NTHREADS;
            int r = (lin * 8) / BK;
            int c = (lin * 8) % BK;
            uint32_t d = smem_addr(&As[stage * ASTAGE + r * ALEAD + c]);
            cp_async_cg(d, &Atile[(size_t) r * K + k0 + c]);
        }
    };
    auto load_B = [&](int stage, int k0) {
#pragma unroll
        for (int i = 0; i < (BK * BN / 8) / NTHREADS; i++) {
            int lin = tid + i * NTHREADS;
            int r = (lin * 8) / BN;
            int c = (lin * 8) % BN;
            uint32_t d = smem_addr(&Bs[stage * BSTAGE + r * BLEAD + c]);
            cp_async_cg(d, &Btile[(size_t)(k0 + r) * N + c]);
        }
    };

    // ---- accumulators ----
    float acc[MMA_M][MMA_N][4];
#pragma unroll
    for (int i = 0; i < MMA_M; i++)
#pragma unroll
        for (int j = 0; j < MMA_N; j++)
#pragma unroll
            for (int t = 0; t < 4; t++) acc[i][j][t] = 0.0f;

    const int numK = K / BK;

    // ---- prologue: issue STAGES-1 loads ----
#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        load_A(s, s * BK);
        load_B(s, s * BK);
        cp_commit();
    }

    // ldmatrix base offsets within a stage
    // A frag addr: As[stage][ (warpRow*WM + mm*16 + lane%16) ][ kk*16 + (lane/16)*8 ]
    // B frag addr: Bs[stage][ kk*16 + lane%16 ][ warpCol*WN + nn*8 + (lane/16)*8 ]  (trans)
    const int aLaneRow = lane & 15;
    const int aLaneCol = (lane >> 4) * 8;

    for (int kIter = 0; kIter < numK; kIter++) {
        cp_wait<STAGES - 2>();
        __syncthreads();

        int readStage = kIter % STAGES;

        uint32_t a[MMA_K][MMA_M][4];
        uint32_t b[MMA_K][MMA_N][2];
#pragma unroll
        for (int kk = 0; kk < MMA_K; kk++) {
            // load A fragments
#pragma unroll
            for (int mm = 0; mm < MMA_M; mm++) {
                int row = warpRow * WM + mm * 16 + aLaneRow;
                int col = kk * 16 + aLaneCol;
                uint32_t ad =
                    smem_addr(&As[readStage * ASTAGE + row * ALEAD + col]);
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                    "{%0,%1,%2,%3}, [%4];\n"
                    : "=r"(a[kk][mm][0]), "=r"(a[kk][mm][1]),
                      "=r"(a[kk][mm][2]), "=r"(a[kk][mm][3])
                    : "r"(ad));
            }
            // load B fragments (2 n8-tiles per ldmatrix.x4.trans)
#pragma unroll
            for (int nn = 0; nn < MMA_N; nn += 2) {
                int row = kk * 16 + aLaneRow;
                int col = warpCol * WN + nn * 8 + aLaneCol;
                uint32_t bd =
                    smem_addr(&Bs[readStage * BSTAGE + row * BLEAD + col]);
                uint32_t r0, r1, r2, r3;
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                    "{%0,%1,%2,%3}, [%4];\n"
                    : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                    : "r"(bd));
                b[kk][nn][0] = r0;
                b[kk][nn][1] = r1;
                b[kk][nn + 1][0] = r2;
                b[kk][nn + 1][1] = r3;
            }
        }

        // issue next load (distributed before mma for simplicity)
        int loadK = kIter + STAGES - 1;
        if (loadK < numK) {
            int wstage = loadK % STAGES;
            load_A(wstage, loadK * BK);
            load_B(wstage, loadK * BK);
        }
        cp_commit();

        // mma
#pragma unroll
        for (int kk = 0; kk < MMA_K; kk++) {
#pragma unroll
            for (int mm = 0; mm < MMA_M; mm++) {
#pragma unroll
                for (int nn = 0; nn < MMA_N; nn++) {
                    float* d = acc[mm][nn];
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                        "{%0,%1,%2,%3};\n"
                        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                        : "r"(a[kk][mm][0]), "r"(a[kk][mm][1]),
                          "r"(a[kk][mm][2]), "r"(a[kk][mm][3]),
                          "r"(b[kk][nn][0]), "r"(b[kk][nn][1]));
                }
            }
        }
        __syncthreads();
    }

    // ---- epilogue: scattered store ----
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
            __half2 lo = __floats2half2_rn(d[0], d[1]);
            __half2 hi = __floats2half2_rn(d[2], d[3]);
            *reinterpret_cast<__half2*>(&C[(size_t) r * N + c]) = lo;
            *reinterpret_cast<__half2*>(&C[(size_t)(r + 8) * N + c]) = hi;
        }
    }
}
}  // namespace v2impl

PLAYGROUND_MATMUL_DEC(float16_t, 2, m, n, k, A, B, C)
{
    using namespace v2impl;
    static bool configured = false;
    size_t smemBytes = STAGES * (ASTAGE + BSTAGE) * sizeof(half);
    if (!configured) {
        cudaFuncSetAttribute(kernel_v2,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int) smemBytes);
        configured = true;
    }
    dim3 block(NTHREADS);
    dim3 grid(n / BN, m / BM);
    kernel_v2<<<grid, block, smemBytes>>>((int) m, (int) n, (int) k, A, B, C);
}

}  // namespace playground
