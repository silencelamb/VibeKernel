#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "playground/matmul.hpp"

namespace playground
{
namespace v10impl
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
constexpr int STAGES = 6;

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
    kernel_v10(int M, int N, int K, const half* __restrict__ A,
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
    // Precompute per-warp/per-lane base shared addresses (bytes); the only
    // runtime-varying term in the inner loop is the stage byte offset.
    const uint32_t aBase =
        smem_addr(&As[(warpRow * WM + aLaneRow) * ALEAD + aLaneCol]);
    const uint32_t bBase =
        smem_addr(&Bs[aLaneRow * BLEAD + warpCol * WN + aLaneCol]);
    constexpr uint32_t HB = sizeof(half);  // 2 bytes

    auto ld_A = [&](int stage, int kk, uint32_t af[][4]) {
        uint32_t base = aBase + (uint32_t) stage * (ASTAGE * HB) + kk * (16 * HB);
#pragma unroll
        for (int mm = 0; mm < MMA_M; mm++)
            ldm_x4(base + mm * (16 * ALEAD * HB), af[mm][0], af[mm][1],
                   af[mm][2], af[mm][3]);
    };
    auto ld_B = [&](int stage, int kk, uint32_t bf[][2]) {
        uint32_t base =
            bBase + (uint32_t) stage * (BSTAGE * HB) + kk * (16 * BLEAD * HB);
#pragma unroll
        for (int nn = 0; nn < MMA_N; nn += 2) {
            uint32_t r0, r1, r2, r3;
            ldm_x4_trans(base + nn * (8 * HB), r0, r1, r2, r3);
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
    static_assert(MMA_K == 2, "pair-barrier loop assumes BK=32");
    const int numMacro = numK / 2;  // process 2 K-tiles per barrier

    auto mma_all = [&](uint32_t a[][4], uint32_t b[][2]) {
#pragma unroll
        for (int mm = 0; mm < MMA_M; mm++)
#pragma unroll
            for (int nn = 0; nn < MMA_N; nn++)
                mma_m16n8k16(acc[mm][nn], a[mm], b[nn]);
    };
    auto load_tile = [&](int tile) {
        int stage = tile % STAGES;
        load_A(stage, tile * BK);
        load_B(stage, tile * BK);
    };

    // prologue: load first 3 pairs (6 tiles) -> 3 pair-slots, 1 group per pair
#pragma unroll
    for (int p = 0; p < 3; p++) {
        load_tile(2 * p);
        load_tile(2 * p + 1);
        cp_commit();
    }
    cp_wait<2>();  // pairs 1,2 in flight; pair 0 ready
    __syncthreads();

    // register double buffer of fragments (ping-pong over the 4 substeps)
    uint32_t af[2][MMA_M][4];
    uint32_t bf[2][MMA_N][2];
    ld_A(0, 0, af[0]);
    ld_B(0, 0, bf[0]);

    for (int mi = 0; mi < numMacro; mi++) {
        int s0 = (2 * mi) % STAGES;
        int s1 = (2 * mi + 1) % STAGES;

        // substep 0: (t0,k0) -> prefetch (t0,k1)
        ld_A(s0, 1, af[1]);
        ld_B(s0, 1, bf[1]);
        mma_all(af[0], bf[0]);
        // substep 1: (t0,k1) -> prefetch (t1,k0) [same published pair, no barrier]
        ld_A(s1, 0, af[0]);
        ld_B(s1, 0, bf[0]);
        mma_all(af[1], bf[1]);
        // substep 2: (t1,k0) -> prefetch (t1,k1)
        ld_A(s1, 1, af[1]);
        ld_B(s1, 1, bf[1]);
        mma_all(af[0], bf[0]);
        // substep 3: (t1,k1) -- pair boundary: load pair mi+3, publish pair mi+1
        int lp = mi + 3;
        if (2 * lp < numK) {
            load_tile(2 * lp);
            load_tile(2 * lp + 1);
        }
        cp_commit();
        cp_wait<2>();
        __syncthreads();
        if (mi + 1 < numMacro) {
            int ns0 = (2 * (mi + 1)) % STAGES;
            ld_A(ns0, 0, af[0]);
            ld_B(ns0, 0, bf[0]);
        }
        mma_all(af[1], bf[1]);
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
}  // namespace v10impl

PLAYGROUND_MATMUL_DEC(float16_t, 10, m, n, k, A, B, C)
{
    using namespace v10impl;
    static bool configured = false;
    size_t smemBytes = STAGES * (ASTAGE + BSTAGE) * sizeof(half);
    if (!configured) {
        cudaFuncSetAttribute(kernel_v10,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int) smemBytes);
        // L2 persistence on B (read by every M-block -> high reuse). Pin up to
        // 25MB (A100 max persisting carveout) contiguously at hitRatio=1.
        constexpr size_t kPersist = 25u * 1024u * 1024u;
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, kPersist);
        size_t bBytes = (size_t) n * k * sizeof(half);
        cudaStreamAttrValue av = {};
        av.accessPolicyWindow.base_ptr = const_cast<half*>(B);
        av.accessPolicyWindow.num_bytes = bBytes < kPersist ? bBytes : kPersist;
        av.accessPolicyWindow.hitRatio = 1.0f;
        av.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
        av.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
        cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &av);
        configured = true;
    }
    dim3 block(NTHREADS);
    dim3 grid(n / BN, m / BM);
    kernel_v10<<<grid, block, smemBytes>>>((int) m, (int) n, (int) k, A, B, C);
}

}  // namespace playground
