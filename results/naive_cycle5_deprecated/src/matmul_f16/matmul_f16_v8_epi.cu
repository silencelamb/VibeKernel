#include <cuda_fp16.h>
#include <cstdint>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{
namespace v8
{
// Block / warp tiling
constexpr int BM = 128, BN = 256, BK = 32;
constexpr int WARPS_M = 2, WARPS_N = 4;
constexpr int NWARPS = WARPS_M * WARPS_N;     // 8
constexpr int NTHREADS = NWARPS * 32;          // 256
constexpr int WARP_M = BM / WARPS_M;           // 64
constexpr int WARP_N = BN / WARPS_N;           // 32
constexpr int MFRAG = WARP_M / 16;             // 4  (m16 tiles)
constexpr int NFRAG = WARP_N / 8;              // 4  (n8 tiles)
constexpr int KSTEP = BK / 16;                 // 2  (k16 steps)
constexpr int SKEW = 8;
constexpr int LDA = BK + SKEW;                 // 40
constexpr int LDB = BN + SKEW;                 // 136
constexpr int STAGES = 3;
constexpr int AS = BM * LDA;                    // halfs per A stage
constexpr int BS = BK * LDB;                    // halfs per B stage

__device__ __forceinline__ uint32_t smem_addr(const void* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(void* d, const void* s)
{
    asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n" ::"r"(smem_addr(d)),
                 "l"(s)
                 : "memory");
}
__device__ __forceinline__ void cp_commit()
{
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}
template <int N>
__device__ __forceinline__ void cp_wait()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}
__device__ __forceinline__ void ldm_x4(uint32_t (&r)[4], uint32_t a)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(a)
        : "memory");
}
__device__ __forceinline__ void ldm_x4_trans(uint32_t (&r)[4], uint32_t a)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(a)
        : "memory");
}
__device__ __forceinline__ void mma16816(uint32_t (&d)[2],
                                         const uint32_t (&a)[4],
                                         const uint32_t (&b)[2])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(d[0]), "+r"(d[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__global__ void __launch_bounds__(NTHREADS, 2)
    kernel(int M, int N, int K, const half* __restrict__ A,
           const half* __restrict__ B, half* __restrict__ C)
{
    extern __shared__ half smem[];
    half* As = smem;
    half* Bs = smem + STAGES * AS;

    const int blockRow = blockIdx.y * BM, blockCol = blockIdx.x * BN;
    const int warpId = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warpRow = (warpId / WARPS_N) * WARP_M;
    const int warpCol = (warpId % WARPS_N) * WARP_N;
    const int tid = threadIdx.x;
    const int numK = K / BK;

    auto load_tile = [&](int kt, int st) {
        int k0 = kt * BK;
        half* Asb = As + st * AS;
        half* Bsb = Bs + st * BS;
#pragma unroll
        for (int i = 0; i < (BM * BK / 8) / NTHREADS; ++i) {
            int idx = tid + i * NTHREADS;
            int row = idx / (BK / 8);
            int col8 = (idx % (BK / 8)) * 8;
            cp_async16(&Asb[row * LDA + col8],
                       &A[(blockRow + row) * K + k0 + col8]);
        }
#pragma unroll
        for (int i = 0; i < (BK * BN / 8) / NTHREADS; ++i) {
            int idx = tid + i * NTHREADS;
            int row = idx / (BN / 8);
            int col8 = (idx % (BN / 8)) * 8;
            cp_async16(&Bsb[row * LDB + col8],
                       &B[(k0 + row) * N + blockCol + col8]);
        }
        cp_commit();
    };

    uint32_t acc[MFRAG][NFRAG][2];
#pragma unroll
    for (int mi = 0; mi < MFRAG; ++mi)
#pragma unroll
        for (int ni = 0; ni < NFRAG; ++ni)
#pragma unroll
            for (int t = 0; t < 2; ++t) acc[mi][ni][t] = 0u;

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < numK)
            load_tile(s, s);
        else
            cp_commit();
    }

    for (int kt = 0; kt < numK; ++kt) {
        cp_wait<STAGES - 2>();
        __syncthreads();

        int cur = kt % STAGES;
        half* Asb = As + cur * AS;
        half* Bsb = Bs + cur * BS;

        // Register double-buffered fragments: prefetch next k-step's
        // ldmatrix while computing current k-step's mma (hides smem latency).
        uint32_t aReg[2][MFRAG][4];
        uint32_t bReg[2][NFRAG][2];

        auto load_frags = [&](int buf, int kk) {
#pragma unroll
            for (int mi = 0; mi < MFRAG; ++mi) {
                int r = warpRow + mi * 16 + (lane % 16);
                int c = kk + (lane / 16) * 8;
                ldm_x4(aReg[buf][mi], smem_addr(&Asb[r * LDA + c]));
            }
#pragma unroll
            for (int n2 = 0; n2 < NFRAG; n2 += 2) {
                int r = kk + (lane % 16);
                int c = warpCol + n2 * 8 + (lane / 16) * 8;
                uint32_t t[4];
                ldm_x4_trans(t, smem_addr(&Bsb[r * LDB + c]));
                bReg[buf][n2][0] = t[0];
                bReg[buf][n2][1] = t[1];
                bReg[buf][n2 + 1][0] = t[2];
                bReg[buf][n2 + 1][1] = t[3];
            }
        };

        // Single-barrier multistage with register-DB; cp.async prefetch as a
        // burst AFTER compute into the buffer freed at kt-1 (WAR safe: the
        // start-of-iteration barrier separates it from kt-1's reads).
        const int nxt = kt + STAGES - 1;

        load_frags(0, 0);
#pragma unroll
        for (int ks = 0; ks < KSTEP; ++ks) {
            int cb = ks & 1;
            if (ks + 1 < KSTEP) load_frags((ks + 1) & 1, (ks + 1) * 16);
#pragma unroll
            for (int mi = 0; mi < MFRAG; ++mi)
#pragma unroll
                for (int ni = 0; ni < NFRAG; ++ni)
                    mma16816(acc[mi][ni], aReg[cb][mi], bReg[cb][ni]);
        }
        if (nxt < numK)
            load_tile(nxt, nxt % STAGES);
        else
            cp_commit();
    }

    // Epilogue: stage the C tile through shared (reuse As/Bs smem, now free)
    // then write to global fully coalesced (128-byte) instead of scattered 4B.
    constexpr int LDC = BN + 8;  // pad -> conflict-free shared C stage
    half* Cs = smem;
    const int groupID = lane >> 2;
    const int tidg = lane & 3;
    __syncthreads();  // no warp still reading As/Bs from the last k-tile
#pragma unroll
    for (int mi = 0; mi < MFRAG; ++mi) {
#pragma unroll
        for (int ni = 0; ni < NFRAG; ++ni) {
            int r = warpRow + mi * 16 + groupID;
            int c = warpCol + ni * 8 + tidg * 2;
            *reinterpret_cast<uint32_t*>(&Cs[r * LDC + c]) = acc[mi][ni][0];
            *reinterpret_cast<uint32_t*>(&Cs[(r + 8) * LDC + c]) =
                acc[mi][ni][1];
        }
    }
    __syncthreads();
    // Coalesced copy Cs[BM][BN] -> C, 8 halfs (float4) per access.
#pragma unroll
    for (int i = 0; i < (BM * BN / 8) / NTHREADS; ++i) {
        int idx = tid + i * NTHREADS;
        int row = idx / (BN / 8);
        int col8 = (idx % (BN / 8)) * 8;
        *reinterpret_cast<float4*>(&C[(blockRow + row) * N + blockCol + col8]) =
            *reinterpret_cast<float4*>(&Cs[row * LDC + col8]);
    }
}

}  // namespace v8

PLAYGROUND_MATMUL_DEC(float16_t, 8, m, n, k, A, B, C)
{
    constexpr int smemBytes =
        v8::STAGES * (v8::AS + v8::BS) * int(sizeof(half));
    static bool attrSet = false;
    if (!attrSet) {
        cudaFuncSetAttribute(v8::kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smemBytes);
        cudaFuncSetAttribute(v8::kernel,
                             cudaFuncAttributePreferredSharedMemoryCarveout,
                             cudaSharedmemCarveoutMaxShared);
        attrSet = true;
    }
    dim3 block(v8::NTHREADS);
    dim3 grid(n / v8::BN, m / v8::BM);
    v8::kernel<<<grid, block, smemBytes>>>(int(m), int(n), int(k), A, B, C);
}

}  // namespace playground
