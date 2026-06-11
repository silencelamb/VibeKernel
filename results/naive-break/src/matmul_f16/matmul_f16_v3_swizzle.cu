// fp16 GEMM v3 — mma.sync + cp.async multi-stage pipeline + XOR-swizzled shared
// memory (conflict-free ldmatrix). Row-major A(m,k)*B(k,n)=C(m,n), f32 accum.
#include <cuda_fp16.h>
#include <cstdint>

#include "playground/matmul.hpp"

namespace playground
{
namespace v3
{

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 64;
constexpr int STAGES = 3;
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 4;
constexpr int NWARPS = WARPS_M * WARPS_N;  // 8
constexpr int NTHREADS = NWARPS * 32;      // 256

constexpr int WM = BM / WARPS_M;  // 64
constexpr int WN = BN / WARPS_N;  // 32
constexpr int MITER = WM / 16;    // 4
constexpr int NITER = WN / 8;     // 4
constexpr int KITER = BK / 16;    // 4

constexpr int A_ATOMS = BK / 8;   // atoms (8-half=16B) per A row  = 8
constexpr int B_ATOMS = BN / 8;   // per B row = 16
constexpr int A_SZ = BM * BK;     // halves per A stage
constexpr int B_SZ = BK * BN;     // halves per B stage

// XOR swizzle on 16B atoms: keeps 8 rows touched by ldmatrix on distinct banks.
template <int CW>
__device__ __forceinline__ int swz(int row, int col)
{
    int atom = col >> 3;
    int phys = atom ^ (row & 7);
    return row * CW + (phys << 3) + (col & 7);
}

__device__ __forceinline__ uint32_t smem_u32(const void* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(
                     smem_u32(smem)),
                 "l"(gmem));
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
__device__ __forceinline__ void ldm_x4(uint32_t (&r)[4], const void* s)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(smem_u32(s)));
}
__device__ __forceinline__ void ldm_x4_t(uint32_t (&r)[4], const void* s)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(smem_u32(s)));
}
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4],
                                    const uint32_t (&b)[2])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__global__ void __launch_bounds__(NTHREADS)
    gemm(const half* __restrict__ A, const half* __restrict__ B,
         half* __restrict__ C, int M, int N, int K)
{
    extern __shared__ half smem[];
    half* sA = smem;                       // STAGES * A_SZ
    half* sB = smem + STAGES * A_SZ;       // STAGES * B_SZ

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int warpId = tid >> 5;
    const int lane = tid & 31;
    const int warpRow = warpId / WARPS_N;
    const int warpCol = warpId % WARPS_N;

    float acc[MITER][NITER][4];
#pragma unroll
    for (int i = 0; i < MITER; ++i)
#pragma unroll
        for (int j = 0; j < NITER; ++j)
#pragma unroll
            for (int t = 0; t < 4; ++t) acc[i][j][t] = 0.0f;

    auto loadStage = [&](int stage, int k0) {
        half* dA = sA + stage * A_SZ;
        half* dB = sB + stage * B_SZ;
        // A: BM*BK/8 atoms over NTHREADS
#pragma unroll
        for (int i = 0; i < (A_SZ / 8) / NTHREADS; ++i) {
            int a = tid + i * NTHREADS;
            int row = a / A_ATOMS;
            int col = (a % A_ATOMS) * 8;
            cp_async16(dA + swz<BK>(row, col),
                       &A[(blockRow + row) * K + k0 + col]);
        }
        // B: BK*BN/8 atoms
#pragma unroll
        for (int i = 0; i < (B_SZ / 8) / NTHREADS; ++i) {
            int a = tid + i * NTHREADS;
            int row = a / B_ATOMS;
            int col = (a % B_ATOMS) * 8;
            cp_async16(dB + swz<BN>(row, col),
                       &B[(k0 + row) * N + blockCol + col]);
        }
    };

    const int numK = K / BK;
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        loadStage(s, s * BK);
        cp_commit();
    }
    cp_wait<STAGES - 2>();
    __syncthreads();

    int cur = 0;
    for (int kt = 0; kt < numK; ++kt) {
        const int kpre = kt + (STAGES - 1);
        const int wstage = kpre % STAGES;
        if (kpre < numK) loadStage(wstage, kpre * BK);
        cp_commit();

        half* rA = sA + cur * A_SZ;
        half* rB = sB + cur * B_SZ;
#pragma unroll
        for (int ks = 0; ks < KITER; ++ks) {
            uint32_t aF[MITER][4];
#pragma unroll
            for (int mi = 0; mi < MITER; ++mi) {
                int row = warpRow * WM + mi * 16 + (lane & 15);
                int col = ks * 16 + (lane >> 4) * 8;
                ldm_x4(aF[mi], rA + swz<BK>(row, col));
            }
            uint32_t bF[NITER][2];
#pragma unroll
            for (int nj = 0; nj < NITER / 2; ++nj) {
                uint32_t tmp[4];
                int row = ks * 16 + (lane & 15);
                int col = warpCol * WN + nj * 16 + (lane >> 4) * 8;
                ldm_x4_t(tmp, rB + swz<BN>(row, col));
                bF[nj * 2 + 0][0] = tmp[0];
                bF[nj * 2 + 0][1] = tmp[1];
                bF[nj * 2 + 1][0] = tmp[2];
                bF[nj * 2 + 1][1] = tmp[3];
            }
#pragma unroll
            for (int mi = 0; mi < MITER; ++mi)
#pragma unroll
                for (int ni = 0; ni < NITER; ++ni)
                    mma(acc[mi][ni], aF[mi], bF[ni]);
        }

        cp_wait<STAGES - 2>();
        __syncthreads();
        cur = (cur + 1) % STAGES;
    }

    const int gid = lane >> 2;
    const int tib = lane & 3;
#pragma unroll
    for (int mi = 0; mi < MITER; ++mi)
#pragma unroll
        for (int ni = 0; ni < NITER; ++ni) {
            int row = blockRow + warpRow * WM + mi * 16;
            int col = blockCol + warpCol * WN + ni * 8 + tib * 2;
            *(half2*) &C[(row + gid) * N + col] =
                __floats2half2_rn(acc[mi][ni][0], acc[mi][ni][1]);
            *(half2*) &C[(row + gid + 8) * N + col] =
                __floats2half2_rn(acc[mi][ni][2], acc[mi][ni][3]);
        }
}
}  // namespace v3

PLAYGROUND_MATMUL_DEC(float16_t, 3, m, n, k, A, B, C)
{
    static int configured = [] {
        int smemBytes =
            v3::STAGES * (v3::A_SZ + v3::B_SZ) * int(sizeof(half));
        cudaFuncSetAttribute(v3::gemm,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smemBytes);
        return smemBytes;
    }();
    int smemBytes = configured;
    dim3 block(v3::NTHREADS);
    dim3 grid(n / v3::BN, m / v3::BM);
    v3::gemm<<<grid, block, smemBytes>>>(A, B, C, int(m), int(n), int(k));
}

}  // namespace playground
