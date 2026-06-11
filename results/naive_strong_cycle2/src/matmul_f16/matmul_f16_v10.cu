// Experiment: force interleaved ldmatrix<->mma issue order via *volatile* asm
// (the compiler batches them otherwise). 128x128 tile has register room so the
// interleave should not spill. Tests whether filling the ldmatrix-burst tensor
// idle window beats v6's 147 TFLOPS despite lower per-warp ILP.
#include <cstdlib>
#include <cuda_fp16.h>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{
namespace v10
{
constexpr int BM = 256;
constexpr int BN = 128;
constexpr int BK = 64;
constexpr int NSTAGE = 3;
constexpr int WARPS_M = 4;
constexpr int WARPS_N = 2;
constexpr int WM = BM / WARPS_M;  // 64
constexpr int WN = BN / WARPS_N;  // 64
constexpr int MTILES = WM / 16;   // 4
constexpr int NTILES = WN / 8;    // 8
constexpr int KSTEPS = BK / 16;   // 4
constexpr int THREADS = WARPS_M * WARPS_N * 32;
constexpr int LDA = BK + 8;
constexpr int LDB = BN + 8;
constexpr int STILE_A = BM * LDA;
constexpr int STILE_B = BK * LDB;
constexpr int A_F4 = BM * BK / 8;
constexpr int B_F4 = BK * BN / 8;
constexpr int SMEM_HALFS = NSTAGE * (STILE_A + STILE_B);

__device__ __forceinline__ unsigned smem_u32(const void* p)
{
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async_cg16(void* dst, const void* src)
{
    unsigned s = smem_u32(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s),
                 "l"(src));
}
__device__ __forceinline__ void cp_async_commit()
{
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void cp_async_wait()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
// volatile ldmatrix / mma to force interleaved issue order.
__device__ __forceinline__ void ldm_x4(uint32_t r[4], const void* p)
{
    unsigned a = smem_u32(p);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(a));
}
__device__ __forceinline__ void ldm_x2_trans(uint32_t r[2], const void* p)
{
    unsigned a = smem_u32(p);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1])
        : "r"(a));
}
__device__ __forceinline__ void mma(float c[4], const uint32_t a[4],
                                    const uint32_t b[2])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__global__ __launch_bounds__(THREADS, 1) void kernel(int m, int n, int k,
                                                     const half* __restrict__ A,
                                                     const half* __restrict__ B,
                                                     half* __restrict__ C)
{
    extern __shared__ half smem[];
    half* As = smem;
    half* Bs = smem + NSTAGE * STILE_A;

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int warpId = tid / 32;
    const int lane = tid % 32;
    const int warpRow = warpId / WARPS_N;
    const int warpCol = warpId % WARPS_N;

    float acc[MTILES][NTILES][4];
#pragma unroll
    for (int i = 0; i < MTILES; ++i)
#pragma unroll
        for (int j = 0; j < NTILES; ++j)
#pragma unroll
            for (int t = 0; t < 4; ++t)
                acc[i][j][t] = 0.0f;

    const int numK = k / BK;

    auto loadStage = [&](int stage, int k0) {
        half* Ad = As + stage * STILE_A;
        half* Bd = Bs + stage * STILE_B;
#pragma unroll
        for (int idx = tid; idx < A_F4; idx += THREADS) {
            int r = idx / (BK / 8);
            int c8 = (idx % (BK / 8)) * 8;
            cp_async_cg16(&Ad[r * LDA + c8],
                          A + (size_t(blockRow + r) * k) + (k0 + c8));
        }
#pragma unroll
        for (int idx = tid; idx < B_F4; idx += THREADS) {
            int r = idx / (BN / 8);
            int c8 = (idx % (BN / 8)) * 8;
            cp_async_cg16(&Bd[r * LDB + c8],
                          B + (size_t(k0 + r) * n) + (blockCol + c8));
        }
        cp_async_commit();
    };

    // Prefetch one k-step's fragments into (na,nb), reading directly into the
    // register arrays so they stay in registers (compile-time c after unroll).
    auto ldA = [&](int stage, int kk, int i, uint32_t na[MTILES][4]) {
        half* Ar = As + stage * STILE_A;
        ldm_x4(na[i], &Ar[(warpRow * WM + i * 16 + (lane % 16)) * LDA + kk +
                          (lane / 16) * 8]);
    };
    auto ldB = [&](int stage, int kk, int j, uint32_t nb[NTILES][2]) {
        half* Br = Bs + stage * STILE_B;
        ldm_x2_trans(nb[j], &Br[(kk + (lane % 16)) * LDB + warpCol * WN + j * 8]);
    };
    auto loadAll = [&](int stage, int kk, uint32_t na[MTILES][4],
                       uint32_t nb[NTILES][2]) {
#pragma unroll
        for (int i = 0; i < MTILES; ++i)
            ldA(stage, kk, i, na);
#pragma unroll
        for (int j = 0; j < NTILES; ++j)
            ldB(stage, kk, j, nb);
    };

#pragma unroll
    for (int s = 0; s < NSTAGE - 1; ++s)
        loadStage(s, s * BK);
    cp_async_wait<NSTAGE - 2>();
    __syncthreads();

    int writeStage = NSTAGE - 1;
    int readStage = 0;

    uint32_t aF[2][MTILES][4];
    uint32_t bF[2][NTILES][2];
    loadAll(readStage, 0, aF[0], bF[0]);

    for (int kt = 0; kt < numK; ++kt) {
#pragma unroll
        for (int ks = 0; ks < KSTEPS; ++ks) {
            int c = ks & 1;
            bool boundary = (ks + 1 == KSTEPS);
            int pstage = readStage, pkk = (ks + 1) * 16;
            bool pref = true;
            if (boundary) {
                int loadKt = kt + (NSTAGE - 1);
                int safeKt = loadKt < numK ? loadKt : (numK - 1);
                loadStage(writeStage, safeKt * BK);
                if (++writeStage == NSTAGE)
                    writeStage = 0;
                if (kt + 1 < numK) {
                    cp_async_wait<NSTAGE - 2>();
                    __syncthreads();
                    if (++readStage == NSTAGE)
                        readStage = 0;
                    pstage = readStage;
                    pkk = 0;
                } else {
                    pref = false;
                }
            }
            // Interleave: per m-tile row, prefetch A[i] and (NTILES/MTILES)
            // B-tiles for the next k-step, spread across this row's mma.
            // Forced order via volatile asm so the tensor pipe stays fed.
            constexpr int BPR = NTILES / MTILES;  // B-loads per row
            constexpr int SPREAD = NTILES / (BPR > 0 ? BPR : 1);  // mma gap
#pragma unroll
            for (int i = 0; i < MTILES; ++i) {
                if (pref)
                    ldA(pstage, pkk, i, aF[c ^ 1]);
#pragma unroll
                for (int j = 0; j < NTILES; ++j) {
                    if (pref && (j % SPREAD) == 0) {
                        int bidx = i * BPR + j / SPREAD;
                        if (bidx < NTILES)
                            ldB(pstage, pkk, bidx, bF[c ^ 1]);
                    }
                    mma(acc[i][j], aF[c][i], bF[c][j]);
                }
            }
        }
    }

    const int groupID = lane >> 2;
    const int tg = lane & 3;
#pragma unroll
    for (int i = 0; i < MTILES; ++i)
#pragma unroll
        for (int j = 0; j < NTILES; ++j) {
            int row0 = blockRow + warpRow * WM + i * 16 + groupID;
            int col = blockCol + warpCol * WN + j * 8 + tg * 2;
            *reinterpret_cast<__half2*>(&C[size_t(row0) * n + col]) =
                __floats2half2_rn(acc[i][j][0], acc[i][j][1]);
            *reinterpret_cast<__half2*>(&C[size_t(row0 + 8) * n + col]) =
                __floats2half2_rn(acc[i][j][2], acc[i][j][3]);
        }
}
}  // namespace v10

PLAYGROUND_MATMUL_DEC(float16_t, 10, m, n, k, A, B, C)
{
    static bool inited = false;
    int smemBytes = v10::SMEM_HALFS * int(sizeof(half));
    if (!inited) {
        cudaFuncSetAttribute(v10::kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smemBytes);
        cudaFuncSetAttribute(v10::kernel,
                             cudaFuncAttributePreferredSharedMemoryCarveout,
                             cudaSharedmemCarveoutMaxShared);
        inited = true;
    }
    dim3 block(v10::THREADS);
    dim3 grid(n / v10::BN, m / v10::BM);
    v10::kernel<<<grid, block, smemBytes>>>(int(m), int(n), int(k), A, B, C);
}

}  // namespace playground
