#include <cuda_fp16.h>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{
namespace v3
{
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int NSTAGE = 3;

constexpr int WARPS_M = 2;
constexpr int WARPS_N = 4;
constexpr int WM = BM / WARPS_M;  // 64
constexpr int WN = BN / WARPS_N;  // 32

constexpr int MTILES = WM / 16;   // 4  (mma m16)
constexpr int NTILES = WN / 8;    // 4  (mma n8)

constexpr int THREADS = WARPS_M * WARPS_N * 32;  // 256

// Padded leading dims to mitigate shared bank conflicts (8 halfs = 16B).
constexpr int LDA = BK + 8;   // 40
constexpr int LDB = BN + 8;   // 136

constexpr int STILE_A = BM * LDA;
constexpr int STILE_B = BK * LDB;
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
__device__ __forceinline__ void mma_m16n8k16(float c[4], const uint32_t a[4],
                                             const uint32_t b[2])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__global__ __launch_bounds__(THREADS) void kernel(int m, int n, int k,
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
        for (int p = 0; p < 2; ++p) {
            int e = p * THREADS + tid;     // 0..511
            int r = e / (BK / 8);          // 0..127
            int c8 = (e % (BK / 8)) * 8;   // 0,8,16,24
            const half* gptr = A + (size_t(blockRow + r) * k) + (k0 + c8);
            cp_async_cg16(&Ad[r * LDA + c8], gptr);
        }
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            int e = p * THREADS + tid;     // 0..511
            int r = e / (BN / 8);          // 0..31
            int c8 = (e % (BN / 8)) * 8;   // 0..120
            const half* gptr = B + (size_t(k0 + r) * n) + (blockCol + c8);
            cp_async_cg16(&Bd[r * LDB + c8], gptr);
        }
        cp_async_commit();
    };

#pragma unroll
    for (int s = 0; s < NSTAGE - 1; ++s)
        loadStage(s, s * BK);

    int writeStage = NSTAGE - 1;
    int readStage = 0;

    for (int kt = 0; kt < numK; ++kt) {
        int loadKt = kt + (NSTAGE - 1);
        if (loadKt < numK)
            loadStage(writeStage, loadKt * BK);

        cp_async_wait<NSTAGE - 2>();
        __syncthreads();

        half* Ar = As + readStage * STILE_A;
        half* Br = Bs + readStage * STILE_B;

#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            uint32_t aF[MTILES][4];
            uint32_t bF[NTILES][2];
#pragma unroll
            for (int i = 0; i < MTILES; ++i) {
                int aRow = warpRow * WM + i * 16 + (lane % 16);
                int aCol = kk + (lane / 16) * 8;
                ldm_x4(aF[i], &Ar[aRow * LDA + aCol]);
            }
#pragma unroll
            for (int j = 0; j < NTILES; ++j) {
                int bRow = kk + (lane % 16);
                int bCol = warpCol * WN + j * 8;
                ldm_x2_trans(bF[j], &Br[bRow * LDB + bCol]);
            }
#pragma unroll
            for (int i = 0; i < MTILES; ++i)
#pragma unroll
                for (int j = 0; j < NTILES; ++j)
                    mma_m16n8k16(acc[i][j], aF[i], bF[j]);
        }
        __syncthreads();

        writeStage = (writeStage + 1) % NSTAGE;
        readStage = (readStage + 1) % NSTAGE;
    }

    // Store back: mma m16n8 C layout.
    const int groupID = lane >> 2;
    const int tg = lane & 3;
#pragma unroll
    for (int i = 0; i < MTILES; ++i) {
#pragma unroll
        for (int j = 0; j < NTILES; ++j) {
            int baseRow = blockRow + warpRow * WM + i * 16;
            int baseCol = blockCol + warpCol * WN + j * 8;
            int row0 = baseRow + groupID;
            int row1 = baseRow + groupID + 8;
            int col = baseCol + tg * 2;
            __half2 v0 =
                __floats2half2_rn(acc[i][j][0], acc[i][j][1]);
            __half2 v1 =
                __floats2half2_rn(acc[i][j][2], acc[i][j][3]);
            *reinterpret_cast<__half2*>(&C[size_t(row0) * n + col]) = v0;
            *reinterpret_cast<__half2*>(&C[size_t(row1) * n + col]) = v1;
        }
    }
}
}  // namespace v3

PLAYGROUND_MATMUL_DEC(float16_t, 3, m, n, k, A, B, C)
{
    static bool inited = false;
    int smemBytes = v3::SMEM_HALFS * int(sizeof(half));
    if (!inited) {
        cudaFuncSetAttribute(v3::kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smemBytes);
        inited = true;
    }
    dim3 block(v3::THREADS);
    dim3 grid(n / v3::BN, m / v3::BM);
    v3::kernel<<<grid, block, smemBytes>>>(int(m), int(n), int(k), A, B, C);
}

}  // namespace playground
