// fp16 Tensor-Core GEMM for A100 (sm_80). Row-major A(m,k), B(k,n), C(m,n).
//
// Design (best config found by sweeping v5):
//   - Threadblock tile 256 x 128, BK = 64.  8 warps (4 warp-rows x 2 warp-cols).
//   - Warp tile 64 x 64 -> 4 x 8 = 32 mma.m16n8k16 tiles, fp32 accumulate.
//   - cp.async 3-stage global->shared pipeline (clamp-always-issue tail so the
//     tile being read each iteration is always a completed cp.async group).
//   - Continuous cross-tile register-prefetch: the next k-step's fragments
//     (and, at a tile boundary, the next tile's first fragments) are loaded via
//     ldmatrix while the tensor cores chew the current ones, so mma flows
//     unbroken across the single per-tile __syncthreads.
//   - Padded shared leading dims (conflict-free ldmatrix; measured 0 bank
//     conflicts).
#include <cuda_fp16.h>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{
namespace v6
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
constexpr int THREADS = WARPS_M * WARPS_N * 32;  // 256

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
__device__ __forceinline__ void ldm_x4(uint32_t r[4], const void* p)
{
    unsigned a = smem_u32(p);
    asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(a));
}
__device__ __forceinline__ void ldm_x2_trans(uint32_t r[2], const void* p)
{
    unsigned a = smem_u32(p);
    asm("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1])
        : "r"(a));
}
__device__ __forceinline__ void mma_m16n8k16(float c[4], const uint32_t a[4],
                                             const uint32_t b[2])
{
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
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

    auto loadFrags = [&](int stage, int kk, uint32_t aF[MTILES][4],
                         uint32_t bF[NTILES][2]) {
        half* Ar = As + stage * STILE_A;
        half* Br = Bs + stage * STILE_B;
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
    loadFrags(readStage, 0, aF[0], bF[0]);

    for (int kt = 0; kt < numK; ++kt) {
#pragma unroll
        for (int ks = 0; ks < KSTEPS; ++ks) {
            int c = ks & 1;
            if (ks + 1 < KSTEPS) {
                loadFrags(readStage, (ks + 1) * 16, aF[c ^ 1], bF[c ^ 1]);
            } else {
                int loadKt = kt + (NSTAGE - 1);
                int safeKt = loadKt < numK ? loadKt : (numK - 1);
                loadStage(writeStage, safeKt * BK);
                writeStage = (writeStage + 1) % NSTAGE;
                if (kt + 1 < numK) {
                    cp_async_wait<NSTAGE - 2>();
                    __syncthreads();
                    readStage = (readStage + 1) % NSTAGE;
                    loadFrags(readStage, 0, aF[c ^ 1], bF[c ^ 1]);
                }
            }
#pragma unroll
            for (int i = 0; i < MTILES; ++i)
#pragma unroll
                for (int j = 0; j < NTILES; ++j)
                    mma_m16n8k16(acc[i][j], aF[c][i], bF[c][j]);
        }
    }

    const int groupID = lane >> 2;
    const int tg = lane & 3;
#pragma unroll
    for (int i = 0; i < MTILES; ++i) {
#pragma unroll
        for (int j = 0; j < NTILES; ++j) {
            int row0 = blockRow + warpRow * WM + i * 16 + groupID;
            int row1 = row0 + 8;
            int col = blockCol + warpCol * WN + j * 8 + tg * 2;
            *reinterpret_cast<__half2*>(&C[size_t(row0) * n + col]) =
                __floats2half2_rn(acc[i][j][0], acc[i][j][1]);
            *reinterpret_cast<__half2*>(&C[size_t(row1) * n + col]) =
                __floats2half2_rn(acc[i][j][2], acc[i][j][3]);
        }
    }
}
}  // namespace v6

PLAYGROUND_MATMUL_DEC(float16_t, 6, m, n, k, A, B, C)
{
    static bool inited = false;
    int smemBytes = v6::SMEM_HALFS * int(sizeof(half));
    if (!inited) {
        cudaFuncSetAttribute(v6::kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smemBytes);
        cudaFuncSetAttribute(v6::kernel,
                             cudaFuncAttributePreferredSharedMemoryCarveout,
                             cudaSharedmemCarveoutMaxShared);
        inited = true;
    }
    dim3 block(v6::THREADS);
    dim3 grid(n / v6::BN, m / v6::BM);
    v6::kernel<<<grid, block, smemBytes>>>(int(m), int(n), int(k), A, B, C);
}

}  // namespace playground
