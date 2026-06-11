#include <cuda_fp16.h>
#include <cuda_pipeline.h>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{
namespace v5
{
// Block tile 128 x 128, 4 warps (2x2), each warp 64x64
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 2;
constexpr int WM = BM / WARPS_M;  // 64
constexpr int WN = BN / WARPS_N;  // 64
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;
constexpr int FM = WM / MMA_M;       // 4
constexpr int FN = WN / MMA_N;       // 8
constexpr int FN16 = WN / 16;        // 4 (number of x4 B loads)

constexpr int NTHREADS = WARPS_M * WARPS_N * 32;  // 128
constexpr int PAD = 8;
constexpr int LDA_S = BK + PAD;   // 40
constexpr int LDB_S = BN + PAD;   // 136
constexpr int NSTAGES = 2;

constexpr int AS_SZ = BM * LDA_S;
constexpr int BS_SZ = BK * LDB_S;
constexpr int SMEM_BYTES = NSTAGES * (AS_SZ + BS_SZ) * (int) sizeof(half);

__device__ __forceinline__ void cp_async16(void* dst, const void* src)
{
    unsigned smem = __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem),
                 "l"(src));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const half* addr)
{
    unsigned s = __cvta_generic_to_shared(addr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(s));
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4],
                                                  const half* addr)
{
    unsigned s = __cvta_generic_to_shared(addr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(s));
}

__device__ __forceinline__ void mma_f16(uint32_t (&d)[2], const uint32_t (&a)[4],
                                        const uint32_t (&b)[2])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(d[0]), "+r"(d[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__global__ void __launch_bounds__(NTHREADS, 4)
    matmul_kernel(int M, int N, int K, const half* __restrict__ A,
                  const half* __restrict__ B, half* __restrict__ C)
{
    extern __shared__ half smem[];
    half* As = smem;
    half* Bs = smem + NSTAGES * AS_SZ;

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    const int warpId = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int warpRow = (warpId / WARPS_N) * WM;
    const int warpCol = (warpId % WARPS_N) * WN;

    uint32_t acc[FM][FN][2];
#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j) {
            acc[i][j][0] = 0;
            acc[i][j][1] = 0;
        }

    const int tid = threadIdx.x;
    const int a_row0 = tid / (BK / 8);
    const int a_col0 = (tid % (BK / 8)) * 8;
    constexpr int A_ROW_STRIDE = NTHREADS / (BK / 8);  // 32
    const int b_row0 = tid / (BN / 8);
    const int b_col0 = (tid % (BN / 8)) * 8;
    constexpr int B_ROW_STRIDE = NTHREADS / (BN / 8);  // 8

    const int nKTiles = K / BK;

    auto load_tile = [&](int kt, int stage) {
        half* Asd = As + stage * AS_SZ;
        half* Bsd = Bs + stage * BS_SZ;
#pragma unroll
        for (int r = 0; r < BM; r += A_ROW_STRIDE) {
            int gr = blockRow + a_row0 + r;
            int gc = kt + a_col0;
            cp_async16(&Asd[(a_row0 + r) * LDA_S + a_col0], &A[gr * K + gc]);
        }
#pragma unroll
        for (int r = 0; r < BK; r += B_ROW_STRIDE) {
            int gr = kt + b_row0 + r;
            int gc = blockCol + b_col0;
            cp_async16(&Bsd[(b_row0 + r) * LDB_S + b_col0], &B[gr * N + gc]);
        }
        __pipeline_commit();
    };

#pragma unroll
    for (int s = 0; s < NSTAGES - 1; ++s) load_tile(s * BK, s);

    int writeStage = NSTAGES - 1;
    int readStage = 0;

    for (int kt = 0; kt < nKTiles; ++kt) {
        int loadKt = kt + (NSTAGES - 1);
        if (loadKt < nKTiles)
            load_tile(loadKt * BK, writeStage);
        else
            __pipeline_commit();
        __pipeline_wait_prior(NSTAGES - 1);
        __syncthreads();

        half* Asd = As + readStage * AS_SZ;
        half* Bsd = Bs + readStage * BS_SZ;
#pragma unroll
        for (int kk = 0; kk < BK; kk += MMA_K) {
            uint32_t a_frag[FM][4];
            uint32_t b_frag[FN][2];
#pragma unroll
            for (int i = 0; i < FM; ++i) {
                const half* addr =
                    &Asd[(warpRow + i * MMA_M + (lane % 16)) * LDA_S + kk +
                         (lane / 16) * 8];
                ldmatrix_x4(a_frag[i], addr);
            }
#pragma unroll
            for (int jj = 0; jj < FN16; ++jj) {
                uint32_t tmp[4];
                const half* addr =
                    &Bsd[(kk + (lane % 16)) * LDB_S + warpCol + jj * 16 +
                         (lane / 16) * 8];
                ldmatrix_x4_trans(tmp, addr);
                b_frag[jj * 2][0] = tmp[0];
                b_frag[jj * 2][1] = tmp[1];
                b_frag[jj * 2 + 1][0] = tmp[2];
                b_frag[jj * 2 + 1][1] = tmp[3];
            }
#pragma unroll
            for (int i = 0; i < FM; ++i)
#pragma unroll
                for (int j = 0; j < FN; ++j)
                    mma_f16(acc[i][j], a_frag[i], b_frag[j]);
        }
        __syncthreads();

        writeStage = (writeStage + 1) % NSTAGES;
        readStage = (readStage + 1) % NSTAGES;
    }

#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j) {
            int cr = blockRow + warpRow + i * MMA_M;
            int cc = blockCol + warpCol + j * MMA_N;
            int row = lane / 4;
            int col = (lane % 4) * 2;
            *reinterpret_cast<uint32_t*>(&C[(cr + row) * N + cc + col]) =
                acc[i][j][0];
            *reinterpret_cast<uint32_t*>(&C[(cr + row + 8) * N + cc + col]) =
                acc[i][j][1];
        }
}

}  // namespace v5

PLAYGROUND_MATMUL_DEC(float16_t, 5, m, n, k, A, B, C)
{
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(v5::matmul_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             v5::SMEM_BYTES);
        configured = true;
    }
    dim3 block(v5::NTHREADS);
    dim3 grid((n + v5::BN - 1) / v5::BN, (m + v5::BM - 1) / v5::BM);
    v5::matmul_kernel<<<grid, block, v5::SMEM_BYTES>>>(int(m), int(n), int(k),
                                                       A, B, C);
}

}  // namespace playground
