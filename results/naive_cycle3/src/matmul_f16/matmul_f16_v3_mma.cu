#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{
namespace v3
{
// Block tile 128x128, BK=32, 3-stage cp.async, mma.sync m16n8k16 + ldmatrix.
// 256 threads = 8 warps, warp grid 2(M) x 4(N) -> warp owns 64x32.
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int STAGES = 3;
constexpr int BLOCK_THREADS = 256;
constexpr int WARP_N = 4;
constexpr int WM = 64, WN = 32;
constexpr int WT_M = WM / 16;  // 4 mma tiles along M
constexpr int WT_N = WN / 8;   // 4 mma tiles along N
constexpr int APAD = 8;
constexpr int BPAD = 8;
constexpr int AS_STRIDE = BK + APAD;  // 40
constexpr int BS_STRIDE = BN + BPAD;  // 136
constexpr int AS_TILE = BM * AS_STRIDE;
constexpr int BS_TILE = BK * BS_STRIDE;

__device__ __forceinline__ uint32_t smem_addr(const void* ptr)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                      \
    asm volatile(                                                              \
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"      \
        : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                               \
        : "r"(addr))

#define LDMATRIX_X4_T(R0, R1, R2, R3, addr)                                    \
    asm volatile(                                                              \
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, "       \
        "[%4];\n"                                                              \
        : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                               \
        : "r"(addr))

#define MMA16816(D0, D1, D2, D3, A0, A1, A2, A3, B0, B1)                       \
    asm volatile(                                                              \
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "                   \
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"              \
        : "+f"(D0), "+f"(D1), "+f"(D2), "+f"(D3)                               \
        : "r"(A0), "r"(A1), "r"(A2), "r"(A3), "r"(B0), "r"(B1))

__device__ __forceinline__ void loadTile(const half* __restrict__ A,
                                         const half* __restrict__ B, half* Asb,
                                         half* Bsb, int rowBase, int colBase,
                                         int tileK, int N, int K, int tid)
{
#pragma unroll
    for (int i = tid; i < BM * BK / 8; i += BLOCK_THREADS) {
        int r = (i * 8) / BK;
        int c = (i * 8) % BK;
        __pipeline_memcpy_async(&Asb[r * AS_STRIDE + c],
                                &A[(rowBase + r) * K + tileK + c], 16);
    }
#pragma unroll
    for (int i = tid; i < BK * BN / 8; i += BLOCK_THREADS) {
        int r = (i * 8) / BN;
        int c = (i * 8) % BN;
        __pipeline_memcpy_async(&Bsb[r * BS_STRIDE + c],
                                &B[(tileK + r) * N + colBase + c], 16);
    }
}

__global__ void kernel(const half* __restrict__ A, const half* __restrict__ B,
                       half* __restrict__ C, int M, int N, int K)
{
    extern __shared__ half smem[];
    half* As = smem;
    half* Bs = smem + STAGES * AS_TILE;

    const int tid = threadIdx.x;
    const int warpId = tid >> 5;
    const int lane = tid & 31;
    const int warpM = warpId / WARP_N;
    const int warpN = warpId % WARP_N;
    const int rowBase = blockIdx.y * BM;
    const int colBase = blockIdx.x * BN;
    const int numTiles = K / BK;

    // ldmatrix per-thread sub-offsets within a 16x16 region.
    const int row16 = (lane & 7) + (((lane >> 3) & 1) << 3);  // 0..15
    const int col8 = ((lane >> 4) & 1) << 3;                  // 0 or 8

    float Cacc[WT_M][WT_N][4];
#pragma unroll
    for (int i = 0; i < WT_M; ++i)
#pragma unroll
        for (int j = 0; j < WT_N; ++j)
#pragma unroll
            for (int t = 0; t < 4; ++t) Cacc[i][j][t] = 0.0f;

    const int warpMrow = warpM * WM;  // 0 or 64
    const int warpNcol = warpN * WN;  // 0,32,64,96

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        loadTile(A, B, As + s * AS_TILE, Bs + s * BS_TILE, rowBase, colBase,
                 s * BK, N, K, tid);
        __pipeline_commit();
    }

    int readStage = 0;
    int writeStage = STAGES - 1;

    for (int tile = 0; tile < numTiles; ++tile) {
        __pipeline_wait_prior(STAGES - 2);
        __syncthreads();

        half* Asb = As + readStage * AS_TILE;
        half* Bsb = Bs + readStage * BS_TILE;

#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            uint32_t Areg[WT_M][4];
            uint32_t Breg[WT_N][2];
#pragma unroll
            for (int mi = 0; mi < WT_M; ++mi) {
                int mrow = warpMrow + mi * 16 + row16;
                uint32_t a = smem_addr(&Asb[mrow * AS_STRIDE + kk + col8]);
                LDMATRIX_X4(Areg[mi][0], Areg[mi][1], Areg[mi][2], Areg[mi][3],
                            a);
            }
#pragma unroll
            for (int nn = 0; nn < WT_N / 2; ++nn) {
                int ncol = warpNcol + nn * 16;
                uint32_t b = smem_addr(&Bsb[(kk + row16) * BS_STRIDE + ncol
                                            + col8]);
                LDMATRIX_X4_T(Breg[nn * 2][0], Breg[nn * 2][1],
                              Breg[nn * 2 + 1][0], Breg[nn * 2 + 1][1], b);
            }
#pragma unroll
            for (int mi = 0; mi < WT_M; ++mi)
#pragma unroll
                for (int ni = 0; ni < WT_N; ++ni)
                    MMA16816(Cacc[mi][ni][0], Cacc[mi][ni][1], Cacc[mi][ni][2],
                             Cacc[mi][ni][3], Areg[mi][0], Areg[mi][1],
                             Areg[mi][2], Areg[mi][3], Breg[ni][0],
                             Breg[ni][1]);
        }

        int loadTileIdx = tile + STAGES - 1;
        if (loadTileIdx < numTiles) {
            loadTile(A, B, As + writeStage * AS_TILE, Bs + writeStage * BS_TILE,
                     rowBase, colBase, loadTileIdx * BK, N, K, tid);
        }
        __pipeline_commit();

        readStage = (readStage + 1) % STAGES;
        writeStage = (writeStage + 1) % STAGES;
    }

    // Store. mma output: c0,c1 at (groupID, 2*(lane%4)); c2,c3 at (groupID+8,..)
    const int groupID = lane >> 2;
    const int colInGroup = (lane & 3) * 2;
#pragma unroll
    for (int mi = 0; mi < WT_M; ++mi) {
#pragma unroll
        for (int ni = 0; ni < WT_N; ++ni) {
            int cRow = rowBase + warpMrow + mi * 16 + groupID;
            int cCol = colBase + warpNcol + ni * 8 + colInGroup;
            half2 lo = __floats2half2_rn(Cacc[mi][ni][0], Cacc[mi][ni][1]);
            half2 hi = __floats2half2_rn(Cacc[mi][ni][2], Cacc[mi][ni][3]);
            *reinterpret_cast<half2*>(&C[cRow * N + cCol]) = lo;
            *reinterpret_cast<half2*>(&C[(cRow + 8) * N + cCol]) = hi;
        }
    }
}

}  // namespace v3

PLAYGROUND_MATMUL_DEC(float16_t, 3, m, n, k, A, B, C)
{
    using namespace v3;
    static bool init = false;
    static int smemBytes = STAGES * (AS_TILE + BS_TILE) * (int) sizeof(half);
    if (!init) {
        cudaFuncSetAttribute(kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smemBytes);
        init = true;
    }
    dim3 block(BLOCK_THREADS);
    dim3 grid(n / BN, m / BM);
    kernel<<<grid, block, smemBytes>>>(A, B, C, int(m), int(n), int(k));
}

}  // namespace playground
