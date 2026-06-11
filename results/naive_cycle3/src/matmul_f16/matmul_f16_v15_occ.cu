#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{
namespace v15
{
// 256x128 block, warp 64x64, fp16 acc, BK=32, 3-stage, XOR-swizzled (no pad).
// Swizzle gives conflict-free ldmatrix AND saves smem -> 3 stages + 16 warps.
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int STAGES = 3;
constexpr int BLOCK_THREADS = 256;
constexpr int WARP_N = 2;
constexpr int WM = 32, WN = 64;
constexpr int WT_M = WM / 16;  // 4
constexpr int WT_N = WN / 8;   // 8
constexpr int AS_TILE = BM * BK;  // 8192 (no padding)
constexpr int BS_TILE = BK * BN;  // 4096

__device__ __forceinline__ uint32_t smem_addr(const void* ptr)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// Swizzle (half-offset within a stage tile). Octet = 8 halfs (16 bytes).
__device__ __forceinline__ int swzA(int r, int c)  // A[r in BM][c in BK]
{
    return r * BK + (((c >> 3) ^ ((r >> 1) & 3)) << 3) + (c & 7);
}
__device__ __forceinline__ int swzB(int k, int n)  // B[k in BK][n in BN]
{
    return k * BN + (((n >> 3) ^ (k & 7)) << 3) + (n & 7);
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

#define MMA16816_F16(D0, D1, A0, A1, A2, A3, B0, B1)                           \
    asm volatile(                                                              \
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "                   \
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"                          \
        : "+r"(D0), "+r"(D1)                                                   \
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
        __pipeline_memcpy_async(&Asb[swzA(r, c)],
                                &A[(rowBase + r) * K + tileK + c], 16);
    }
#pragma unroll
    for (int i = tid; i < BK * BN / 8; i += BLOCK_THREADS) {
        int r = (i * 8) / BN;
        int c = (i * 8) % BN;
        __pipeline_memcpy_async(&Bsb[swzB(r, c)],
                                &B[(tileK + r) * N + colBase + c], 16);
    }
}

__global__ void __launch_bounds__(BLOCK_THREADS, 3)
    kernel(const half* __restrict__ A, const half* __restrict__ B,
           half* __restrict__ C, int M, int N, int K)
{
    extern __shared__ half smem[];
    half* As = smem;
    half* Bs = smem + STAGES * AS_TILE;
    const uint32_t AsBase = smem_addr(As);
    const uint32_t BsBase = smem_addr(Bs);

    const int tid = threadIdx.x;
    const int warpId = tid >> 5;
    const int lane = tid & 31;
    const int warpM = warpId / WARP_N;
    const int warpN = warpId % WARP_N;
    const int rowBase = blockIdx.y * BM;
    const int colBase = blockIdx.x * BN;
    const int numTiles = K / BK;

    const int row16 = (lane & 7) + (((lane >> 3) & 1) << 3);
    const int col8 = ((lane >> 4) & 1) << 3;
    const int warpMrow = warpM * WM;
    const int warpNcol = warpN * WN;

    // Precompute swizzled half-offsets for ldmatrix (per fragment, both kk).
    int aSwz[WT_M][2];  // [mi][kkIdx]
#pragma unroll
    for (int mi = 0; mi < WT_M; ++mi) {
        int r = warpMrow + mi * 16 + row16;
        aSwz[mi][0] = swzA(r, 0 + col8);
        aSwz[mi][1] = swzA(r, 16 + col8);
    }
    int bSwz[WT_N / 2][2];  // [nn][kkIdx]
#pragma unroll
    for (int nn = 0; nn < WT_N / 2; ++nn) {
        int n = warpNcol + nn * 16 + col8;
        bSwz[nn][0] = swzB(0 + row16, n);
        bSwz[nn][1] = swzB(16 + row16, n);
    }

    uint32_t Cacc[WT_M][WT_N][2];
#pragma unroll
    for (int i = 0; i < WT_M; ++i)
#pragma unroll
        for (int j = 0; j < WT_N; ++j) {
            Cacc[i][j][0] = 0u;
            Cacc[i][j][1] = 0u;
        }

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

        uint32_t aStage = AsBase + uint32_t(readStage * AS_TILE * 2);
        uint32_t bStage = BsBase + uint32_t(readStage * BS_TILE * 2);

#pragma unroll
        for (int kkIdx = 0; kkIdx < 2; ++kkIdx) {
            uint32_t Areg[WT_M][4];
            uint32_t Breg[WT_N][2];
#pragma unroll
            for (int mi = 0; mi < WT_M; ++mi)
                LDMATRIX_X4(Areg[mi][0], Areg[mi][1], Areg[mi][2], Areg[mi][3],
                            aStage + uint32_t(aSwz[mi][kkIdx] * 2));
#pragma unroll
            for (int nn = 0; nn < WT_N / 2; ++nn)
                LDMATRIX_X4_T(Breg[nn * 2][0], Breg[nn * 2][1],
                              Breg[nn * 2 + 1][0], Breg[nn * 2 + 1][1],
                              bStage + uint32_t(bSwz[nn][kkIdx] * 2));
#pragma unroll
            for (int mi = 0; mi < WT_M; ++mi)
#pragma unroll
                for (int ni = 0; ni < WT_N; ++ni)
                    MMA16816_F16(Cacc[mi][ni][0], Cacc[mi][ni][1], Areg[mi][0],
                                 Areg[mi][1], Areg[mi][2], Areg[mi][3],
                                 Breg[ni][0], Breg[ni][1]);
        }

        int loadIdx = tile + STAGES - 1;
        if (loadIdx < numTiles) {
            loadTile(A, B, As + writeStage * AS_TILE, Bs + writeStage * BS_TILE,
                     rowBase, colBase, loadIdx * BK, N, K, tid);
        }
        __pipeline_commit();
        readStage = (readStage + 1) % STAGES;
        writeStage = (writeStage + 1) % STAGES;
    }

    const int groupID = lane >> 2;
    const int colInGroup = (lane & 3) * 2;
#pragma unroll
    for (int mi = 0; mi < WT_M; ++mi) {
#pragma unroll
        for (int ni = 0; ni < WT_N; ++ni) {
            int cRow = rowBase + warpMrow + mi * 16 + groupID;
            int cCol = colBase + warpNcol + ni * 8 + colInGroup;
            *reinterpret_cast<uint32_t*>(&C[cRow * N + cCol]) = Cacc[mi][ni][0];
            *reinterpret_cast<uint32_t*>(&C[(cRow + 8) * N + cCol]) =
                Cacc[mi][ni][1];
        }
    }
}

}  // namespace v15

PLAYGROUND_MATMUL_DEC(float16_t, 15, m, n, k, A, B, C)
{
    using namespace v15;
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
