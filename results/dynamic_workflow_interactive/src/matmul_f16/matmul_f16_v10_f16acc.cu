// @file: task-1/src/matmul_f16/matmul_f16_v8_sweep.cu
//
// v10: fp16-accumulate variant of v9 (half the accum regs -> higher occupancy + vectorized 32-bit epilogue). Configurable.
//     design space. Pick a config with -DCFG=<id> (default below). All the
//     proven pieces from v3/v4/v7: padded smem (conflict-free ldmatrix),
//     NSTAGE cp.async pipeline, ldmatrix.x4 (A) + x4.trans (B), fp32 accum.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "playground/matmul.hpp"

#ifndef CFG
#define CFG 11
#endif

namespace playground
{
namespace
{
struct Config {
    int BM, BN, BK, WM_CNT, WN_CNT, NSTAGE;
};
// id:        BM   BN   BK  WMc WNc  ST
constexpr Config CFGS[] = {
    {128, 128, 32, 2, 4, 4},  // 0: = v4 (133), warp 64x32, 2 blk/SM
    {128, 128, 32, 4, 4, 4},  // 1: warp 32x32, 16 warps -> more occupancy
    {128, 128, 64, 2, 4, 4},  // 2: BK=64, K16=4, fewer barriers
    {256, 128, 32, 4, 2, 4},  // 3: warp 64x64, 16 warps (512 thr)
    {128, 128, 32, 4, 2, 4},  // 4: warp 32x64, 16 warps
    {128, 128, 32, 2, 4, 3},  // 5: 3 stages (more smem headroom)
    {256, 128, 32, 4, 4, 3},  // 6: big block, warp 64x32, 16 warps
    {128, 64, 32, 2, 2, 4},   // 7: warp 64x32, 4 warps small block
    {128, 128, 64, 4, 4, 3},  // 8: BK=64, warp 32x32, 16 warps
    {256, 64, 32, 4, 2, 4},   // 9: warp 64x32, 16 warps tall
    {256, 128, 32, 4, 2, 5},  // 10: CFG3 + 5 stages (deeper latency hide)
    {256, 128, 64, 4, 2, 3},  // 11: CFG3 + BK=64 (longer inner pipeline)
    {128, 256, 32, 2, 4, 4},  // 12: warp 64x64, transposed block 128x256
    {256, 128, 64, 4, 2, 2},  // 13: BK=64, 2 stages
    {256, 256, 32, 4, 4, 3},  // 14: warp 64x64, 16 warps, big block
    {192, 128, 32, 3, 2, 4},  // 15: warp 64x64, 6 warps (invalid geom)
    {128, 128, 32, 2, 2, 4},  // 16: warp 64x64, 4 warps/128thr -> 2 blk/SM
    {128, 128, 64, 2, 2, 2},  // 17: warp 64x64, BK64, 2 stage -> 2 blk/SM
    {128, 128, 64, 2, 2, 3},  // 18: warp 64x64, BK64, 3 stage (1 blk)
    {128, 128, 128, 2, 2, 2},  // 19: warp 64x64, BK128, amortize bubble
    {64, 256, 64, 1, 4, 3},   // 20: warp 64x64, tall-N
    {256, 128, 64, 4, 2, 3},  // 21: == 11 reference re-pin
    {256, 128, 64, 2, 2, 3},  // 22: warp 128x64, 4 warps, 64 acc tiles/warp
    {128, 256, 64, 2, 2, 3},  // 23: warp 64x128, 4 warps
    {128, 128, 32, 2, 2, 8},  // 24: warp 64x64, 8-stage deep pipeline
    {256, 128, 32, 2, 2, 4},  // 25: warp 128x64, BK32, 4 stage
};
constexpr Config CF = CFGS[CFG];
constexpr int BM = CF.BM;
constexpr int BN = CF.BN;
constexpr int BK = CF.BK;
constexpr int WM_CNT = CF.WM_CNT;
constexpr int WN_CNT = CF.WN_CNT;
constexpr int NSTAGE = CF.NSTAGE;
constexpr int NWARPS = WM_CNT * WN_CNT;
constexpr int NTHREADS = NWARPS * 32;
constexpr int WM = BM / WM_CNT;
constexpr int WN = BN / WN_CNT;
constexpr int WMITER = WM / 16;
constexpr int WNITER = WN / 8;
constexpr int K16 = BK / 16;
constexpr int AP = 8;
constexpr int ALDM = BK + AP;
constexpr int BLDM = BN + AP;
constexpr int A_STAGE = BM * ALDM;
constexpr int B_STAGE = BK * BLDM;
constexpr int SMEM_HALFS = NSTAGE * (A_STAGE + B_STAGE);
constexpr int VA = BM * BK / 8;  // 16B vectors for A tile
constexpr int VB = BK * BN / 8;  // 16B vectors for B tile

__device__ __forceinline__ uint32_t smem_u32(const void* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem)
{
    uint32_t s = smem_u32(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s),
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
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* smem)
{
    uint32_t s = smem_u32(smem);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(s));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4],
                                                  const void* smem)
{
    uint32_t s = smem_u32(smem);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(s));
}
// fp16 accumulate: C/D fragment is 4 fp16 = 2 .b32 regs (half of fp32 accum).
__device__ __forceinline__ void mma_m16n8k16(uint32_t (&acc)[2],
                                             const uint32_t (&a)[4],
                                             const uint32_t (&b)[2])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(acc[0]), "+r"(acc[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__global__ void __launch_bounds__(NTHREADS)
matmul_v10_kernel(int M, int N, int K, const half* __restrict__ A,
                 const half* __restrict__ B, half* __restrict__ C)
{
    extern __shared__ half smem[];
    half* As = smem;
    half* Bs = smem + NSTAGE * A_STAGE;

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int warpId = tid / 32;
    const int lane = tid % 32;
    const int warpM = warpId / WN_CNT;
    const int warpN = warpId % WN_CNT;

    uint32_t acc[WMITER][WNITER][2];  // fp16 accum: 2 packed-fp16 regs / tile
#pragma unroll
    for (int i = 0; i < WMITER; ++i)
#pragma unroll
        for (int j = 0; j < WNITER; ++j)
#pragma unroll
            for (int t = 0; t < 2; ++t) acc[i][j][t] = 0u;

    auto AsP = [&](int s, int m, int k) -> half* {
        return &As[s * A_STAGE + m * ALDM + k];
    };
    auto BsP = [&](int s, int k, int n) -> half* {
        return &Bs[s * B_STAGE + k * BLDM + n];
    };
    constexpr int AVPR = BK / 8;  // A vectors per row
    constexpr int BVPR = BN / 8;  // B vectors per row
    constexpr int A_ITERS = VA / NTHREADS;  // compile-time -> full unroll
    constexpr int B_ITERS = VB / NTHREADS;
    static_assert(A_ITERS * NTHREADS == VA && B_ITERS * NTHREADS == VB,
                  "tile must divide evenly across threads");
    auto loadStage = [&](int s, int k0) {
#pragma unroll
        for (int i = 0; i < A_ITERS; ++i) {
            const int v = tid + i * NTHREADS;
            const int row = v / AVPR;
            const int col = (v % AVPR) * 8;
            cp_async16(AsP(s, row, col),
                       &A[(blockRow + row) * K + (k0 + col)]);
        }
#pragma unroll
        for (int i = 0; i < B_ITERS; ++i) {
            const int v = tid + i * NTHREADS;
            const int row = v / BVPR;
            const int col = (v % BVPR) * 8;
            cp_async16(BsP(s, row, col),
                       &B[(k0 + row) * N + (blockCol + col)]);
        }
    };

    const int numTiles = K / BK;
#pragma unroll
    for (int s = 0; s < NSTAGE - 1; ++s) {
        loadStage(s, s * BK);
        cp_commit();
    }

    int readStage = 0;
    int writeStage = NSTAGE - 1;

    // Register double buffer for operands: ldmatrix(ki+1) overlaps mma(ki).
    uint32_t aF[2][WMITER][4];
    uint32_t bF[2][WNITER][2];
    auto loadKi = [&](int buf, int stage, int ki) {
        const int kB = ki * 16;
#pragma unroll
        for (int mi = 0; mi < WMITER; ++mi)
            ldmatrix_x4(aF[buf][mi],
                        AsP(stage, warpM * WM + mi * 16 + (lane % 16),
                            kB + (lane / 16) * 8));
#pragma unroll
        for (int ni = 0; ni < WNITER; ni += 2) {
            uint32_t r[4];
            ldmatrix_x4_trans(r, BsP(stage, kB + (lane % 16),
                                     warpN * WN + ni * 8 + (lane / 16) * 8));
            bF[buf][ni][0] = r[0];
            bF[buf][ni][1] = r[1];
            bF[buf][ni + 1][0] = r[2];
            bF[buf][ni + 1][1] = r[3];
        }
    };

    for (int tile = 0; tile < numTiles; ++tile) {
        cp_wait<NSTAGE - 2>();
        __syncthreads();

        loadKi(0, readStage, 0);
#pragma unroll
        for (int ki = 0; ki < K16; ++ki) {
            const int cur = ki & 1;
            if (ki + 1 < K16) loadKi((ki + 1) & 1, readStage, ki + 1);
            // The last K-step issues the global prefetch so the cp.async issue
            // overlaps this final mma group (NOT before compute -> no MIO clog).
            if (ki == K16 - 1) {
                const int nextTile = tile + (NSTAGE - 1);
                if (nextTile < numTiles) loadStage(writeStage, nextTile * BK);
                cp_commit();
            }
#pragma unroll
            for (int mi = 0; mi < WMITER; ++mi)
#pragma unroll
                for (int ni = 0; ni < WNITER; ++ni)
                    mma_m16n8k16(acc[mi][ni], aF[cur][mi], bF[cur][ni]);
        }

        readStage = (readStage + 1) % NSTAGE;
        writeStage = (writeStage + 1) % NSTAGE;
    }

#pragma unroll
    for (int mi = 0; mi < WMITER; ++mi) {
#pragma unroll
        for (int ni = 0; ni < WNITER; ++ni) {
            const int baseM = blockRow + warpM * WM + mi * 16;
            const int baseN = blockCol + warpN * WN + ni * 8;
            const int row0 = baseM + lane / 4;
            const int col0 = baseN + (lane % 4) * 2;
            // acc[..][0] packs (row0,col0)+(row0,col0+1); [1] packs row0+8.
            // Adjacent cols in row-major C -> one 32-bit store each.
            *reinterpret_cast<uint32_t*>(&C[row0 * N + col0]) = acc[mi][ni][0];
            *reinterpret_cast<uint32_t*>(&C[(row0 + 8) * N + col0]) =
                acc[mi][ni][1];
        }
    }
}
}  // namespace

PLAYGROUND_MATMUL_DEC(float16_t, 10, m, n, k, A, B, C)
{
    static bool attrSet = false;
    const int smemBytes = SMEM_HALFS * int(sizeof(half));
    if (!attrSet) {
        cudaFuncSetAttribute(matmul_v10_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smemBytes);
        attrSet = true;
    }
    dim3 block(NTHREADS);
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    matmul_v10_kernel<<<grid, block, smemBytes>>>(int(m), int(n), int(k), A, B,
                                                 C);
}

}  // namespace playground
