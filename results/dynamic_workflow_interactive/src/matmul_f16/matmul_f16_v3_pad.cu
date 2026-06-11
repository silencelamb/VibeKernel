// @file: task-1/src/matmul_f16/matmul_f16_v2_mma.cu
//
// v3: v2 + smem padding (bank-conflict-free ldmatrix) Tensor-Core GEMM with the Ampere primitives:
//   - mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32  (TC, fp32 accum)
//   - ldmatrix.sync.aligned.m8n8.x4 / .x2.trans          (smem -> reg)
//   - cp.async.cg                                        (gmem -> smem, async)
//   - 2-stage (double-buffered) software pipeline.
//
// Tiling: block 128x128, BK=32. 8 warps (2x4) -> warp tile 64x32.
//   Per warp: WMITER=4 (M/16) x WNITER=4 (N/8) = 16 mma tiles, 64 fp32 accum.
// A,B,C row-major device pointers. A,B stored row-major in smem (cp.async-
// friendly); A via ldmatrix.x4, B via ldmatrix.x2.trans.

#include <cuda_fp16.h>

#include "playground/matmul.hpp"

namespace playground
{
namespace
{
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int NTHREADS = 256;            // 8 warps
constexpr int WN_CNT = 4;                // warp grid: N
constexpr int WM_CNT = 2;                // warp grid: M
constexpr int WM = BM / WM_CNT;          // 64
constexpr int WN = BN / WN_CNT;          // 32
constexpr int WMITER = WM / 16;          // 4
constexpr int WNITER = WN / 8;           // 4
constexpr int K16 = BK / 16;             // 2

// ---- PTX helpers -----------------------------------------------------------
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
__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t (&r)[2],
                                                  const void* smem)
{
    uint32_t s = smem_u32(smem);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1])
        : "r"(s));
}
__device__ __forceinline__ void mma_m16n8k16(float (&acc)[4],
                                             const uint32_t (&a)[4],
                                             const uint32_t (&b)[2])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__global__ void __launch_bounds__(NTHREADS)
matmul_v3_kernel(int M, int N, int K, const half* __restrict__ A,
                 const half* __restrict__ B, half* __restrict__ C)
{
    // Pad inner (leading) dim by 8 fp16 (16B) so each 8-row ldmatrix matrix
    // hits 8 distinct shared-memory bank groups (conflict-free), while keeping
    // every cp.async target 16B-aligned. 80B / 272B row strides give distinct
    // residues mod 128 for 8 consecutive rows.
    constexpr int AP = 8;
    __shared__ half As[2][BM][BK + AP];  // 2*128*40*2 = 20KB
    __shared__ half Bs[2][BK][BN + AP];  // 2*32*136*2 = 17KB

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    const int tid = threadIdx.x;
    const int warpId = tid / 32;
    const int lane = tid % 32;
    const int warpM = warpId / WN_CNT;  // 0..1
    const int warpN = warpId % WN_CNT;  // 0..3

    float acc[WMITER][WNITER][4];
#pragma unroll
    for (int i = 0; i < WMITER; ++i)
#pragma unroll
        for (int j = 0; j < WNITER; ++j)
#pragma unroll
            for (int t = 0; t < 4; ++t) acc[i][j][t] = 0.0f;

    // ---- global -> smem loader (cp.async, 16B = 8 fp16 vectors) ----
    // As: 128x32 = 512 vectors, 256 threads -> 2 each. 4 vec/row, 128 rows.
    // Bs: 32x128 = 512 vectors, 256 threads -> 2 each. 16 vec/row, 32 rows.
    auto loadTile = [&](int stage, int k0) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int v = tid + i * NTHREADS;       // 0..511
            const int row = v / 4;                  // 0..127
            const int col = (v % 4) * 8;            // 0,8,16,24
            cp_async16(&As[stage][row][col],
                       &A[(blockRow + row) * K + (k0 + col)]);
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int v = tid + i * NTHREADS;       // 0..511
            const int row = v / 16;                 // 0..31
            const int col = (v % 16) * 8;           // 0..120
            cp_async16(&Bs[stage][row][col],
                       &B[(k0 + row) * N + (blockCol + col)]);
        }
    };

    int stage = 0;
    loadTile(0, 0);
    cp_commit();

    for (int k0 = 0; k0 < K; k0 += BK) {
        const int next = k0 + BK;
        const bool hasNext = next < K;
        if (hasNext) {
            loadTile(stage ^ 1, next);
            cp_commit();
            cp_wait<1>();  // keep the just-issued group pending
        } else {
            cp_wait<0>();
        }
        __syncthreads();

        uint32_t aFrag[K16][WMITER][4];
        uint32_t bFrag[K16][WNITER][2];
#pragma unroll
        for (int ki = 0; ki < K16; ++ki) {
            const int kBase = ki * 16;
#pragma unroll
            for (int mi = 0; mi < WMITER; ++mi) {
                const int mRow = warpM * WM + mi * 16;
                ldmatrix_x4(aFrag[ki][mi],
                            &As[stage][mRow + (lane % 16)][kBase +
                                                           (lane / 16) * 8]);
            }
#pragma unroll
            for (int ni = 0; ni < WNITER; ++ni) {
                const int nCol = warpN * WN + ni * 8;
                ldmatrix_x2_trans(bFrag[ki][ni],
                                  &Bs[stage][kBase + (lane % 16)][nCol]);
            }
        }
#pragma unroll
        for (int ki = 0; ki < K16; ++ki)
#pragma unroll
            for (int mi = 0; mi < WMITER; ++mi)
#pragma unroll
                for (int ni = 0; ni < WNITER; ++ni)
                    mma_m16n8k16(acc[mi][ni], aFrag[ki][mi], bFrag[ki][ni]);

        __syncthreads();
        stage ^= 1;
    }

    // ---- store: each thread holds 4 fp32 per (mi,ni) tile ----
#pragma unroll
    for (int mi = 0; mi < WMITER; ++mi) {
#pragma unroll
        for (int ni = 0; ni < WNITER; ++ni) {
            const int baseM = blockRow + warpM * WM + mi * 16;
            const int baseN = blockCol + warpN * WN + ni * 8;
            const int row0 = baseM + lane / 4;
            const int col0 = baseN + (lane % 4) * 2;
            C[row0 * N + col0] = __float2half(acc[mi][ni][0]);
            C[row0 * N + col0 + 1] = __float2half(acc[mi][ni][1]);
            C[(row0 + 8) * N + col0] = __float2half(acc[mi][ni][2]);
            C[(row0 + 8) * N + col0 + 1] = __float2half(acc[mi][ni][3]);
        }
    }
}
}  // namespace

PLAYGROUND_MATMUL_DEC(float16_t, 3, m, n, k, A, B, C)
{
    dim3 block(NTHREADS);
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    matmul_v3_kernel<<<grid, block>>>(int(m), int(n), int(k), A, B, C);
}

}  // namespace playground
