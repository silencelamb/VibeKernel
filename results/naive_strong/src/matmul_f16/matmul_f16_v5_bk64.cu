#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{
namespace
{
// -----------------------------------------------------------------------------
// v5: BK=64 -> A gets 8 chunks/row, fully swizzled (conflict-free A & B).
//   Block 128x128, 8 warps (2x4), warp tile 64x32. 4 k-substeps per tile.
// -----------------------------------------------------------------------------
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 64;
constexpr int STAGES = 3;

constexpr int WARP_ROWS = 2;
constexpr int WARP_COLS = 4;
constexpr int WM = BM / WARP_ROWS;  // 64
constexpr int WN = BN / WARP_COLS;  // 32
constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;
constexpr int MTILES = WM / MMA_M;  // 4
constexpr int NTILES = WN / MMA_N;  // 4

constexpr int APITCH = BM * BK;  // 8192
constexpr int BPITCH = BK * BN;  // 8192

__device__ __forceinline__ int swzA(int row, int col)
{
    int chunk = (col >> 3) ^ (row & 7);  // 8 chunks per row (BK/8)
    return row * BK + (chunk << 3) + (col & 7);
}
__device__ __forceinline__ int swzB(int row, int col)
{
    int chunk = (col >> 3) ^ (row & 15);  // 16 chunks per row (BN/8)
    return row * BN + (chunk << 3) + (col & 7);
}

__device__ __forceinline__ unsigned smem_addr(const void* p)
{
    return (unsigned) __cvta_generic_to_shared(p);
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem)
{
    unsigned s = smem_addr(smem);
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
__device__ __forceinline__ void ldm_x4(uint32_t (&r)[4], const void* s)
{
    unsigned a = smem_addr(s);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(a));
}
__device__ __forceinline__ void ldm_x2_trans(uint32_t (&r)[2], const void* s)
{
    unsigned a = smem_addr(s);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1])
        : "r"(a));
}
__device__ __forceinline__ void mma_m16n8k16(float (&d)[4],
                                             const uint32_t (&a)[4],
                                             const uint32_t (&b)[2])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__global__ void __launch_bounds__(256)
    hgemm_v5(const half* __restrict__ A, const half* __restrict__ B,
             half* __restrict__ C, int M, int N, int K)
{
    extern __shared__ half smem[];
    half* As = smem;
    half* Bs = smem + STAGES * APITCH;

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warpId = tid >> 5;
    const int warpRow = warpId / WARP_COLS;
    const int warpCol = warpId % WARP_COLS;

    // cp.async mapping: 4 float4 per thread for each of A, B.
    const int aRowB = tid / 8;          // 0..31 (+0/32/64/96)
    const int aCol = (tid % 8) * 8;     // 0..56
    const int bRowB = tid / 16;         // 0..15 (+0/16/32/48)
    const int bCol = (tid % 16) * 8;    // 0..120

    float acc[MTILES][NTILES][4];
#pragma unroll
    for (int i = 0; i < MTILES; ++i)
#pragma unroll
        for (int j = 0; j < NTILES; ++j)
#pragma unroll
            for (int t = 0; t < 4; ++t) acc[i][j][t] = 0.0f;

    const int numTiles = K / BK;

    auto loadTile = [&](int stage, int kt) {
        const int k0 = kt * BK;
        half* asp = As + stage * APITCH;
        half* bsp = Bs + stage * BPITCH;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            int arow = aRowB + r * 32;
            cp_async16(&asp[swzA(arow, aCol)],
                       A + (blockRow + arow) * K + (k0 + aCol));
        }
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            int brow = bRowB + r * 16;
            cp_async16(&bsp[swzB(brow, bCol)],
                       B + (k0 + brow) * N + (blockCol + bCol));
        }
    };

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        loadTile(s, s);
        cp_commit();
    }

    int kTile = STAGES - 1;
    for (int ki = 0; ki < numTiles; ++ki) {
        cp_wait<STAGES - 2>();
        __syncthreads();

        const int rs = ki % STAGES;
        half* asp = As + rs * APITCH;
        half* bsp = Bs + rs * BPITCH;

#pragma unroll
        for (int ks = 0; ks < BK; ks += MMA_K) {
            uint32_t aF[MTILES][4];
            uint32_t bF[NTILES][2];
#pragma unroll
            for (int i = 0; i < MTILES; ++i) {
                int row = warpRow * WM + i * MMA_M + (lane % 16);
                int col = ks + (lane / 16) * 8;
                ldm_x4(aF[i], &asp[swzA(row, col)]);
            }
#pragma unroll
            for (int j = 0; j < NTILES; ++j) {
                int row = ks + (lane % 16);
                int col = warpCol * WN + j * MMA_N;
                ldm_x2_trans(bF[j], &bsp[swzB(row, col)]);
            }
#pragma unroll
            for (int i = 0; i < MTILES; ++i)
#pragma unroll
                for (int j = 0; j < NTILES; ++j)
                    mma_m16n8k16(acc[i][j], aF[i], bF[j]);
        }

        if (kTile < numTiles) loadTile(kTile % STAGES, kTile);
        cp_commit();
        ++kTile;
    }

    const int group = lane / 4;
    const int tid4 = lane % 4;
#pragma unroll
    for (int i = 0; i < MTILES; ++i) {
#pragma unroll
        for (int j = 0; j < NTILES; ++j) {
            int rowBase = blockRow + warpRow * WM + i * MMA_M;
            int colBase = blockCol + warpCol * WN + j * MMA_N + tid4 * 2;
            half2 lo = __floats2half2_rn(acc[i][j][0], acc[i][j][1]);
            half2 hi = __floats2half2_rn(acc[i][j][2], acc[i][j][3]);
            *reinterpret_cast<half2*>(&C[(rowBase + group) * N + colBase]) = lo;
            *reinterpret_cast<half2*>(&C[(rowBase + group + 8) * N + colBase]) =
                hi;
        }
    }
}

}  // namespace

PLAYGROUND_MATMUL_DEC(float16_t, 5, m, n, k, A, B, C)
{
    static bool configured = false;
    const int smemBytes = STAGES * (APITCH + BPITCH) * int(sizeof(half));
    if (!configured) {
        cudaError_t e = cudaFuncSetAttribute(
            hgemm_v5, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes);
        if (e != cudaSuccess)
            fprintf(stderr, "[v5] setAttr(%d) failed: %s\n", smemBytes,
                    cudaGetErrorString(e));
        configured = true;
    }
    dim3 block(256);
    dim3 grid(n / BN, m / BM);
    hgemm_v5<<<grid, block, smemBytes>>>(A, B, C, int(m), int(n), int(k));
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess)
        fprintf(stderr, "[v5] launch failed: %s\n", cudaGetErrorString(le));
}

}  // namespace playground
