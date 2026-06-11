#include "playground/matmul.hpp"
#include "playground/system.hpp"

#include <cstdint>
#include <cuda_fp16.h>

namespace playground
{
namespace v56
{
constexpr int BM = 128, BN = 256, BK = 64;
constexpr int WARPS_M = 2, WARPS_N = 4, NWARPS = 8, NTHREADS = 256;
constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N;  // 64, 64
constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;
constexpr int M_ITER = WM / MMA_M, N_ITER = WN / MMA_N, K_ITER = BK / MMA_K;
constexpr int APAD = 8, BPAD = 8;
constexpr int AS_STRIDE = BK + APAD, AS_TILE = BM * AS_STRIDE;
constexpr int BS_STRIDE = BN + BPAD, BS_TILE = BK * BS_STRIDE;
constexpr int NSTAGES = 3;
constexpr int N_CHUNK = WN / 16;

__device__ __forceinline__ uint32_t cvta(const void* p)
{
    return (uint32_t) __cvta_generic_to_shared(p);
}
__device__ __forceinline__ void cp16(uint32_t s, const void* g)
{
    asm volatile("cp.async.cg.shared.global [%0],[%1],16;\n" ::"r"(s), "l"(g));
}
__device__ __forceinline__ void commit()
{
    asm volatile("cp.async.commit_group;\n");
}
template <int N>
__device__ __forceinline__ void waitg()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void ldm4(uint32_t r[4], uint32_t a)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(a));
}
__device__ __forceinline__ void ldm4t(uint32_t r[4], uint32_t a)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(a));
}
__device__ __forceinline__ void mma(float d[4], const uint32_t a[4],
                                    const uint32_t b[2])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__global__ __launch_bounds__(NTHREADS) void kernel(const half* __restrict__ A,
                                                    const half* __restrict__ B,
                                                    half* __restrict__ C, int M,
                                                    int N, int K)
{
    extern __shared__ half smem[];
    half* As = smem;
    half* Bs = smem + NSTAGES * AS_TILE;
    const int tid = threadIdx.x;
    const int warpId = tid >> 5, laneId = tid & 31;
    const int warpM = warpId / WARPS_N, warpN = warpId % WARPS_N;
    const int blocksM = M / BM, blocksN = N / BN;
    constexpr int GROUP_M = 8;
    const int linear = blockIdx.y * blocksN + blockIdx.x;
    const int width = GROUP_M * blocksN;
    const int group_id = linear / width;
    const int first_row = group_id * GROUP_M;
    const int group_rows = min(blocksM - first_row, GROUP_M);
    const int by = first_row + (linear % group_rows);
    const int bx = (linear % width) / group_rows;
    const int rowBase = by * BM, colBase = bx * BN;
    const int numTiles = K / BK;

    float acc[M_ITER][N_ITER][4];
#pragma unroll
    for (int i = 0; i < M_ITER; i++)
#pragma unroll
        for (int j = 0; j < N_ITER; j++)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

    const uint32_t smemA = cvta(As), smemB = cvta(Bs);
    const uint32_t aLaneOff =
        ((warpM * WM + (laneId & 15)) * AS_STRIDE + (laneId >> 4) * 8) * 2;
    const uint32_t bLaneOff =
        ((laneId & 15) * BS_STRIDE + warpN * WN + (laneId >> 4) * 8) * 2;
    uint32_t aTileBase[NSTAGES], bTileBase[NSTAGES];
#pragma unroll
    for (int s = 0; s < NSTAGES; s++) {
        aTileBase[s] = smemA + s * AS_TILE * 2 + aLaneOff;
        bTileBase[s] = smemB + s * BS_TILE * 2 + bLaneOff;
    }

    constexpr int ACP = BM * BK / 8 / NTHREADS;
    constexpr int BCP = BK * BN / 8 / NTHREADS;
    // Per-thread row/col for each copy (compile-time small ints, few registers).
    auto aR = [&](int j) { return ((tid + j * NTHREADS) * 8) / BK; };
    auto aC = [&](int j) { return ((tid + j * NTHREADS) * 8) % BK; };
    auto bR = [&](int j) { return ((tid + j * NTHREADS) * 8) / BN; };
    auto bC = [&](int j) { return ((tid + j * NTHREADS) * 8) % BN; };

    // Interleaved cp.async: issue ~1/K_ITER of the copies per inner slice.
    constexpr int A_PER = (ACP + K_ITER - 1) / K_ITER;
    constexpr int B_PER = (BCP + K_ITER - 1) / K_ITER;
    auto load_chunk = [&](int st, int k0, int chunk) {
        uint32_t aoff = st * AS_TILE * 2, boff = st * BS_TILE * 2;
#pragma unroll
        for (int jj = 0; jj < A_PER; jj++) {
            int j = chunk * A_PER + jj;
            if (j < ACP) {
                int r = aR(j), c = aC(j);
                cp16(smemA + (r * AS_STRIDE + c) * 2 + aoff,
                     A + (rowBase + r) * (size_t) K + k0 + c);
            }
        }
#pragma unroll
        for (int jj = 0; jj < B_PER; jj++) {
            int j = chunk * B_PER + jj;
            if (j < BCP) {
                int r = bR(j), c = bC(j);
                cp16(smemB + (r * BS_STRIDE + c) * 2 + boff,
                     B + (size_t) (k0 + r) * N + colBase + c);
            }
        }
    };
    auto load_stage = [&](int st, int k0) {
#pragma unroll
        for (int chunk = 0; chunk < K_ITER; chunk++)
            load_chunk(st, k0, chunk);
    };

    auto load_frags = [&](uint32_t aTile, uint32_t bTile, int kk,
                          uint32_t aa[M_ITER][4], uint32_t bb[N_ITER][2]) {
#pragma unroll
        for (int mi = 0; mi < M_ITER; mi++)
            ldm4(aa[mi], aTile + (mi * MMA_M * AS_STRIDE + kk) * 2);
#pragma unroll
        for (int chunk = 0; chunk < N_CHUNK; chunk++) {
            uint32_t r[4];
            ldm4t(r, bTile + (kk * BS_STRIDE + chunk * 16) * 2);
            bb[chunk * 2 + 0][0] = r[0];
            bb[chunk * 2 + 0][1] = r[1];
            bb[chunk * 2 + 1][0] = r[2];
            bb[chunk * 2 + 1][1] = r[3];
        }
    };

    // Prologue: issue NSTAGES-1 stages of cp.async, then wait for the first.
#pragma unroll
    for (int s = 0; s < NSTAGES - 1; s++) {
        load_stage(s, s * BK);
        commit();
    }
    waitg<NSTAGES - 2>();
    __syncthreads();

    uint32_t a[2][M_ITER][4];
    uint32_t b[2][N_ITER][2];
    load_frags(aTileBase[0], bTileBase[0], 0, a[0], b[0]);

    int rbuf = 0;
    for (int t = 0; t < numTiles; t++) {
        int cur = t % NSTAGES;
        int pf = t + NSTAGES - 1;
        int pfStage = pf % NSTAGES, pfK0 = pf * BK;

#pragma unroll
        for (int kki = 0; kki < K_ITER; kki++) {
            // Interleave next-stage cp.async issue across the inner slices.
            if (pf < numTiles)
                load_chunk(pfStage, pfK0, kki);
            if (kki == K_ITER - 1)
                commit();

            int nrb = rbuf ^ 1;
            if (kki + 1 < K_ITER) {
                load_frags(aTileBase[cur], bTileBase[cur], (kki + 1) * MMA_K,
                           a[nrb], b[nrb]);
            } else if (t + 1 < numTiles) {
                waitg<NSTAGES - 2>();
                __syncthreads();
                int nx = (t + 1) % NSTAGES;
                load_frags(aTileBase[nx], bTileBase[nx], 0, a[nrb], b[nrb]);
            }
#pragma unroll
            for (int mi = 0; mi < M_ITER; mi++)
#pragma unroll
                for (int ni = 0; ni < N_ITER; ni++)
                    mma(acc[mi][ni], a[rbuf][mi], b[rbuf][ni]);
            rbuf = nrb;
        }
    }

    const int group = laneId / 4, base = (laneId % 4) * 2;
#pragma unroll
    for (int mi = 0; mi < M_ITER; mi++)
#pragma unroll
        for (int ni = 0; ni < N_ITER; ni++) {
            int r0 = rowBase + warpM * WM + mi * MMA_M + group;
            int cc = colBase + warpN * WN + ni * MMA_N + base;
            half2 lo = __floats2half2_rn(acc[mi][ni][0], acc[mi][ni][1]);
            half2 hi = __floats2half2_rn(acc[mi][ni][2], acc[mi][ni][3]);
            *reinterpret_cast<half2*>(&C[r0 * (size_t) N + cc]) = lo;
            *reinterpret_cast<half2*>(&C[(r0 + 8) * (size_t) N + cc]) = hi;
        }
}

}  // namespace v56

PLAYGROUND_MATMUL_DEC(float16_t, 56, m, n, k, A, B, C)
{
    static bool inited = false;
    constexpr int smemBytes =
        (v56::NSTAGES * (v56::AS_TILE + v56::BS_TILE)) * sizeof(half);
    if (!inited) {
        cudaFuncSetAttribute((const void*) v56::kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smemBytes);
        inited = true;
    }
    dim3 block(v56::NTHREADS), grid(n / v56::BN, m / v56::BM);
    v56::kernel<<<grid, block, smemBytes>>>(A, B, C, int(m), int(n), int(k));
}

}  // namespace playground
