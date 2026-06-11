#include <cuda_fp16.h>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{
namespace v7
{
constexpr int BM = 128;
constexpr int BN = 256;
constexpr int BK = 32;

constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;

constexpr int WARPS_M = 2;
constexpr int WARPS_N = 4;
constexpr int NUM_WARPS = WARPS_M * WARPS_N;  // 8
constexpr int NUM_THREADS = NUM_WARPS * 32;   // 256

constexpr int WM = BM / WARPS_M;    // 64
constexpr int WN = BN / WARPS_N;    // 32
constexpr int WMITER = WM / MMA_M;  // 4
constexpr int WNITER = WN / MMA_N;  // 4
constexpr int KSUB = BK / MMA_K;    // 2

constexpr int SKEW = 8;
constexpr int STAGES = 4;
constexpr int WAITN = STAGES - 3;  // pending cp.async groups during compute
constexpr int As_stride = BK + SKEW;
constexpr int Bs_stride = BN + SKEW;
constexpr int As_tile = BM * As_stride;
constexpr int Bs_tile = BK * Bs_stride;
constexpr int SMEM_HALFS = STAGES * (As_tile + Bs_tile);

__device__ __forceinline__ uint32_t smem_u32(const void* ptr)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}
__device__ __forceinline__ void cp_async_cg(void* smem, const void* gmem)
{
    uint32_t s = smem_u32(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s),
                 "l"(gmem)
                 : "memory");
}
__device__ __forceinline__ void cp_async_commit()
{
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}
template <int N>
__device__ __forceinline__ void cp_async_wait()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1,
                                            uint32_t& r2, uint32_t& r3,
                                            uint32_t addr)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr)
        : "memory");
}
__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t& r0, uint32_t& r1,
                                                  uint32_t addr)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r0), "=r"(r1)
        : "r"(addr)
        : "memory");
}
__device__ __forceinline__ void mma_m16n8k16(float& c0, float& c1, float& c2,
                                             float& c3, uint32_t a0, uint32_t a1,
                                             uint32_t a2, uint32_t a3,
                                             uint32_t b0, uint32_t b1)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__global__ __launch_bounds__(NUM_THREADS) void kernel(
    int M, int N, int K, const half* __restrict__ A, const half* __restrict__ B,
    half* __restrict__ C)
{
    extern __shared__ half smem[];
    half* As = smem;
    half* Bs = smem + STAGES * As_tile;

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int warpId = tid / 32;
    const int lane = tid % 32;
    const int warpRow = warpId / WARPS_N;
    const int warpCol = warpId % WARPS_N;

    float acc[WMITER][WNITER][4];
#pragma unroll
    for (int i = 0; i < WMITER; ++i)
#pragma unroll
        for (int j = 0; j < WNITER; ++j)
#pragma unroll
            for (int t = 0; t < 4; ++t) acc[i][j][t] = 0.0f;

    auto loadA = [&](int stage, int k0) {
        half* dst = As + stage * As_tile;
#pragma unroll
        for (int l = tid; l < (BM * BK) / 8; l += NUM_THREADS) {
            int row = l / (BK / 8);
            int col = (l % (BK / 8)) * 8;
            cp_async_cg(&dst[row * As_stride + col],
                        &A[(blockRow + row) * K + k0 + col]);
        }
    };
    auto loadB = [&](int stage, int k0) {
        half* dst = Bs + stage * Bs_tile;
#pragma unroll
        for (int l = tid; l < (BK * BN) / 8; l += NUM_THREADS) {
            int row = l / (BN / 8);
            int col = (l % (BN / 8)) * 8;
            cp_async_cg(&dst[row * Bs_stride + col],
                        &B[(k0 + row) * N + blockCol + col]);
        }
    };

    // Register fragment double buffer.
    uint32_t aF[2][WMITER][4];
    uint32_t bF[2][WNITER][2];
    auto load_frag = [&](int slot, int stage, int ks) {
        const half* As_s = As + stage * As_tile;
        const half* Bs_s = Bs + stage * Bs_tile;
#pragma unroll
        for (int i = 0; i < WMITER; ++i) {
            int aRow = warpRow * WM + i * MMA_M + (lane % 16);
            int aCol = ks + (lane / 16) * 8;
            ldmatrix_x4(aF[slot][i][0], aF[slot][i][1], aF[slot][i][2],
                        aF[slot][i][3], smem_u32(&As_s[aRow * As_stride + aCol]));
        }
#pragma unroll
        for (int j = 0; j < WNITER; ++j) {
            int bRow = ks + (lane % 16);
            int bCol = warpCol * WN + j * MMA_N;
            ldmatrix_x2_trans(bF[slot][j][0], bF[slot][j][1],
                              smem_u32(&Bs_s[bRow * Bs_stride + bCol]));
        }
    };

    // Prologue: issue STAGES-1 tiles.
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        loadA(s, s * BK);
        loadB(s, s * BK);
        cp_async_commit();
    }
    cp_async_wait<WAITN>();
    __syncthreads();

    int read_stage = 0;
    int write_stage = STAGES - 1;
    int load_k = (STAGES - 1) * BK;
    const int num_k_tiles = K / BK;

    // preload first substep's fragments
    load_frag(0, 0, 0);
    int slot = 0;

    for (int kt = 0; kt < num_k_tiles; ++kt) {
        // issue global prefetch for a future tile
        if (load_k < K) {
            loadA(write_stage, load_k);
            loadB(write_stage, load_k);
            load_k += BK;
        }
        cp_async_commit();
        write_stage = (write_stage + 1) % STAGES;
        const int next_read = (read_stage + 1) % STAGES;

        // cross-boundary register double buffer (read_stage & next_read both
        // resident thanks to WAITN lookahead)
#pragma unroll
        for (int ksi = 0; ksi < KSUB; ++ksi) {
            int nstage = (ksi + 1 < KSUB) ? read_stage : next_read;
            int nks = (ksi + 1 < KSUB) ? (ksi + 1) * MMA_K : 0;
            load_frag(slot ^ 1, nstage, nks);
#pragma unroll
            for (int i = 0; i < WMITER; ++i)
#pragma unroll
                for (int j = 0; j < WNITER; ++j)
                    mma_m16n8k16(acc[i][j][0], acc[i][j][1], acc[i][j][2],
                                 acc[i][j][3], aF[slot][i][0], aF[slot][i][1],
                                 aF[slot][i][2], aF[slot][i][3], bF[slot][j][0],
                                 bF[slot][j][1]);
            slot ^= 1;
        }

        read_stage = next_read;
        cp_async_wait<WAITN>();
        __syncthreads();
    }

    const int gid = lane / 4;
    const int tg = lane % 4;
#pragma unroll
    for (int i = 0; i < WMITER; ++i) {
#pragma unroll
        for (int j = 0; j < WNITER; ++j) {
            int cRow = blockRow + warpRow * WM + i * MMA_M + gid;
            int cCol = blockCol + warpCol * WN + j * MMA_N + tg * 2;
            half2 lo = __floats2half2_rn(acc[i][j][0], acc[i][j][1]);
            half2 hi = __floats2half2_rn(acc[i][j][2], acc[i][j][3]);
            *reinterpret_cast<half2*>(&C[cRow * N + cCol]) = lo;
            *reinterpret_cast<half2*>(&C[(cRow + 8) * N + cCol]) = hi;
        }
    }
}
}  // namespace v7

PLAYGROUND_MATMUL_DEC(float16_t, 7, m, n, k, A, B, C)
{
    static bool init = false;
    if (!init) {
        cudaFuncSetAttribute(v7::kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             v7::SMEM_HALFS * (int) sizeof(half));
        init = true;
    }
    dim3 block(v7::NUM_THREADS);
    dim3 grid(n / v7::BN, m / v7::BM);
    v7::kernel<<<grid, block, v7::SMEM_HALFS * sizeof(half)>>>(
        int(m), int(n), int(k), A, B, C);
}

}  // namespace playground
