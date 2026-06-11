#include <cuda_fp16.h>
#include <cstdint>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{
namespace v7
{
// Block 128x256, warp tile 64x64 (8 warps, 2x4). BK=32.
// gmem->smem: STAGES-deep cp.async pipeline.
// smem->reg : 2-deep register pipeline (double-buffered fragments).
constexpr int BM = 128;
constexpr int BN = 256;
constexpr int BK = 64;
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 4;
constexpr int N_WARPS = WARPS_M * WARPS_N;
constexpr int N_THREADS = N_WARPS * 32;  // 256
constexpr int WARP_M = BM / WARPS_M;     // 64
constexpr int WARP_N = BN / WARPS_N;     // 64
constexpr int MI = WARP_M / 16;          // 4
constexpr int NI = WARP_N / 8;           // 8
constexpr int KS = BK / 16;              // 4
constexpr int PAD = 8;
constexpr int AS_STRIDE = BK + PAD;   // 72
constexpr int BS_STRIDE = BN + PAD;   // 264
constexpr int STAGES = 3;

constexpr int A_TILE = BM * AS_STRIDE;
constexpr int B_TILE = BK * BS_STRIDE;
constexpr int SMEM_HALFS = STAGES * (A_TILE + B_TILE);

__device__ __forceinline__ uint32_t smem_addr(const void* ptr)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem)
{
    uint32_t s = smem_addr(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* smem)
{
    uint32_t a = smem_addr(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], const void* smem)
{
    uint32_t a = smem_addr(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void mma_m16n8k16(float (&d)[4], const uint32_t (&a)[4],
                                             const uint32_t (&b)[2])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__global__ void __launch_bounds__(N_THREADS)
    gemm(const half* __restrict__ A, const half* __restrict__ B,
         half* __restrict__ C, int M, int N, int K)
{
    extern __shared__ half smem[];
    half* As = smem;
    half* Bs = smem + STAGES * A_TILE;

    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;
    const int tid = threadIdx.x;
    const int warpId = tid >> 5;
    const int lane = tid & 31;
    const int Rb = (warpId / WARPS_N) * WARP_M;
    const int Cb = (warpId % WARPS_N) * WARP_N;
    const int groupID = lane >> 2;
    const int tig = lane & 3;

    float acc[MI][NI][4];
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < NI; ++j)
#pragma unroll
            for (int t = 0; t < 4; ++t) acc[i][j][t] = 0.0f;

    const int numTiles = K / BK;

    auto load_tile = [&](int kt, int buf) {
        int kk = kt * BK;
        half* Adst = As + buf * A_TILE;
        half* Bdst = Bs + buf * B_TILE;
        // A tile: BM x BK = 128 x 64 = 1024 float4 -> 4 per thread.
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int idx = i * N_THREADS + tid;
            int r = idx >> 3;          // /(BK/8)=8
            int c = (idx & 7) << 3;    // *8
            cp_async16(Adst + r * AS_STRIDE + c, A + (blockRow * BM + r) * K + kk + c);
        }
        // B tile: BK x BN = 64 x 256 = 2048 float4 -> 8 per thread.
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            int idx = i * N_THREADS + tid;
            int r = idx >> 5;          // /(BN/8)=32
            int c = (idx & 31) << 3;   // *8
            cp_async16(Bdst + r * BS_STRIDE + c, B + (kk + r) * N + blockCol * BN + c);
        }
    };

    // `kss` is the K-substep index (0..KS-1); the column/row offset is kss*16.
    auto ldA = [&](uint32_t (&af)[MI][4], int buf, int kss) {
        half* Acur = As + buf * A_TILE;
        int koff = kss * 16;
#pragma unroll
        for (int mi = 0; mi < MI; ++mi) {
            int row = Rb + mi * 16 + (lane & 15);
            int col = koff + ((lane >> 4) << 3);
            ldmatrix_x4(af[mi], Acur + row * AS_STRIDE + col);
        }
    };
    // One ldmatrix.x4.trans loads B for two adjacent n-tiles (16x16 of k x n).
    auto ldB = [&](uint32_t (&bf)[NI][2], int buf, int kss) {
        half* Bcur = Bs + buf * B_TILE;
        int row = kss * 16 + (lane & 15);
        int noff = (lane >> 4) << 3;
#pragma unroll
        for (int p = 0; p < NI / 2; ++p) {
            uint32_t t[4];
            ldmatrix_x4_trans(t, Bcur + row * BS_STRIDE + Cb + p * 16 + noff);
            bf[2 * p][0] = t[0];
            bf[2 * p][1] = t[1];
            bf[2 * p + 1][0] = t[2];
            bf[2 * p + 1][1] = t[3];
        }
    };

    // Prologue: prefetch STAGES-1 tiles.
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) { load_tile(s, s); cp_commit(); }
    cp_wait<STAGES - 2>();
    __syncthreads();

    uint32_t af[2][MI][4];
    uint32_t bf[2][NI][2];
    ldA(af[0], 0, 0);
    ldB(bf[0], 0, 0);

    for (int kt = 0; kt < numTiles; ++kt) {
        int cur_buf = kt % STAGES;
        // Prefetch a future tile (kt+STAGES-1) into its buffer.
        int lt = kt + STAGES - 1;
        if (lt < numTiles) load_tile(lt, lt % STAGES);
        cp_commit();

#pragma unroll
        for (int ks = 0; ks < KS; ++ks) {
            int c = (kt * KS + ks) & 1;
            int n = c ^ 1;
            if (ks + 1 < KS) {
                ldA(af[n], cur_buf, ks + 1);
                ldB(bf[n], cur_buf, ks + 1);
            } else if (kt + 1 < numTiles) {
                cp_wait<STAGES - 2>();
                __syncthreads();
                ldA(af[n], (kt + 1) % STAGES, 0);
                ldB(bf[n], (kt + 1) % STAGES, 0);
            }
#pragma unroll
            for (int mi = 0; mi < MI; ++mi)
#pragma unroll
                for (int nj = 0; nj < NI; ++nj)
                    mma_m16n8k16(acc[mi][nj], af[c][mi], bf[c][nj]);
        }
    }

    const int R0 = blockRow * BM + Rb;
    const int C0 = blockCol * BN + Cb;
#pragma unroll
    for (int mi = 0; mi < MI; ++mi) {
#pragma unroll
        for (int nj = 0; nj < NI; ++nj) {
            int r = R0 + mi * 16 + groupID;
            int c = C0 + nj * 8 + tig * 2;
            __half2 lo = __floats2half2_rn(acc[mi][nj][0], acc[mi][nj][1]);
            __half2 hi = __floats2half2_rn(acc[mi][nj][2], acc[mi][nj][3]);
            *reinterpret_cast<__half2*>(&C[r * N + c]) = lo;
            *reinterpret_cast<__half2*>(&C[(r + 8) * N + c]) = hi;
        }
    }
}

}  // namespace v7

PLAYGROUND_MATMUL_DEC(float16_t, 7, m, n, k, A, B, C)
{
    static bool configured = false;
    int smemBytes = v7::SMEM_HALFS * int(sizeof(half));
    if (!configured) {
        cudaFuncSetAttribute(v7::gemm, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             smemBytes);
        configured = true;
    }
    dim3 block(v7::N_THREADS);
    dim3 grid(n / v7::BN, m / v7::BM);
    v7::gemm<<<grid, block, smemBytes>>>(A, B, C, int(m), int(n), int(k));
}

}  // namespace playground
