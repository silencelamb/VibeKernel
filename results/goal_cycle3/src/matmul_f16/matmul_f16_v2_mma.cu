#include <cuda_fp16.h>
#include <cstdint>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{
namespace v2
{
// Block tile 128x128, K-step 32. 8 warps (256 threads).
// Warp layout 2(M) x 4(N) -> warp tile 64(M) x 32(N).
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 4;
constexpr int N_WARPS = WARPS_M * WARPS_N;  // 8
constexpr int N_THREADS = N_WARPS * 32;     // 256
constexpr int WARP_M = BM / WARPS_M;        // 64
constexpr int WARP_N = BN / WARPS_N;        // 32
constexpr int MI = WARP_M / 16;             // 4  (mma M=16)
constexpr int NI = WARP_N / 8;              // 4  (mma N=8)
constexpr int PAD = 8;                      // smem padding (halfs)
constexpr int AS_STRIDE = BK + PAD;         // 40
constexpr int BS_STRIDE = BN + PAD;         // 136

__device__ __forceinline__ uint32_t smem_addr(const void* ptr)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem)
{
    uint32_t s = smem_addr(smem);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }

template <int N>
__device__ __forceinline__ void cp_wait()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* smem)
{
    uint32_t a = smem_addr(smem);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(a));
}

__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t (&r)[2],
                                                  const void* smem)
{
    uint32_t a = smem_addr(smem);
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

__global__ void __launch_bounds__(N_THREADS)
    gemm(const half* __restrict__ A, const half* __restrict__ B,
         half* __restrict__ C, int M, int N, int K)
{
    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;
    const int tid = threadIdx.x;
    const int warpId = tid >> 5;
    const int lane = tid & 31;
    const int warpRow = warpId / WARPS_N;  // 0..1
    const int warpCol = warpId % WARPS_N;  // 0..3
    const int Rb = warpRow * WARP_M;       // 0 or 64
    const int Cb = warpCol * WARP_N;       // 0,32,64,96
    const int groupID = lane >> 2;
    const int tig = lane & 3;

    __shared__ half As[2][BM][AS_STRIDE];
    __shared__ half Bs[2][BK][BS_STRIDE];

    float acc[MI][NI][4];
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < NI; ++j)
#pragma unroll
            for (int t = 0; t < 4; ++t) acc[i][j][t] = 0.0f;

    const int numTiles = K / BK;

    // Async-load A[128x32] and B[32x128] for tile `kt` into buffer `buf`.
    auto load_tile = [&](int kt, int buf) {
        int kk = kt * BK;
        // A: 512 float4, 256 threads -> 2 each. idx = tid, tid+256.
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            int idx = i * N_THREADS + tid;  // 0..511
            int r = idx >> 2;               // /4 (BK/8=4)
            int c = (idx & 3) << 3;          // *8
            const half* src = A + (blockRow * BM + r) * K + kk + c;
            cp_async16(&As[buf][r][c], src);
        }
        // B: 512 float4 -> 2 each.
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            int idx = i * N_THREADS + tid;
            int r = idx >> 4;          // /16 (BN/8=16)
            int c = (idx & 15) << 3;   // *8
            const half* src = B + (kk + r) * N + blockCol * BN + c;
            cp_async16(&Bs[buf][r][c], src);
        }
    };

    load_tile(0, 0);
    cp_commit();

    for (int kt = 0; kt < numTiles; ++kt) {
        cp_wait<0>();
        __syncthreads();
        int cur = kt & 1;
        if (kt + 1 < numTiles) {
            load_tile(kt + 1, (kt + 1) & 1);
            cp_commit();
        }

        // Compute over BK in steps of 16
#pragma unroll
        for (int ks = 0; ks < BK; ks += 16) {
            uint32_t a[MI][4];
            uint32_t b[NI][2];
#pragma unroll
            for (int mi = 0; mi < MI; ++mi) {
                int row = Rb + mi * 16 + (lane & 15);
                int col = ks + ((lane >> 4) << 3);
                ldmatrix_x4(a[mi], &As[cur][row][col]);
            }
#pragma unroll
            for (int nj = 0; nj < NI; ++nj) {
                int tile = (lane & 15) >> 3;     // 0 or 1
                int krow = (lane & 7);
                int row = ks + tile * 8 + krow;
                int col = Cb + nj * 8;
                ldmatrix_x2_trans(b[nj], &Bs[cur][row][col]);
            }
#pragma unroll
            for (int mi = 0; mi < MI; ++mi)
#pragma unroll
                for (int nj = 0; nj < NI; ++nj)
                    mma_m16n8k16(acc[mi][nj], a[mi], b[nj]);
        }
    }

    // Store C. D-fragment layout: c0,c1 -> (groupID, tig*2+{0,1});
    //                              c2,c3 -> (groupID+8, tig*2+{0,1}).
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

}  // namespace v2

PLAYGROUND_MATMUL_DEC(float16_t, 2, m, n, k, A, B, C)
{
    dim3 block(v2::N_THREADS);
    dim3 grid(n / v2::BN, m / v2::BM);
    v2::gemm<<<grid, block>>>(A, B, C, int(m), int(n), int(k));
}

}  // namespace playground
