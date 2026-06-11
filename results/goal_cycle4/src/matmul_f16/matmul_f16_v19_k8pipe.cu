#include <cuda_fp16.h>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{
namespace v19
{
constexpr int BM = 128;
constexpr int BN = 256;
constexpr int BK = 32;

constexpr int WARPS_M = 2;
constexpr int WARPS_N = 4;
constexpr int WARP_M = BM / WARPS_M;  // 64
constexpr int WARP_N = BN / WARPS_N;  // 64

constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 8;             // m16n8k8
constexpr int MT = WARP_M / MMA_M;   // 4
constexpr int NT = WARP_N / MMA_N;   // 8
constexpr int KK = BK / MMA_K;       // 4 k8-steps per stage

constexpr int THREADS = WARPS_M * WARPS_N * 32;  // 256
constexpr int APAD = 8;
constexpr int BPAD = 8;
constexpr int STAGES = 3;

constexpr int AS_STRIDE = BK + APAD;
constexpr int BS_STRIDE = BN + BPAD;
constexpr int AS_TILE = BM * AS_STRIDE;
constexpr int BS_TILE = BK * BS_STRIDE;
constexpr int SMEM_HALFS = STAGES * (AS_TILE + BS_TILE);

constexpr int A_PER = (BM * BK / 8) / THREADS;
constexpr int B_PER = (BK * BN / 8) / THREADS;

__device__ __forceinline__ unsigned smem_addr(const void* p)
{
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async_cg(void* smem, const void* gmem)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(
                     smem_addr(smem)),
                 "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit()
{
    asm volatile("cp.async.commit_group;\n");
}
template <int N>
__device__ __forceinline__ void cp_async_wait()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void ldm_x4(uint32_t (&r)[4], const half* p)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(smem_addr(p)));
}
__device__ __forceinline__ void ldm_x4_trans(uint32_t (&r)[4], const half* p)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(smem_addr(p)));
}
// m16n8k8 f16-accumulate: D(2 regs) = A(2 regs) * B(1 reg) + C(2 regs)
__device__ __forceinline__ void mma(uint32_t (&d)[2], const uint32_t (&a)[2],
                                    uint32_t b)
{
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3}, {%4}, {%0,%1};\n"
        : "+r"(d[0]), "+r"(d[1])
        : "r"(a[0]), "r"(a[1]), "r"(b));
}

__global__ void __launch_bounds__(THREADS, 2)
    kernel(int M, int N, int K, const half* __restrict__ A,
           const half* __restrict__ B, half* __restrict__ C)
{
    extern __shared__ half smem[];
    half* Asb = smem;
    half* Bsb = smem + STAGES * AS_TILE;

    auto As = [&](int s, int r, int c) -> half& {
        return Asb[s * AS_TILE + r * AS_STRIDE + c];
    };
    auto Bs = [&](int s, int r, int c) -> half& {
        return Bsb[s * BS_TILE + r * BS_STRIDE + c];
    };

    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;
    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;
    const int warpRow = warpId / WARPS_N;
    const int warpCol = warpId % WARPS_N;
    const int tid = threadIdx.x;

    const int rowBase = blockRow * BM;
    const int colBase = blockCol * BN;

    uint32_t acc[MT][NT][2];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            acc[i][j][0] = 0;
            acc[i][j][1] = 0;
        }

    const int warpArow = warpRow * WARP_M;
    const int warpBcol = warpCol * WARP_N;
    // ldmatrix addressing for k8:
    //  A x4: lane -> row offset (0..31), col = kk8 (K=8 wide)
    //  B x4.trans: lane -> row (kk8 + lane&7), col group (lane>>3)*8
    const int aRow = laneId;             // 0..31
    const int bRow = laneId & 7;         // 0..7
    const int bColG = (laneId >> 3) * 8;  // 0,8,16,24

    auto load_stage = [&](int stage, int k0) {
#pragma unroll
        for (int it = 0; it < A_PER; ++it) {
            int f = tid + it * THREADS;
            int row = f / (BK / 8);
            int col = (f % (BK / 8)) * 8;
            cp_async_cg(&As(stage, row, col),
                        &A[(rowBase + row) * K + k0 + col]);
        }
#pragma unroll
        for (int it = 0; it < B_PER; ++it) {
            int f = tid + it * THREADS;
            int row = f / (BN / 8);
            int col = (f % (BN / 8)) * 8;
            cp_async_cg(&Bs(stage, row, col),
                        &B[(k0 + row) * N + colBase + col]);
        }
        cp_async_commit();
    };

    auto load_frag = [&](int read, int kk8, uint32_t (&a)[MT][2],
                         uint32_t (&b)[NT][1]) {
        uint32_t rA[2][4];
#pragma unroll
        for (int g = 0; g < 2; ++g)
            ldm_x4(rA[g], &As(read, warpArow + g * 32 + aRow, kk8));
#pragma unroll
        for (int g = 0; g < 2; ++g) {
            a[g * 2 + 0][0] = rA[g][0];
            a[g * 2 + 0][1] = rA[g][1];
            a[g * 2 + 1][0] = rA[g][2];
            a[g * 2 + 1][1] = rA[g][3];
        }
        uint32_t rB[2][4];
#pragma unroll
        for (int g = 0; g < 2; ++g)
            ldm_x4_trans(rB[g], &Bs(read, kk8 + bRow, warpBcol + g * 32 + bColG));
#pragma unroll
        for (int g = 0; g < 2; ++g) {
            b[g * 4 + 0][0] = rB[g][0];
            b[g * 4 + 1][0] = rB[g][1];
            b[g * 4 + 2][0] = rB[g][2];
            b[g * 4 + 3][0] = rB[g][3];
        }
    };

    const int nK = K / BK;
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s)
        load_stage(s, s * BK);
    cp_async_wait<STAGES - 2>();
    __syncthreads();

    uint32_t af[2][MT][2];
    uint32_t bf[2][NT][1];
    load_frag(0, 0, af[0], bf[0]);

    int buf = 0;
    int read = 0;
    int write = STAGES - 1;
    for (int kt = 0; kt < nK; ++kt) {
        if (kt + STAGES - 1 < nK)
            load_stage(write, (kt + STAGES - 1) * BK);

#pragma unroll
        for (int ki = 0; ki < KK; ++ki) {
            bool last_ki = (ki == KK - 1);
            int n_read = last_ki ? (read + 1) % STAGES : read;
            int n_kk = last_ki ? 0 : (ki + 1) * MMA_K;
            bool has_next = !(kt == nK - 1 && last_ki);

            if (last_ki && kt + 1 < nK) {
                cp_async_wait<STAGES - 2>();
                __syncthreads();
            }
            if (has_next)
                load_frag(n_read, n_kk, af[buf ^ 1], bf[buf ^ 1]);

#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int j = 0; j < NT; ++j)
                    mma(acc[i][j], af[buf][i], bf[buf][j][0]);
            buf ^= 1;
        }
        read = (read + 1) % STAGES;
        write = (write + 1) % STAGES;
    }

    const int gid = laneId >> 2;
    const int tig = laneId & 3;
#pragma unroll
    for (int i = 0; i < MT; ++i) {
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            int base_r = rowBase + warpArow + i * MMA_M;
            int base_c = colBase + warpBcol + j * MMA_N;
            *reinterpret_cast<uint32_t*>(
                &C[(base_r + gid) * N + base_c + tig * 2]) = acc[i][j][0];
            *reinterpret_cast<uint32_t*>(
                &C[(base_r + gid + 8) * N + base_c + tig * 2]) = acc[i][j][1];
        }
    }
}
}  // namespace v19

PLAYGROUND_MATMUL_DEC(float16_t, 19, m, n, k, A, B, C)
{
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(v19::kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             v19::SMEM_HALFS * (int) sizeof(half));
        configured = true;
    }
    dim3 block(v19::THREADS);
    dim3 grid(n / v19::BN, m / v19::BM);
    v19::kernel<<<grid, block, v19::SMEM_HALFS * sizeof(half)>>>(
        (int) m, (int) n, (int) k, A, B, C);
}

}  // namespace playground
