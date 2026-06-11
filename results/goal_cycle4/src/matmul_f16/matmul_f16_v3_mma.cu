#include <cuda_fp16.h>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{
namespace v3
{
// Block tile.
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;

// Warp tile: 8 warps laid out 2 (M) x 4 (N).
constexpr int WARPS_M = 2;
constexpr int WARPS_N = 4;
constexpr int WARP_M = BM / WARPS_M;  // 64
constexpr int WARP_N = BN / WARPS_N;  // 32

// mma m16n8k16
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;
constexpr int MT = WARP_M / MMA_M;  // 4  (m-tiles per warp)
constexpr int NT = WARP_N / MMA_N;  // 4  (n-tiles per warp)

constexpr int THREADS = WARPS_M * WARPS_N * 32;  // 256
constexpr int APAD = 8;
constexpr int BPAD = 8;
constexpr int STAGES = 2;

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
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4],
                                    const uint32_t (&b)[2])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__global__ void __launch_bounds__(THREADS)
    kernel(int M, int N, int K, const half* __restrict__ A,
           const half* __restrict__ B, half* __restrict__ C)
{
    __shared__ half As[STAGES][BM][BK + APAD];
    __shared__ half Bs[STAGES][BK][BN + BPAD];

    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;
    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;
    const int warpRow = warpId / WARPS_N;  // 0..1
    const int warpCol = warpId % WARPS_N;  // 0..3
    const int tid = threadIdx.x;

    const int rowBase = blockRow * BM;
    const int colBase = blockCol * BN;

    float acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int t = 0; t < 4; ++t)
                acc[i][j][t] = 0.0f;

    const int lr = laneId & 15;          // local row for ldmatrix
    const int lc = (laneId & 16) >> 1;   // local col extra (0 or 8)

    auto load_stage = [&](int stage, int k0) {
#pragma unroll
        for (int it = 0; it < 2; ++it) {
            int f = tid + it * THREADS;
            int row = f / (BK / 8);
            int col = (f % (BK / 8)) * 8;
            cp_async_cg(&As[stage][row][col],
                        &A[(rowBase + row) * K + k0 + col]);
        }
#pragma unroll
        for (int it = 0; it < 2; ++it) {
            int f = tid + it * THREADS;
            int row = f / (BN / 8);
            int col = (f % (BN / 8)) * 8;
            cp_async_cg(&Bs[stage][row][col],
                        &B[(k0 + row) * N + colBase + col]);
        }
        cp_async_commit();
    };

    const int nK = K / BK;
    load_stage(0, 0);

    int cur = 0;
    for (int kt = 0; kt < nK; ++kt) {
        if (kt + 1 < nK)
            load_stage((cur + 1) % STAGES, (kt + 1) * BK);
        cp_async_wait < STAGES - 1 > ();
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; kk += MMA_K) {
            uint32_t a[MT][4];
            uint32_t b[NT][2];
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                int ar = warpRow * WARP_M + i * MMA_M;
                ldm_x4(a[i], &As[cur][ar + lr][kk + lc]);
            }
#pragma unroll
            for (int p = 0; p < NT / 2; ++p) {
                int nc0 = warpCol * WARP_N + p * 16;
                uint32_t r[4];
                ldm_x4_trans(r, &Bs[cur][kk + lr][nc0 + lc]);
                b[2 * p][0] = r[0];
                b[2 * p][1] = r[1];
                b[2 * p + 1][0] = r[2];
                b[2 * p + 1][1] = r[3];
            }
#pragma unroll
            for (int i = 0; i < MT; ++i)
#pragma unroll
                for (int j = 0; j < NT; ++j)
                    mma(acc[i][j], a[i], b[j]);
        }
        __syncthreads();
        cur = (cur + 1) % STAGES;
    }

    // Store. mma D fragment per thread: d0=(gid,tig*2), d1=(gid,tig*2+1),
    // d2=(gid+8,tig*2), d3=(gid+8,tig*2+1). gid=lane/4, tig=lane%4.
    const int gid = laneId >> 2;
    const int tig = laneId & 3;
#pragma unroll
    for (int i = 0; i < MT; ++i) {
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            int base_r = rowBase + warpRow * WARP_M + i * MMA_M;
            int base_c = colBase + warpCol * WARP_N + j * MMA_N;
            half2 lo = __floats2half2_rn(acc[i][j][0], acc[i][j][1]);
            half2 hi = __floats2half2_rn(acc[i][j][2], acc[i][j][3]);
            *reinterpret_cast<half2*>(&C[(base_r + gid) * N + base_c + tig * 2]) =
                lo;
            *reinterpret_cast<half2*>(
                &C[(base_r + gid + 8) * N + base_c + tig * 2]) = hi;
        }
    }
}
}  // namespace v3

PLAYGROUND_MATMUL_DEC(float16_t, 3, m, n, k, A, B, C)
{
    dim3 block(v3::THREADS);
    dim3 grid(n / v3::BN, m / v3::BM);
    v3::kernel<<<grid, block>>>((int) m, (int) n, (int) k, A, B, C);
}

}  // namespace playground
