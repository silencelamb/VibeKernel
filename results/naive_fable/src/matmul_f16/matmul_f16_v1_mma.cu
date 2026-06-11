// v1: hand-written Ampere fp16 GEMM.
//   - CTA tile 128x128, K-tile 32, 3-stage cp.async pipeline (48KB smem)
//   - 8 warps (2x4), warp tile 64x32 -> mma.m16n8k16, fp32 accumulate
//   - XOR-swizzled shared memory, conflict-free for both cp.async stores
//     and ldmatrix loads
#include <cuda_runtime.h>

#include "playground/matmul.hpp"

namespace playground
{

namespace v1
{

constexpr uint32_t BM = 128, BN = 128, BK = 32;
constexpr uint32_t STAGES = 3;
constexpr uint32_t WARPS_M = 2, WARPS_N = 4;  // warp tile 64x32
constexpr uint32_t MT = 4;                    // 16-row mma tiles per warp
constexpr uint32_t NT = 4;                    // 8-col mma tiles per warp

constexpr uint32_t ASTAGE = BM * BK * 2;  // bytes per A stage (8KB)
constexpr uint32_t BSTAGE = BK * BN * 2;  // bytes per B stage (8KB)

// A smem: [BM][BK] halves; rows of 4 16B-chunks. 128B segment = 2 rows.
// Swizzle position-in-segment by segment index -> conflict-free ldmatrix
// column reads and cp.async row writes.
__device__ __forceinline__ uint32_t swzA(uint32_t r, uint32_t c)
{
    uint32_t seg = r >> 1;
    uint32_t p = ((r & 1) << 2) | c;
    return (seg << 7) | ((p ^ (seg & 7)) << 4);
}

// B smem: [BK][BN] halves; rows of 16 chunks (2 segments of 8).
__device__ __forceinline__ uint32_t swzB(uint32_t r, uint32_t c)
{
    uint32_t cs = (c & 8) | ((c & 7) ^ (r & 7));
    return (r << 8) | (cs << 4);
}

__device__ __forceinline__ void ldmatrixX4(uint32_t& r0, uint32_t& r1,
                                           uint32_t& r2, uint32_t& r3,
                                           uint32_t addr)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr));
}

__device__ __forceinline__ void ldmatrixX4T(uint32_t& r0, uint32_t& r1,
                                            uint32_t& r2, uint32_t& r3,
                                            uint32_t addr)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr));
}

__device__ __forceinline__ void mma16816(float* d, const uint32_t* a,
                                         const uint32_t* b)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

#define CP_ASYNC_CG(dst, src)                                                 \
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst),    \
                 "l"(src))
#define CP_ASYNC_COMMIT() asm volatile("cp.async.commit_group;\n")
#define CP_ASYNC_WAIT(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))

__global__ void __launch_bounds__(256) hgemmV1(uint32_t M, uint32_t N,
                                               uint32_t K,
                                               const half* __restrict__ A,
                                               const half* __restrict__ B,
                                               half* __restrict__ C)
{
    __shared__ __align__(128) uint8_t smem[STAGES * (ASTAGE + BSTAGE)];

    const uint32_t sABase =
        static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    const uint32_t sBBase = sABase + STAGES * ASTAGE;

    const uint32_t bm = blockIdx.y * BM;
    const uint32_t bn = blockIdx.x * BN;
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31;
    const uint32_t warp = tid >> 5;
    const uint32_t wm = warp >> 2;  // 0..1
    const uint32_t wn = warp & 3;   // 0..3

    // ---- cp.async source/dest for this thread ----
    // A: 512 chunks/stage, 2 per thread: rows tid/4 and tid/4+64, chunk tid%4
    const uint32_t aRow = tid >> 2, aCol = tid & 3;
    const half* gA = A + (bm + aRow) * K + aCol * 8;
    const uint32_t swA = sABase + swzA(aRow, aCol);  // +64 rows = +4096B
    // B: rows tid/16 and tid/16+16, chunk tid%16
    const uint32_t bRow = tid >> 4, bCol = tid & 15;
    const half* gB = B + bRow * N + bn + bCol * 8;
    const uint32_t swB = sBBase + swzB(bRow, bCol);  // +16 rows = +4096B

    auto loadStage = [&](uint32_t s, uint32_t kt) {
        const half* a = gA + kt * BK;
        const half* b = gB + kt * BK * N;
        CP_ASYNC_CG(swA + s * ASTAGE, a);
        CP_ASYNC_CG(swA + s * ASTAGE + 4096, a + 64 * K);
        CP_ASYNC_CG(swB + s * BSTAGE, b);
        CP_ASYNC_CG(swB + s * BSTAGE + 4096, b + 16 * N);
    };

    const uint32_t nkt = K / BK;
    loadStage(0, 0);
    CP_ASYNC_COMMIT();
    loadStage(1, 1);
    CP_ASYNC_COMMIT();

    // ---- ldmatrix lane addressing ----
    // x4: lanes 0-7 rows 0-7 (left), 8-15 rows 8-15 (left), 16-23 rows 0-7
    // (right), 24-31 rows 8-15 (right)
    const uint32_t ldRow = (lane & 7) | (lane & 8);  // row within 16-row tile
    const uint32_t ldSel = lane >> 4;                // 0: cols 0-7, 1: 8-15

    float acc[MT][NT][4] = {};

    for (uint32_t kt = 0; kt < nkt; ++kt) {
        CP_ASYNC_WAIT(1);
        __syncthreads();

        if (kt + 2 < nkt) {
            loadStage((kt + 2) % STAGES, kt + 2);
        }
        CP_ASYNC_COMMIT();

        const uint32_t sa = sABase + (kt % STAGES) * ASTAGE;
        const uint32_t sb = sBBase + (kt % STAGES) * BSTAGE;

#pragma unroll
        for (uint32_t ks = 0; ks < 2; ++ks) {
            uint32_t af[MT][4];
            uint32_t bf[NT][2];
#pragma unroll
            for (uint32_t mi = 0; mi < MT; ++mi) {
                uint32_t r = wm * 64 + mi * 16 + ldRow;
                uint32_t c = ks * 2 + ldSel;
                ldmatrixX4(af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                           sa + swzA(r, c));
            }
#pragma unroll
            for (uint32_t nq = 0; nq < NT / 2; ++nq) {
                uint32_t kr = ks * 16 + ldRow;
                uint32_t c = wn * 4 + nq * 2 + ldSel;
                ldmatrixX4T(bf[2 * nq][0], bf[2 * nq][1], bf[2 * nq + 1][0],
                            bf[2 * nq + 1][1], sb + swzB(kr, c));
            }
#pragma unroll
            for (uint32_t mi = 0; mi < MT; ++mi) {
#pragma unroll
                for (uint32_t ni = 0; ni < NT; ++ni) {
                    mma16816(acc[mi][ni], af[mi], bf[ni]);
                }
            }
        }
        __syncthreads();
    }

    // ---- epilogue: fp32 -> fp16, direct global stores (half2) ----
    const uint32_t cRow0 = bm + wm * 64 + (lane >> 2);
    const uint32_t cCol0 = bn + wn * 32 + (lane & 3) * 2;
#pragma unroll
    for (uint32_t mi = 0; mi < MT; ++mi) {
#pragma unroll
        for (uint32_t ni = 0; ni < NT; ++ni) {
            uint32_t r = cRow0 + mi * 16;
            uint32_t c = cCol0 + ni * 8;
            __half2 lo =
                __float22half2_rn({acc[mi][ni][0], acc[mi][ni][1]});
            __half2 hi =
                __float22half2_rn({acc[mi][ni][2], acc[mi][ni][3]});
            *reinterpret_cast<__half2*>(C + r * N + c) = lo;
            *reinterpret_cast<__half2*>(C + (r + 8) * N + c) = hi;
        }
    }
}

}  // namespace v1

PLAYGROUND_MATMUL_DEC(float16_t, 1, m, n, k, A, B, C)
{
    dim3 grid(static_cast<uint32_t>(n) / v1::BN,
              static_cast<uint32_t>(m) / v1::BM);
    v1::hgemmV1<<<grid, 256>>>(static_cast<uint32_t>(m),
                               static_cast<uint32_t>(n),
                               static_cast<uint32_t>(k), A, B, C);
}

}  // namespace playground
