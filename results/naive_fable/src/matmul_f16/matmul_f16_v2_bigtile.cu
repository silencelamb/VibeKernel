// v2: bigger tiles to cut shared-memory traffic per FLOP.
//   - CTA tile 128x256, K-tile 32, 4-stage cp.async pipeline (96KB dyn smem)
//   - 8 warps (2x4), warp tile 64x64 (mma:ldmatrix ratio 4:1 vs 2.7:1 in v1)
//   - single __syncthreads per K-tile, grouped CTA rasterization for L2
#include <cuda_runtime.h>

#include "playground/matmul.hpp"

namespace playground
{

namespace v2
{

constexpr uint32_t BM = 128, BN = 256, BK = 32;
constexpr uint32_t STAGES = 4;
constexpr uint32_t MT = 4;  // 16-row mma tiles per warp (64 rows)
constexpr uint32_t NT = 8;  // 8-col mma tiles per warp (64 cols)

constexpr uint32_t ASTAGE = BM * BK * 2;            // 8KB
constexpr uint32_t BSTAGE = BK * BN * 2;            // 16KB
constexpr uint32_t SMEM_B = STAGES * (ASTAGE + BSTAGE);  // 96KB

// A smem [BM][BK]: rows of 4 chunks, 128B segment = 2 rows.
__device__ __forceinline__ uint32_t swzA(uint32_t r, uint32_t c)
{
    uint32_t seg = r >> 1;
    uint32_t p = ((r & 1) << 2) | c;
    return (seg << 7) | ((p ^ (seg & 7)) << 4);
}

// B smem [BK][BN]: rows of 32 chunks (4 segments of 8).
__device__ __forceinline__ uint32_t swzB(uint32_t r, uint32_t c)
{
    uint32_t cs = (c & 24) | ((c & 7) ^ (r & 7));
    return (r << 9) | (cs << 4);
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

__global__ void __launch_bounds__(256, 1) hgemmV2(uint32_t M, uint32_t N,
                                                  uint32_t K,
                                                  const half* __restrict__ A,
                                                  const half* __restrict__ B,
                                                  half* __restrict__ C)
{
    extern __shared__ __align__(128) uint8_t smem[];

    const uint32_t sABase =
        static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    const uint32_t sBBase = sABase + STAGES * ASTAGE;

    // Grouped rasterization: walk M-tiles fastest within a group of
    // GROUP_M rows so the ~108 concurrent CTAs touch a compact A/B set.
    const uint32_t gridM = M / BM, gridN = N / BN;
    constexpr uint32_t GROUP_M = 16;
    const uint32_t bid = blockIdx.x;
    const uint32_t groupSize = GROUP_M * gridN;
    const uint32_t group = bid / groupSize;
    const uint32_t inGroup = bid % groupSize;
    const uint32_t groupRows = min(GROUP_M, gridM - group * GROUP_M);
    const uint32_t bmIdx = group * GROUP_M + inGroup % groupRows;
    const uint32_t bnIdx = inGroup / groupRows;
    const uint32_t bm = bmIdx * BM;
    const uint32_t bn = bnIdx * BN;

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31;
    const uint32_t warp = tid >> 5;
    const uint32_t wm = warp >> 2;  // 0..1
    const uint32_t wn = warp & 3;   // 0..3

    // ---- cp.async addressing ----
    // A: 512 chunks/stage -> 2 per thread (rows tid/4, tid/4+64)
    const uint32_t aRow = tid >> 2, aCol = tid & 3;
    const half* gA = A + (bm + aRow) * K + aCol * 8;
    const uint32_t swA = sABase + swzA(aRow, aCol);
    // B: 1024 chunks/stage -> 4 per thread (rows tid/32 + {0,8,16,24})
    const uint32_t bRow = tid >> 5, bCol = tid & 31;
    const half* gB = B + bRow * N + bn + bCol * 8;
    const uint32_t swB = sBBase + swzB(bRow, bCol);

    auto loadStage = [&](uint32_t s, uint32_t kt) {
        const half* a = gA + kt * BK;
        const half* b = gB + kt * BK * N;
        CP_ASYNC_CG(swA + s * ASTAGE, a);
        CP_ASYNC_CG(swA + s * ASTAGE + 4096, a + 64 * K);
        uint32_t db = swB + s * BSTAGE;
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i) {
            CP_ASYNC_CG(db + i * 4096, b + i * 8 * N);
        }
    };

    const uint32_t nkt = K / BK;
#pragma unroll
    for (uint32_t s = 0; s < STAGES - 1; ++s) {
        loadStage(s, s);
        CP_ASYNC_COMMIT();
    }

    const uint32_t ldRow = (lane & 7) | (lane & 8);
    const uint32_t ldSel = lane >> 4;

    float acc[MT][NT][4] = {};

    for (uint32_t kt = 0; kt < nkt; ++kt) {
        CP_ASYNC_WAIT(STAGES - 2);
        __syncthreads();

        if (kt + STAGES - 1 < nkt) {
            loadStage((kt + STAGES - 1) % STAGES, kt + STAGES - 1);
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
                ldmatrixX4(af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                           sa + swzA(wm * 64 + mi * 16 + ldRow,
                                     ks * 2 + ldSel));
            }
#pragma unroll
            for (uint32_t nq = 0; nq < NT / 2; ++nq) {
                ldmatrixX4T(bf[2 * nq][0], bf[2 * nq][1], bf[2 * nq + 1][0],
                            bf[2 * nq + 1][1],
                            sb + swzB(ks * 16 + ldRow,
                                      wn * 8 + nq * 2 + ldSel));
            }
#pragma unroll
            for (uint32_t mi = 0; mi < MT; ++mi) {
#pragma unroll
                for (uint32_t ni = 0; ni < NT; ++ni) {
                    mma16816(acc[mi][ni], af[mi], bf[ni]);
                }
            }
        }
    }

    // ---- epilogue ----
    const uint32_t cRow0 = bm + wm * 64 + (lane >> 2);
    const uint32_t cCol0 = bn + wn * 64 + (lane & 3) * 2;
#pragma unroll
    for (uint32_t mi = 0; mi < MT; ++mi) {
#pragma unroll
        for (uint32_t ni = 0; ni < NT; ++ni) {
            uint32_t r = cRow0 + mi * 16;
            uint32_t c = cCol0 + ni * 8;
            __half2 lo = __float22half2_rn({acc[mi][ni][0], acc[mi][ni][1]});
            __half2 hi = __float22half2_rn({acc[mi][ni][2], acc[mi][ni][3]});
            *reinterpret_cast<__half2*>(C + r * N + c) = lo;
            *reinterpret_cast<__half2*>(C + (r + 8) * N + c) = hi;
        }
    }
}

}  // namespace v2

PLAYGROUND_MATMUL_DEC(float16_t, 2, m, n, k, A, B, C)
{
    static bool inited = false;
    if (!inited) {
        cudaFuncSetAttribute(v2::hgemmV2,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             v2::SMEM_B);
        inited = true;
    }
    const uint32_t grid =
        (static_cast<uint32_t>(m) / v2::BM) * (static_cast<uint32_t>(n) / v2::BN);
    v2::hgemmV2<<<grid, 256, v2::SMEM_B>>>(static_cast<uint32_t>(m),
                                           static_cast<uint32_t>(n),
                                           static_cast<uint32_t>(k), A, B, C);
}

}  // namespace playground
