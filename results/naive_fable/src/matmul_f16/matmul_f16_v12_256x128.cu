// v12: CUTLASS-favored tile shape probe — CTA 256x128, BK=32, 3 stages
// (72KB smem), 8 warps in 4x2, warp tile 64x64, fp16 accumulate.
// Same machinery as v9: sliced cp.async + rotating stage pointers + register
// frag double-buffer + padded-smem staged epilogue.
#include <cuda_runtime.h>

#include "playground/matmul.hpp"

namespace playground
{

namespace v12
{

constexpr uint32_t BM = 256, BN = 128, BK = 32;
constexpr uint32_t STAGES = 3;
constexpr uint32_t KSTEPS = BK / 16;  // 2
constexpr uint32_t MT = 4;            // warp tile 64x64
constexpr uint32_t NT = 8;

constexpr uint32_t ASTAGE = BM * BK * 2;                 // 16KB
constexpr uint32_t BSTAGE = BK * BN * 2;                 // 8KB
constexpr uint32_t SMEM_B = STAGES * (ASTAGE + BSTAGE);  // 72KB

// A smem [BM][BK=32]: row = 4 chunks; 128B segment = 2 rows.
__device__ __forceinline__ uint32_t swzA(uint32_t r, uint32_t c)
{
    uint32_t seg = r >> 1;
    uint32_t p = ((r & 1) << 2) | c;
    return (seg << 7) | ((p ^ (seg & 7)) << 4);
}

// B smem [BK=32][BN=128]: row = 16 chunks = 2 segments of 8.
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

__device__ __forceinline__ void mma16816(uint32_t* d, const uint32_t* a,
                                         const uint32_t* b)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(d[0]), "+r"(d[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

#define CP_ASYNC_CG(dst, src)                                                 \
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::    \
                     "r"(dst),                                                \
                 "l"(src))
#define CP_ASYNC_COMMIT() asm volatile("cp.async.commit_group;\n")
#define CP_ASYNC_WAIT(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))

struct Frags {
    uint32_t a[MT][4];
    uint32_t b[NT][2];
};

__global__ void __launch_bounds__(256, 1) hgemmV12(uint32_t M, uint32_t N,
                                                   uint32_t K,
                                                   const half* __restrict__ A,
                                                   const half* __restrict__ B,
                                                   half* __restrict__ C)
{
    extern __shared__ __align__(128) uint8_t smem[];

    const uint32_t sABase =
        static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    const uint32_t sBBase = sABase + STAGES * ASTAGE;

    const uint32_t gridM = M / BM, gridN = N / BN;
    constexpr uint32_t GROUP_M = 8;
    const uint32_t bid = blockIdx.x;
    const uint32_t groupSize = GROUP_M * gridN;
    const uint32_t group = bid / groupSize;
    const uint32_t inGroup = bid % groupSize;
    const uint32_t groupRows = min(GROUP_M, gridM - group * GROUP_M);
    const uint32_t bm = (group * GROUP_M + inGroup % groupRows) * BM;
    const uint32_t bn = (inGroup / groupRows) * BN;

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31;
    const uint32_t warp = tid >> 5;
    const uint32_t wm = warp >> 1;  // 0..3
    const uint32_t wn = warp & 1;   // 0..1

    // cp.async: A 1024 chunks/stage -> 4/thread (rows tid/4 + {0,64,128,192})
    const uint32_t aRow = tid >> 2, aCol = tid & 3;
    const half* gA = A + (bm + aRow) * (size_t)K + aCol * 8;
    const uint32_t swAoff = swzA(aRow, aCol);
    // B 512 chunks/stage -> 2/thread (rows tid/16 + {0,16})
    const uint32_t bRow = tid >> 4, bCol = tid & 15;
    const half* gB = B + bRow * (size_t)N + bn + bCol * 8;
    const uint32_t swBoff = swzB(bRow, bCol);

    auto loadStage = [&](uint32_t st, uint32_t kt) {
        const half* a = gA + kt * BK;
        const half* b = gB + kt * BK * (size_t)N;
        uint32_t da = sABase + st * ASTAGE + swAoff;
        uint32_t db = sBBase + st * BSTAGE + swBoff;
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i) {
            CP_ASYNC_CG(da + i * 4096, a + i * 64 * K);
        }
#pragma unroll
        for (uint32_t i = 0; i < 2; ++i) {
            CP_ASYNC_CG(db + i * 4096, b + i * 16 * N);
        }
    };

    // slice 0: A rows 0-127 + B half; slice 1: A rows 128-255 + B half
    auto loadSlice2 = [&](uint32_t saBase, uint32_t sbBase, uint32_t kt,
                          uint32_t slice) {
        const half* a = gA + kt * BK + slice * 128 * (size_t)K;
        uint32_t da = saBase + swAoff + slice * 8192;
#pragma unroll
        for (uint32_t i = 0; i < 2; ++i) {
            CP_ASYNC_CG(da + i * 4096, a + i * 64 * K);
        }
        const half* b =
            gB + kt * BK * (size_t)N + slice * 16 * (size_t)N;
        uint32_t db = sbBase + swBoff + slice * 4096;
        CP_ASYNC_CG(db, b);
    };

    const uint32_t nkt = K / BK;
    loadStage(0, 0);
    CP_ASYNC_COMMIT();
    loadStage(1, 1);
    CP_ASYNC_COMMIT();

    const uint32_t ldRow = (lane & 7) | (lane & 8);
    const uint32_t ldSel = lane >> 4;

    uint32_t aOff[KSTEPS];
#pragma unroll
    for (uint32_t ks = 0; ks < KSTEPS; ++ks) {
        aOff[ks] = swzA(wm * 64 + ldRow, ks * 2 + ldSel);
    }
    uint32_t bOff[NT / 2];
#pragma unroll
    for (uint32_t nq = 0; nq < NT / 2; ++nq) {
        bOff[nq] = swzB(ldRow, wn * 8 + nq * 2 + ldSel);
    }

    Frags fr[2];
    uint32_t acc[MT][NT][2] = {};

    auto ldFragsMix = [&](Frags& f, uint32_t saStage, uint32_t sbStage,
                          uint32_t ks) {
        ldmatrixX4T(f.b[0][0], f.b[0][1], f.b[1][0], f.b[1][1],
                    sbStage + bOff[0] + ks * (16 * 256));
        ldmatrixX4T(f.b[2][0], f.b[2][1], f.b[3][0], f.b[3][1],
                    sbStage + bOff[1] + ks * (16 * 256));
        ldmatrixX4(f.a[0][0], f.a[0][1], f.a[0][2], f.a[0][3],
                   saStage + aOff[ks]);
        ldmatrixX4(f.a[1][0], f.a[1][1], f.a[1][2], f.a[1][3],
                   saStage + aOff[ks] + 16 * 64);
        ldmatrixX4T(f.b[4][0], f.b[4][1], f.b[5][0], f.b[5][1],
                    sbStage + bOff[2] + ks * (16 * 256));
        ldmatrixX4T(f.b[6][0], f.b[6][1], f.b[7][0], f.b[7][1],
                    sbStage + bOff[3] + ks * (16 * 256));
        ldmatrixX4(f.a[2][0], f.a[2][1], f.a[2][2], f.a[2][3],
                   saStage + aOff[ks] + 2 * 16 * 64);
        ldmatrixX4(f.a[3][0], f.a[3][1], f.a[3][2], f.a[3][3],
                   saStage + aOff[ks] + 3 * 16 * 64);
    };
    auto dommaAll = [&](const Frags& f) {
#pragma unroll
        for (uint32_t mi = 0; mi < MT; ++mi) {
#pragma unroll
            for (uint32_t ni = 0; ni < NT; ++ni) {
                mma16816(acc[mi][ni], f.a[mi], f.b[ni]);
            }
        }
    };

    CP_ASYNC_WAIT(1);
    __syncthreads();
    ldFragsMix(fr[0], sABase, sBBase, 0);

    uint32_t sa = sABase, sb = sBBase;
    uint32_t saN = sABase + ASTAGE, sbN = sBBase + BSTAGE;
    uint32_t saL = sABase + 2 * ASTAGE, sbL = sBBase + 2 * BSTAGE;

    for (uint32_t kt = 0; kt < nkt; ++kt) {
#pragma unroll
        for (uint32_t ks = 0; ks < KSTEPS; ++ks) {
            if (ks < 2) {
                if (kt + 2 < nkt) {
                    loadSlice2(saL, sbL, kt + 2, ks);
                }
                if (ks == 1) {
                    CP_ASYNC_COMMIT();
                }
            }
            if (ks + 1 < KSTEPS) {
                ldFragsMix(fr[(ks + 1) & 1], sa, sb, ks + 1);
            } else if (kt + 1 < nkt) {
                CP_ASYNC_WAIT(1);
                __syncthreads();
                ldFragsMix(fr[0], saN, sbN, 0);
            }
            dommaAll(fr[ks & 1]);
        }
        const uint32_t ta = sa, tb = sb;
        sa = saN;
        sb = sbN;
        saN = saL;
        sbN = sbL;
        saL = ta;
        sbL = tb;
    }

    // ---- epilogue: staged through padded smem ----
    __syncthreads();
    half* sC = reinterpret_cast<half*>(smem);
    constexpr uint32_t CSTRIDE = BN + 8;  // 136 halves
    {
        const uint32_t cr = wm * 64 + (lane >> 2);
        const uint32_t cc = wn * 64 + (lane & 3) * 2;
#pragma unroll
        for (uint32_t mi = 0; mi < MT; ++mi) {
#pragma unroll
            for (uint32_t ni = 0; ni < NT; ++ni) {
                uint32_t r = cr + mi * 16;
                uint32_t c = cc + ni * 8;
                *reinterpret_cast<uint32_t*>(sC + r * CSTRIDE + c) =
                    acc[mi][ni][0];
                *reinterpret_cast<uint32_t*>(sC + (r + 8) * CSTRIDE + c) =
                    acc[mi][ni][1];
            }
        }
    }
    __syncthreads();
    const uint32_t srow = tid >> 3;
    const uint32_t schunk = tid & 7;
#pragma unroll
    for (uint32_t rr = 0; rr < 8; ++rr) {
        uint32_t r = srow + rr * 32;
#pragma unroll
        for (uint32_t h = 0; h < 2; ++h) {
            uint32_t c = (schunk + h * 8) * 8;
            uint4 v = *reinterpret_cast<const uint4*>(sC + r * CSTRIDE + c);
            *reinterpret_cast<uint4*>(C + (bm + r) * (size_t)N + bn + c) = v;
        }
    }
}

}  // namespace v12

PLAYGROUND_MATMUL_DEC(float16_t, 12, m, n, k, A, B, C)
{
    static bool inited = false;
    if (!inited) {
        cudaFuncSetAttribute(v12::hgemmV12,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             v12::SMEM_B);
        inited = true;
    }
    const uint32_t grid = (static_cast<uint32_t>(m) / v12::BM) *
                          (static_cast<uint32_t>(n) / v12::BN);
    v12::hgemmV12<<<grid, 256, v12::SMEM_B>>>(static_cast<uint32_t>(m),
                                              static_cast<uint32_t>(n),
                                              static_cast<uint32_t>(k), A, B,
                                              C);
}

}  // namespace playground
