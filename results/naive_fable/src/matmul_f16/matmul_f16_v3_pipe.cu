// v3: deeper software pipelining.
//   - CTA 128x256, K-tile 64 (4 mma k-steps per barrier), 3-stage cp.async
//     (144KB dynamic smem)
//   - register-level fragment double buffering: ldmatrix for step ks+1
//     overlaps mma of step ks; cross-tile handoff loads next tile's first
//     fragments right after the barrier
//   - one __syncthreads per 64-deep K-tile
#include <cuda_runtime.h>

#include "playground/matmul.hpp"

namespace playground
{

namespace v3
{

constexpr uint32_t BM = 128, BN = 256, BK = 64;
constexpr uint32_t STAGES = 3;
constexpr uint32_t KSTEPS = BK / 16;  // 4
constexpr uint32_t MT = 4;            // warp tile 64x64
constexpr uint32_t NT = 8;

constexpr uint32_t ASTAGE = BM * BK * 2;                 // 16KB
constexpr uint32_t BSTAGE = BK * BN * 2;                 // 32KB
constexpr uint32_t SMEM_B = STAGES * (ASTAGE + BSTAGE);  // 144KB

// A smem [BM][BK=64]: row = 8 chunks = one 128B segment.
__device__ __forceinline__ uint32_t swzA(uint32_t r, uint32_t c)
{
    return (r << 7) | ((c ^ (r & 7)) << 4);
}

// B smem [BK][BN=256]: row = 32 chunks = 4 segments of 8.
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

struct Frags {
    uint32_t a[MT][4];
    uint32_t b[NT][2];
};

__global__ void __launch_bounds__(256, 1) hgemmV3(uint32_t M, uint32_t N,
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
    constexpr uint32_t GROUP_M = 16;
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
    const uint32_t wm = warp >> 2;
    const uint32_t wn = warp & 3;

    // cp.async: A 1024 chunks -> 4/thread (rows tid/8 + {0,32,64,96})
    const uint32_t aRow = tid >> 3, aCol = tid & 7;
    const half* gA = A + (bm + aRow) * K + aCol * 8;
    const uint32_t swA = sABase + swzA(aRow, aCol);
    // B 2048 chunks -> 8/thread (rows tid/32 + {0,8,..,56})
    const uint32_t bRow = tid >> 5, bCol = tid & 31;
    const half* gB = B + bRow * N + bn + bCol * 8;
    const uint32_t swB = sBBase + swzB(bRow, bCol);

    auto loadStage = [&](uint32_t s, uint32_t kt) {
        const half* a = gA + kt * BK;
        const half* b = gB + kt * BK * N;
        uint32_t da = swA + s * ASTAGE;
        uint32_t db = swB + s * BSTAGE;
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i) {
            CP_ASYNC_CG(da + i * 4096, a + i * 32 * K);
        }
#pragma unroll
        for (uint32_t i = 0; i < 8; ++i) {
            CP_ASYNC_CG(db + i * 4096, b + i * 8 * N);
        }
    };

    const uint32_t nkt = K / BK;
    loadStage(0, 0);
    CP_ASYNC_COMMIT();
    loadStage(1, 1);
    CP_ASYNC_COMMIT();

    const uint32_t ldRow = (lane & 7) | (lane & 8);
    const uint32_t ldSel = lane >> 4;

    // Per-thread ldmatrix base offsets (within a stage), excluding stage base.
    // A tile (mi, ks): swzA(wm*64 + mi*16 + ldRow, ks*2 + ldSel)
    // B tile (nq, ks): swzB(ks*16 + ldRow, wn*8 + nq*2 + ldSel)
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
    float acc[MT][NT][4] = {};

    auto ldFragsA = [&](Frags& f, uint32_t saStage, uint32_t ks) {
#pragma unroll
        for (uint32_t mi = 0; mi < MT; ++mi) {
            ldmatrixX4(f.a[mi][0], f.a[mi][1], f.a[mi][2], f.a[mi][3],
                       saStage + aOff[ks] + mi * (16 * 128));
        }
    };
    auto ldFragsB = [&](Frags& f, uint32_t sbStage, uint32_t ks) {
#pragma unroll
        for (uint32_t nq = 0; nq < NT / 2; ++nq) {
            ldmatrixX4T(f.b[2 * nq][0], f.b[2 * nq][1], f.b[2 * nq + 1][0],
                        f.b[2 * nq + 1][1],
                        sbStage + bOff[nq] + ks * (16 * 512));
        }
    };
    auto domma = [&](const Frags& f) {
#pragma unroll
        for (uint32_t mi = 0; mi < MT; ++mi) {
#pragma unroll
            for (uint32_t ni = 0; ni < NT; ++ni) {
                mma16816(acc[mi][ni], f.a[mi], f.b[ni]);
            }
        }
    };

    // Prologue: stage 0 ready, load first fragments.
    CP_ASYNC_WAIT(1);
    __syncthreads();
    ldFragsA(fr[0], sABase, 0);
    ldFragsB(fr[0], sBBase, 0);

    for (uint32_t kt = 0; kt < nkt; ++kt) {
        const uint32_t sa = sABase + (kt % STAGES) * ASTAGE;
        const uint32_t sb = sBBase + (kt % STAGES) * BSTAGE;
        const uint32_t saN = sABase + ((kt + 1) % STAGES) * ASTAGE;
        const uint32_t sbN = sBBase + ((kt + 1) % STAGES) * BSTAGE;

#pragma unroll
        for (uint32_t ks = 0; ks < KSTEPS; ++ks) {
            if (ks == 0) {
                if (kt + 2 < nkt) {
                    loadStage((kt + 2) % STAGES, kt + 2);
                }
                CP_ASYNC_COMMIT();
            }
            if (ks + 1 < KSTEPS) {
                ldFragsA(fr[(ks + 1) & 1], sa, ks + 1);
                ldFragsB(fr[(ks + 1) & 1], sb, ks + 1);
            } else if (kt + 1 < nkt) {
                CP_ASYNC_WAIT(1);
                __syncthreads();
                ldFragsA(fr[0], saN, 0);
                ldFragsB(fr[0], sbN, 0);
            }
            domma(fr[ks & 1]);
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

}  // namespace v3

PLAYGROUND_MATMUL_DEC(float16_t, 3, m, n, k, A, B, C)
{
    static bool inited = false;
    if (!inited) {
        cudaFuncSetAttribute(v3::hgemmV3,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             v3::SMEM_B);
        inited = true;
    }
    const uint32_t grid = (static_cast<uint32_t>(m) / v3::BM) *
                          (static_cast<uint32_t>(n) / v3::BN);
    v3::hgemmV3<<<grid, 256, v3::SMEM_B>>>(static_cast<uint32_t>(m),
                                           static_cast<uint32_t>(n),
                                           static_cast<uint32_t>(k), A, B, C);
}

}  // namespace playground
