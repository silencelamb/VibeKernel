// v6: HYBRID stream-K on top of v5.
//   512 tiles = 432 data-parallel CTAs (4 full waves, grouped raster for L2)
//   + the 80-tile remainder spread evenly over 108 stream-K CTAs (~47 K-tile
//   units each) so the last wave is balanced instead of 80/108 quantized.
//   The DP path is instantiated WITHOUT the stream-K outer loop (template
//   lambda) so its inner-loop codegen stays identical to v5; only the 108
//   SK CTAs run the generic multi-tile loop.
//   Split tiles combine through per-CTA scratch slots + epoch flags. SK CTAs
//   are first in blockIdx order => co-resident in wave 1, and every CTA
//   writes its tail partial before any head wait => acyclic, ~zero spin.
#include <cstdlib>
#include <cuda_runtime.h>

#include "playground/matmul.hpp"

namespace playground
{

namespace v6
{

constexpr uint32_t BM = 128, BN = 256, BK = 64;
constexpr uint32_t STAGES = 3;
constexpr uint32_t KSTEPS = BK / 16;  // 4
constexpr uint32_t MT = 4;            // warp tile 64x64
constexpr uint32_t NT = 8;

constexpr uint32_t ASTAGE = BM * BK * 2;                 // 16KB
constexpr uint32_t BSTAGE = BK * BN * 2;                 // 32KB
constexpr uint32_t SMEM_B = STAGES * (ASTAGE + BSTAGE);  // 144KB

__device__ __forceinline__ uint32_t swzA(uint32_t r, uint32_t c)
{
    return (r << 7) | ((c ^ (r & 7)) << 4);
}

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

__device__ __forceinline__ uint32_t hadd2u(uint32_t a, uint32_t b)
{
    __half2 r = __hadd2(*reinterpret_cast<__half2*>(&a),
                        *reinterpret_cast<__half2*>(&b));
    return *reinterpret_cast<uint32_t*>(&r);
}

__global__ void __launch_bounds__(256, 1) hgemmV6(
    uint32_t M, uint32_t N, uint32_t K, const half* __restrict__ A,
    const half* __restrict__ B, half* __restrict__ C, half* scratch,
    uint32_t* flags, uint32_t epoch, uint32_t dpTiles)
{
    extern __shared__ __align__(128) uint8_t smem[];
    __shared__ uint32_t sPartners[8];
    __shared__ uint32_t sPCount;

    const uint32_t sABase =
        static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    const uint32_t sBBase = sABase + STAGES * ASTAGE;

    const uint32_t gridM = M / BM, gridN = N / BN;
    constexpr uint32_t GROUP_M = 16;
    const uint32_t groupSize = GROUP_M * gridN;

    const uint32_t nkt = K / BK;
    const uint32_t tiles = gridM * gridN;
    const uint32_t p = blockIdx.x;
    const uint32_t skCtas = gridDim.x - dpTiles;
    const uint32_t base = skCtas ? (tiles - dpTiles) * nkt / skCtas : 0;
    const uint32_t rem = skCtas ? (tiles - dpTiles) * nkt % skCtas : 0;
    const uint32_t dpEnd = dpTiles * nkt;

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31;
    const uint32_t warp = tid >> 5;
    const uint32_t wm = warp >> 2;
    const uint32_t wn = warp & 3;

    const uint32_t aRow = tid >> 3, aCol = tid & 7;
    const uint32_t swA = sABase + swzA(aRow, aCol);
    const uint32_t bRow = tid >> 5, bCol = tid & 31;
    const uint32_t swB = sBBase + swzB(bRow, bCol);

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

    // ---- one output tile (or a K-slice of one): mainloop + epilogue ----
    // The SK=false instantiation has no surrounding loop and fully-folded
    // epilogue branches -> v5-identical codegen for the DP CTAs.
    auto processTile = [&]<bool SK>(uint32_t t, uint32_t k0, uint32_t kEnd) {
        const uint32_t g = t / groupSize;
        const uint32_t ing = t % groupSize;
        const uint32_t rows = min(GROUP_M, gridM - g * GROUP_M);
        const uint32_t bm = (g * GROUP_M + ing % rows) * BM;
        const uint32_t bn = (ing / rows) * BN;

        const half* gA = A + (bm + aRow) * (size_t)K + aCol * 8;
        const half* gB = B + bRow * (size_t)N + bn + bCol * 8;

        auto loadStage = [&](uint32_t st, uint32_t kt) {
            const half* a = gA + kt * BK;
            const half* b = gB + kt * BK * (size_t)N;
            uint32_t da = swA + st * ASTAGE;
            uint32_t db = swB + st * BSTAGE;
#pragma unroll
            for (uint32_t i = 0; i < 4; ++i) {
                CP_ASYNC_CG(da + i * 4096, a + i * 32 * K);
            }
#pragma unroll
            for (uint32_t i = 0; i < 8; ++i) {
                CP_ASYNC_CG(db + i * 4096, b + i * 8 * N);
            }
        };
        auto loadSlice = [&](uint32_t st, uint32_t kt, uint32_t slice) {
            if (slice == 0) {
                const half* a = gA + kt * BK;
                uint32_t da = swA + st * ASTAGE;
#pragma unroll
                for (uint32_t i = 0; i < 4; ++i) {
                    CP_ASYNC_CG(da + i * 4096, a + i * 32 * K);
                }
            } else {
                const half* b =
                    gB + kt * BK * (size_t)N + (slice - 1) * 32 * (size_t)N;
                uint32_t db = swB + st * BSTAGE + (slice - 1) * 16384;
#pragma unroll
                for (uint32_t i = 0; i < 4; ++i) {
                    CP_ASYNC_CG(db + i * 4096, b + i * 8 * N);
                }
            }
        };

        Frags fr[2];
        uint32_t acc[MT][NT][2] = {};

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
                ldmatrixX4T(f.b[2 * nq][0], f.b[2 * nq][1],
                            f.b[2 * nq + 1][0], f.b[2 * nq + 1][1],
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

        if constexpr (SK) {
            __syncthreads();  // previous tile's epilogue may still read sC
        }

        const uint32_t L = kEnd - k0;
        loadStage(0, k0);
        CP_ASYNC_COMMIT();
        if (!SK || L > 1) {
            loadStage(1, k0 + 1);
        }
        CP_ASYNC_COMMIT();

        CP_ASYNC_WAIT(1);
        __syncthreads();
        ldFragsA(fr[0], sABase, 0);
        ldFragsB(fr[0], sBBase, 0);

        for (uint32_t lk = 0; lk < L; ++lk) {
            const uint32_t sa = sABase + (lk % STAGES) * ASTAGE;
            const uint32_t sb = sBBase + (lk % STAGES) * BSTAGE;
            const uint32_t saN = sABase + ((lk + 1) % STAGES) * ASTAGE;
            const uint32_t sbN = sBBase + ((lk + 1) % STAGES) * BSTAGE;

#pragma unroll
            for (uint32_t ks = 0; ks < KSTEPS; ++ks) {
                if (ks < 3) {
                    if (lk + 2 < L) {
                        loadSlice((lk + 2) % STAGES, k0 + lk + 2, ks);
                    }
                    if (ks == 2) {
                        CP_ASYNC_COMMIT();
                    }
                }
                if (ks + 1 < KSTEPS) {
                    ldFragsA(fr[(ks + 1) & 1], sa, ks + 1);
                    ldFragsB(fr[(ks + 1) & 1], sb, ks + 1);
                } else if (lk + 1 < L) {
                    CP_ASYNC_WAIT(1);
                    __syncthreads();
                    ldFragsA(fr[0], saN, 0);
                    ldFragsB(fr[0], sbN, 0);
                }
                domma(fr[ks & 1]);
            }
        }

        // ---- epilogue ----
        __syncthreads();
        half* sC = reinterpret_cast<half*>(smem);
        constexpr uint32_t CSTRIDE = BN + 8;
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

        const uint32_t srow = tid >> 4;
        const uint32_t schunk = tid & 15;
        const bool isTail = SK && (k0 > 0);
        const bool isHead = SK && (kEnd < nkt);

        if (isTail) {
            half* dst = scratch + (size_t)p * BM * BN;
#pragma unroll
            for (uint32_t rr = 0; rr < 8; ++rr) {
                uint32_t r = srow + rr * 16;
#pragma unroll
                for (uint32_t h = 0; h < 2; ++h) {
                    uint32_t c = (schunk + h * 16) * 8;
                    uint4 v =
                        *reinterpret_cast<const uint4*>(sC + r * CSTRIDE + c);
                    *reinterpret_cast<uint4*>(dst + r * BN + c) = v;
                }
            }
            __threadfence();
            __syncthreads();
            if (tid == 0) {
                atomicExch(&flags[p], epoch);
            }
        } else if (isHead) {
            if (tid == 0) {
                uint32_t cnt = 0;
                uint32_t su = (t - dpTiles) * nkt + kEnd;  // SK-space unit
                const uint32_t tileEndSu = (t - dpTiles + 1) * nkt;
                while (su < tileEndSu) {
                    uint32_t qq = (su >= rem * (base + 1))
                                      ? rem + (su - rem * (base + 1)) / base
                                      : su / (base + 1);
                    sPartners[cnt++] = qq;
                    su = (qq + 1) * base + min(qq + 1, rem);
                }
                sPCount = cnt;
                for (uint32_t i = 0; i < cnt; ++i) {
                    while (atomicAdd(&flags[sPartners[i]], 0u) != epoch) {
                    }
                }
            }
            __syncthreads();
            __threadfence();
            const uint32_t cnt = sPCount;
#pragma unroll
            for (uint32_t rr = 0; rr < 8; ++rr) {
                uint32_t r = srow + rr * 16;
#pragma unroll
                for (uint32_t h = 0; h < 2; ++h) {
                    uint32_t c = (schunk + h * 16) * 8;
                    uint4 acc4 =
                        *reinterpret_cast<const uint4*>(sC + r * CSTRIDE + c);
                    for (uint32_t i = 0; i < cnt; ++i) {
                        const half* psrc =
                            scratch + (size_t)sPartners[i] * BM * BN;
                        uint4 other = *reinterpret_cast<const uint4*>(
                            psrc + r * BN + c);
                        acc4.x = hadd2u(acc4.x, other.x);
                        acc4.y = hadd2u(acc4.y, other.y);
                        acc4.z = hadd2u(acc4.z, other.z);
                        acc4.w = hadd2u(acc4.w, other.w);
                    }
                    *reinterpret_cast<uint4*>(C + (bm + r) * (size_t)N + bn +
                                              c) = acc4;
                }
            }
        } else {
#pragma unroll
            for (uint32_t rr = 0; rr < 8; ++rr) {
                uint32_t r = srow + rr * 16;
#pragma unroll
                for (uint32_t h = 0; h < 2; ++h) {
                    uint32_t c = (schunk + h * 16) * 8;
                    uint4 v =
                        *reinterpret_cast<const uint4*>(sC + r * CSTRIDE + c);
                    *reinterpret_cast<uint4*>(C + (bm + r) * (size_t)N + bn +
                                              c) = v;
                }
            }
        }
    };

    if (p >= skCtas) {
        // data-parallel: exactly one full tile, loop-free instantiation
        processTile.template operator()<false>(p - skCtas, 0u, nkt);
    } else {
        uint32_t s = dpEnd + p * base + min(p, rem);
        const uint32_t e = s + base + (p < rem ? 1u : 0u);
        uint32_t t = s / nkt;
        uint32_t k0 = s - t * nkt;
        while (s < e) {
            const uint32_t kEnd = min(nkt, k0 + (e - s));
            processTile.template operator()<true>(t, k0, kEnd);
            s += kEnd - k0;
            ++t;
            k0 = 0;
        }
    }
}

}  // namespace v6

PLAYGROUND_MATMUL_DEC(float16_t, 6, m, n, k, A, B, C)
{
    static half* scratch = nullptr;
    static uint32_t* flags = nullptr;
    static uint32_t epoch = 0;
    static int nsm = 0;
    if (scratch == nullptr) {
        cudaFuncSetAttribute(v6::hgemmV6,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             v6::SMEM_B);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
        cudaMalloc(&scratch, (size_t)nsm * v6::BM * v6::BN * sizeof(half));
        cudaMalloc(&flags, (size_t)nsm * sizeof(uint32_t));
        cudaMemset(flags, 0, (size_t)nsm * sizeof(uint32_t));
    }
    ++epoch;
    const uint32_t tiles = (static_cast<uint32_t>(m) / v6::BM) *
                           (static_cast<uint32_t>(n) / v6::BN);
    const uint32_t R = getenv("V6_DP_ONLY") ? 0u : tiles % static_cast<uint32_t>(nsm);
    const uint32_t dpTiles = tiles - R;
    const uint32_t P = dpTiles + (R ? static_cast<uint32_t>(nsm) : 0);
    v6::hgemmV6<<<P, 256, v6::SMEM_B>>>(
        static_cast<uint32_t>(m), static_cast<uint32_t>(n),
        static_cast<uint32_t>(k), A, B, C, scratch, flags, epoch, dpTiles);
}

}  // namespace playground
