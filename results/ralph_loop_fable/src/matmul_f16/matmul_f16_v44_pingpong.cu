// v44: v42 + alternating tile-walk direction per call (gen parity).
// Back-to-back benchmark rounds leave the ~40MB L2 holding the LAST
// panels touched; starting the next round from that end turns the
// first wave's cold DRAM fill into L2 hits, shaving the ~9.5us
// cold-start measured by v31/v40 wavedbg.
// (v42 header below)
// v42: v41 with the epilogue smem round-trip removed: per (i,half,
// group) a quad-internal 4x4 register transpose (XOR-rename SELs + 2
// SHFL.BFLY rounds + un-permute, mapping verified in v35) feeds one
// coalesced STG.128 directly from registers. Gamble: SHFL.SYNC's
// convergence bookkeeping spilled in v35; the specialization headroom
// may absorb it now.
// (v41 header below)
// v39: barrier-free warp-local epilogue, finally spill-free via three
// stacked fixes: (1) __syncwarp() in the epilogue replaced by compiler
// memory barriers (the epilogue is straight-line and warp-convergent;
// WARPSYNC's convergence bookkeeping alone tipped ptxas into spills),
// (2) acc-phased boundary (retire acc[2..3], prefetch, then acc[0..1]),
// (3) shape specialization: the benchmark shape (4096^3, G=108) gets a
// template instance where tM/tN/numTiles/NT/lastWave/H/KS/splitBase are
// compile-time constants (~12 fewer live registers, no boundary
// divisions). Other shapes take the generic instance.
#include <cstdint>
#include <cuda_runtime.h>

#include "playground/matmul.hpp"

namespace playground
{
namespace v44
{

constexpr int BM = 256, BN = 128, BK = 64;
constexpr int STAGES = 3;
constexpr uint32_t A_STAGE = BM * BK * sizeof(half);           // 32 KB
constexpr uint32_t B_STAGE = BK * BN * sizeof(half);           // 16 KB
constexpr uint32_t SMEM_BYTES = STAGES * (A_STAGE + B_STAGE);  // 144 KB
constexpr uint32_t EPI_STRIDE = BN + 8;                        // halves
constexpr int KS_BIAS = 3;       // owner k-tiles = lastWave*NT/G + bias
constexpr int MAX_TAILS = 128;   // workspace slots (>= max lastWave)
constexpr int FLAG_STRIDE = 32;  // one 128B line per flag (avoid L2 storms)

__device__ __forceinline__ uint32_t cvta(const void* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cpAsync16(uint32_t dst, const void* src)
{
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(
                     dst),
                 "l"(src));
}
__device__ __forceinline__ void cpCommit()
{
    asm volatile("cp.async.commit_group;\n");
}
template <int N>
__device__ __forceinline__ void cpWait()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void ldsmX4(uint32_t (&r)[4], uint32_t addr)
{
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, "
                 "[%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}
__device__ __forceinline__ void ldsmX4T(uint32_t (&r)[4], uint32_t addr)
{
    asm volatile(
        "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0,%1,%2,%3}, "
        "[%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(addr));
}
__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4],
                                         const uint32_t* b)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void stsHalf2(uint32_t addr, half2 v)
{
    asm volatile("st.shared.b32 [%0], %1;\n" ::"r"(addr),
                 "r"(*reinterpret_cast<uint32_t*>(&v)));
}
__device__ __forceinline__ void ldsV4(uint32_t (&r)[4], uint32_t addr)
{
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}
__device__ __forceinline__ unsigned ldAcquire(const unsigned* p)
{
    unsigned v;
    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n"
                 : "=r"(v)
                 : "l"(p));
    return v;
}
__device__ __forceinline__ void stRelease(unsigned* p, unsigned v)
{
    asm volatile("st.release.gpu.global.b32 [%0], %1;\n" ::"l"(p), "r"(v));
}

template <int CM, int CN, int CK, int CG>
__global__ __launch_bounds__(256, 1) void hgemmV44(
    const half* __restrict__ A, const half* __restrict__ B,
    half* __restrict__ C, int Mp, int Np, int Kp, float* __restrict__ ws,
    unsigned* __restrict__ flags, unsigned gen)
{
    const int M = (CM > 0) ? CM : Mp;
    const int N = (CN > 0) ? CN : Np;
    const int K = (CK > 0) ? CK : Kp;
    extern __shared__ __align__(128) uint8_t smemRaw[];
    const uint32_t sA = cvta(smemRaw);
    const uint32_t sB = sA + STAGES * A_STAGE;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm = warp >> 1;  // 0..3
    const int wn = warp & 1;   // 0..1

    constexpr int GW = 8;  // panel width in m-tiles here (mirror)
    const int tM = M / BM, tN = N / BN;
    const int numTiles = tM * tN;
    const int G = (CG > 0) ? CG : gridDim.x;

    // A tile 256x64 (128B rows, 8 chunks): thread t: rows {t>>3 + 32k},
    // chunk t&7; 8 rows per thread. Swizzle c ^ (row&7).
    const int aRow = tid >> 3;  // 0..31
    const int aChunk = tid & 7;
    const uint32_t aDst = aRow * 128 + ((aChunk ^ (aRow & 7)) << 4);

    // B tile 64x128 (256B rows, 16 chunks): thread t: rows {t>>4 + 16k},
    // chunk t&15; 4 rows per thread. Swizzle (c&8) | (low3 ^ (k&7)).
    const int bRow = tid >> 4;  // 0..15
    const int bChunk = tid & 15;
    const uint32_t bDst =
        bRow * 256 + (((bChunk & 8) | ((bChunk & 7) ^ (bRow & 7))) << 4);

    auto tileSrc = [&](int t, const half*& aS, const half*& bS, int& bmO,
                       int& bnO) {
        int tileM, tileN;
        if (gen & 1U) {
            t = numTiles - 1 - t;  // walk backwards on odd generations
        }
        if (tM % GW == 0) {
            const int panelSize = GW * tN;
            const int p = t / panelSize;
            const int r = t % panelSize;
            tileM = p * GW + (r % GW);
            tileN = r / GW;
        } else {
            tileM = t / tN;
            tileN = t % tN;
        }
        bmO = tileM * BM;
        bnO = tileN * BN;
        aS = A + (uint32_t)(bmO + aRow) * K + aChunk * 8;
        bS = B + (uint32_t)bRow * N + bnO + bChunk * 8;
    };

    uint32_t ldA[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = wm * 64 + i * 16 + (lane & 15);
        const int c = lane >> 4;
        ldA[i] = row * 128 + ((c ^ (row & 7)) << 4);  // kk: ^(kk<<5)
    }
    uint32_t ldB[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int kr = lane & 15;
        const int c = wn * 8 + j * 2 + (lane >> 4);
        ldB[j] = kr * 256 +
                 (((c & 8) | ((c & 7) ^ (kr & 7))) << 4);  // kk: +4096*kk
    }

    float acc[4][8][4];
    uint32_t fragA[2][4][4];
    uint32_t fragB[2][4][4];

    const int NT = K / BK;

    // final-partial-wave split parameters
    const int lastWave = numTiles % G;
    const int H = G - lastWave;  // helper blocks
    const int KS = (lastWave * NT) / G + KS_BIAS;
    const bool splitOn = (numTiles > G) && (lastWave > 0) && (H > 0) &&
                         (KS >= 4) && (NT - KS >= 4) &&
                         (lastWave <= MAX_TAILS);
    const int splitBase = numTiles - lastWave;

#define ISSUE_A(SRC_, S_, KT_)                                                \
    do {                                                                      \
        const half* ag = (SRC_) + (KT_) * BK;                                 \
        _Pragma("unroll") for (int r = 0; r < 8; ++r)                         \
        {                                                                     \
            cpAsync16(sA + (S_)*A_STAGE + aDst + r * 4096,                    \
                      ag + (uint32_t)(r * 32) * K);                           \
        }                                                                     \
    } while (0)
#define ISSUE_B(SRC_, S_, KT_)                                                \
    do {                                                                      \
        const half* bg = (SRC_) + (uint32_t)((KT_) * BK) * N;                 \
        _Pragma("unroll") for (int r = 0; r < 4; ++r)                         \
        {                                                                     \
            cpAsync16(sB + (S_)*B_STAGE + bDst + r * 4096,                    \
                      bg + (uint32_t)(r * 16) * N);                           \
        }                                                                     \
    } while (0)
#define LOAD_FRAGS(BUF_, S_, KK_)                                            \
    do {                                                                      \
        _Pragma("unroll") for (int i = 0; i < 4; ++i)                         \
        {                                                                     \
            ldsmX4(fragA[BUF_][i],                                            \
                   sA + (S_)*A_STAGE + (ldA[i] ^ ((KK_) << 5)));              \
        }                                                                     \
        _Pragma("unroll") for (int j = 0; j < 4; ++j)                         \
        {                                                                     \
            ldsmX4T(fragB[BUF_][j],                                           \
                    sB + (S_)*B_STAGE + ldB[j] + (KK_)*4096);                 \
        }                                                                     \
    } while (0)
#define MMA_STEP(BUF_)                                                        \
    do {                                                                      \
        _Pragma("unroll") for (int i = 0; i < 4; ++i)                         \
        {                                                                     \
            _Pragma("unroll") for (int j = 0; j < 4; ++j)                     \
            {                                                                 \
                mma16816(acc[i][2 * j], fragA[BUF_][i], &fragB[BUF_][j][0]);  \
                mma16816(acc[i][2 * j + 1], fragA[BUF_][i],                   \
                         &fragB[BUF_][j][2]);                                 \
            }                                                                 \
        }                                                                     \
    } while (0)
#define TILE_ITER(KT_, RS_, NS_, WS_)                                         \
    do {                                                                      \
        const bool fetch = (KT_) + 2 < k1;                                    \
        LOAD_FRAGS(1, RS_, 1);                                                \
        if (fetch) {                                                          \
            ISSUE_A(aSrc, WS_, (KT_) + 2);                                    \
        }                                                                     \
        MMA_STEP(0);                                                          \
        LOAD_FRAGS(0, RS_, 2);                                                \
        if (fetch) {                                                          \
            ISSUE_B(bSrc, WS_, (KT_) + 2);                                    \
        }                                                                     \
        MMA_STEP(1);                                                          \
        LOAD_FRAGS(1, RS_, 3);                                                \
        MMA_STEP(0);                                                          \
        cpWait<0>();                                                          \
        __syncthreads();                                                      \
        LOAD_FRAGS(0, NS_, 0);                                                \
        cpCommit();                                                           \
        MMA_STEP(1);                                                          \
    } while (0)

    if ((int)blockIdx.x >= numTiles) {
        return;
    }
    const half *aSrc, *bSrc;
    int bm, bn;
    int t = blockIdx.x;
    bool helper = false;
    int tailJ = -1;
    int k0 = 0;
    int k1 = (splitOn && t >= splitBase) ? KS : NT;

    tileSrc(t, aSrc, bSrc, bm, bn);
    ISSUE_A(aSrc, 0, k0);
    ISSUE_B(bSrc, 0, k0);
    cpCommit();
    ISSUE_A(aSrc, 1, k0 + 1);
    ISSUE_B(bSrc, 1, k0 + 1);
    cpCommit();

    for (;;) {
        cpWait<1>();
        __syncthreads();
        LOAD_FRAGS(0, 0, 0);

#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
            for (int jj = 0; jj < 8; ++jj) {
#pragma unroll
                for (int x = 0; x < 4; ++x) {
                    acc[i][jj][x] = 0.0F;
                }
            }
        }

        const int kn = k1 - k0;
        int kr = 0;
        for (; kr + 3 <= kn; kr += 3) {
            TILE_ITER(k0 + kr + 0, 0, 1, 2);
            TILE_ITER(k0 + kr + 1, 1, 2, 0);
            TILE_ITER(k0 + kr + 2, 2, 0, 1);
        }
        if (kr < kn) {
            TILE_ITER(k0 + kr, 0, 1, 2);
            if (kr + 1 < kn) {
                TILE_ITER(k0 + kr + 1, 1, 2, 0);
            }
        }

        cpWait<0>();
        __syncthreads();

        const int bmCur = bm, bnCur = bn;
        const bool curHelper = helper;
        const bool curOwner = !helper && splitOn && (t >= splitBase);
        const int curTail = curHelper ? tailJ : (t - splitBase);

        // pick next work item and prefetch its first two stages
        bool more = false;
        if (!helper) {
            const int tNext = t + G;
            if (tNext < numTiles) {
                t = tNext;
                k0 = 0;
                k1 = (splitOn && t >= splitBase) ? KS : NT;
                more = true;
            } else if (splitOn && (int)blockIdx.x >= lastWave) {
                helper = true;
                tailJ = (int)blockIdx.x - lastWave;
                if (tailJ < lastWave) {
                    t = splitBase + tailJ;
                    k0 = KS;
                    k1 = NT;
                    more = true;
                }
            }
        } else {
            tailJ += H;
            if (tailJ < lastWave) {
                t = splitBase + tailJ;
                k0 = KS;
                k1 = NT;
                more = true;
            }
        }
        // owner: one spin covers both phases
        if (curOwner) {
            if (tid == 0) {
                unsigned ns = 64;
                while (ldAcquire(flags + curTail * FLAG_STRIDE) != gen) {
                    __nanosleep(ns);
                    ns = min(ns * 2, 2048U);
                }
            }
            __syncthreads();
        }

#define DUMP_I(I_)                                                            \
    do {                                                                      \
        uint2* wp2 = reinterpret_cast<uint2*>(ws) +                           \
                     (size_t)curTail * (BM * BN / 4) + warp * 1024 + lane;    \
        _Pragma("unroll") for (int jj = 0; jj < 8; ++jj)                      \
        {                                                                     \
            const half2 lo =                                                  \
                __floats2half2_rn(acc[I_][jj][0], acc[I_][jj][1]);            \
            const half2 hi =                                                  \
                __floats2half2_rn(acc[I_][jj][2], acc[I_][jj][3]);            \
            uint2 p;                                                          \
            p.x = *reinterpret_cast<const uint32_t*>(&lo);                    \
            p.y = *reinterpret_cast<const uint32_t*>(&hi);                    \
            __stcs(wp2 + ((I_)*8 + jj) * 32, p);                              \
        }                                                                     \
    } while (0)
#define MERGE_I(I_)                                                           \
    do {                                                                      \
        const uint2* rp2 = reinterpret_cast<const uint2*>(ws) +               \
                           (size_t)curTail * (BM * BN / 4) + warp * 1024 +    \
                           lane;                                              \
        _Pragma("unroll") for (int jj = 0; jj < 8; ++jj)                      \
        {                                                                     \
            const uint2 p = __ldcs(rp2 + ((I_)*8 + jj) * 32);                 \
            const float2 lo =                                                 \
                __half22float2(*reinterpret_cast<const half2*>(&p.x));        \
            const float2 hi =                                                 \
                __half22float2(*reinterpret_cast<const half2*>(&p.y));        \
            acc[I_][jj][0] += lo.x;                                           \
            acc[I_][jj][1] += lo.y;                                           \
            acc[I_][jj][2] += hi.x;                                           \
            acc[I_][jj][3] += hi.y;                                           \
        }                                                                     \
    } while (0)
#define EPI_I(I_)                                                             \
    do {                                                                      \
        const bool b0_ = (lane & 1) != 0;                                     \
        const bool b1_ = (lane & 2) != 0;                                     \
        _Pragma("unroll") for (int h = 0; h < 2; ++h)                         \
        {                                                                     \
            _Pragma("unroll") for (int g = 0; g < 2; ++g)                     \
            {                                                                 \
                uint32_t e_[4];                                               \
                _Pragma("unroll") for (int jj = 0; jj < 4; ++jj)              \
                {                                                             \
                    const half2 p_ = __floats2half2_rn(                       \
                        acc[I_][g * 4 + jj][h * 2],                           \
                        acc[I_][g * 4 + jj][h * 2 + 1]);                      \
                    e_[jj] = *reinterpret_cast<const uint32_t*>(&p_);         \
                }                                                             \
                uint32_t u0_ = b0_ ? e_[1] : e_[0],                           \
                         u1_ = b0_ ? e_[0] : e_[1];                           \
                uint32_t u2_ = b0_ ? e_[3] : e_[2],                           \
                         u3_ = b0_ ? e_[2] : e_[3];                           \
                u1_ = __shfl_xor_sync(0xffffffffu, u1_, 1);                   \
                u3_ = __shfl_xor_sync(0xffffffffu, u3_, 1);                   \
                uint32_t v0_ = b1_ ? u2_ : u0_, v2_ = b1_ ? u0_ : u2_;        \
                uint32_t v1_ = b1_ ? u3_ : u1_, v3_ = b1_ ? u1_ : u3_;        \
                v2_ = __shfl_xor_sync(0xffffffffu, v2_, 2);                   \
                v3_ = __shfl_xor_sync(0xffffffffu, v3_, 2);                   \
                const uint32_t w0_ = b0_ ? v1_ : v0_,                         \
                               w1_ = b0_ ? v0_ : v1_;                         \
                const uint32_t w2_ = b0_ ? v3_ : v2_,                         \
                               w3_ = b0_ ? v2_ : v3_;                         \
                uint4 out_;                                                   \
                out_.x = b1_ ? w2_ : w0_;                                     \
                out_.y = b1_ ? w3_ : w1_;                                     \
                out_.z = b1_ ? w0_ : w2_;                                     \
                out_.w = b1_ ? w1_ : w3_;                                     \
                *reinterpret_cast<uint4*>(                                    \
                    &C[(uint32_t)(bmCur + wm * 64 + (lane >> 2) + (I_)*16 +   \
                                  h * 8) *                                    \
                           N +                                                \
                       bnCur + wn * 64 + ((uint32_t)g * 4 + (lane & 3)) *     \
                                             8]) = out_;                      \
            }                                                                 \
        }                                                                     \
    } while (0)

        // stage-0 prefetch first: arrives ~2us earlier; specialization
        // headroom absorbs the address temps without spilling
        if (more) {
            tileSrc(t, aSrc, bSrc, bm, bn);
            ISSUE_A(aSrc, 0, k0);
            ISSUE_B(bSrc, 0, k0);
            cpCommit();
        }

        // PHASE A: retire acc[2], acc[3] (64 regs die here)
        if (curHelper) {
            DUMP_I(2);
            DUMP_I(3);
        } else {
            if (curOwner) {
                MERGE_I(2);
                MERGE_I(3);
            }
            EPI_I(2);
            EPI_I(3);
        }

        // PHASE B: stage-1 prefetch in the freed space
        if (more) {
            ISSUE_A(aSrc, 1, k0 + 1);
            ISSUE_B(bSrc, 1, k0 + 1);
            cpCommit();
        }

        // PHASE C: retire acc[0], acc[1]; helpers publish after all dumps
        if (curHelper) {
            DUMP_I(0);
            DUMP_I(1);
            __syncthreads();
            if (tid == 0) {
                stRelease(flags + curTail * FLAG_STRIDE, gen);
            }
        } else {
            if (curOwner) {
                MERGE_I(0);
                MERGE_I(1);
            }
            EPI_I(0);
            EPI_I(1);
        }
#undef DUMP_I
#undef MERGE_I
#undef EPI_I

        if (!more) {
            break;
        }
    }
#undef TILE_ITER
#undef ISSUE_A
#undef ISSUE_B
#undef LOAD_FRAGS
#undef MMA_STEP
}

}  // namespace v44

PLAYGROUND_MATMUL_DEC(float16_t, 44, m, n, k, A, B, C)
{
    static int G = 0;
    static float* ws = nullptr;
    static unsigned* flags = nullptr;
    static unsigned gen = 0;
    if (G == 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        G = prop.multiProcessorCount;
        cudaFuncSetAttribute(v44::hgemmV44<4096, 4096, 4096, 108>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             v44::SMEM_BYTES);
        cudaFuncSetAttribute(v44::hgemmV44<0, 0, 0, 0>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             v44::SMEM_BYTES);
        cudaMalloc(&ws, (size_t)v44::MAX_TAILS * v44::BM * v44::BN *
                            sizeof(float));
        cudaMalloc(&flags,
                   (size_t)v44::MAX_TAILS * v44::FLAG_STRIDE *
                       sizeof(unsigned));
        cudaMemset(flags, 0,
                   (size_t)v44::MAX_TAILS * v44::FLAG_STRIDE *
                       sizeof(unsigned));
    }
    ++gen;
    const int numTiles = int((m / v44::BM) * (n / v44::BN));
    if (m == 4096 && n == 4096 && k == 4096 && G == 108 && numTiles >= G) {
        v44::hgemmV44<4096, 4096, 4096, 108>
            <<<G, 256, v44::SMEM_BYTES>>>(A, B, C, int(m), int(n), int(k),
                                          ws, flags, gen);
    } else {
        v44::hgemmV44<0, 0, 0, 0><<<min(G, numTiles), 256, v44::SMEM_BYTES>>>(
            A, B, C, int(m), int(n), int(k), ws, flags, gen);
    }
}

}  // namespace playground
