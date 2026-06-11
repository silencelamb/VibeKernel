// v9: persistent hybrid stream-K.
//  Phase 1: first floor(tiles/G)*G tiles data-parallel, wave-strided
//           (template instance without any tail bookkeeping -> v8-level regs).
//  Phase 2: remaining tail tiles' k-units split evenly across blocks;
//           partials -> f32 workspace slots; last arriver (per-tile counter)
//           merges and writes C. Deterministic (no float atomics).
//  Segment mainloop = v8: BM128 BN256 BK32, 8 warps 64x64, f32 acc,
//  5-stage cp.async, split-issue, smem-staged epilogue for full tiles.
#include <cstdint>
#include <cuda_runtime.h>

#include "playground/matmul.hpp"

namespace playground
{
namespace v9
{

constexpr int BM = 128, BN = 256, BK = 32;
constexpr int STAGES = 5;
constexpr int GW = 8;  // panel width (n-tiles) for tile-order swizzle
constexpr uint32_t A_STAGE = BM * BK * sizeof(half);
constexpr uint32_t B_STAGE = BK * BN * sizeof(half);
constexpr uint32_t SMEM_BYTES = STAGES * (A_STAGE + B_STAGE);  // 120 KB
constexpr uint32_t EPI_STRIDE = BN + 8;
static_assert(128 * EPI_STRIDE * 2 <= SMEM_BYTES);

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

struct Geom {
    uint32_t sA, sB;
    int tid, lane, warp, wm, wn;
    int aRow, aChunk, bRow, bChunk;
    uint32_t aDst, bDst;
    uint32_t ldA[4], ldB[4];
};

// One (tile, k-range) segment. FULL_ONLY=true compiles out the partial path.
template <bool FULL_ONLY>
__device__ __forceinline__ void runSegment(
    const Geom& g, const half* __restrict__ A, const half* __restrict__ B,
    half* __restrict__ C, float* __restrict__ ws, unsigned* __restrict__ cnt,
    unsigned* sArrived, int M, int N, int K, int NT, int tM, int tN, int t,
    int k0, int k1, long base, long tailU, int G, int b)
{
    const int NTk = k1 - k0;
    // tile -> (bm, bn), GW-wide panel order
    int tileM, tileN;
    if (tN % GW == 0) {
        const int panelSize = GW * tM;
        const int p = t / panelSize;
        const int r = t % panelSize;
        tileN = p * GW + (r % GW);
        tileM = r / GW;
    } else {
        tileM = t / tN;
        tileN = t % tN;
    }
    const int bm = tileM * BM;
    const int bn = tileN * BN;

    const half* aSrc =
        A + (uint32_t)(bm + g.aRow) * K + (uint32_t)k0 * BK + g.aChunk * 8;
    const half* bSrc =
        B + (uint32_t)(k0 * BK + g.bRow) * N + bn + g.bChunk * 8;
    const uint32_t sA = g.sA, sB = g.sB;

    float acc[4][8][4] = {};
    uint32_t fragA[2][4][4];
    uint32_t fragB[2][4][4];

#define ISSUE_H1(S_, KT_)                                                     \
    do {                                                                      \
        const half* ag = aSrc + (KT_) * BK;                                   \
        cpAsync16(sA + (S_)*A_STAGE + g.aDst, ag);                            \
        cpAsync16(sA + (S_)*A_STAGE + g.aDst + 4096, ag + (uint32_t)64 * K);  \
        const half* bg = bSrc + (uint32_t)((KT_) * BK) * N;                   \
        cpAsync16(sB + (S_)*B_STAGE + g.bDst, bg);                            \
        cpAsync16(sB + (S_)*B_STAGE + g.bDst + 8 * 512,                       \
                  bg + (uint32_t)8 * N);                                      \
    } while (0)
#define ISSUE_H2(S_, KT_)                                                     \
    do {                                                                      \
        const half* bg = bSrc + (uint32_t)((KT_) * BK + 16) * N;              \
        cpAsync16(sB + (S_)*B_STAGE + g.bDst + 16 * 512, bg);                 \
        cpAsync16(sB + (S_)*B_STAGE + g.bDst + 24 * 512,                      \
                  bg + (uint32_t)8 * N);                                      \
    } while (0)
#define LOAD_FRAGS(BUF_, S_, KK_)                                            \
    do {                                                                      \
        _Pragma("unroll") for (int i = 0; i < 4; ++i)                         \
        {                                                                     \
            ldsmX4(fragA[BUF_][i],                                            \
                   sA + (S_)*A_STAGE + (g.ldA[i] ^ ((KK_) << 5)));            \
        }                                                                     \
        _Pragma("unroll") for (int j = 0; j < 4; ++j)                         \
        {                                                                     \
            ldsmX4T(fragB[BUF_][j],                                           \
                    sB + (S_)*B_STAGE + g.ldB[j] + (KK_)*8192);               \
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

    if (NTk >= 1) {
        ISSUE_H1(0, 0);
        ISSUE_H2(0, 0);
    }
    cpCommit();
    if (NTk >= 2) {
        ISSUE_H1(1, 1);
        ISSUE_H2(1, 1);
    }
    cpCommit();
    if (NTk >= 3) {
        ISSUE_H1(2, 2);
        ISSUE_H2(2, 2);
    }
    cpCommit();
    if (NTk >= 4) {
        ISSUE_H1(3, 3);
        ISSUE_H2(3, 3);
    }
    cpCommit();
    cpWait<STAGES - 2>();
    __syncthreads();
    LOAD_FRAGS(0, 0, 0);

#define TILE_ITER(KT_, RS_, NS_, WS_)                                         \
    do {                                                                      \
        if ((KT_) + 4 < NTk) {                                                \
            ISSUE_H1(WS_, (KT_) + 4);                                         \
        }                                                                     \
        LOAD_FRAGS(1, RS_, 1);                                                \
        MMA_STEP(0);                                                          \
        cpWait<STAGES - 3>();                                                 \
        __syncthreads();                                                      \
        LOAD_FRAGS(0, NS_, 0);                                                \
        if ((KT_) + 4 < NTk) {                                                \
            ISSUE_H2(WS_, (KT_) + 4);                                         \
        }                                                                     \
        cpCommit(); /* one commit per iter; empties pad tail */               \
        MMA_STEP(1);                                                          \
    } while (0)

    int kt = 0;
    for (; kt + 5 <= NTk; kt += 5) {
        TILE_ITER(kt + 0, 0, 1, 4);
        TILE_ITER(kt + 1, 1, 2, 0);
        TILE_ITER(kt + 2, 2, 3, 1);
        TILE_ITER(kt + 3, 3, 4, 2);
        TILE_ITER(kt + 4, 4, 0, 3);
    }
    if (kt < NTk) {
        TILE_ITER(kt, 0, 1, 4);
        if (kt + 1 < NTk) {
            TILE_ITER(kt + 1, 1, 2, 0);
        }
        if (kt + 2 < NTk) {
            TILE_ITER(kt + 2, 2, 3, 1);
        }
        if (kt + 3 < NTk) {
            TILE_ITER(kt + 3, 3, 4, 2);
        }
    }
#undef TILE_ITER
#undef ISSUE_H1
#undef ISSUE_H2
#undef LOAD_FRAGS
#undef MMA_STEP

    cpWait<0>();
    __syncthreads();  // pipeline smem free

    if (FULL_ONLY || NTk == NT) {
        // full tile: stage through smem, coalesced 16B stores
        const int r0e = g.wm * 64 + (g.lane >> 2);
        const int c0e = g.wn * 64 + (g.lane & 3) * 2;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
            for (int jj = 0; jj < 8; ++jj) {
                const float* c4 = acc[i][jj];
                const uint32_t a0 =
                    sA + ((r0e + i * 16) * EPI_STRIDE + c0e + jj * 8) * 2;
                stsHalf2(a0, __floats2half2_rn(c4[0], c4[1]));
                stsHalf2(a0 + 8 * EPI_STRIDE * 2,
                         __floats2half2_rn(c4[2], c4[3]));
            }
        }
        __syncthreads();
#pragma unroll
        for (int it = 0; it < 16; ++it) {
            const int row = it * 8 + g.warp;
            uint32_t v[4];
            ldsV4(v, sA + (row * EPI_STRIDE + g.lane * 8) * 2);
            *reinterpret_cast<uint4*>(
                &C[(uint32_t)(bm + row) * N + bn + g.lane * 8]) =
                *reinterpret_cast<uint4*>(v);
        }
        __syncthreads();  // smem reusable for next segment
    } else {
        // partial tile: write f32 partial to my slot; last arriver merges.
        const long myR0 = base + (long)b * tailU / G;
        const int tFirst = (int)(myR0 / NT);
        const int mySlot = (t == tFirst) ? 2 * b : 2 * b + 1;
        float* slot = ws + (long)mySlot * (BM * BN);
        const int r0e = g.wm * 64 + (g.lane >> 2);
        const int c0e = g.wn * 64 + (g.lane & 3) * 2;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
            for (int jj = 0; jj < 8; ++jj) {
                float* p = slot + (r0e + i * 16) * BN + c0e + jj * 8;
                __stcg(reinterpret_cast<float2*>(p),
                       make_float2(acc[i][jj][0], acc[i][jj][1]));
                __stcg(reinterpret_cast<float2*>(p + 8 * BN),
                       make_float2(acc[i][jj][2], acc[i][jj][3]));
            }
        }
        const long uF = (long)t * NT, uL = (long)(t + 1) * NT - 1;
        const int bf = (int)(((uF - base + 1) * G - 1) / tailU);
        const int bl = min(G - 1, (int)(((uL - base + 1) * G - 1) / tailU));
        const unsigned nContrib = (unsigned)(bl - bf + 1);
        __threadfence();  // every thread: make slot writes globally visible
        __syncthreads();  // all threads fenced before counting
        if (g.tid == 0) {
            *sArrived = atomicAdd(&cnt[t], 1U);
        }
        __syncthreads();
        if (*sArrived == nContrib - 1) {
            __threadfence();  // acquire side
#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int jj = 0; jj < 8; ++jj) {
                    float2 s01 = make_float2(acc[i][jj][0], acc[i][jj][1]);
                    float2 s23 = make_float2(acc[i][jj][2], acc[i][jj][3]);
                    for (int cb = bf; cb <= bl; ++cb) {
                        if (cb == b) {
                            continue;
                        }
                        const long cr0 = base + (long)cb * tailU / G;
                        const int ctF = (int)(cr0 / NT);
                        const int cSlot = (t == ctF) ? 2 * cb : 2 * cb + 1;
                        const float* sp = ws + (long)cSlot * (BM * BN) +
                                          (r0e + i * 16) * BN + c0e + jj * 8;
                        const float2 p01 =
                            __ldcg(reinterpret_cast<const float2*>(sp));
                        const float2 p23 =
                            __ldcg(reinterpret_cast<const float2*>(sp + 8 * BN));
                        s01.x += p01.x;
                        s01.y += p01.y;
                        s23.x += p23.x;
                        s23.y += p23.y;
                    }
                    const uint32_t cr =
                        (uint32_t)(bm + r0e + i * 16) * N + bn + c0e + jj * 8;
                    *reinterpret_cast<half2*>(&C[cr]) =
                        __floats2half2_rn(s01.x, s01.y);
                    *reinterpret_cast<half2*>(&C[cr + 8 * N]) =
                        __floats2half2_rn(s23.x, s23.y);
                }
            }
            __syncthreads();
            if (g.tid == 0) {
                cnt[t] = 0;  // reset for next launch
            }
        }
        __syncthreads();
    }
}

__device__ __forceinline__ void initGeom(Geom& g, uint8_t* smemRaw)
{
    g.sA = cvta(smemRaw);
    g.sB = g.sA + STAGES * A_STAGE;
    g.tid = threadIdx.x;
    g.lane = g.tid & 31;
    g.warp = g.tid >> 5;
    g.wm = g.warp >> 2;
    g.wn = g.warp & 3;
    g.aRow = g.tid >> 2;
    g.aChunk = g.tid & 3;
    g.aDst = g.aRow * 64 + ((g.aChunk ^ ((g.aRow >> 1) & 3)) << 4);
    g.bRow = g.warp;
    g.bChunk = g.lane;
    g.bDst = g.bRow * 512 +
             (((g.bChunk & 24) | ((g.bChunk & 7) ^ (g.bRow & 7))) << 4);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = g.wm * 64 + i * 16 + (g.lane & 15);
        const int c = g.lane >> 4;
        g.ldA[i] = row * 64 + ((c ^ ((row >> 1) & 3)) << 4);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int kr = g.lane & 15;
        const int c = g.wn * 8 + j * 2 + (g.lane >> 4);
        g.ldB[j] = kr * 512 + (((c & 24) | ((c & 7) ^ (kr & 7))) << 4);
    }
}

// Kernel A: data-parallel full tiles 0..dpTiles-1, one block per tile.
__global__ __launch_bounds__(256, 1) void hgemmV9dp(const half* __restrict__ A,
                                                    const half* __restrict__ B,
                                                    half* __restrict__ C,
                                                    int M, int N, int K)
{
    extern __shared__ __align__(128) uint8_t smemRaw[];
    __shared__ unsigned sArrived;
    Geom g;
    initGeom(g, smemRaw);
    const int NT = K / BK;
    const int tM = M / BM, tN = N / BN;
    runSegment<true>(g, A, B, C, nullptr, nullptr, &sArrived, M, N, K, NT, tM,
                     tN, (int)blockIdx.x, 0, NT, 0, 1, 1, 0);
}

// Kernel B: stream-K over tail tiles [dpTiles, numTiles).
__global__ __launch_bounds__(256, 1) void hgemmV9sk(
    const half* __restrict__ A, const half* __restrict__ B,
    half* __restrict__ C, float* __restrict__ ws, unsigned* __restrict__ cnt,
    int M, int N, int K, int dpTiles)
{
    extern __shared__ __align__(128) uint8_t smemRaw[];
    __shared__ unsigned sArrived;
    Geom g;
    initGeom(g, smemRaw);
    const int NT = K / BK;
    const int tM = M / BM, tN = N / BN;
    const int numTiles = tM * tN;
    const int G = gridDim.x;
    const int b = blockIdx.x;
    const long tailU = (long)(numTiles - dpTiles) * NT;
    const long base = (long)dpTiles * NT;
    long r0 = base + (long)b * tailU / G;
    const long r1 = base + (long)(b + 1) * tailU / G;
    while (r0 < r1) {
        const int t = (int)(r0 / NT);
        const int k0 = (int)(r0 % NT);
        const int k1 = (int)min((long)NT, r1 - (long)t * NT);
        runSegment<false>(g, A, B, C, ws, cnt, &sArrived, M, N, K, NT, tM, tN,
                          t, k0, k1, base, tailU, G, b);
        r0 = (long)t * NT + k1;
    }
}

}  // namespace v9

PLAYGROUND_MATMUL_DEC(float16_t, 9, m, n, k, A, B, C)
{
    static float* ws = nullptr;
    static unsigned* cnt = nullptr;
    static int G = 0;
    static size_t cntCap = 0;
    if (G == 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        G = prop.multiProcessorCount;
        cudaFuncSetAttribute(v9::hgemmV9dp,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             v9::SMEM_BYTES);
        cudaFuncSetAttribute(v9::hgemmV9sk,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             v9::SMEM_BYTES);
        cudaMalloc(&ws, (size_t)2 * G * v9::BM * v9::BN * sizeof(float));
    }
    const int numTiles = int((m / v9::BM) * (n / v9::BN));
    if ((size_t)numTiles > cntCap) {
        if (cnt) {
            cudaFree(cnt);
        }
        cudaMalloc(&cnt, numTiles * sizeof(unsigned));
        cudaMemset(cnt, 0, numTiles * sizeof(unsigned));
        cntCap = numTiles;
    }
    const int dpTiles = (numTiles / G) * G;
    if (dpTiles > 0) {
        v9::hgemmV9dp<<<dpTiles, 256, v9::SMEM_BYTES>>>(A, B, C, int(m),
                                                        int(n), int(k));
    }
    if (numTiles > dpTiles) {
        v9::hgemmV9sk<<<G, 256, v9::SMEM_BYTES>>>(
            A, B, C, ws, cnt, int(m), int(n), int(k), dpTiles);
    }
}

}  // namespace playground
