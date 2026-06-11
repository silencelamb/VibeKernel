// v8: v4 skeleton (BM128 BN256 BK32, 8 warps 64x64, f32 acc) +
//  - 5-stage cp.async pipeline (120KB smem)
//  - copies split across both k16 steps (smooth LSU/LDGSTS pressure)
//  - epilogue staged through smem: fully-coalesced 512B/warp global stores
#include <cstdint>
#include <cuda_runtime.h>

#include "playground/matmul.hpp"

namespace playground
{
namespace v8
{

constexpr int BM = 128, BN = 256, BK = 32;
constexpr int STAGES = 5;
constexpr uint32_t A_STAGE = BM * BK * sizeof(half);            // 8 KB
constexpr uint32_t B_STAGE = BK * BN * sizeof(half);            // 16 KB
constexpr uint32_t SMEM_BYTES = STAGES * (A_STAGE + B_STAGE);   // 120 KB
// epilogue staging: 128 rows x (256+8) halves, padded for bank rotation
constexpr uint32_t EPI_STRIDE = BN + 8;  // halves
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

__global__ __launch_bounds__(256, 1) void hgemmV8(const half* __restrict__ A,
                                                  const half* __restrict__ B,
                                                  half* __restrict__ C, int M,
                                                  int N, int K)
{
    extern __shared__ __align__(128) uint8_t smemRaw[];
    const uint32_t sA = cvta(smemRaw);
    const uint32_t sB = sA + STAGES * A_STAGE;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm = warp >> 2;  // 0..1
    const int wn = warp & 3;   // 0..3

    // Panel swizzle (GW=8) for L2 locality.
    constexpr int GW = 8;
    int tileM, tileN;
    if (gridDim.x % GW == 0) {
        const int bid = blockIdx.y * gridDim.x + blockIdx.x;
        const int panelSize = GW * gridDim.y;
        const int p = bid / panelSize;
        const int r = bid % panelSize;
        tileN = p * GW + (r % GW);
        tileM = r / GW;
    } else {
        tileM = blockIdx.y;
        tileN = blockIdx.x;
    }
    const int bm = tileM * BM;
    const int bn = tileN * BN;

    const int aRow = tid >> 2;
    const int aChunk = tid & 3;
    const half* aSrc = A + (uint32_t)(bm + aRow) * K + aChunk * 8;
    const uint32_t aDst = aRow * 64 + ((aChunk ^ ((aRow >> 1) & 3)) << 4);

    const int bRow = warp;
    const int bChunk = lane;
    const half* bSrc = B + (uint32_t)bRow * N + bn + bChunk * 8;
    const uint32_t bDst =
        bRow * 512 + (((bChunk & 24) | ((bChunk & 7) ^ (bRow & 7))) << 4);

    uint32_t ldA[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = wm * 64 + i * 16 + (lane & 15);
        const int c = lane >> 4;
        ldA[i] = row * 64 + ((c ^ ((row >> 1) & 3)) << 4);
    }
    uint32_t ldB[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int kr = lane & 15;
        const int c = wn * 8 + j * 2 + (lane >> 4);
        ldB[j] = kr * 512 + (((c & 24) | ((c & 7) ^ (kr & 7))) << 4);
    }

    float acc[4][8][4] = {};
    uint32_t fragA[2][4][4];
    uint32_t fragB[2][4][4];

    const int NT = K / BK;

// First half of a stage's copies: A (2) + B rows {w, w+8} (2).
#define ISSUE_H1(S_, KT_)                                                     \
    do {                                                                      \
        const half* ag = aSrc + (KT_) * BK;                                   \
        cpAsync16(sA + (S_)*A_STAGE + aDst, ag);                              \
        cpAsync16(sA + (S_)*A_STAGE + aDst + 4096, ag + (uint32_t)64 * K);    \
        const half* bg = bSrc + (uint32_t)((KT_) * BK) * N;                   \
        cpAsync16(sB + (S_)*B_STAGE + bDst, bg);                              \
        cpAsync16(sB + (S_)*B_STAGE + bDst + 8 * 512, bg + (uint32_t)8 * N);  \
    } while (0)

// Second half: B rows {w+16, w+24} (2).
#define ISSUE_H2(S_, KT_)                                                     \
    do {                                                                      \
        const half* bg = bSrc + (uint32_t)((KT_) * BK + 16) * N;              \
        cpAsync16(sB + (S_)*B_STAGE + bDst + 16 * 512, bg);                   \
        cpAsync16(sB + (S_)*B_STAGE + bDst + 24 * 512, bg + (uint32_t)8 * N); \
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
                    sB + (S_)*B_STAGE + ldB[j] + (KK_)*8192);                 \
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

    // ---- prologue: fully issue stages 0..3
    ISSUE_H1(0, 0);
    ISSUE_H2(0, 0);
    cpCommit();
    ISSUE_H1(1, 1);
    ISSUE_H2(1, 1);
    cpCommit();
    ISSUE_H1(2, 2);
    ISSUE_H2(2, 2);
    cpCommit();
    ISSUE_H1(3, 3);
    ISSUE_H2(3, 3);
    cpCommit();
    cpWait<STAGES - 2>();
    __syncthreads();
    LOAD_FRAGS(0, 0, 0);

// Tile iteration; stage T+4 copies: H1 at step0, H2 at step1 (post-BAR of
// this tile, which orders them after all reads of the slot they overwrite),
// committed at next tile's step0.
#define TILE_ITER(KT_, RS_, NS_, WS_)                                         \
    do {                                                                      \
        if ((KT_) + 4 < NT) {                                                 \
            ISSUE_H1(WS_, (KT_) + 4);                                         \
        }                                                                     \
        LOAD_FRAGS(1, RS_, 1);                                                \
        MMA_STEP(0);                                                          \
        cpWait<STAGES - 3>();                                                 \
        __syncthreads();                                                      \
        LOAD_FRAGS(0, NS_, 0);                                                \
        if ((KT_) + 4 < NT) {                                                 \
            ISSUE_H2(WS_, (KT_) + 4);                                         \
        }                                                                     \
        cpCommit(); /* one commit per iter; empties pad tail */               \
        MMA_STEP(1);                                                          \
    } while (0)

    // NT divisible by 5? 128 = 25*5+3 -> peel: main loop in chunks of 5.
    int kt = 0;
    for (; kt + 5 <= NT; kt += 5) {
        TILE_ITER(kt + 0, 0, 1, 4);
        TILE_ITER(kt + 1, 1, 2, 0);
        TILE_ITER(kt + 2, 2, 3, 1);
        TILE_ITER(kt + 3, 3, 4, 2);
        TILE_ITER(kt + 4, 4, 0, 3);
    }
    if (kt < NT) {
        TILE_ITER(kt, 0, 1, 4);
        if (kt + 1 < NT) {
            TILE_ITER(kt + 1, 1, 2, 0);
        }
        if (kt + 2 < NT) {
            TILE_ITER(kt + 2, 2, 3, 1);
        }
        if (kt + 3 < NT) {
            TILE_ITER(kt + 3, 3, 4, 2);
        }
    }
#undef TILE_ITER
#undef ISSUE_H1
#undef ISSUE_H2
#undef LOAD_FRAGS
#undef MMA_STEP

    // ---- epilogue: stage C tile through smem, then 16B coalesced stores
    cpWait<0>();
    __syncthreads();  // all smem reads done; safe to reuse as staging

    const uint32_t epiBase = sA;  // reuse pipeline smem
    {
        const int r0 = wm * 64 + (lane >> 2);
        const int c0 = wn * 64 + (lane & 3) * 2;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
            for (int jj = 0; jj < 8; ++jj) {
                const float* c4 = acc[i][jj];
                const uint32_t a0 =
                    epiBase +
                    ((r0 + i * 16) * EPI_STRIDE + c0 + jj * 8) * 2;
                stsHalf2(a0, __floats2half2_rn(c4[0], c4[1]));
                stsHalf2(a0 + 8 * EPI_STRIDE * 2,
                         __floats2half2_rn(c4[2], c4[3]));
            }
        }
    }
    __syncthreads();
    // 256 threads copy 128x256 halves: warp w covers row (it*8 + w) fully
    // (32 lanes x 16B = 512B contiguous in gmem).
    {
        const int roww = warp;
        const int chunk = lane;
#pragma unroll
        for (int it = 0; it < 16; ++it) {
            const int row = it * 8 + roww;
            uint32_t v[4];
            ldsV4(v, epiBase + (row * EPI_STRIDE + chunk * 8) * 2);
            *reinterpret_cast<uint4*>(
                &C[(uint32_t)(bm + row) * N + bn + chunk * 8]) =
                *reinterpret_cast<uint4*>(v);
        }
    }
}

}  // namespace v8

PLAYGROUND_MATMUL_DEC(float16_t, 8, m, n, k, A, B, C)
{
    static bool smemConfigured = false;
    if (!smemConfigured) {
        cudaFuncSetAttribute(v8::hgemmV8,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             v8::SMEM_BYTES);
        smemConfigured = true;
    }
    dim3 grid(n / v8::BN, m / v8::BM);
    v8::hgemmV8<<<grid, 256, v8::SMEM_BYTES>>>(A, B, C, int(m), int(n),
                                               int(k));
}

}  // namespace playground
