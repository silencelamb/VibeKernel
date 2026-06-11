// v3: BK 32->64 (half the barriers; 4 k16-steps per tile), 3-stage cp.async
// (144KB smem), copies spread across k-steps to smooth LSU pressure.
// BM=128 BN=256, 8 warps (2x4), warp tile 64x64, f32 accumulators.
#include <cstdint>
#include <cuda_runtime.h>

#include "playground/matmul.hpp"

namespace playground
{
namespace v3
{

constexpr int BM = 128, BN = 256, BK = 64;
constexpr int STAGES = 3;
constexpr uint32_t A_STAGE = BM * BK * sizeof(half);  // 16 KB
constexpr uint32_t B_STAGE = BK * BN * sizeof(half);  // 32 KB
constexpr uint32_t SMEM_BYTES = STAGES * (A_STAGE + B_STAGE);  // 144 KB

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

__global__ __launch_bounds__(256, 1) void hgemmV3(const half* __restrict__ A,
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

    const int bm = blockIdx.y * BM;
    const int bn = blockIdx.x * BN;

    // ---- A tile 128x64 halves: rows of 128B = 8 chunks. Swizzle c^(row&7).
    // thread t: row t>>3 (+32,+64,+96), chunk t&7.
    const int aRow = tid >> 3;
    const int aChunk = tid & 7;
    const half* aSrc = A + (uint32_t)(bm + aRow) * K + aChunk * 8;
    const uint32_t aDst = aRow * 128 + ((aChunk ^ (aRow & 7)) << 4);
    // row+32k: swizzle unchanged, +4096 bytes each.

    // ---- B tile 64x256: rows of 512B = 32 chunks. Swizzle low3(c)^(k&7).
    // warp w: k-rows {w+8i}, lane = chunk.
    const int bRow = warp;
    const int bChunk = lane;
    const half* bSrc = B + (uint32_t)bRow * N + bn + bChunk * 8;
    const uint32_t bDst =
        bRow * 512 + (((bChunk & 24) | ((bChunk & 7) ^ (bRow & 7))) << 4);

    // ---- ldmatrix lane offsets (kk=0); kk in 0..3
    // A: addr(kk) = ldA0 ^ (kk<<5)
    uint32_t ldA[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = wm * 64 + i * 16 + (lane & 15);
        const int c = lane >> 4;
        ldA[i] = row * 128 + ((c ^ (row & 7)) << 4);
    }
    // B: addr(kk) = ldB0 + kk*8192
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

    uint32_t aBase[STAGES], bBase[STAGES];
#pragma unroll
    for (int s = 0; s < STAGES; ++s) {
        aBase[s] = sA + s * A_STAGE;
        bBase[s] = sB + s * B_STAGE;
    }

#define ISSUE_A(S_, KT_)                                                      \
    do {                                                                      \
        const half* ag = aSrc + (KT_) * BK;                                   \
        cpAsync16(aBase[S_] + aDst, ag);                                      \
        cpAsync16(aBase[S_] + aDst + 4096, ag + (uint32_t)32 * K);            \
        cpAsync16(aBase[S_] + aDst + 8192, ag + (uint32_t)64 * K);            \
        cpAsync16(aBase[S_] + aDst + 12288, ag + (uint32_t)96 * K);           \
    } while (0)

#define ISSUE_B(S_, KT_, I0_)                                                 \
    do {                                                                      \
        const half* bg = bSrc + (uint32_t)((KT_) * BK + (I0_)*8) * N;         \
        cpAsync16(bBase[S_] + bDst + (I0_)*4096, bg);                         \
        cpAsync16(bBase[S_] + bDst + ((I0_) + 1) * 4096,                      \
                  bg + (uint32_t)8 * N);                                      \
        cpAsync16(bBase[S_] + bDst + ((I0_) + 2) * 4096,                      \
                  bg + (uint32_t)16 * N);                                     \
        cpAsync16(bBase[S_] + bDst + ((I0_) + 3) * 4096,                      \
                  bg + (uint32_t)24 * N);                                     \
    } while (0)

#define LOAD_FRAGS(BUF_, S_, KK_)                                            \
    do {                                                                      \
        _Pragma("unroll") for (int i = 0; i < 4; ++i)                         \
        {                                                                     \
            ldsmX4(fragA[BUF_][i], aBase[S_] + (ldA[i] ^ ((KK_) << 5)));      \
        }                                                                     \
        _Pragma("unroll") for (int j = 0; j < 4; ++j)                         \
        {                                                                     \
            ldsmX4T(fragB[BUF_][j], bBase[S_] + ldB[j] + (KK_)*8192);         \
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

    // ---- prologue: stages 0,1
    ISSUE_A(0, 0);
    ISSUE_B(0, 0, 0);
    ISSUE_B(0, 0, 4);
    cpCommit();
    ISSUE_A(1, 1);
    ISSUE_B(1, 1, 0);
    ISSUE_B(1, 1, 4);
    cpCommit();
    cpWait<STAGES - 2>();
    __syncthreads();
    LOAD_FRAGS(0, 0, 0);

    int rs = 0, ns = 1, ws = 2;
    for (int kt = 0; kt < NT; ++kt) {
        const bool fetch = (kt + 2 < NT);
        // ---- k16 step 0
        LOAD_FRAGS(1, rs, 1);
        if (fetch) {
            ISSUE_A(ws, kt + 2);
        }
        MMA_STEP(0);
        // ---- k16 step 1
        LOAD_FRAGS(0, rs, 2);
        if (fetch) {
            ISSUE_B(ws, kt + 2, 0);
        }
        MMA_STEP(1);
        // ---- k16 step 2
        LOAD_FRAGS(1, rs, 3);
        if (fetch) {
            ISSUE_B(ws, kt + 2, 4);
        }
        cpCommit();
        MMA_STEP(0);
        // ---- k16 step 3
        cpWait<STAGES - 2>();
        __syncthreads();
        LOAD_FRAGS(0, ns, 0);  // next tile kk=0 (garbage on last iter, unused)
        MMA_STEP(1);
        const int t = rs;
        rs = ns;
        ns = ws;
        ws = t;
    }

#undef ISSUE_A
#undef ISSUE_B
#undef LOAD_FRAGS
#undef MMA_STEP

    // ---- epilogue: direct half2 stores
    const int cRow = bm + wm * 64 + (lane >> 2);
    const int cCol = bn + wn * 64 + (lane & 3) * 2;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int jj = 0; jj < 8; ++jj) {
            const float* c4 = acc[i][jj];
            const uint32_t r0 = (uint32_t)(cRow + i * 16) * N + cCol + jj * 8;
            *reinterpret_cast<half2*>(&C[r0]) = __floats2half2_rn(c4[0], c4[1]);
            *reinterpret_cast<half2*>(&C[r0 + 8 * N]) =
                __floats2half2_rn(c4[2], c4[3]);
        }
    }
}

}  // namespace v3

PLAYGROUND_MATMUL_DEC(float16_t, 3, m, n, k, A, B, C)
{
    static bool smemConfigured = false;
    if (!smemConfigured) {
        cudaFuncSetAttribute(v3::hgemmV3,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             v3::SMEM_BYTES);
        smemConfigured = true;
    }
    dim3 grid(n / v3::BN, m / v3::BM);
    v3::hgemmV3<<<grid, 256, v3::SMEM_BYTES>>>(A, B, C, int(m), int(n),
                                               int(k));
}

}  // namespace playground
