// v5: v2 with stage-unrolled main loop (compile-time stage ids). handwritten mma.sync(m16n8k16) + cp.async 3-stage pipeline.
// BM=128 BN=256 BK=32, 8 warps (2x4), warp tile 64x64, f32 accumulators.
// XOR-swizzled smem so both ldmatrix and cp.async stores are bank-conflict
// free; A/B fragments double-buffered in registers across k16 steps.
#include <cstdint>
#include <cuda_runtime.h>

#include "playground/matmul.hpp"

namespace playground
{
namespace v5
{

constexpr int BM = 128, BN = 256, BK = 32;
constexpr int STAGES = 3;
constexpr uint32_t A_STAGE = BM * BK * sizeof(half);  // 8 KB
constexpr uint32_t B_STAGE = BK * BN * sizeof(half);  // 16 KB
constexpr uint32_t SMEM_BYTES = STAGES * (A_STAGE + B_STAGE);

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

__global__ __launch_bounds__(256, 1) void hgemmV5(const half* __restrict__ A,
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
    const int wm = warp >> 2;  // 0..1  (m dimension)
    const int wn = warp & 3;   // 0..3  (n dimension)

    const int bm = blockIdx.y * BM;
    const int bn = blockIdx.x * BN;

    // ---------------- global -> shared (cp.async) thread mapping ----------
    // A tile 128x32 halves: thread t covers rows (t>>2, t>>2+64), 16B chunk
    // c = t&3. Swizzle: physical chunk = c ^ ((row>>1)&3).
    const int aRow = tid >> 2;
    const int aChunk = tid & 3;
    const half* aSrc = A + (uint32_t)(bm + aRow) * K + aChunk * 8;
    const uint32_t aDst =
        aRow * 64 + ((aChunk ^ ((aRow >> 1) & 3)) << 4);  // +4096 for row+64

    // B tile 32x256 halves: warp w covers k-rows {w, w+8, w+16, w+24},
    // lane = 16B chunk (32 per row). Swizzle: low 3 bits of chunk ^ (k&7).
    const int bRow = warp;
    const int bChunk = lane;
    const half* bSrc = B + (uint32_t)bRow * N + bn + bChunk * 8;
    const uint32_t bDst =
        bRow * 512 + (((bChunk & 24) | ((bChunk & 7) ^ (bRow & 7))) << 4);

    // ---------------- ldmatrix per-lane base offsets (kk=0) ---------------
    // A frag i (m16): row = wm*64 + i*16 + (lane&15), chunk = lane>>4.
    // kk=1 toggles chunk bit1 -> XOR 32 on the offset.
    uint32_t ldA[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = wm * 64 + i * 16 + (lane & 15);
        const int c = lane >> 4;
        ldA[i] = row * 64 + ((c ^ ((row >> 1) & 3)) << 4);
    }
    // B frag j (n16): k-row = lane&15, chunk = wn*8 + j*2 + (lane>>4).
    // kk=1 -> k-row += 16 -> +16*512 bytes (swizzle bits unchanged).
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

    // stage bases
    uint32_t aBase[STAGES], bBase[STAGES];
#pragma unroll
    for (int s = 0; s < STAGES; ++s) {
        aBase[s] = sA + s * A_STAGE;
        bBase[s] = sB + s * B_STAGE;
    }

#define ISSUE_STAGE(S_, KT_)                                                  \
    do {                                                                      \
        const half* ag = aSrc + (KT_) * BK;                                   \
        cpAsync16(aBase[S_] + aDst, ag);                                      \
        cpAsync16(aBase[S_] + aDst + 4096, ag + (uint32_t)64 * K);            \
        const half* bg = bSrc + (uint32_t)((KT_) * BK) * N;                   \
        cpAsync16(bBase[S_] + bDst, bg);                                      \
        cpAsync16(bBase[S_] + bDst + 8 * 512, bg + (uint32_t)8 * N);          \
        cpAsync16(bBase[S_] + bDst + 16 * 512, bg + (uint32_t)16 * N);        \
        cpAsync16(bBase[S_] + bDst + 24 * 512, bg + (uint32_t)24 * N);        \
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

    // ---------------- prologue: fill stages 0..STAGES-2 -------------------
    ISSUE_STAGE(0, 0);
    cpCommit();
    ISSUE_STAGE(1, 1);
    cpCommit();
    cpWait<STAGES - 2>();
    __syncthreads();
    LOAD_FRAGS(0, 0, 0);

// One k-tile iteration with compile-time stage ids (no dynamic indexing).
#define TILE_ITER(KT_, RS_, NS_, WS_)                                         \
    do {                                                                      \
        if ((KT_) + 2 < NT) {                                                 \
            ISSUE_STAGE(WS_, (KT_) + 2);                                      \
        }                                                                     \
        cpCommit();                                                           \
        LOAD_FRAGS(1, RS_, 1);                                                \
        MMA_STEP(0);                                                          \
        cpWait<STAGES - 2>();                                                 \
        __syncthreads();                                                      \
        LOAD_FRAGS(0, NS_, 0);                                                \
        MMA_STEP(1);                                                          \
    } while (0)

    int kt = 0;
    for (; kt + 3 <= NT; kt += 3) {
        TILE_ITER(kt + 0, 0, 1, 2);
        TILE_ITER(kt + 1, 1, 2, 0);
        TILE_ITER(kt + 2, 2, 0, 1);
    }
    if (kt < NT) {  // remainder (kt % 3 == 0 here)
        TILE_ITER(kt, 0, 1, 2);
        if (kt + 1 < NT) {
            TILE_ITER(kt + 1, 1, 2, 0);
        }
    }
#undef TILE_ITER

#undef ISSUE_STAGE
#undef LOAD_FRAGS
#undef MMA_STEP

    // ---------------- epilogue: f32 acc -> half2 direct stores ------------
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

}  // namespace v5

PLAYGROUND_MATMUL_DEC(float16_t, 5, m, n, k, A, B, C)
{
    static bool smemConfigured = false;
    if (!smemConfigured) {
        cudaFuncSetAttribute(v5::hgemmV5,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             v5::SMEM_BYTES);
        smemConfigured = true;
    }
    dim3 grid(n / v5::BN, m / v5::BM);
    v5::hgemmV5<<<grid, 256, v5::SMEM_BYTES>>>(A, B, C, int(m), int(n),
                                               int(k));
}

}  // namespace playground
