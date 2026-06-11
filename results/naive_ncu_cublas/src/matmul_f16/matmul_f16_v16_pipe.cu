// fp16 GEMM v15 — CUTLASS-like deep pipeline (mirrors cuBLAS's
// h16816gemm_256x128_64x3). Big 256x128 tile, BK=64, 3-stage cp.async pipeline,
// register double-buffering over the 4 k-slices, ~159KB smem, 1 CTA/SM.
// Latency is hidden by the deep software pipeline, not by occupancy.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{
namespace v16
{

constexpr int BM = 256;
constexpr int BN = 128;
constexpr int BK = 64;
constexpr int APAD = 8;
constexpr int BPAD = 8;
constexpr int WARPS_M = 4;
constexpr int WARPS_N = 2;
constexpr int NWARPS = WARPS_M * WARPS_N;  // 8
constexpr int NTHREADS = NWARPS * 32;      // 256
constexpr int WARP_M = BM / WARPS_M;       // 64
constexpr int WARP_N = BN / WARPS_N;       // 64
constexpr int WMITER = WARP_M / 16;        // 4
constexpr int WNITER = WARP_N / 8;         // 8
constexpr int KITER = BK / 16;             // 4
constexpr int NSTAGES = 3;

constexpr int AS_STRIDE = BK + APAD;  // 72
constexpr int BS_STRIDE = BN + BPAD;  // 136
constexpr int AS_STAGE = BM * AS_STRIDE;
constexpr int BS_STAGE = BK * BS_STRIDE;
constexpr int SMEM_BYTES = NSTAGES * (AS_STAGE + BS_STAGE) * 2;

constexpr int A_VECS = BM * BK / 8;
constexpr int B_VECS = BK * BN / 8;

__device__ __forceinline__ uint32_t smem_u32(const void* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst),
                 "l"(src));
}
template <int N>
__device__ __forceinline__ void cp_async_wait()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void ldm_x4(uint32_t& r0, uint32_t& r1,
                                        uint32_t& r2, uint32_t& r3, uint32_t a)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(a));
}
__device__ __forceinline__ void ldm_x4_trans(uint32_t& r0, uint32_t& r1,
                                              uint32_t& r2, uint32_t& r3,
                                              uint32_t a)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(a));
}
__device__ __forceinline__ void mma16816(uint32_t* d, const uint32_t* a,
                                          const uint32_t* b)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%0,%1};\n"
        : "+r"(d[0]), "+r"(d[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__global__ void __launch_bounds__(NTHREADS, 1)
    gemm(const half* __restrict__ A, const half* __restrict__ B,
         half* __restrict__ C, int M, int N, int K)
{
    extern __shared__ half smem[];
    half* As = smem;
    half* Bs = smem + NSTAGES * AS_STAGE;

    const int tid = threadIdx.x;
    const int warpId = tid >> 5;
    const int lane = tid & 31;
    const int warp_m = warpId / WARPS_N;
    const int warp_n = warpId % WARPS_N;

    // threadblock swizzle for L2 reuse
    constexpr int GROUP_M = 8;
    const int gridM = M / BM;
    const int gridN = N / BN;
    const int pid = blockIdx.x;
    const int numInGroup = GROUP_M * gridN;
    const int groupId = pid / numInGroup;
    const int firstM = groupId * GROUP_M;
    const int gsize = min(gridM - firstM, GROUP_M);
    const int pid_m = firstM + (pid % gsize);
    const int pid_n = (pid % numInGroup) / gsize;
    const int blockRow = pid_m * BM;
    const int blockCol = pid_n * BN;
    const int warpRowBase = warp_m * WARP_M;
    const int warpColBase = warp_n * WARP_N;

    uint32_t RC[WMITER][WNITER][2];
#pragma unroll
    for (int i = 0; i < WMITER; i++)
#pragma unroll
        for (int j = 0; j < WNITER; j++)
            RC[i][j][0] = RC[i][j][1] = 0u;

    auto loadTile = [&](int stage, int kk0) {
        half* Asd = As + stage * AS_STAGE;
        half* Bsd = Bs + stage * BS_STAGE;
#pragma unroll
        for (int it = 0; it < A_VECS / NTHREADS; it++) {
            int v = tid + it * NTHREADS;
            int r = v / (BK / 8);
            int c = (v % (BK / 8)) << 3;
            cp_async16(smem_u32(Asd + r * AS_STRIDE + c),
                       A + (size_t)(blockRow + r) * K + (kk0 + c));
        }
#pragma unroll
        for (int it = 0; it < B_VECS / NTHREADS; it++) {
            int v = tid + it * NTHREADS;
            int r = v / (BN / 8);
            int c = (v % (BN / 8)) << 3;
            cp_async16(smem_u32(Bsd + r * BS_STRIDE + c),
                       B + (size_t)(kk0 + r) * N + (blockCol + c));
        }
    };

    const int numTiles = K / BK;
#pragma unroll
    for (int s = 0; s < NSTAGES - 1; s++) {
        loadTile(s, s * BK);
        asm volatile("cp.async.commit_group;\n");
    }

    // Per-lane ldmatrix base byte-offsets for k-slice 0; later slices add a
    // compile-time constant (A: +ks*32B along K; B: +ks*16 rows).
    uint32_t aoff0[WMITER];
    uint32_t boff0[WARP_N / 16];
#pragma unroll
    for (int mi = 0; mi < WMITER; mi++) {
        int row = warpRowBase + mi * 16 + (lane & 15);
        int col = (lane >> 4) << 3;
        aoff0[mi] = (row * AS_STRIDE + col) * 2;
    }
#pragma unroll
    for (int ng = 0; ng < WARP_N / 16; ng++) {
        int row = (lane & 15);
        int col = warpColBase + ng * 16 + ((lane >> 4) << 3);
        boff0[ng] = (row * BS_STRIDE + col) * 2;
    }
    const uint32_t AsBase = smem_u32(As);
    const uint32_t BsBase = smem_u32(Bs);
    constexpr uint32_t A_KS = 16 * 2;             // K advances 16 halves
    constexpr uint32_t B_KS = 16 * BS_STRIDE * 2;  // 16 rows of B

    uint32_t RA[2][WMITER][4];
    uint32_t RB[2][WNITER][2];

    auto ldFrag = [&](uint32_t aStage, uint32_t bStage, int ks, int buf) {
#pragma unroll
        for (int mi = 0; mi < WMITER; mi++)
            ldm_x4(RA[buf][mi][0], RA[buf][mi][1], RA[buf][mi][2],
                   RA[buf][mi][3], aStage + aoff0[mi] + ks * A_KS);
#pragma unroll
        for (int ng = 0; ng < WARP_N / 16; ng++) {
            uint32_t r0, r1, r2, r3;
            ldm_x4_trans(r0, r1, r2, r3, bStage + boff0[ng] + ks * B_KS);
            RB[buf][ng * 2 + 0][0] = r0;
            RB[buf][ng * 2 + 0][1] = r1;
            RB[buf][ng * 2 + 1][0] = r2;
            RB[buf][ng * 2 + 1][1] = r3;
        }
    };
    auto stageA = [&](int t) { return AsBase + (t % NSTAGES) * AS_STAGE * 2; };
    auto stageB = [&](int t) { return BsBase + (t % NSTAGES) * BS_STAGE * 2; };

    // tile 0 is ready (prologue waited); preload its first k-slice.
    cp_async_wait<NSTAGES - 2>();
    __syncthreads();
    ldFrag(stageA(0), stageB(0), 0, 0);

    // Fully software-pipelined mainloop: registers stay one k-slice ahead,
    // continuously across tile boundaries; cp.async + barrier happen mid-tile.
    for (int t = 0; t < numTiles; t++) {
        uint32_t aStage = stageA(t);
        uint32_t bStage = stageB(t);
#pragma unroll
        for (int ks = 0; ks < KITER; ks++) {
            int slice = t * KITER + ks;
            int cur = slice & 1;
            int nxt = cur ^ 1;
            if (ks == 0) {
                int tp = t + NSTAGES - 1;
                if (tp < numTiles)
                    loadTile(tp % NSTAGES, tp * BK);
                asm volatile("cp.async.commit_group;\n");
            }
            if (ks < KITER - 1) {
                ldFrag(aStage, bStage, ks + 1, nxt);
            } else {
                cp_async_wait<NSTAGES - 2>();
                __syncthreads();
                if (t + 1 < numTiles)
                    ldFrag(stageA(t + 1), stageB(t + 1), 0, nxt);
            }
#pragma unroll
            for (int mi = 0; mi < WMITER; mi++)
#pragma unroll
                for (int ni = 0; ni < WNITER; ni++)
                    mma16816(RC[mi][ni], RA[cur][mi], RB[cur][ni]);
        }
    }

    const int groupID = lane >> 2;
    const int tig = lane & 3;
#pragma unroll
    for (int mi = 0; mi < WMITER; mi++)
#pragma unroll
        for (int ni = 0; ni < WNITER; ni++) {
            int row0 = blockRow + warpRowBase + mi * 16 + groupID;
            int col0 = blockCol + warpColBase + ni * 8 + tig * 2;
            *reinterpret_cast<uint32_t*>(&C[(size_t)row0 * N + col0]) =
                RC[mi][ni][0];
            *reinterpret_cast<uint32_t*>(&C[(size_t)(row0 + 8) * N + col0]) =
                RC[mi][ni][1];
        }
}

}  // namespace v16

PLAYGROUND_MATMUL_DEC(float16_t, 16, m, n, k, A, B, C)
{
    static bool inited = false;
    if (!inited) {
        cudaFuncSetAttribute(v16::gemm,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             v16::SMEM_BYTES);
        inited = true;
    }
    dim3 block(v16::NTHREADS);
    dim3 grid((n / v16::BN) * (m / v16::BM));
    v16::gemm<<<grid, block, v16::SMEM_BYTES>>>(A, B, C, (int) m, (int) n,
                                                (int) k);
}

}  // namespace playground
