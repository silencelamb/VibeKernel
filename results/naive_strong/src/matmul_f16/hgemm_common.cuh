#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

// =============================================================================
// Shared infrastructure for hand-written fp16 Tensor-Core GEMM kernels.
//   mma.sync.m16n8k16 + ldmatrix + multi-stage cp.async + XOR-swizzled smem.
//   A templated kernel parameterized by tile/stage/warp config so we can sweep.
// =============================================================================
namespace playground
{
namespace hg
{

__device__ __forceinline__ unsigned smem_addr(const void* p)
{
    return (unsigned) __cvta_generic_to_shared(p);
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem)
{
    unsigned s = smem_addr(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s),
                 "l"(gmem));
}
__device__ __forceinline__ void cp_commit()
{
    asm volatile("cp.async.commit_group;\n");
}
template <int N>
__device__ __forceinline__ void cp_wait()
{
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void ldm_x4(uint32_t (&r)[4], const void* s)
{
    unsigned a = smem_addr(s);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(a));
}
__device__ __forceinline__ void ldm_x2_trans(uint32_t (&r)[2], const void* s)
{
    unsigned a = smem_addr(s);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1])
        : "r"(a));
}
// x4.trans loads a 16x16 B region -> two n8 tiles: {r0,r1}=tile0, {r2,r3}=tile1
__device__ __forceinline__ void ldm_x4_trans(uint32_t (&r)[4], const void* s)
{
    unsigned a = smem_addr(s);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(a));
}
__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4],
                                    const uint32_t (&b)[2])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void mma_f16(uint32_t (&d)[2],
                                        const uint32_t (&a)[4],
                                        const uint32_t (&b)[2])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(d[0]), "+r"(d[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// swizzle: permute 8-half chunk within a row.  A: period 8, B: period 16.
template <int COLS>
__device__ __forceinline__ int swz(int row, int col, int mask)
{
    int chunk = (col >> 3) ^ (row & mask);
    return row * COLS + (chunk << 3) + (col & 7);
}

// threadblock rasterization: traverse GW block-rows across all columns before
// advancing, keeping B column-panels hot in L2.  GW<=0 disables (identity).
template <int GW>
__device__ __forceinline__ int2 raster_block()
{
    if (GW <= 0) return make_int2(blockIdx.x, blockIdx.y);
    int bn = gridDim.x, bm = gridDim.y;
    int bid = blockIdx.y * bn + blockIdx.x;
    int per = GW * bn;
    int cg = bid / per;
    int idx = bid % per;
    int rows = min(GW, bm - cg * GW);
    int by = cg * GW + (idx % rows);
    int bx = idx / rows;
    return make_int2(bx, by);
}

// -----------------------------------------------------------------------------
template <int BM, int BN, int BK, int STAGES, int WARP_ROWS, int WARP_COLS,
          int MINB = 1, int RAST = 0>
__global__ void __launch_bounds__(WARP_ROWS* WARP_COLS * 32, MINB)
    hgemm_kernel(const half* __restrict__ A, const half* __restrict__ B,
                 half* __restrict__ C, int M, int N, int K)
{
    constexpr int THREADS = WARP_ROWS * WARP_COLS * 32;
    constexpr int WM = BM / WARP_ROWS;
    constexpr int WN = BN / WARP_COLS;
    constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;
    constexpr int MTILES = WM / MMA_M;
    constexpr int NTILES = WN / MMA_N;
    constexpr int KSUB = BK / MMA_K;
    constexpr int APITCH = BM * BK;
    constexpr int BPITCH = BK * BN;
    constexpr int A_ROW_F4 = BK / 8;
    constexpr int B_ROW_F4 = BN / 8;
    constexpr int A_F4_PT = (BM * BK / 8) / THREADS;
    constexpr int B_F4_PT = (BK * BN / 8) / THREADS;
    constexpr int AMASK = (BK / 8 < 8 ? BK / 8 : 8) - 1;
    constexpr int BMASK = (BN / 8 < 16 ? BN / 8 : 16) - 1;

    extern __shared__ half smem[];
    half* As = smem;
    half* Bs = smem + STAGES * APITCH;

    int2 _blk = raster_block<RAST>();
    const int blockRow = _blk.y * BM;
    const int blockCol = _blk.x * BN;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warpId = tid >> 5;
    const int warpRow = warpId / WARP_COLS;
    const int warpCol = warpId % WARP_COLS;

    float acc[MTILES][NTILES][4];
#pragma unroll
    for (int i = 0; i < MTILES; ++i)
#pragma unroll
        for (int j = 0; j < NTILES; ++j)
#pragma unroll
            for (int t = 0; t < 4; ++t) acc[i][j][t] = 0.0f;

    const int numTiles = K / BK;

    auto loadTile = [&](int stage, int kt) {
        const int k0 = kt * BK;
        half* asp = As + stage * APITCH;
        half* bsp = Bs + stage * BPITCH;
#pragma unroll
        for (int t = 0; t < A_F4_PT; ++t) {
            int f = tid + t * THREADS;
            int row = f / A_ROW_F4;
            int col = (f % A_ROW_F4) * 8;
            cp_async16(&asp[swz<BK>(row, col, AMASK)],
                       A + (blockRow + row) * K + (k0 + col));
        }
#pragma unroll
        for (int t = 0; t < B_F4_PT; ++t) {
            int f = tid + t * THREADS;
            int row = f / B_ROW_F4;
            int col = (f % B_ROW_F4) * 8;
            cp_async16(&bsp[swz<BN>(row, col, BMASK)],
                       B + (k0 + row) * N + (blockCol + col));
        }
    };

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        loadTile(s, s);
        cp_commit();
    }
    cp_wait<STAGES - 2>();
    __syncthreads();

    int kTile = STAGES - 1;
    for (int ki = 0; ki < numTiles; ++ki) {
        if (kTile < numTiles) loadTile(kTile % STAGES, kTile);
        cp_commit();
        ++kTile;

        const int rs = ki % STAGES;
        half* asp = As + rs * APITCH;
        half* bsp = Bs + rs * BPITCH;

#pragma unroll
        for (int ks = 0; ks < BK; ks += MMA_K) {
            uint32_t aF[MTILES][4];
            uint32_t bF[NTILES][2];
#pragma unroll
            for (int i = 0; i < MTILES; ++i) {
                int row = warpRow * WM + i * MMA_M + (lane % 16);
                int col = ks + (lane / 16) * 8;
                ldm_x4(aF[i], &asp[swz<BK>(row, col, AMASK)]);
            }
            // B: x4.trans loads two n8 tiles per instruction
#pragma unroll
            for (int j = 0; j < NTILES; j += 2) {
                int row = ks + (lane % 16);
                int col = warpCol * WN + j * MMA_N + (lane / 16) * 8;
                uint32_t t[4];
                ldm_x4_trans(t, &bsp[swz<BN>(row, col, BMASK)]);
                bF[j][0] = t[0];
                bF[j][1] = t[1];
                bF[j + 1][0] = t[2];
                bF[j + 1][1] = t[3];
            }
#pragma unroll
            for (int i = 0; i < MTILES; ++i)
#pragma unroll
                for (int j = 0; j < NTILES; ++j) mma(acc[i][j], aF[i], bF[j]);
        }

        cp_wait<STAGES - 2>();
        __syncthreads();
    }

    const int group = lane / 4;
    const int tid4 = lane % 4;
#pragma unroll
    for (int i = 0; i < MTILES; ++i) {
#pragma unroll
        for (int j = 0; j < NTILES; ++j) {
            int rowBase = blockRow + warpRow * WM + i * MMA_M;
            int colBase = blockCol + warpCol * WN + j * MMA_N + tid4 * 2;
            half2 lo = __floats2half2_rn(acc[i][j][0], acc[i][j][1]);
            half2 hi = __floats2half2_rn(acc[i][j][2], acc[i][j][3]);
            *reinterpret_cast<half2*>(&C[(rowBase + group) * N + colBase]) = lo;
            *reinterpret_cast<half2*>(&C[(rowBase + group + 8) * N + colBase]) =
                hi;
        }
    }
}

// Launch helper: sets the dynamic-smem opt-in (with a sync so it takes effect
// before the first launch) once, then launches.
template <int BM, int BN, int BK, int STAGES, int WARP_ROWS, int WARP_COLS,
          int MINB = 1, int RAST = 0>
inline void launch(size_t m, size_t n, size_t k, const half* A, const half* B,
                   half* C)
{
    constexpr int smemBytes =
        STAGES * (BM * BK + BK * BN) * int(sizeof(half));
    auto kern =
        hgemm_kernel<BM, BN, BK, STAGES, WARP_ROWS, WARP_COLS, MINB, RAST>;
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(
            kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes);
        cudaDeviceSynchronize();
        configured = true;
    }
    dim3 block(WARP_ROWS * WARP_COLS * 32);
    dim3 grid(n / BN, m / BM);
    kern<<<grid, block, smemBytes>>>(A, B, C, int(m), int(n), int(k));
}

// -----------------------------------------------------------------------------
// f16-accumulate variant: halves accumulator registers (2 regs/tile vs 4).
// -----------------------------------------------------------------------------
template <int BM, int BN, int BK, int STAGES, int WARP_ROWS, int WARP_COLS,
          int RAST = 0>
__global__ void hgemm_kernel_f16(const half* __restrict__ A,
                                 const half* __restrict__ B,
                                 half* __restrict__ C, int M, int N, int K)
{
    constexpr int THREADS = WARP_ROWS * WARP_COLS * 32;
    constexpr int WM = BM / WARP_ROWS;
    constexpr int WN = BN / WARP_COLS;
    constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;
    constexpr int MTILES = WM / MMA_M;
    constexpr int NTILES = WN / MMA_N;
    constexpr int APITCH = BM * BK;
    constexpr int BPITCH = BK * BN;
    constexpr int A_ROW_F4 = BK / 8;
    constexpr int B_ROW_F4 = BN / 8;
    constexpr int A_F4_PT = (BM * BK / 8) / THREADS;
    constexpr int B_F4_PT = (BK * BN / 8) / THREADS;
    constexpr int AMASK = (BK / 8 < 8 ? BK / 8 : 8) - 1;
    constexpr int BMASK = (BN / 8 < 16 ? BN / 8 : 16) - 1;

    extern __shared__ half smem[];
    half* As = smem;
    half* Bs = smem + STAGES * APITCH;

    int2 _blk = raster_block<RAST>();
    const int blockRow = _blk.y * BM;
    const int blockCol = _blk.x * BN;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warpId = tid >> 5;
    const int warpRow = warpId / WARP_COLS;
    const int warpCol = warpId % WARP_COLS;

    uint32_t acc[MTILES][NTILES][2];
#pragma unroll
    for (int i = 0; i < MTILES; ++i)
#pragma unroll
        for (int j = 0; j < NTILES; ++j) {
            acc[i][j][0] = 0;
            acc[i][j][1] = 0;
        }

    const int numTiles = K / BK;

    auto loadTile = [&](int stage, int kt) {
        const int k0 = kt * BK;
        half* asp = As + stage * APITCH;
        half* bsp = Bs + stage * BPITCH;
#pragma unroll
        for (int t = 0; t < A_F4_PT; ++t) {
            int f = tid + t * THREADS;
            int row = f / A_ROW_F4;
            int col = (f % A_ROW_F4) * 8;
            cp_async16(&asp[swz<BK>(row, col, AMASK)],
                       A + (blockRow + row) * K + (k0 + col));
        }
#pragma unroll
        for (int t = 0; t < B_F4_PT; ++t) {
            int f = tid + t * THREADS;
            int row = f / B_ROW_F4;
            int col = (f % B_ROW_F4) * 8;
            cp_async16(&bsp[swz<BN>(row, col, BMASK)],
                       B + (k0 + row) * N + (blockCol + col));
        }
    };

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        loadTile(s, s);
        cp_commit();
    }
    cp_wait<STAGES - 2>();
    __syncthreads();

    int kTile = STAGES - 1;
    for (int ki = 0; ki < numTiles; ++ki) {
        if (kTile < numTiles) loadTile(kTile % STAGES, kTile);
        cp_commit();
        ++kTile;

        const int rs = ki % STAGES;
        half* asp = As + rs * APITCH;
        half* bsp = Bs + rs * BPITCH;

#pragma unroll
        for (int ks = 0; ks < BK; ks += MMA_K) {
            uint32_t aF[MTILES][4];
            uint32_t bF[NTILES][2];
#pragma unroll
            for (int i = 0; i < MTILES; ++i) {
                int row = warpRow * WM + i * MMA_M + (lane % 16);
                int col = ks + (lane / 16) * 8;
                ldm_x4(aF[i], &asp[swz<BK>(row, col, AMASK)]);
            }
#pragma unroll
            for (int j = 0; j < NTILES; j += 2) {
                int row = ks + (lane % 16);
                int col = warpCol * WN + j * MMA_N + (lane / 16) * 8;
                uint32_t t[4];
                ldm_x4_trans(t, &bsp[swz<BN>(row, col, BMASK)]);
                bF[j][0] = t[0];
                bF[j][1] = t[1];
                bF[j + 1][0] = t[2];
                bF[j + 1][1] = t[3];
            }
#pragma unroll
            for (int j = 0; j < NTILES; ++j)
#pragma unroll
                for (int i = 0; i < MTILES; ++i)
                    mma_f16(acc[i][j], aF[i], bF[j]);
        }

        cp_wait<STAGES - 2>();
        __syncthreads();
    }

    const int group = lane / 4;
    const int tid4 = lane % 4;
#pragma unroll
    for (int i = 0; i < MTILES; ++i) {
#pragma unroll
        for (int j = 0; j < NTILES; ++j) {
            int rowBase = blockRow + warpRow * WM + i * MMA_M;
            int colBase = blockCol + warpCol * WN + j * MMA_N + tid4 * 2;
            *reinterpret_cast<uint32_t*>(&C[(rowBase + group) * N + colBase]) =
                acc[i][j][0];
            *reinterpret_cast<uint32_t*>(
                &C[(rowBase + group + 8) * N + colBase]) = acc[i][j][1];
        }
    }
}

template <int BM, int BN, int BK, int STAGES, int WARP_ROWS, int WARP_COLS,
          int RAST = 0>
inline void launch_f16(size_t m, size_t n, size_t k, const half* A,
                       const half* B, half* C)
{
    constexpr int smemBytes =
        STAGES * (BM * BK + BK * BN) * int(sizeof(half));
    auto kern = hgemm_kernel_f16<BM, BN, BK, STAGES, WARP_ROWS, WARP_COLS, RAST>;
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(
            kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes);
        cudaDeviceSynchronize();
        configured = true;
    }
    dim3 block(WARP_ROWS * WARP_COLS * 32);
    dim3 grid(n / BN, m / BM);
    kern<<<grid, block, smemBytes>>>(A, B, C, int(m), int(n), int(k));
}

// -----------------------------------------------------------------------------
// f16-accumulate + register double-buffered fragments: prefetch next k-substep's
// ldmatrix while issuing current mma -> overlaps shared-load latency with TC.
// -----------------------------------------------------------------------------
template <int BM, int BN, int BK, int STAGES, int WARP_ROWS, int WARP_COLS,
          int RAST = 0>
__global__ void hgemm_kernel_f16_db(const half* __restrict__ A,
                                    const half* __restrict__ B,
                                    half* __restrict__ C, int M, int N, int K)
{
    constexpr int THREADS = WARP_ROWS * WARP_COLS * 32;
    constexpr int WM = BM / WARP_ROWS;
    constexpr int WN = BN / WARP_COLS;
    constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;
    constexpr int MTILES = WM / MMA_M;
    constexpr int NTILES = WN / MMA_N;
    constexpr int KSUB = BK / MMA_K;
    constexpr int APITCH = BM * BK;
    constexpr int BPITCH = BK * BN;
    constexpr int A_ROW_F4 = BK / 8;
    constexpr int B_ROW_F4 = BN / 8;
    constexpr int A_F4_PT = (BM * BK / 8) / THREADS;
    constexpr int B_F4_PT = (BK * BN / 8) / THREADS;
    constexpr int AMASK = (BK / 8 < 8 ? BK / 8 : 8) - 1;
    constexpr int BMASK = (BN / 8 < 16 ? BN / 8 : 16) - 1;

    extern __shared__ half smem[];
    half* As = smem;
    half* Bs = smem + STAGES * APITCH;

    int2 _blk = raster_block<RAST>();
    const int blockRow = _blk.y * BM;
    const int blockCol = _blk.x * BN;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warpId = tid >> 5;
    const int warpRow = warpId / WARP_COLS;
    const int warpCol = warpId % WARP_COLS;

    uint32_t acc[MTILES][NTILES][2];
#pragma unroll
    for (int i = 0; i < MTILES; ++i)
#pragma unroll
        for (int j = 0; j < NTILES; ++j) {
            acc[i][j][0] = 0;
            acc[i][j][1] = 0;
        }

    const int numTiles = K / BK;

    auto loadTile = [&](int stage, int kt) {
        const int k0 = kt * BK;
        half* asp = As + stage * APITCH;
        half* bsp = Bs + stage * BPITCH;
#pragma unroll
        for (int t = 0; t < A_F4_PT; ++t) {
            int f = tid + t * THREADS;
            int row = f / A_ROW_F4;
            int col = (f % A_ROW_F4) * 8;
            cp_async16(&asp[swz<BK>(row, col, AMASK)],
                       A + (blockRow + row) * K + (k0 + col));
        }
#pragma unroll
        for (int t = 0; t < B_F4_PT; ++t) {
            int f = tid + t * THREADS;
            int row = f / B_ROW_F4;
            int col = (f % B_ROW_F4) * 8;
            cp_async16(&bsp[swz<BN>(row, col, BMASK)],
                       B + (k0 + row) * N + (blockCol + col));
        }
    };

    uint32_t aF[2][MTILES][4];
    uint32_t bF[2][NTILES][2];
    auto loadFrag = [&](int buf, half* asp, half* bsp, int ks) {
#pragma unroll
        for (int i = 0; i < MTILES; ++i) {
            int row = warpRow * WM + i * MMA_M + (lane % 16);
            int col = ks + (lane / 16) * 8;
            ldm_x4(aF[buf][i], &asp[swz<BK>(row, col, AMASK)]);
        }
#pragma unroll
        for (int j = 0; j < NTILES; j += 2) {
            int row = ks + (lane % 16);
            int col = warpCol * WN + j * MMA_N + (lane / 16) * 8;
            uint32_t t[4];
            ldm_x4_trans(t, &bsp[swz<BN>(row, col, BMASK)]);
            bF[buf][j][0] = t[0];
            bF[buf][j][1] = t[1];
            bF[buf][j + 1][0] = t[2];
            bF[buf][j + 1][1] = t[3];
        }
    };

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        loadTile(s, s);
        cp_commit();
    }
    cp_wait<STAGES - 2>();
    __syncthreads();

    int kTile = STAGES - 1;
    for (int ki = 0; ki < numTiles; ++ki) {
        if (kTile < numTiles) loadTile(kTile % STAGES, kTile);
        cp_commit();
        ++kTile;

        const int rs = ki % STAGES;
        half* asp = As + rs * APITCH;
        half* bsp = Bs + rs * BPITCH;

        loadFrag(0, asp, bsp, 0);
#pragma unroll
        for (int s = 0; s < KSUB; ++s) {
            const int cur = s & 1;
            if (s + 1 < KSUB) loadFrag((s + 1) & 1, asp, bsp, (s + 1) * MMA_K);
#pragma unroll
            for (int i = 0; i < MTILES; ++i)
#pragma unroll
                for (int j = 0; j < NTILES; ++j)
                    mma_f16(acc[i][j], aF[cur][i], bF[cur][j]);
        }

        cp_wait<STAGES - 2>();
        __syncthreads();
    }

    const int group = lane / 4;
    const int tid4 = lane % 4;
#pragma unroll
    for (int i = 0; i < MTILES; ++i) {
#pragma unroll
        for (int j = 0; j < NTILES; ++j) {
            int rowBase = blockRow + warpRow * WM + i * MMA_M;
            int colBase = blockCol + warpCol * WN + j * MMA_N + tid4 * 2;
            *reinterpret_cast<uint32_t*>(&C[(rowBase + group) * N + colBase]) =
                acc[i][j][0];
            *reinterpret_cast<uint32_t*>(
                &C[(rowBase + group + 8) * N + colBase]) = acc[i][j][1];
        }
    }
}

template <int BM, int BN, int BK, int STAGES, int WARP_ROWS, int WARP_COLS,
          int RAST = 0>
inline void launch_f16_db(size_t m, size_t n, size_t k, const half* A,
                          const half* B, half* C)
{
    constexpr int smemBytes =
        STAGES * (BM * BK + BK * BN) * int(sizeof(half));
    auto kern = hgemm_kernel_f16_db<BM, BN, BK, STAGES, WARP_ROWS, WARP_COLS, RAST>;
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(
            kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes);
        cudaDeviceSynchronize();
        configured = true;
    }
    dim3 block(WARP_ROWS * WARP_COLS * 32);
    dim3 grid(n / BN, m / BM);
    kern<<<grid, block, smemBytes>>>(A, B, C, int(m), int(n), int(k));
}

// -----------------------------------------------------------------------------
// f16-accumulate + cp.async issue SPREAD across k-substeps (smooths LSU/MIO so
// LDGSTS don't burst-contend with LDSM at the tile start).
// -----------------------------------------------------------------------------
template <int BM, int BN, int BK, int STAGES, int WARP_ROWS, int WARP_COLS,
          int RAST = 0>
__global__ void hgemm_kernel_f16_sw(const half* __restrict__ A,
                                    const half* __restrict__ B,
                                    half* __restrict__ C, int M, int N, int K)
{
    constexpr int THREADS = WARP_ROWS * WARP_COLS * 32;
    constexpr int WM = BM / WARP_ROWS;
    constexpr int WN = BN / WARP_COLS;
    constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;
    constexpr int MTILES = WM / MMA_M;
    constexpr int NTILES = WN / MMA_N;
    constexpr int KSUB = BK / MMA_K;
    constexpr int APITCH = BM * BK;
    constexpr int BPITCH = BK * BN;
    constexpr int A_ROW_F4 = BK / 8;
    constexpr int B_ROW_F4 = BN / 8;
    constexpr int A_F4_PT = (BM * BK / 8) / THREADS;
    constexpr int B_F4_PT = (BK * BN / 8) / THREADS;
    constexpr int AMASK = (BK / 8 < 8 ? BK / 8 : 8) - 1;
    constexpr int BMASK = (BN / 8 < 16 ? BN / 8 : 16) - 1;

    extern __shared__ half smem[];
    half* As = smem;
    half* Bs = smem + STAGES * APITCH;

    int2 _blk = raster_block<RAST>();
    const int blockRow = _blk.y * BM;
    const int blockCol = _blk.x * BN;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warpId = tid >> 5;
    const int warpRow = warpId / WARP_COLS;
    const int warpCol = warpId % WARP_COLS;

    uint32_t acc[MTILES][NTILES][2];
#pragma unroll
    for (int i = 0; i < MTILES; ++i)
#pragma unroll
        for (int j = 0; j < NTILES; ++j) {
            acc[i][j][0] = 0;
            acc[i][j][1] = 0;
        }
    const int numTiles = K / BK;

    auto loadFull = [&](int stage, int kt) {
        const int k0 = kt * BK;
        half* asp = As + stage * APITCH;
        half* bsp = Bs + stage * BPITCH;
#pragma unroll
        for (int t = 0; t < A_F4_PT; ++t) {
            int f = tid + t * THREADS;
            cp_async16(&asp[swz<BK>(f / A_ROW_F4, (f % A_ROW_F4) * 8, AMASK)],
                       A + (blockRow + f / A_ROW_F4) * K +
                           (k0 + (f % A_ROW_F4) * 8));
        }
#pragma unroll
        for (int t = 0; t < B_F4_PT; ++t) {
            int f = tid + t * THREADS;
            cp_async16(&bsp[swz<BN>(f / B_ROW_F4, (f % B_ROW_F4) * 8, BMASK)],
                       B + (k0 + f / B_ROW_F4) * N +
                           (blockCol + (f % B_ROW_F4) * 8));
        }
    };
    // issue only the cp.async whose linear index maps to substep `s`
    auto loadSlice = [&](int stage, int kt, int s) {
        const int k0 = kt * BK;
        half* asp = As + stage * APITCH;
        half* bsp = Bs + stage * BPITCH;
#pragma unroll
        for (int t = 0; t < A_F4_PT; ++t)
            if (t % KSUB == s) {
                int f = tid + t * THREADS;
                cp_async16(
                    &asp[swz<BK>(f / A_ROW_F4, (f % A_ROW_F4) * 8, AMASK)],
                    A + (blockRow + f / A_ROW_F4) * K +
                        (k0 + (f % A_ROW_F4) * 8));
            }
#pragma unroll
        for (int t = 0; t < B_F4_PT; ++t)
            if (t % KSUB == s) {
                int f = tid + t * THREADS;
                cp_async16(
                    &bsp[swz<BN>(f / B_ROW_F4, (f % B_ROW_F4) * 8, BMASK)],
                    B + (k0 + f / B_ROW_F4) * N +
                        (blockCol + (f % B_ROW_F4) * 8));
            }
    };

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        loadFull(s, s);
        cp_commit();
    }
    cp_wait<STAGES - 2>();
    __syncthreads();

    int kTile = STAGES - 1;
    for (int ki = 0; ki < numTiles; ++ki) {
        const int rs = ki % STAGES;
        half* asp = As + rs * APITCH;
        half* bsp = Bs + rs * BPITCH;
        const int wstage = kTile % STAGES;
        const bool doLoad = kTile < numTiles;

#pragma unroll
        for (int s = 0; s < KSUB; ++s) {
            int ks = s * MMA_K;
            uint32_t aF[MTILES][4];
            uint32_t bF[NTILES][2];
#pragma unroll
            for (int i = 0; i < MTILES; ++i) {
                int row = warpRow * WM + i * MMA_M + (lane % 16);
                int col = ks + (lane / 16) * 8;
                ldm_x4(aF[i], &asp[swz<BK>(row, col, AMASK)]);
            }
#pragma unroll
            for (int j = 0; j < NTILES; j += 2) {
                int row = ks + (lane % 16);
                int col = warpCol * WN + j * MMA_N + (lane / 16) * 8;
                uint32_t t[4];
                ldm_x4_trans(t, &bsp[swz<BN>(row, col, BMASK)]);
                bF[j][0] = t[0];
                bF[j][1] = t[1];
                bF[j + 1][0] = t[2];
                bF[j + 1][1] = t[3];
            }
            if (doLoad) loadSlice(wstage, kTile, s);
#pragma unroll
            for (int i = 0; i < MTILES; ++i)
#pragma unroll
                for (int j = 0; j < NTILES; ++j)
                    mma_f16(acc[i][j], aF[i], bF[j]);
        }
        cp_commit();
        ++kTile;
        cp_wait<STAGES - 2>();
        __syncthreads();
    }

    const int group = lane / 4;
    const int tid4 = lane % 4;
#pragma unroll
    for (int i = 0; i < MTILES; ++i) {
#pragma unroll
        for (int j = 0; j < NTILES; ++j) {
            int rowBase = blockRow + warpRow * WM + i * MMA_M;
            int colBase = blockCol + warpCol * WN + j * MMA_N + tid4 * 2;
            *reinterpret_cast<uint32_t*>(&C[(rowBase + group) * N + colBase]) =
                acc[i][j][0];
            *reinterpret_cast<uint32_t*>(
                &C[(rowBase + group + 8) * N + colBase]) = acc[i][j][1];
        }
    }
}

template <int BM, int BN, int BK, int STAGES, int WARP_ROWS, int WARP_COLS,
          int RAST = 0>
inline void launch_f16_sw(size_t m, size_t n, size_t k, const half* A,
                          const half* B, half* C)
{
    constexpr int smemBytes =
        STAGES * (BM * BK + BK * BN) * int(sizeof(half));
    auto kern =
        hgemm_kernel_f16_sw<BM, BN, BK, STAGES, WARP_ROWS, WARP_COLS, RAST>;
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(
            kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes);
        cudaDeviceSynchronize();
        configured = true;
    }
    dim3 block(WARP_ROWS * WARP_COLS * 32);
    dim3 grid(n / BN, m / BM);
    kern<<<grid, block, smemBytes>>>(A, B, C, int(m), int(n), int(k));
}

// -----------------------------------------------------------------------------
// f16-accumulate PERSISTENT: launch exactly (blocksPerSM * numSMs) blocks, each
// grid-strides over output tiles -> no ramp-up/tail wave quantization.
// -----------------------------------------------------------------------------
template <int BM, int BN, int BK, int STAGES, int WARP_ROWS, int WARP_COLS>
__global__ void hgemm_kernel_f16_persist(const half* __restrict__ A,
                                         const half* __restrict__ B,
                                         half* __restrict__ C, int M, int N,
                                         int K)
{
    constexpr int THREADS = WARP_ROWS * WARP_COLS * 32;
    constexpr int WM = BM / WARP_ROWS;
    constexpr int WN = BN / WARP_COLS;
    constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;
    constexpr int MTILES = WM / MMA_M;
    constexpr int NTILES = WN / MMA_N;
    constexpr int APITCH = BM * BK;
    constexpr int BPITCH = BK * BN;
    constexpr int A_ROW_F4 = BK / 8;
    constexpr int B_ROW_F4 = BN / 8;
    constexpr int A_F4_PT = (BM * BK / 8) / THREADS;
    constexpr int B_F4_PT = (BK * BN / 8) / THREADS;
    constexpr int AMASK = (BK / 8 < 8 ? BK / 8 : 8) - 1;
    constexpr int BMASK = (BN / 8 < 16 ? BN / 8 : 16) - 1;

    extern __shared__ half smem[];
    half* As = smem;
    half* Bs = smem + STAGES * APITCH;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warpId = tid >> 5;
    const int warpRow = warpId / WARP_COLS;
    const int warpCol = warpId % WARP_COLS;
    const int numTilesN = N / BN;
    const int totalTiles = (M / BM) * numTilesN;
    const int numKTiles = K / BK;

    for (int tile = blockIdx.x; tile < totalTiles; tile += gridDim.x) {
        const int blockRow = (tile / numTilesN) * BM;
        const int blockCol = (tile % numTilesN) * BN;

        uint32_t acc[MTILES][NTILES][2];
#pragma unroll
        for (int i = 0; i < MTILES; ++i)
#pragma unroll
            for (int j = 0; j < NTILES; ++j) {
                acc[i][j][0] = 0;
                acc[i][j][1] = 0;
            }

        auto loadTile = [&](int stage, int kt) {
            const int k0 = kt * BK;
            half* asp = As + stage * APITCH;
            half* bsp = Bs + stage * BPITCH;
#pragma unroll
            for (int t = 0; t < A_F4_PT; ++t) {
                int f = tid + t * THREADS;
                cp_async16(
                    &asp[swz<BK>(f / A_ROW_F4, (f % A_ROW_F4) * 8, AMASK)],
                    A + (blockRow + f / A_ROW_F4) * K + (k0 + (f % A_ROW_F4) * 8));
            }
#pragma unroll
            for (int t = 0; t < B_F4_PT; ++t) {
                int f = tid + t * THREADS;
                cp_async16(
                    &bsp[swz<BN>(f / B_ROW_F4, (f % B_ROW_F4) * 8, BMASK)],
                    B + (k0 + f / B_ROW_F4) * N + (blockCol + (f % B_ROW_F4) * 8));
            }
        };

#pragma unroll
        for (int s = 0; s < STAGES - 1; ++s) {
            loadTile(s, s);
            cp_commit();
        }
        cp_wait<STAGES - 2>();
        __syncthreads();

        int kTile = STAGES - 1;
        for (int ki = 0; ki < numKTiles; ++ki) {
            if (kTile < numKTiles) loadTile(kTile % STAGES, kTile);
            cp_commit();
            ++kTile;
            const int rs = ki % STAGES;
            half* asp = As + rs * APITCH;
            half* bsp = Bs + rs * BPITCH;
#pragma unroll
            for (int ks = 0; ks < BK; ks += MMA_K) {
                uint32_t aF[MTILES][4];
                uint32_t bF[NTILES][2];
#pragma unroll
                for (int i = 0; i < MTILES; ++i) {
                    int row = warpRow * WM + i * MMA_M + (lane % 16);
                    int col = ks + (lane / 16) * 8;
                    ldm_x4(aF[i], &asp[swz<BK>(row, col, AMASK)]);
                }
#pragma unroll
                for (int j = 0; j < NTILES; j += 2) {
                    int row = ks + (lane % 16);
                    int col = warpCol * WN + j * MMA_N + (lane / 16) * 8;
                    uint32_t t[4];
                    ldm_x4_trans(t, &bsp[swz<BN>(row, col, BMASK)]);
                    bF[j][0] = t[0];
                    bF[j][1] = t[1];
                    bF[j + 1][0] = t[2];
                    bF[j + 1][1] = t[3];
                }
#pragma unroll
                for (int i = 0; i < MTILES; ++i)
#pragma unroll
                    for (int j = 0; j < NTILES; ++j)
                        mma_f16(acc[i][j], aF[i], bF[j]);
            }
            cp_wait<STAGES - 2>();
            __syncthreads();
        }

        const int group = lane / 4;
        const int tid4 = lane % 4;
#pragma unroll
        for (int i = 0; i < MTILES; ++i) {
#pragma unroll
            for (int j = 0; j < NTILES; ++j) {
                int rowBase = blockRow + warpRow * WM + i * MMA_M;
                int colBase = blockCol + warpCol * WN + j * MMA_N + tid4 * 2;
                *reinterpret_cast<uint32_t*>(
                    &C[(rowBase + group) * N + colBase]) = acc[i][j][0];
                *reinterpret_cast<uint32_t*>(
                    &C[(rowBase + group + 8) * N + colBase]) = acc[i][j][1];
            }
        }
        __syncthreads();  // ensure smem reuse safe across output tiles
    }
}

template <int BM, int BN, int BK, int STAGES, int WARP_ROWS, int WARP_COLS,
          int BLOCKS_PER_SM = 2>
inline void launch_f16_persist(size_t m, size_t n, size_t k, const half* A,
                               const half* B, half* C)
{
    constexpr int smemBytes =
        STAGES * (BM * BK + BK * BN) * int(sizeof(half));
    auto kern =
        hgemm_kernel_f16_persist<BM, BN, BK, STAGES, WARP_ROWS, WARP_COLS>;
    static int grid = 0;
    if (grid == 0) {
        cudaFuncSetAttribute(
            kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes);
        int dev = 0;
        cudaGetDevice(&dev);
        int sms = 108;
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        grid = sms * BLOCKS_PER_SM;
        cudaDeviceSynchronize();
    }
    dim3 block(WARP_ROWS * WARP_COLS * 32);
    kern<<<grid, block, smemBytes>>>(A, B, C, int(m), int(n), int(k));
}

}  // namespace hg
}  // namespace playground
