#pragma once
#include <cuda_fp16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>

namespace playground::mmagemm
{

__device__ __forceinline__ void cp_async16(void* dst, const void* src)
{
    unsigned smem = __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem),
                 "l"(src));
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const half* addr)
{
    unsigned s = __cvta_generic_to_shared(addr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(s));
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4],
                                                  const half* addr)
{
    unsigned s = __cvta_generic_to_shared(addr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(s));
}

__device__ __forceinline__ void mma_f16(uint32_t (&d)[2], const uint32_t (&a)[4],
                                        const uint32_t (&b)[2])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(d[0]), "+r"(d[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void mma_f32(float (&d)[4], const uint32_t (&a)[4],
                                        const uint32_t (&b)[2])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Templated f16-accumulate mma GEMM with cp.async multistage + warp-level
// double-buffered fragment prefetch. C = A*B, all row-major.
template <int BM, int BN, int BK, int WARPS_M, int WARPS_N, int NSTAGES,
          int MINBLK, bool DBUF = true>
__global__ void __launch_bounds__(WARPS_M* WARPS_N * 32, MINBLK)
    kernel(int M, int N, int K, const half* __restrict__ A,
           const half* __restrict__ B, half* __restrict__ C)
{
    constexpr int WM = BM / WARPS_M;
    constexpr int WN = BN / WARPS_N;
    constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;
    constexpr int FM = WM / MMA_M;
    constexpr int FN = WN / MMA_N;
    constexpr int FN16 = WN / 16;
    constexpr int KSTEPS = BK / MMA_K;
    constexpr int NTHREADS = WARPS_M * WARPS_N * 32;
    constexpr int PAD = 8;
    constexpr int LDA_S = BK + PAD;
    constexpr int LDB_S = BN + PAD;
    constexpr int AS_SZ = BM * LDA_S;
    constexpr int BS_SZ = BK * LDB_S;

    extern __shared__ half smem[];
    half* As = smem;
    half* Bs = smem + NSTAGES * AS_SZ;

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;
    const int warpId = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int warpRow = (warpId / WARPS_N) * WM;
    const int warpCol = (warpId % WARPS_N) * WN;

    uint32_t acc[FM][FN][2];
#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j) { acc[i][j][0] = 0; acc[i][j][1] = 0; }

    const int tid = threadIdx.x;
    const int a_row0 = tid / (BK / 8);
    const int a_col0 = (tid % (BK / 8)) * 8;
    constexpr int A_ROW_STRIDE = NTHREADS / (BK / 8);
    const int b_row0 = tid / (BN / 8);
    const int b_col0 = (tid % (BN / 8)) * 8;
    constexpr int B_ROW_STRIDE = NTHREADS / (BN / 8);

    const int nKTiles = K / BK;

    auto load_tile = [&](int kt, int stage) {
        half* Asd = As + stage * AS_SZ;
        half* Bsd = Bs + stage * BS_SZ;
#pragma unroll
        for (int r = 0; r < BM; r += A_ROW_STRIDE)
            cp_async16(&Asd[(a_row0 + r) * LDA_S + a_col0],
                       &A[(blockRow + a_row0 + r) * K + kt + a_col0]);
#pragma unroll
        for (int r = 0; r < BK; r += B_ROW_STRIDE)
            cp_async16(&Bsd[(b_row0 + r) * LDB_S + b_col0],
                       &B[(kt + b_row0 + r) * N + blockCol + b_col0]);
        __pipeline_commit();
    };

    auto ld_frag = [&](uint32_t (&a_buf)[FM][4], uint32_t (&b_buf)[FN][2],
                       int stage, int kk) {
        half* Asd = As + stage * AS_SZ;
        half* Bsd = Bs + stage * BS_SZ;
#pragma unroll
        for (int i = 0; i < FM; ++i)
            ldmatrix_x4(a_buf[i], &Asd[(warpRow + i * MMA_M + (lane % 16)) * LDA_S
                                       + kk + (lane / 16) * 8]);
#pragma unroll
        for (int jj = 0; jj < FN16; ++jj) {
            uint32_t tmp[4];
            ldmatrix_x4_trans(tmp, &Bsd[(kk + (lane % 16)) * LDB_S + warpCol +
                                        jj * 16 + (lane / 16) * 8]);
            b_buf[jj * 2][0] = tmp[0];
            b_buf[jj * 2][1] = tmp[1];
            b_buf[jj * 2 + 1][0] = tmp[2];
            b_buf[jj * 2 + 1][1] = tmp[3];
        }
    };

#pragma unroll
    for (int s = 0; s < NSTAGES - 1; ++s) load_tile(s * BK, s);

    int readStage = 0;
    int writeStage = NSTAGES - 1;
    __pipeline_wait_prior(NSTAGES - 2);
    __syncthreads();

    if constexpr (DBUF) {
        uint32_t a_buf[2][FM][4];
        uint32_t b_buf[2][FN][2];
        ld_frag(a_buf[0], b_buf[0], readStage, 0);

        for (int kt = 0; kt < nKTiles; ++kt) {
#pragma unroll
            for (int ks = 0; ks < KSTEPS; ++ks) {
                int slot = ks & 1;
                if (ks == 0) {
                    int loadKt = kt + (NSTAGES - 1);
                    if (loadKt < nKTiles) load_tile(loadKt * BK, writeStage);
                    else __pipeline_commit();
                }
                if (ks + 1 < KSTEPS) {
                    ld_frag(a_buf[slot ^ 1], b_buf[slot ^ 1], readStage,
                            (ks + 1) * MMA_K);
                } else {
                    readStage = (readStage + 1) % NSTAGES;
                    writeStage = (writeStage + 1) % NSTAGES;
                    __pipeline_wait_prior(NSTAGES - 2);
                    __syncthreads();
                    if (kt + 1 < nKTiles)
                        ld_frag(a_buf[slot ^ 1], b_buf[slot ^ 1], readStage, 0);
                }
#pragma unroll
                for (int i = 0; i < FM; ++i)
#pragma unroll
                    for (int j = 0; j < FN; ++j)
                        mma_f16(acc[i][j], a_buf[slot][i], b_buf[slot][j]);
            }
        }
    } else {
        // Single-buffered fragments: lower register pressure, higher occupancy.
        uint32_t a_buf[FM][4];
        uint32_t b_buf[FN][2];
        for (int kt = 0; kt < nKTiles; ++kt) {
            int loadKt = kt + (NSTAGES - 1);
            if (loadKt < nKTiles) load_tile(loadKt * BK, writeStage);
            else __pipeline_commit();
#pragma unroll
            for (int ks = 0; ks < KSTEPS; ++ks) {
                ld_frag(a_buf, b_buf, readStage, ks * MMA_K);
#pragma unroll
                for (int i = 0; i < FM; ++i)
#pragma unroll
                    for (int j = 0; j < FN; ++j)
                        mma_f16(acc[i][j], a_buf[i], b_buf[j]);
            }
            readStage = (readStage + 1) % NSTAGES;
            writeStage = (writeStage + 1) % NSTAGES;
            __pipeline_wait_prior(NSTAGES - 2);
            __syncthreads();
        }
    }

#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j) {
            int cr = blockRow + warpRow + i * MMA_M;
            int cc = blockCol + warpCol + j * MMA_N;
            int row = lane / 4;
            int col = (lane % 4) * 2;
            *reinterpret_cast<uint32_t*>(&C[(cr + row) * N + cc + col]) =
                acc[i][j][0];
            *reinterpret_cast<uint32_t*>(&C[(cr + row + 8) * N + cc + col]) =
                acc[i][j][1];
        }
}

// Map a linear block id to (blockRow, blockCol) with a column-major
// super-grouping over GM tile-rows to improve L2 reuse.
template <int BM, int BN, int GM>
__device__ __forceinline__ void swizzle_block(int& blockRow, int& blockCol,
                                              int M, int N)
{
    int blocksN = (N + BN - 1) / BN;
    int blocksM = (M + BM - 1) / BM;
    int bid = blockIdx.x + blockIdx.y * gridDim.x;
    int perGroup = GM * blocksN;
    int gid = bid / perGroup;
    int firstM = gid * GM;
    int sizeM = blocksM - firstM;
    if (sizeM > GM) sizeM = GM;
    int local = bid - gid * perGroup;
    int row = firstM + (local % sizeM);
    int col = local / sizeM;
    blockRow = row * BM;
    blockCol = col * BN;
}

// ---- split parity-bank accumulator (two f16 banks summed in f32) ----
template <int BM, int BN, int BK, int WARPS_M, int WARPS_N, int NSTAGES,
          int MINBLK, int SWZ = 1>
__global__ void __launch_bounds__(WARPS_M* WARPS_N * 32, MINBLK)
    kernel_split(int M, int N, int K, const half* __restrict__ A,
                 const half* __restrict__ B, half* __restrict__ C)
{
    constexpr int WM = BM / WARPS_M;
    constexpr int WN = BN / WARPS_N;
    constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;
    constexpr int FM = WM / MMA_M;
    constexpr int FN = WN / MMA_N;
    constexpr int FN16 = WN / 16;
    constexpr int KSTEPS = BK / MMA_K;  // must be 2 (parity == slot)
    constexpr int NTHREADS = WARPS_M * WARPS_N * 32;
    constexpr int PAD = 8;
    constexpr int LDA_S = BK + PAD;
    constexpr int LDB_S = BN + PAD;
    constexpr int AS_SZ = BM * LDA_S;
    constexpr int BS_SZ = BK * LDB_S;

    extern __shared__ half smem[];
    half* As = smem;
    half* Bs = smem + NSTAGES * AS_SZ;

    int blockRow, blockCol;
    if constexpr (SWZ > 1) swizzle_block<BM, BN, SWZ>(blockRow, blockCol, M, N);
    else { blockRow = blockIdx.y * BM; blockCol = blockIdx.x * BN; }
    const int warpId = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int warpRow = (warpId / WARPS_N) * WM;
    const int warpCol = (warpId % WARPS_N) * WN;

    uint32_t acc[2][FM][FN][2];
#pragma unroll
    for (int p = 0; p < 2; ++p)
#pragma unroll
        for (int i = 0; i < FM; ++i)
#pragma unroll
            for (int j = 0; j < FN; ++j) { acc[p][i][j][0] = 0; acc[p][i][j][1] = 0; }

    const int tid = threadIdx.x;
    const int a_row0 = tid / (BK / 8);
    const int a_col0 = (tid % (BK / 8)) * 8;
    constexpr int A_ROW_STRIDE = NTHREADS / (BK / 8);
    const int b_row0 = tid / (BN / 8);
    const int b_col0 = (tid % (BN / 8)) * 8;
    constexpr int B_ROW_STRIDE = NTHREADS / (BN / 8);
    const int nKTiles = K / BK;

    auto load_tile = [&](int kt, int stage) {
        half* Asd = As + stage * AS_SZ;
        half* Bsd = Bs + stage * BS_SZ;
#pragma unroll
        for (int r = 0; r < BM; r += A_ROW_STRIDE)
            cp_async16(&Asd[(a_row0 + r) * LDA_S + a_col0],
                       &A[(blockRow + a_row0 + r) * K + kt + a_col0]);
#pragma unroll
        for (int r = 0; r < BK; r += B_ROW_STRIDE)
            cp_async16(&Bsd[(b_row0 + r) * LDB_S + b_col0],
                       &B[(kt + b_row0 + r) * N + blockCol + b_col0]);
        __pipeline_commit();
    };
    auto ld_frag = [&](uint32_t (&a_buf)[FM][4], uint32_t (&b_buf)[FN][2],
                       int stage, int kk) {
        half* Asd = As + stage * AS_SZ;
        half* Bsd = Bs + stage * BS_SZ;
#pragma unroll
        for (int i = 0; i < FM; ++i)
            ldmatrix_x4(a_buf[i], &Asd[(warpRow + i * MMA_M + (lane % 16)) * LDA_S
                                       + kk + (lane / 16) * 8]);
#pragma unroll
        for (int jj = 0; jj < FN16; ++jj) {
            uint32_t tmp[4];
            ldmatrix_x4_trans(tmp, &Bsd[(kk + (lane % 16)) * LDB_S + warpCol +
                                        jj * 16 + (lane / 16) * 8]);
            b_buf[jj * 2][0] = tmp[0]; b_buf[jj * 2][1] = tmp[1];
            b_buf[jj * 2 + 1][0] = tmp[2]; b_buf[jj * 2 + 1][1] = tmp[3];
        }
    };

#pragma unroll
    for (int s = 0; s < NSTAGES - 1; ++s) load_tile(s * BK, s);
    int readStage = 0, writeStage = NSTAGES - 1;
    __pipeline_wait_prior(NSTAGES - 2);
    __syncthreads();

    uint32_t a_buf[2][FM][4];
    uint32_t b_buf[2][FN][2];
    ld_frag(a_buf[0], b_buf[0], readStage, 0);

    for (int kt = 0; kt < nKTiles; ++kt) {
#pragma unroll
        for (int ks = 0; ks < KSTEPS; ++ks) {
            int slot = ks & 1;
            if (ks == 0) {
                int loadKt = kt + (NSTAGES - 1);
                if (loadKt < nKTiles) load_tile(loadKt * BK, writeStage);
                else __pipeline_commit();
            }
            if (ks + 1 < KSTEPS) {
                ld_frag(a_buf[slot ^ 1], b_buf[slot ^ 1], readStage,
                        (ks + 1) * MMA_K);
            } else {
                readStage = (readStage + 1) % NSTAGES;
                writeStage = (writeStage + 1) % NSTAGES;
                __pipeline_wait_prior(NSTAGES - 2);
                __syncthreads();
                if (kt + 1 < nKTiles)
                    ld_frag(a_buf[slot ^ 1], b_buf[slot ^ 1], readStage, 0);
            }
#pragma unroll
            for (int i = 0; i < FM; ++i)
#pragma unroll
                for (int j = 0; j < FN; ++j)
                    mma_f16(acc[slot][i][j], a_buf[slot][i], b_buf[slot][j]);
        }
    }

#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j) {
            int cr = blockRow + warpRow + i * MMA_M;
            int cc = blockCol + warpCol + j * MMA_N;
            int row = lane / 4;
            int col = (lane % 4) * 2;
            half2 e0 = *reinterpret_cast<half2*>(&acc[0][i][j][0]);
            half2 o0 = *reinterpret_cast<half2*>(&acc[1][i][j][0]);
            half2 e1 = *reinterpret_cast<half2*>(&acc[0][i][j][1]);
            half2 o1 = *reinterpret_cast<half2*>(&acc[1][i][j][1]);
            float2 s0 = __half22float2(e0), t0 = __half22float2(o0);
            float2 s1 = __half22float2(e1), t1 = __half22float2(o1);
            half2 r0 = __floats2half2_rn(s0.x + t0.x, s0.y + t0.y);
            half2 r1 = __floats2half2_rn(s1.x + t1.x, s1.y + t1.y);
            *reinterpret_cast<half2*>(&C[(cr + row) * N + cc + col]) = r0;
            *reinterpret_cast<half2*>(&C[(cr + row + 8) * N + cc + col]) = r1;
        }
}

template <int BM, int BN, int BK, int WARPS_M, int WARPS_N, int NSTAGES,
          int MINBLK, int SWZ = 1>
void launch_split(int m, int n, int k, const half* A, const half* B, half* C)
{
    constexpr int PAD = 8;
    constexpr int SMEM =
        NSTAGES * (BM * (BK + PAD) + BK * (BN + PAD)) * (int) sizeof(half);
    constexpr int NTHREADS = WARPS_M * WARPS_N * 32;
    auto fn = kernel_split<BM, BN, BK, WARPS_M, WARPS_N, NSTAGES, MINBLK, SWZ>;
    static bool cfg = false;
    if (!cfg) {
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             SMEM);
        cfg = true;
    }
    dim3 block(NTHREADS);
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    fn<<<grid, block, SMEM>>>(m, n, k, A, B, C);
}

template <int BM, int BN, int BK, int WARPS_M, int WARPS_N, int NSTAGES,
          int MINBLK, bool DBUF = true>
void launch(int m, int n, int k, const half* A, const half* B, half* C)
{
    constexpr int PAD = 8;
    constexpr int AS_SZ = BM * (BK + PAD);
    constexpr int BS_SZ = BK * (BN + PAD);
    constexpr int SMEM = NSTAGES * (AS_SZ + BS_SZ) * (int) sizeof(half);
    constexpr int NTHREADS = WARPS_M * WARPS_N * 32;
    auto fn = kernel<BM, BN, BK, WARPS_M, WARPS_N, NSTAGES, MINBLK, DBUF>;
    static bool cfg = false;
    if (!cfg) {
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             SMEM);
        cfg = true;
    }
    dim3 block(NTHREADS);
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    fn<<<grid, block, SMEM>>>(m, n, k, A, B, C);
}

// ---- f32-accumulate variant (double-buffered) for precision safety ----
template <int BM, int BN, int BK, int WARPS_M, int WARPS_N, int NSTAGES,
          int MINBLK, int SWZ = 1>
__global__ void __launch_bounds__(WARPS_M* WARPS_N * 32, MINBLK)
    kernel_f32(int M, int N, int K, const half* __restrict__ A,
               const half* __restrict__ B, half* __restrict__ C)
{
    constexpr int WM = BM / WARPS_M;
    constexpr int WN = BN / WARPS_N;
    constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;
    constexpr int FM = WM / MMA_M;
    constexpr int FN = WN / MMA_N;
    constexpr int FN16 = WN / 16;
    constexpr int KSTEPS = BK / MMA_K;
    constexpr int NTHREADS = WARPS_M * WARPS_N * 32;
    constexpr int PAD = 8;
    constexpr int LDA_S = BK + PAD;
    constexpr int LDB_S = BN + PAD;
    constexpr int AS_SZ = BM * LDA_S;
    constexpr int BS_SZ = BK * LDB_S;

    extern __shared__ half smem[];
    half* As = smem;
    half* Bs = smem + NSTAGES * AS_SZ;

    int blockRow, blockCol;
    if constexpr (SWZ > 1) {
        swizzle_block<BM, BN, SWZ>(blockRow, blockCol, M, N);
    } else {
        blockRow = blockIdx.y * BM;
        blockCol = blockIdx.x * BN;
    }
    const int warpId = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int warpRow = (warpId / WARPS_N) * WM;
    const int warpCol = (warpId % WARPS_N) * WN;

    float acc[FM][FN][4];
#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j)
#pragma unroll
            for (int t = 0; t < 4; ++t) acc[i][j][t] = 0.0f;

    const int tid = threadIdx.x;
    const int a_row0 = tid / (BK / 8);
    const int a_col0 = (tid % (BK / 8)) * 8;
    constexpr int A_ROW_STRIDE = NTHREADS / (BK / 8);
    const int b_row0 = tid / (BN / 8);
    const int b_col0 = (tid % (BN / 8)) * 8;
    constexpr int B_ROW_STRIDE = NTHREADS / (BN / 8);
    const int nKTiles = K / BK;

    auto load_tile = [&](int kt, int stage) {
        half* Asd = As + stage * AS_SZ;
        half* Bsd = Bs + stage * BS_SZ;
#pragma unroll
        for (int r = 0; r < BM; r += A_ROW_STRIDE)
            cp_async16(&Asd[(a_row0 + r) * LDA_S + a_col0],
                       &A[(blockRow + a_row0 + r) * K + kt + a_col0]);
#pragma unroll
        for (int r = 0; r < BK; r += B_ROW_STRIDE)
            cp_async16(&Bsd[(b_row0 + r) * LDB_S + b_col0],
                       &B[(kt + b_row0 + r) * N + blockCol + b_col0]);
        __pipeline_commit();
    };
    auto ld_frag = [&](uint32_t (&a_buf)[FM][4], uint32_t (&b_buf)[FN][2],
                       int stage, int kk) {
        half* Asd = As + stage * AS_SZ;
        half* Bsd = Bs + stage * BS_SZ;
#pragma unroll
        for (int i = 0; i < FM; ++i)
            ldmatrix_x4(a_buf[i], &Asd[(warpRow + i * MMA_M + (lane % 16)) * LDA_S
                                       + kk + (lane / 16) * 8]);
#pragma unroll
        for (int jj = 0; jj < FN16; ++jj) {
            uint32_t tmp[4];
            ldmatrix_x4_trans(tmp, &Bsd[(kk + (lane % 16)) * LDB_S + warpCol +
                                        jj * 16 + (lane / 16) * 8]);
            b_buf[jj * 2][0] = tmp[0]; b_buf[jj * 2][1] = tmp[1];
            b_buf[jj * 2 + 1][0] = tmp[2]; b_buf[jj * 2 + 1][1] = tmp[3];
        }
    };

#pragma unroll
    for (int s = 0; s < NSTAGES - 1; ++s) load_tile(s * BK, s);
    int readStage = 0, writeStage = NSTAGES - 1;
    __pipeline_wait_prior(NSTAGES - 2);
    __syncthreads();

    uint32_t a_buf[2][FM][4];
    uint32_t b_buf[2][FN][2];
    ld_frag(a_buf[0], b_buf[0], readStage, 0);

    for (int kt = 0; kt < nKTiles; ++kt) {
#pragma unroll
        for (int ks = 0; ks < KSTEPS; ++ks) {
            int slot = ks & 1;
            if (ks == 0) {
                int loadKt = kt + (NSTAGES - 1);
                if (loadKt < nKTiles) load_tile(loadKt * BK, writeStage);
                else __pipeline_commit();
            }
            if (ks + 1 < KSTEPS) {
                ld_frag(a_buf[slot ^ 1], b_buf[slot ^ 1], readStage,
                        (ks + 1) * MMA_K);
            } else {
                readStage = (readStage + 1) % NSTAGES;
                writeStage = (writeStage + 1) % NSTAGES;
                __pipeline_wait_prior(NSTAGES - 2);
                __syncthreads();
                if (kt + 1 < nKTiles)
                    ld_frag(a_buf[slot ^ 1], b_buf[slot ^ 1], readStage, 0);
            }
#pragma unroll
            for (int i = 0; i < FM; ++i)
#pragma unroll
                for (int j = 0; j < FN; ++j)
                    mma_f32(acc[i][j], a_buf[slot][i], b_buf[slot][j]);
        }
    }

#pragma unroll
    for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j) {
            int cr = blockRow + warpRow + i * MMA_M;
            int cc = blockCol + warpCol + j * MMA_N;
            int row = lane / 4;
            int col = (lane % 4) * 2;
            half2 lo = __floats2half2_rn(acc[i][j][0], acc[i][j][1]);
            half2 hi = __floats2half2_rn(acc[i][j][2], acc[i][j][3]);
            *reinterpret_cast<half2*>(&C[(cr + row) * N + cc + col]) = lo;
            *reinterpret_cast<half2*>(&C[(cr + row + 8) * N + cc + col]) = hi;
        }
}

template <int BM, int BN, int BK, int WARPS_M, int WARPS_N, int NSTAGES,
          int MINBLK, int SWZ = 1>
void launch_f32(int m, int n, int k, const half* A, const half* B, half* C)
{
    constexpr int PAD = 8;
    constexpr int SMEM =
        NSTAGES * (BM * (BK + PAD) + BK * (BN + PAD)) * (int) sizeof(half);
    constexpr int NTHREADS = WARPS_M * WARPS_N * 32;
    auto fn = kernel_f32<BM, BN, BK, WARPS_M, WARPS_N, NSTAGES, MINBLK, SWZ>;
    static bool cfg = false;
    if (!cfg) {
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             SMEM);
        cfg = true;
    }
    dim3 block(NTHREADS);
    dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM);
    fn<<<grid, block, SMEM>>>(m, n, k, A, B, C);
}

}  // namespace playground::mmagemm
