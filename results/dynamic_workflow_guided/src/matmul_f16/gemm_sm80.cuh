// gemm_sm80.cuh — parameterized SM80 fp16 tensor-core GEMM.
// ONE kernel, every optimization ingredient is a compile-time toggle so that
// (a) tile-shape tuning and (b) leave-one-out ablations are single-config flips
// on an otherwise byte-identical kernel (fair ablation; the compute math is a
// shared device function used by every variant).
//
// Ingredients / toggles:
//   tile shape / register blocking : BM,BN,BK, WARPS_M,WARPS_N  (warp tile = BM/WARPS_M x BN/WARPS_N)
//   multi-stage software pipeline  : STAGES (>=1; 1 = single buffer, no overlap)
//   cp.async global->shared        : CP_ASYNC (false = synchronous; register-prefetch double buffer when STAGES>1)
//   shared-memory XOR swizzle      : SWIZZLE  (false = contiguous, bank-conflict-laden)
//   128-bit vectorized load        : COPY_BYTES (16 = float4/cp.async.cg; <16 = narrow cp.async.ca)
//   ldmatrix -> mma                : LDMATRIX (false = manual per-thread fragment loads)
//
// Assumes m,n,k multiples of the block tile (task fixes 4096^3). fp32 accumulate.
#pragma once
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace gemm_sm80
{
using half = ::half;

__device__ __forceinline__ uint32_t smem_u32(const void* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// XOR swizzle at 8-half (16-byte) segment granularity. COLS = halves per row.
template <int COLS, bool SW>
__device__ __forceinline__ int swz(int row, int col)
{
    if constexpr (!SW) {
        return row * COLS + col;
    } else {
        constexpr int NSEG = COLS >> 3;
        int seg = (col >> 3) ^ (row & (NSEG - 1));
        return row * COLS + (seg << 3) + (col & 7);
    }
}

template <int CB>
__device__ __forceinline__ void cp_async(uint32_t d, const void* s)
{
    if constexpr (CB == 16) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d),
                     "l"(s));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::"r"(d),
                     "l"(s), "n"(CB));
    }
}

__device__ __forceinline__ void ldm_x4(uint32_t (&r)[4], uint32_t a)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(a));
}

__device__ __forceinline__ void ldm_x2_trans(uint32_t (&r)[2], uint32_t a)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1])
        : "r"(a));
}

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4],
                                          const uint32_t (&b)[2],
                                          const float (&c)[4])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// ---- shared -> register fragments + MMA over one BK-tile (shared by all variants) ----
template <int BK, int BN, int WARPS_N, int WMITER, int WNITER, int KSTEPS,
          bool SWIZZLE, bool LDMATRIX, bool FRAG_PIPE>
__device__ __forceinline__ void compute_tile(const half* rA, const half* rB,
                                             float acc[WMITER][WNITER][4],
                                             int lane, int warp_m, int warp_n)
{
    constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;
    const int ar = (lane & 7) + ((lane >> 3) & 1) * 8;
    const int ac = ((lane >> 3) >> 1) * 8;
    const int br = lane & 15;
    const int gid = lane >> 2;
    const int t4 = lane & 3;

    // Load all A/B fragments for k-substep `ks` (ldmatrix or manual).
    auto load_frag = [&](uint32_t (&af)[WMITER][4], uint32_t (&bf)[WNITER][2],
                         int ks) {
        const int kk = ks * MMA_K;
        if constexpr (LDMATRIX) {
#pragma unroll
            for (int mi = 0; mi < WMITER; ++mi)
                ldm_x4(af[mi], smem_u32(&rA[swz<BK, SWIZZLE>(
                                   warp_m + mi * MMA_M + ar, kk + ac)]));
#pragma unroll
            for (int ni = 0; ni < WNITER; ++ni)
                ldm_x2_trans(bf[ni], smem_u32(&rB[swz<BN, SWIZZLE>(
                                         kk + br, warp_n + ni * MMA_N)]));
        } else {
            // Manual fragment loads (no ldmatrix). Fill the exact m16n8k16 layout.
#pragma unroll
            for (int mi = 0; mi < WMITER; ++mi) {
                const int r0 = warp_m + mi * MMA_M;
                half2 a0 = *reinterpret_cast<const half2*>(
                    &rA[swz<BK, SWIZZLE>(r0 + gid, kk + 2 * t4)]);
                half2 a1 = *reinterpret_cast<const half2*>(
                    &rA[swz<BK, SWIZZLE>(r0 + gid + 8, kk + 2 * t4)]);
                half2 a2 = *reinterpret_cast<const half2*>(
                    &rA[swz<BK, SWIZZLE>(r0 + gid, kk + 2 * t4 + 8)]);
                half2 a3 = *reinterpret_cast<const half2*>(
                    &rA[swz<BK, SWIZZLE>(r0 + gid + 8, kk + 2 * t4 + 8)]);
                af[mi][0] = *reinterpret_cast<uint32_t*>(&a0);
                af[mi][1] = *reinterpret_cast<uint32_t*>(&a1);
                af[mi][2] = *reinterpret_cast<uint32_t*>(&a2);
                af[mi][3] = *reinterpret_cast<uint32_t*>(&a3);
            }
#pragma unroll
            for (int ni = 0; ni < WNITER; ++ni) {
                const int c0 = warp_n + ni * MMA_N + gid;
                half b0a = rB[swz<BN, SWIZZLE>(kk + 2 * t4, c0)];
                half b0b = rB[swz<BN, SWIZZLE>(kk + 2 * t4 + 1, c0)];
                half b1a = rB[swz<BN, SWIZZLE>(kk + 2 * t4 + 8, c0)];
                half b1b = rB[swz<BN, SWIZZLE>(kk + 2 * t4 + 9, c0)];
                half2 b0 = __halves2half2(b0a, b0b);
                half2 b1 = __halves2half2(b1a, b1b);
                bf[ni][0] = *reinterpret_cast<uint32_t*>(&b0);
                bf[ni][1] = *reinterpret_cast<uint32_t*>(&b1);
            }
        }
    };
    auto do_mma = [&](uint32_t (&af)[WMITER][4], uint32_t (&bf)[WNITER][2]) {
#pragma unroll
        for (int mi = 0; mi < WMITER; ++mi)
#pragma unroll
            for (int ni = 0; ni < WNITER; ++ni)
                mma16816(acc[mi][ni], af[mi], bf[ni], acc[mi][ni]);
    };

    if constexpr (FRAG_PIPE && KSTEPS > 1) {
        // Register-level double buffer: prefetch ks+1 fragments (ldmatrix) while
        // the tensor cores consume ks -> overlaps shared loads with MMA.
        uint32_t af[2][WMITER][4];
        uint32_t bf[2][WNITER][2];
        load_frag(af[0], bf[0], 0);
#pragma unroll
        for (int ks = 0; ks < KSTEPS; ++ks) {
            if (ks + 1 < KSTEPS)
                load_frag(af[(ks + 1) & 1], bf[(ks + 1) & 1], ks + 1);
            do_mma(af[ks & 1], bf[ks & 1]);
        }
    } else {
#pragma unroll
        for (int ks = 0; ks < KSTEPS; ++ks) {
            uint32_t af[WMITER][4];
            uint32_t bf[WNITER][2];
            load_frag(af, bf, ks);
            do_mma(af, bf);
        }
    }
}

template <int BM, int BN, int BK, int WARPS_M, int WARPS_N, int STAGES,
          bool CP_ASYNC, bool SWIZZLE, int COPY_BYTES, bool LDMATRIX,
          bool FRAG_PIPE = true>
__global__ void gemm(const half* __restrict__ A, const half* __restrict__ B,
                     half* __restrict__ C, int M, int N, int K)
{
    constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N;
    constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;
    constexpr int WMITER = WM / MMA_M, WNITER = WN / MMA_N, KSTEPS = BK / MMA_K;
    constexpr int THREADS = WARPS_M * WARPS_N * 32;
    constexpr int ASIZE = BM * BK, BSIZE = BK * BN;
    constexpr int NBUF = CP_ASYNC ? STAGES : (STAGES > 1 ? 2 : 1);
    constexpr int ASEG_PR = BK / 8;        // A segments per row
    constexpr int BSEG_PR = BN / 8;        // B segments per row
    constexpr int ASEGS = BM * ASEG_PR;    // total A 16B-segments in a tile
    constexpr int BSEGS = BK * BSEG_PR;
    constexpr int SEG_A = ASEGS / THREADS;  // A segments per thread
    constexpr int SEG_B = BSEGS / THREADS;

    extern __shared__ half smem[];
    half* sA = smem;
    half* sB = smem + NBUF * ASIZE;

    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int warp_m = (warp / WARPS_N) * WM;
    const int warp_n = (warp % WARPS_N) * WN;
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    float acc[WMITER][WNITER][4];
#pragma unroll
    for (int i = 0; i < WMITER; ++i)
#pragma unroll
        for (int j = 0; j < WNITER; ++j)
#pragma unroll
            for (int t = 0; t < 4; ++t)
                acc[i][j][t] = 0.0f;

    const int NTILES = K / BK;

    // cp.async load of one BK-tile into shared buffer `buf`.
    auto cp_tile = [&](int buf, int k0) {
        half* dA = sA + buf * ASIZE;
        half* dB = sB + buf * BSIZE;
#pragma unroll
        for (int i = 0; i < SEG_A; ++i) {
            int s = tid + i * THREADS;
            int row = s / ASEG_PR, col = (s % ASEG_PR) * 8;
            uint32_t d = smem_u32(&dA[swz<BK, SWIZZLE>(row, col)]);
            const char* g = reinterpret_cast<const char*>(
                &A[(block_row + row) * K + (k0 + col)]);
#pragma unroll
            for (int j = 0; j < 16 / COPY_BYTES; ++j)
                cp_async<COPY_BYTES>(d + j * COPY_BYTES, g + j * COPY_BYTES);
        }
#pragma unroll
        for (int i = 0; i < SEG_B; ++i) {
            int s = tid + i * THREADS;
            int row = s / BSEG_PR, col = (s % BSEG_PR) * 8;
            uint32_t d = smem_u32(&dB[swz<BN, SWIZZLE>(row, col)]);
            const char* g = reinterpret_cast<const char*>(
                &B[(k0 + row) * N + (block_col + col)]);
#pragma unroll
            for (int j = 0; j < 16 / COPY_BYTES; ++j)
                cp_async<COPY_BYTES>(d + j * COPY_BYTES, g + j * COPY_BYTES);
        }
    };

    if constexpr (CP_ASYNC) {
        constexpr int WAIT = STAGES >= 2 ? STAGES - 2 : 0;
        if constexpr (STAGES == 1) {
            for (int tile = 0; tile < NTILES; ++tile) {
                cp_tile(0, tile * BK);
                asm volatile("cp.async.commit_group;\n");
                asm volatile("cp.async.wait_group 0;\n");
                __syncthreads();
                compute_tile<BK, BN, WARPS_N, WMITER, WNITER, KSTEPS, SWIZZLE,
                             LDMATRIX, FRAG_PIPE>(sA, sB, acc, lane, warp_m, warp_n);
                __syncthreads();
            }
        } else {
#pragma unroll
            for (int s = 0; s < STAGES - 1; ++s) {
                cp_tile(s, s * BK);
                asm volatile("cp.async.commit_group;\n");
            }
            for (int tile = 0; tile < NTILES; ++tile) {
                asm volatile("cp.async.wait_group %0;\n" ::"n"(WAIT));
                __syncthreads();
                int read_buf = tile % STAGES;
                int pf = tile + STAGES - 1;
                if (pf < NTILES)
                    cp_tile(pf % STAGES, pf * BK);
                asm volatile("cp.async.commit_group;\n");
                compute_tile<BK, BN, WARPS_N, WMITER, WNITER, KSTEPS, SWIZZLE,
                             LDMATRIX, FRAG_PIPE>(sA + read_buf * ASIZE,
                                       sB + read_buf * BSIZE, acc, lane, warp_m,
                                       warp_n);
            }
        }
    } else {
        // Synchronous global->register load + register->shared store. Scoped to
        // this branch so the float4 scratch never costs registers on the cp.async path.
        float4 ra[SEG_A > 0 ? SEG_A : 1], rb[SEG_B > 0 ? SEG_B : 1];
        auto ld_glob = [&](int k0) {
#pragma unroll
            for (int i = 0; i < SEG_A; ++i) {
                int s = tid + i * THREADS;
                int row = s / ASEG_PR, col = (s % ASEG_PR) * 8;
                ra[i] = *reinterpret_cast<const float4*>(
                    &A[(block_row + row) * K + (k0 + col)]);
            }
#pragma unroll
            for (int i = 0; i < SEG_B; ++i) {
                int s = tid + i * THREADS;
                int row = s / BSEG_PR, col = (s % BSEG_PR) * 8;
                rb[i] = *reinterpret_cast<const float4*>(
                    &B[(k0 + row) * N + (block_col + col)]);
            }
        };
        auto st_shared = [&](int buf) {
            half* dA = sA + buf * ASIZE;
            half* dB = sB + buf * BSIZE;
#pragma unroll
            for (int i = 0; i < SEG_A; ++i) {
                int s = tid + i * THREADS;
                int row = s / ASEG_PR, col = (s % ASEG_PR) * 8;
                *reinterpret_cast<float4*>(&dA[swz<BK, SWIZZLE>(row, col)]) =
                    ra[i];
            }
#pragma unroll
            for (int i = 0; i < SEG_B; ++i) {
                int s = tid + i * THREADS;
                int row = s / BSEG_PR, col = (s % BSEG_PR) * 8;
                *reinterpret_cast<float4*>(&dB[swz<BN, SWIZZLE>(row, col)]) =
                    rb[i];
            }
        };
        if constexpr (STAGES == 1) {
            for (int tile = 0; tile < NTILES; ++tile) {
                ld_glob(tile * BK);
                st_shared(0);
                __syncthreads();
                compute_tile<BK, BN, WARPS_N, WMITER, WNITER, KSTEPS, SWIZZLE,
                             LDMATRIX, FRAG_PIPE>(sA, sB, acc, lane, warp_m, warp_n);
                __syncthreads();
            }
        } else {
            // register-prefetch double buffer (faithful non-async pipeline)
            ld_glob(0);
            st_shared(0);
            __syncthreads();
            for (int tile = 0; tile < NTILES; ++tile) {
                int cur = tile & 1;
                if (tile + 1 < NTILES)
                    ld_glob((tile + 1) * BK);  // LDG overlaps compute
                compute_tile<BK, BN, WARPS_N, WMITER, WNITER, KSTEPS, SWIZZLE,
                             LDMATRIX, FRAG_PIPE>(sA + cur * ASIZE, sB + cur * BSIZE, acc,
                                       lane, warp_m, warp_n);
                if (tile + 1 < NTILES) {
                    st_shared((tile + 1) & 1);
                    __syncthreads();
                }
            }
        }
    }

    // ---- store C (row-major, m16n8k16 C layout) ----
    const int gid = lane >> 2, t4 = lane & 3;
#pragma unroll
    for (int mi = 0; mi < WMITER; ++mi)
#pragma unroll
        for (int ni = 0; ni < WNITER; ++ni) {
            int row = block_row + warp_m + mi * MMA_M + gid;
            int col = block_col + warp_n + ni * MMA_N + t4 * 2;
            __half2 lo = __floats2half2_rn(acc[mi][ni][0], acc[mi][ni][1]);
            __half2 hi = __floats2half2_rn(acc[mi][ni][2], acc[mi][ni][3]);
            *reinterpret_cast<__half2*>(&C[row * N + col]) = lo;
            *reinterpret_cast<__half2*>(&C[(row + 8) * N + col]) = hi;
        }
}

template <int BM, int BN, int BK, int WARPS_M, int WARPS_N, int STAGES,
          bool CP_ASYNC, bool SWIZZLE, int COPY_BYTES, bool LDMATRIX,
          bool FRAG_PIPE = true>
void launch(size_t m, size_t n, size_t k, const half* A, const half* B, half* C)
{
    constexpr int THREADS = WARPS_M * WARPS_N * 32;
    constexpr int NBUF = CP_ASYNC ? STAGES : (STAGES > 1 ? 2 : 1);
    constexpr int smem_bytes = NBUF * (BM * BK + BK * BN) * (int) sizeof(half);
    auto kern = gemm<BM, BN, BK, WARPS_M, WARPS_N, STAGES, CP_ASYNC, SWIZZLE,
                     COPY_BYTES, LDMATRIX, FRAG_PIPE>;
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(
            kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        configured = true;
    }
    dim3 block(THREADS);
    dim3 grid((unsigned) (n / BN), (unsigned) (m / BM));
    kern<<<grid, block, smem_bytes>>>(A, B, C, (int) m, (int) n, (int) k);
}

}  // namespace gemm_sm80
