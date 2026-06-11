// v1: simple wmma baseline — correctness anchor.
// 128x128 block tile, BK=32, 8 warps each computing 64x32 via wmma 16x16x16.
// Padded smem to dodge bank conflicts; fp16 accumulate.
#include <cuda_runtime.h>
#include <mma.h>

#include "playground/matmul.hpp"

namespace playground
{
namespace v1
{
using namespace nvcuda;

constexpr int BM = 128, BN = 128, BK = 32;
constexpr int APAD = 8, BPAD = 8;

__global__ void hgemm_v1(const half* __restrict__ A, const half* __restrict__ B,
                         half* __restrict__ C, int M, int N, int K)
{
    __shared__ half sA[BM][BK + APAD];
    __shared__ half sB[BK][BN + BPAD];

    const int bm = blockIdx.y * BM;
    const int bn = blockIdx.x * BN;
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    // warp grid 2 (m) x 4 (n); warp tile 64x32
    const int wrow = (wid >> 2) * 64;
    const int wcol = (wid & 3) * 32;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> fc[4][2];

    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 2; ++j)
            wmma::fill_fragment(fc[i][j], __float2half(0.0F));

    // A tile loads: 128x32 halves; 8 halves (16B) per load.
    // thread t -> row t/4, col (t%4)*8 ; two rows per thread (r, r+64)
    const int arow = tid >> 2;
    const int acol = (tid & 3) << 3;
    // B tile loads: 32x128: row t/16, col (t%16)*8 ; two rows (r, r+16)
    const int brow = tid >> 4;
    const int bcol = (tid & 15) << 3;

    for (int kt = 0; kt < K; kt += BK) {
        *reinterpret_cast<float4*>(&sA[arow][acol]) =
            *reinterpret_cast<const float4*>(&A[(bm + arow) * K + kt + acol]);
        *reinterpret_cast<float4*>(&sA[arow + 64][acol]) =
            *reinterpret_cast<const float4*>(
                &A[(bm + arow + 64) * K + kt + acol]);
        *reinterpret_cast<float4*>(&sB[brow][bcol]) =
            *reinterpret_cast<const float4*>(&B[(kt + brow) * N + bn + bcol]);
        *reinterpret_cast<float4*>(&sB[brow + 16][bcol]) =
            *reinterpret_cast<const float4*>(
                &B[(kt + brow + 16) * N + bn + bcol]);
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
#pragma unroll
            for (int i = 0; i < 4; ++i)
                wmma::load_matrix_sync(fa[i], &sA[wrow + i * 16][kk],
                                       BK + APAD);
#pragma unroll
            for (int j = 0; j < 2; ++j)
                wmma::load_matrix_sync(fb[j], &sB[kk][wcol + j * 16],
                                       BN + BPAD);
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 2; ++j)
                    wmma::mma_sync(fc[i][j], fa[i], fb[j], fc[i][j]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
            wmma::store_matrix_sync(
                &C[(bm + wrow + i * 16) * N + bn + wcol + j * 16], fc[i][j], N,
                wmma::mem_row_major);
}

}  // namespace v1

PLAYGROUND_MATMUL_DEC(float16_t, 1, m, n, k, A, B, C)
{
    dim3 grid(n / v1::BN, m / v1::BM);
    v1::hgemm_v1<<<grid, 256>>>(A, B, C, int(m), int(n), int(k));
}

}  // namespace playground
