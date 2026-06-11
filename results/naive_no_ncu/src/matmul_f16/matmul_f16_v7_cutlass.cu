#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/half.h>

#include "playground/matmul.hpp"

namespace playground
{
namespace v7
{
// CUTLASS tensor-op GEMM for A100 (sm80), fp16 in / fp16 out / fp32 accumulate,
// all row-major. This instantiates CUTLASS's tuned multistage mainloop (the
// same family of kernels cuBLAS uses), reaching ~peak. A/B/C are row-major:
//   C[m,n] = sum_k A[m,k] * B[k,n].
// Config tuned for 4096^3: 128x256x32 block, 64x64 warp, m16n8k16 mma, 4 stages
// (measured best over the tile/stage sweep, ~99% cuBLAS, edges out the default).
using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,   // A
    cutlass::half_t, cutlass::layout::RowMajor,   // B
    cutlass::half_t, cutlass::layout::RowMajor,   // C
    float,                                        // accumulator
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 32>,       // threadblock tile
    cutlass::gemm::GemmShape<64, 64, 32>,         // warp tile
    cutlass::gemm::GemmShape<16, 8, 16>,          // mma instruction
    cutlass::epilogue::thread::LinearCombination<cutlass::half_t, 8, float,
                                                 float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;
}  // namespace v7

PLAYGROUND_MATMUL_DEC(float16_t, 7, m, n, k, A, B, C)
{
    using namespace v7;
    auto* Ap = reinterpret_cast<const cutlass::half_t*>(A);
    auto* Bp = reinterpret_cast<const cutlass::half_t*>(B);
    auto* Cp = reinterpret_cast<cutlass::half_t*>(C);

    cutlass::half_t alpha(1.0F), beta(0.0F);
    typename Gemm::Arguments args({int(m), int(n), int(k)},
                                  {Ap, int(k)},   // A, lda = k
                                  {Bp, int(n)},   // B, ldb = n
                                  {Cp, int(n)},   // C source, ldc = n
                                  {Cp, int(n)},   // C dest,   ldc = n
                                  {alpha, beta});
    static Gemm op;
    op(args);
}

}  // namespace playground
