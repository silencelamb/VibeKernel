// ============================================================================
// matmul_f16_v10 — FINAL fp16 tensor-core GEMM for A100 (SM80).
//
// Result: ~172 TFLOPS sustained @ 4096^3, 100-round avg = 55% of the 312 TFLOPS
// fp16 Tensor-Core peak. Pure hand-written (mma.sync + ldmatrix + cp.async);
// no cuBLAS / CUTLASS / CuTe / cuDNN. fp32 accumulate, err ~4e-5 vs cBLAS GT.
//
// Config picked by the mandated backward-elimination search (NOT forward-greedy):
//   block 128x128x64, warps 2x2 (64x64 warp tile), STAGES=2, 2 blocks/SM.
// All six ingredients are present and every one is load-bearing (leave-one-out
// drop, TFLOPS): swizzle -115, vectorize -54, cp.async -41, ldmatrix -32,
// register-blocking -30, multi-stage pipeline -10. Four of them (cp.async,
// pipeline, ldmatrix, vectorize) have ~0 / negative STANDALONE value — a
// forward-greedy search would have discarded them; leave-one-out rescued them.
//
// Implementation lives in gemm_sm80.cuh (one parameterized kernel; each
// ingredient is a compile-time toggle so ablations are single-config flips).
// ============================================================================
#include "gemm_sm80.cuh"

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{
PLAYGROUND_MATMUL_DEC(float16_t, 10, m, n, k, A, B, C)
{
    gemm_sm80::launch</*BM*/ 128, /*BN*/ 128, /*BK*/ 64,
                      /*WARPS_M*/ 2, /*WARPS_N*/ 2, /*STAGES*/ 2,
                      /*CP_ASYNC*/ true, /*SWIZZLE*/ true, /*COPY_BYTES*/ 16,
                      /*LDMATRIX*/ true, /*FRAG_PIPE*/ false>(m, n, k, A, B, C);
}
}  // namespace playground
