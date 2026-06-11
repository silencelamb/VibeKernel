// v3 — FULL STACK via the parameterized template (should reproduce v2).
// Config: 128x128x32, 8 warps, STAGES=3, cp.async, swizzle, 16B float4, ldmatrix.
#include "gemm_sm80.cuh"

#include "playground/matmul.hpp"
#include "playground/system.hpp"

namespace playground
{
PLAYGROUND_MATMUL_DEC(float16_t, 3, m, n, k, A, B, C)
{
    gemm_sm80::launch</*BM*/ 128, /*BN*/ 128, /*BK*/ 32, /*WARPS_M*/ 2,
                      /*WARPS_N*/ 4, /*STAGES*/ 3, /*CP_ASYNC*/ true,
                      /*SWIZZLE*/ true, /*COPY_BYTES*/ 16, /*LDMATRIX*/ true>(
        m, n, k, A, B, C);
}
}  // namespace playground
