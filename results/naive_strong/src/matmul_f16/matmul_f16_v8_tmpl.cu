#include "hgemm_common.cuh"
#include "playground/matmul.hpp"
#include "playground/system.hpp"

// v8: templated kernel @ v7 config (128x256x64, STAGES=3, warp 2x4 -> 64x64).
//   Parity check for the reusable infrastructure.
namespace playground
{
PLAYGROUND_MATMUL_DEC(float16_t, 8, m, n, k, A, B, C)
{
    hg::launch<128, 256, 64, 3, 2, 4>(m, n, k, A, B, C);
}
}  // namespace playground
