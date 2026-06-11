#include "playground/matmul.hpp"
#include "playground/mma_gemm.cuh"
namespace playground {
PLAYGROUND_MATMUL_DEC(float16_t, 33, m, n, k, A, B, C)
{ mmagemm::launch_f32<128,256,32, 2,4, 4, 1, 4>(int(m),int(n),int(k),A,B,C); } // v28 + swizzle4
}
