#include "playground/matmul.hpp"
#include "playground/mma_gemm.cuh"
namespace playground {
PLAYGROUND_MATMUL_DEC(float16_t, 9, m, n, k, A, B, C)
{ mmagemm::launch<128,128,32, 2,4, 3, 2>(int(m),int(n),int(k),A,B,C); }
}
