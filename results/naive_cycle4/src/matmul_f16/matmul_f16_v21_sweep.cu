#include "playground/matmul.hpp"
#include "playground/mma_gemm.cuh"
namespace playground {
PLAYGROUND_MATMUL_DEC(float16_t, 21, m, n, k, A, B, C)
{ mmagemm::launch<256,128,32, 4,2, 4, 1, true>(int(m),int(n),int(k),A,B,C); } // 64x64 8w NS4 DB
}
