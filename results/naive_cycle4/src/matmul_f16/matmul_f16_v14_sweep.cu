#include "playground/matmul.hpp"
#include "playground/mma_gemm.cuh"
namespace playground {
PLAYGROUND_MATMUL_DEC(float16_t, 14, m, n, k, A, B, C)
{ mmagemm::launch<256,128,32, 4,2, 3, 1>(int(m),int(n),int(k),A,B,C); }   // 64x64, 8 warps, 256x128
}
