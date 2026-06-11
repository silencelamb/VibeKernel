#include "playground/matmul.hpp"
#include "playground/mma_gemm.cuh"
namespace playground {
PLAYGROUND_MATMUL_DEC(float16_t, 9, m, n, k, A, B, C) {
    pg_mma::launch<128, 256, 64, 4, 4, 3>(m, n, k, A, B, C);
}
}
