#include "playground/matmul.hpp"
#include "playground/mma_gemm.cuh"
namespace playground {
PLAYGROUND_MATMUL_DEC(float16_t, 12, m, n, k, A, B, C) {
    pg_mma::launch<128, 128, 64, 4, 4, 3, 1>(m, n, k, A, B, C);
}
}
