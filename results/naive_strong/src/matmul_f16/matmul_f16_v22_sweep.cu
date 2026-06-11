#include "hgemm_common.cuh"
#include "playground/matmul.hpp"
#include "playground/system.hpp"
namespace playground {
PLAYGROUND_MATMUL_DEC(float16_t, 22, m, n, k, A, B, C) { hg::launch<128,128,64,2,2,4,2>(m,n,k,A,B,C); }
}
