#include "hgemm_common.cuh"
#include "playground/matmul.hpp"
#include "playground/system.hpp"
namespace playground {
PLAYGROUND_MATMUL_DEC(float16_t, 15, m, n, k, A, B, C) { hg::launch_f16<128,128,64,2,2,4>(m,n,k,A,B,C); }
}
