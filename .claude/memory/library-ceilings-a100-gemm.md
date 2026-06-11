---
name: library-ceilings-a100-gemm
description: "这台A100上4096³ fp16库天花板实测:cuBLAS f16-acc=219.85(err.018)、cuBLAS fp32-acc=218.74(err3e-5)、CUTLASS fp32-acc=217.9(同goal配置);cuBLAS f16≈fp32(只差1→f16-acc几乎不提速,手写f16>fp32只是1-block register-wall产物);CUTLASS纯C++/PTX无手写SASS→手写离库~5-8%是结构成熟度非汇编;比库先对齐accumulate精度档"
metadata: 
  node_type: memory
  type: reference
  originSessionId: efbd0bee-80cb-44d8-a7a2-221db97a0819
---

这台 A100(1155MHz sustained,真峰 ~256,见 [[a100-sustained-clock]])上 **4096³ fp16、task1 100 轮口径**的库天花板实测:

| 库 | accumulate | TFLOPS | err | 出处 |
|---|---|---|---|---|
| **cuBLAS** | **f16** | **219.85** | 0.018 | `cublasGemmEx(..., computeType=CUDA_R_16F)`(最新实测;早先一次测 222) |
| **cuBLAS** | **f32** | **218.74** | 3.4e-5 | `computeType=CUDA_R_32F`(最新实测)——与 goal/naive fp32 同档可比 |
| **CUTLASS** | **f32** | **217.9** | 3.2e-5 | `cutlass::gemm::device::Gemm<half_t,…,float acc,Sm80,128×256/64×64/16×8×16>`;`results/naive_no_ncu/src/.../v7_*.cu` |

**⭐ cuBLAS f16(219.85)≈ cuBLAS fp32(218.74),只差 ~1** → 对调好的库 **f16-acc 几乎不提速**。所以手写里"f16-acc 明显 > fp32"(naive_cycle6 203→208、deprecated cycle5 188→218)是**手写 fp32 卡在 1-block register-wall(128 累加寄存器)的产物**,不是 f16>fp32 的本质;库的 fp32 没这个 wall、两档齐平。另:`naive_cycle6`(干净串行)**独立**走到 f16-acc(208.3,err.018)→ naive 自己能找到 f16-acc 这条轴(见 [[f16-accumulate-precision-confound]])。

**关键结论(几轮讨论落档,2026-06-08):**
1. **比库务必先对齐 accumulate 精度档**:cuBLAS 222 是 f16 累加(err 0.02);CUTLASS 217.9 是 fp32 累加(err 3e-5)。手写 kernel 要和哪条比,先看自己是 f16-acc 还是 fp32-acc。错配档 = 苹果比橘子(naive_cycle5 的 218 f16-acc 冒充 fp32 战绩就是这错,已 deprecate,见 [[f16-accumulate-precision-confound]])。
2. **CUTLASS fp32 217.9 的 tile 配置 = 128×256/64×64/BK32,与 goal_cycle3 手写完全相同** → 同档同配置对标:goal_cycle3 fp32 201.6 = CUTLASS 的 **92.5%**,goal_cycle1 206.8 = 95%。那 ~5–7% 是**实现结构成熟度**。
3. **差距≠手写 SASS。** CUTLASS 开源、**纯 C++ 模板 + inline PTX(`mma.sync`/`cp.async`/`ldmatrix`/mbarrier via `cute`/`arch`),SASS 调度+寄存器分配交给 ptxas,零手写汇编**——它不碰 SASS 也到 217.9。所以手写离库那 ~6% **在可读 C++/PTX 里就能拿**(更优 multistage mainloop 结构),不是够不着的汇编层。("剩 10% 是库 SASS 调度"那个旧说法据此修正。)
4. **逼近库不需要 mbarrier-无-sync**:有人手写 f16-acc + 普通 `__syncthreads` + XOR swizzle + 256×128 = 214(cuBLAS f16 档 96%)。mbarrier-无-barrier 是 Hopper(sm_90+TMA)范式,A100 上可选、只值 ~13%。
5. **任务规则禁库**(`check_handwritten.sh` 扫 cutlass/cublas/cute 判违规)→ 这些库只能当**天花板参照 + 读源学技法**,agent 不能 `#include`。

应用:写 result.md / 报"占库百分比"时,引这张表 + 标精度档。详见 `results/SUMMARY.md` §库天花板对照、`results/goal_cycle3/result.md` §库对照。
