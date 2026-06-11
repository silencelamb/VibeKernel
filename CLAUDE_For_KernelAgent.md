# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# CUDA GEMM (playground task-1)

你是 **CUDA kernel 专家**。本仓库是 NVIDIA A100 上的 **fp16 GEMM**(走 Tensor Core)kernel 实现任务；正确性以 cBLAS(CPU) 为 ground truth，**追求极致性能、不断逼近硬件峰值**（A100 理论算力 fp16 312 TFlops）——**无库基线对标**。完整说明见 `task-1/README.md`。下面是构建、运行、profile 与新增 kernel 版本的约定。

> 🎯 **目标 kernel = fp16（`float16_t`，走 Tensor Core），在 `task-1/src/matmul_f16/` 下从 `--ver 1` 起手写，逼近 A100 fp16 峰值 312 TFLOPS。** 所有命令一律 `--float f16`、版本迭代都在 `matmul_f16/`；`--ver 0` 的 cBLAS 仅作正确性 GT。

## 硬性约束：从零手写，禁用现成 GEMM 库

> 本任务考核的是**你亲手写出高性能 kernel 的能力**，不是调库能力。以下是任务的硬性要求，违反则该版本无效：

- `--ver ≥ 1` 的 kernel **必须完全手写**：tiling、Tensor Core（`mma.sync`/`wmma`）、`cp.async`、swizzle、寄存器/共享内存流水等全部自己实现。
- **禁止调用或包含任何现成 GEMM 实现**：CUTLASS / CuTe（`#include <cutlass/...>`、`cutlass::`）、cuBLAS / cuBLASLt（`cublas*`）、cuDNN，或任何把 GEMM 主体外包给库的写法。（`--ver 0` cBLAS 只作正确性 ground truth、不算你的成果；除此之外**没有任何库基线**——目标是逼近硬件峰值，不与任何库比较。）
- **自己分析、自己 profile**：用 `ncu` 读真实指标（SM / tensor-pipe 利用率、shared bank conflict、warp stall 原因、occupancy、寄存器数）定位瓶颈，**不要靠猜**；需要时查官方文档（PTX ISA、CUDA C++ Programming Guide）最后一节有提供文档资料。
- 本任务会用 `scripts/check_handwritten.sh` 扫描源码 include，**检测到 cutlass/cublas 等会判该版本无效**——别在这上面浪费力气。

## 性能口径：只认 task1 的 100 轮平均，禁止自建 profiling

> 这是唯一合法的性能成绩口径，违反则该数字无效：

- 性能**必须**用 task1 自带的计分方式：`./task1.sh run --float f16 --ver N`（或 `./build/src/task1_float16_vN -m 4096 -n 4096 -k 4096 -t 100`），即 **10 warmup + 100 轮取平均**、固定 4096³。报的就是它打印的 `[Playground] Result >>> TFLOPS: ...; Average Error: ...`。
- **禁止自己写计时 / 热峰值** 之类的"另一套 profiling"来报成绩。 不要自己加 -t 跑更多轮——长跑暖机时钟会 boost从而虚高。那种短爆发满 boost 时钟的数字会虚高 10~20%（连厂商库实现都能被量出 +10%），不是真实持续性能，**判无效**。
- `ncu` 只用来**看指标、定位瓶颈**，不用来产出"性能数字"。报成绩一律用 task1 的 100 轮口径。

## 运行环境

- 已在 GPU docker 容器内。GPU 已由环境变量 `CUDA_VISIBLE_DEVICES` 绑定（启动时已设好），直接构建 / 运行 / profile 即可，**别自行改动这个绑定**。
- 工具链：CUDA 13.0（`nvcc` / `ncu` 在 `/usr/local/cuda/bin`）、vcpkg 在 `/opt/vcpkg`（`$VCPKG_HOME` 已设）、CMake ≥ 3.30 + Ninja、C++20 / CUDA20。
- **环境已验证**：构建链、`./task1.sh run/build/profile`、以及 **`ncu` profiling（容器已具备 `CAP_SYS_ADMIN`，GPU 性能计数器可读）均已实测可跑**。启动后**不要花 turn 去 验证工具链**；真有某步报错再就地处理即可。

## 常用命令（在仓库根目录执行）

```bash
./task1.sh run     --float f16 --ver 2   # 构建+运行+自动落 logs/
./task1.sh build   --float f16 --ver 2   # 仅构建
./task1.sh profile --float f16 --ver 2   # RelWithDebInfo 构建 + Nsight Compute（--set full）
# 直接跑二进制、自定义 shape（默认 m=n=k=4096，10 warmup，100 test）
./build/src/task1_float16_v2 -m 4096 -n 4096 -k 4096 -t 100
./task1.sh clean        # 清 build
./task1.sh clean-logs   # 清 logs
```

基准版本：`--ver 0` = cBLAS（CPU，正确性 ground truth，唯一非手写版本）。自写 kernel 从 `--ver 1` 起。（本仓库已无 cuBLAS，不存在库性能基线。）

## 注册一个新 kernel 版本

在 `task-1/src/matmul_f16/`下**直接**新建一个 `.cu`(glob 非递归,别放子目录)、**文件名要带版本号 `_vN`**(如 `matmul_f16_v2.cu` 或 `matmul_f16_v2_mykernel.cu`——构建按文件名里的 `_vN` 挑当前 `--ver N` 的源,不带就选不中),在 `namespace playground` 内用宏定义模板特化：

```cpp
#include "playground/matmul.hpp"
namespace playground {
PLAYGROUND_MATMUL_DEC(float16_t, 2, m, n, k, A, B, C) {
    // 实现 matmul<float16_t, 2>；A / B / C 均为 row-major 设备指针
}
}
```

- A、B、C 均为**行主序**设备指针。若你借列主序思路实现(如交换 A/B、用 `n, m, k`)，务必保证最终写回的 C 是行主序、与 cBLAS ground truth 对得上。
- 调用哪个版本由 CMake 的 `MATMUL_VERSION` 宏决定，`main.cpp` 据此调用对应特化。

## 关键构建坑（`src/CMakeLists.txt`）

`src/CMakeLists.txt` 只把 **`main.cpp` + `v0`（cBLAS 参照）+ 当前 `--ver N` 选中的那个版本文件**编进二进制（按文件名里的 `_vN_` / `_vN.` 选;不再 GLOB 全目录）。因此：

- 同一 `(dtype, version)` 只能被定义一次，否则重复符号 / ODR 冲突。
- **别的版本文件编译坏了不影响你**——只有 v0 + 你当前在跑的版本会被编;废弃 / 试验版本留着也没关系（不选就不编），当然想删也行。
- 改**已存在**版本的 `.cu` 直接 `./task1.sh run --float f16 --ver N` 即可（跳过 CMake 重配置、直接 ninja 增量编，很快）。**给同一个 version 新增第二个文件**时,glob 只在重配置时更新 → 先 `./task1.sh clean`（或换 `--ver`）触发一次重配置。

## 正确性与性能口径

- 输入为 N(0,1) 随机；ground truth 由 v0(cBLAS) 在 CPU 算。正确性 = 平均相对误差 `mean(|GT−C| / |GT|)`，出现 inf/nan 立即中止；f16 正确 kernel 的误差量级约 1e-2（f16 累加 ≲ 0.02 即正常）。
- 性能 = `2·m·n·k / time`（TFLOPS），CUDA event 计时。**目标是向 A100 fp16 峰值 312 TFLOPS 逼近，不设百分比门槛、无库基线对标**（不与 cuBLAS 或任何库比较）。
- `ncu --set full` 会把默认「100 轮 × 4096³」里每个 kernel 重放很多次，极慢。想快速看指标，直接对二进制只跑 1 轮：`ncu --set full -o out ./build/src/task1_float16_v2 -t 1`。`.ncu-rep` 默认落在 `logs/profiles/`。

## 每轮简报

每当你用 **task1 计分口径**（`./task1.sh run --float f16 --ver N` 或 `./build/src/task1_float16_vN -t 100`，即 100 轮均值）成功完成一个版本后，写一行人类可读简报：**当前 TFLOPS、Average Error、本轮改了什么、下一步试什么**——然后立刻继续下一轮。

- 性能数字**必须**是 task1 100 轮口径（见上「性能口径」）——不要报临时 sweep / best-of-N 的"热峰值"（短爆发满 boost 时钟会虚高 10~20%，不是真实持续性能）。
- 不需要输出任何结构化标记；性能与版本由编排侧从 task1 工具输出自动解析。

## 文档资料

| #   | Resource                                  | URL                                                                                                     | Format for ingest                                                     | What it's for                                                                                                                                                                       |
| --- | ----------------------------------------- | ------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **CUDA C++ Best Practices Guide**         | https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/                                               | PDF: https://docs.nvidia.com/cuda/pdf/CUDA_C_Best_Practices_Guide.pdf | The optimization heuristics source. Has explicit **High/Medium/Low Priority** tags — pre-ranked rules, ideal for an agent.                                                          |
| 2   | **CUDA C++ Programming Guide**            | https://docs.nvidia.com/cuda/cuda-c-programming-guide/                                                  | PDF: https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf    | Ground truth for the programming model, memory model, `cp.async`, CC 8.0 specs (Appendix). Big — chunk it.                                                                          |
| 3   | **Nsight Compute Kernel Profiling Guide** | https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html                                        | HTML (no clean PDF for latest)                                        | **Metric definitions** + the SOL/roofline model + replay/overhead mechanics. The "Metrics Reference" + "Metrics Decoder" subsections are what let the agent *interpret ncu output*. |
| 3b  | Nsight Compute CLI reference              | https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html                                      | HTML                                                                  | `--set` / `--section` / `--clock-control` flag semantics. Pair with #3.                                                                                                             |
| 4   | PTX ISA reference                         | https://docs.nvidia.com/cuda/parallel-thread-execution/                                                 | PDF: https://docs.nvidia.com/cuda/pdf/ptx_isa_9.3.pdf                 | When inspects/reasons about SASS/PTX.                                                                                                                                               |
| 5   | NVIDIA A100 architecture whitepaper       | https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf | PDF: https://docs.nvidia.com/cuda/pdf/ptx_isa_9.3.pdf                 | Authoritative SM80 microarch numbers.                                                                                                                                               |