# Runbook — naive baseline

> worker session 的指令源。在 `playground-naive/` 根目录开**全新** headless 进程；它同时会自动加载根 `CLAUDE.md`（更多 how 在那、及它指向的 `task-1/README.md`）。
> **零脚手架、零人工干预**：`--dangerously-skip-permissions` 让它无人值守地改 / build / profile，但**不开 `/goal` / `/loop` / workflow、也绝不 re-nudge**。
> 给一次 prompt 后观察它**自己能持续多久、冲到多少 TFLOPS 才停**——这就是 naive 要测的（内在持续力）。
> 本文件不透露这是一次方法对比。

## seed prompt（喂给 worker 的 `-p` 全文）

```text
你是 CUDA kernel 专家。你的任务是在 A100 上做高性能 fp16 GEMM kernel，不断优化性能。
**NEVER STOP**：不断追求更好的性能，追求极致，（A100 fp16 理论峰值 312 TFLOPS）。持续地做、尝试各种底层优化（包括但不限于shared-mem / register tiling、双缓冲 / 多级流水、cp.async、swizzle 避免 bank conflict、提高 L2 命中、warp 排布等）、细致 profiling，绝不停下、不要回头问、可以查找CUDA编程的官方资料（比如ptx：https://docs.nvidia.com/cuda/parallel-thread-execution/ 、cuda https://docs.nvidia.com/cuda/cuda-programming-guide/index.html）。

具体约定：
每轮简报：当前 TFLOPS、Average Error、本轮改了什么、下一步试什么——然后立刻继续下一轮，不要停、不要问我。
```

## 启动命令（在 VibeKernel 项目根目录下执行）

```bash
# 当前在 VibeKernel 项目根目录
cd playground-naive
cp ../CLAUDE.md ./CLAUDE.md       # 每次启动刷新 fork 内 CLAUDE.md（最保险：防 master 改动后 fork 内过期）
source ../env.sh                 # 绑定 GPU（env.sh 指定），worker 子进程自动继承
# SEED = 上面「## seed prompt」整段
claude -p "$SEED" \
  --model claude-opus-4-8 --effort max \
  --dangerously-skip-permissions \
  --output-format stream-json --verbose \
  > ../results/naive_cycle1.jsonl 2>&1
```

- 不加 `--add-dir ..` → 文件工具够不到 VibeKernel 根（META / 其它 fork / results），隔离更稳。
- 不设 `--max-turns` → 让它自己跑到停。
- 结尾 `result` 事件含 `usage`(tokens) / `duration_ms` → 填 CSV 的 `tokens` / `wall_clock`。

## 该方法的 harness

- **naive**：零续跑机制；不装 skill、不人工 re-nudge。测「纯 prompt（NEVER STOP）下模型自身能持续多久 / 冲到多少」。
- 它早晚自己停 —— 停在第几轮 / 多少 wall_clock / tokens / TFLOPS 就是结果，不要续。
- 记录到 `results/naive.csv` 是编排侧的事，不写进 worker。
