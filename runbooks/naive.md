# Runbook — naive baseline

> **唯一可执行真源 = [`scripts/launch_naive.sh`](../scripts/launch_naive.sh)**：启动参数、路径都在那（detached/setsid 跑，落 `results/naive/run.jsonl`）。**worker 指令(seed)已抽到共享文件 [`scripts/seed_gemm.txt`](../scripts/seed_gemm.txt)，naive 与 /goal 同读一份、文字完全一致(公平对比的前提)**——改 seed 改那一个文件。本文件只讲 naive **是什么**与 harness 约束；**别在本文件复制 seed/命令**（两处复制必漂移——之前就漂过）。
> worker session 在**本次 worktree** `worktrees/<run-name>/` 根开**全新** headless 进程，自动加载 worktree 的 `CLAUDE.md`（启动时由脚本从根 `CLAUDE_For_KernelAgent.md` cp 来；更多 how 在它 + 它指向的 `task-1/README.md`）。跑完用 `scripts/finish_run.sh <run-name>` 一键归档 + 删 worktree（见 META §5）。
> 本文件不透露这是一次方法对比。

## naive 是什么（被测的「内在持续力」）

- **零脚手架、零人工干预**：一次 prompt 喂下去，`--dangerously-skip-permissions` 让它无人值守地改 / build / profile；**不开 `/goal` / `/loop` / workflow、不装 skill、绝不 re-nudge**。
- **NEVER STOP** + 每轮简报（当前 TFLOPS、Average Error、本轮改了什么、下一步）；性能一律 task1 100 轮计分口径，**禁 sweep / best-of-N 热峰**。
- 目标 = 向 A100 fp16 峰值 312 TFLOPS 逼近，**无 cuBLAS 基线**（基座已去 cuBLAS；正确性以 v0 cBLAS 为 ground truth，自写 kernel 从 v1 起）。
- 它早晚**自己停** —— 停在第几轮 / 多少 wall_clock / tokens / TFLOPS 就是结果，**不要续**。这就是 naive 要测的（纯 prompt 下模型自身能持续多久 / 冲到多少）。
- 固定 `--model claude-opus-4-8 --effort max`（跨方法可比）。

## 启动（在 VibeKernel 项目根）

```bash
bash scripts/launch_naive.sh [run-name]   # run-name 默认 naive；多跑传 naive_cycle2 / naive_cycle3 …
```

脚本依次（经 `_run_common.sh`）：①校验基座 `playground-base/` 在 + **防覆盖闸**（worktree 已存在 或 `results/<run-name>` 已有结果 → 拒跑、提示换 run-name）→ ②`git worktree add worktrees/<run-name>`（从基座 HEAD 开干净树）+ cp `CLAUDE.md` → ③`source env.sh` 绑 GPU → ④`setsid` detached 起 worker，落 `results/<run-name>/run.jsonl` + `run.pid`。

- 实时看：`results/watch_run.sh results/<run-name>/run.jsonl`
- 停整棵进程树：`kill -TERM -- -$(cat results/<run-name>/run.pid)`
- 不加 `--add-dir ..`（文件工具够不到 VibeKernel 根 / 其它方法 / results，隔离更稳）；不设 `--max-turns`（让它自己跑到停）。

## 跑完（编排侧，不写进 worker）

1. **一键收尾**：`bash scripts/finish_run.sh <run-name>` —— 自动：归档 transcript（token/曲线权威源）→ `check_handwritten`（用库判无效）→ 快照 `src`/`include`/`logs`/`worker.patch` 进 `results/<run-name>/` → `parse_run` 出 `result.csv`/`result_table.md`/`curve.png`（每版标最佳点、固定画 cuBLAS~225 与 312 参照线，约定见 META §6）→ **删 worktree**（基座没动）。
2. 套 `results/TEMPLATE.md` 写 `results/<run-name>/result.md`（结尾 `result` 事件含 usage/duration = 权威总量）。薄 dispatcher 源文件名不带技法的版本，想图上显示技法写 `results/<run-name>/labels.json`。
3. LLM 有随机性，多跑几轮报均值/方差 —— 直接 `bash scripts/launch_naive.sh naive_cycle2` 再来一轮（独立 worktree + slug + results，零串味）。
