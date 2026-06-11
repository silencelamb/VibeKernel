# Runbook — naive baseline（Fable 5 臂）

> **唯一可执行真源 = [`scripts/launch_naive_fable.sh`](../scripts/launch_naive_fable.sh)**。它与 [`scripts/launch_naive.sh`](../scripts/launch_naive.sh) **逐字相同**，唯一差异是主 worker 模型 `claude-opus-4-8` → `claude-fable-5`。seed 仍同读共享文件 [`scripts/seed_gemm.txt`](../scripts/seed_gemm.txt)（naive/Opus、/goal、Fable 全用这一份，文字完全一致——公平对比前提）；改 seed 改那一个文件。本文件只讲这一臂**是什么**与它的单变量定位；**别在本文件复制 seed/命令**（两处复制必漂移）。
> worker session 在**本次 worktree** `worktrees/<run-name>/` 根开**全新** headless 进程，自动加载该 worktree 的 `CLAUDE.md`（脚本启动时从根 `CLAUDE_For_KernelAgent.md` cp 来）。跑完用 `scripts/finish_run.sh <run-name>` 一键归档 + 删 worktree（见 META §5）。
> 本文件不透露这是一次方法对比。

## 这一臂是什么（被测的「模型差异」）

- **= naive，换模型**：和 naive 共用同一套被测对象——零脚手架、零人工干预、`--dangerously-skip-permissions` 无人值守改/build/profile、不开 `/goal`/`/loop`/workflow、不装 skill、绝不 re-nudge、**NEVER STOP** 每轮简报。**唯一改动 = 主 worker 模型 `claude-fable-5`（Anthropic 最新）**，`--effort max` 不变。
- **单变量 = 模型**。因此结果直接与 **naive(Opus) 家族**（`naive`/`naive_cycle2…6` 等，`claude-opus-4-8 --effort max`）横比：相同 prompt、相同 harness、相同 effort、相同精度口径下，**换模型能冲到多高 / 能自己持续多久 / 花多少 token-cost**。
- 性能一律 task1 100 轮计分口径（`./task1.sh run --float f16 --ver N`），**禁 sweep / best-of-N 热峰**；目标向 A100 fp16 峰值 312 TFLOPS 逼近，**无 cuBLAS 基线**（基座已去 cuBLAS；正确性以 v0 cBLAS 为 ground truth，自写 kernel 从 v1 起）。
- 它早晚**自己停**——停在第几轮 / 多少 wall_clock / tokens / TFLOPS 就是结果，**不要续**（续 = 变成 /goal 那一臂，混淆变量）。
- ⚠️ 模型已 preflight 验过 `claude -p --model claude-fable-5 --effort max` 可用（2026-06-10）。Fable 5 的 token 单价 / 持续力 / 范式倾向都可能与 Opus 不同——这正是要测的，**别因 token/cost 数字与 Opus 不同就以为 harness 坏了**。

## 启动（在 VibeKernel 项目根）

```bash
bash scripts/launch_naive_fable.sh [run-name]   # run-name 默认 naive_fable；多跑传 naive_fable_cycle2 …
```

脚本依次（经 `_run_common.sh`）：①校验基座 `playground-base/` 在 + **防覆盖闸**（worktree 已存在 或 `results/<run-name>` 已有结果 → 拒跑）→ ②`git worktree add worktrees/<run-name>`（从基座 HEAD 开干净树）+ cp `CLAUDE.md` → ③`source env.sh` 绑 GPU → ④`setsid` detached 起 worker（`--model claude-fable-5`），落 `results/<run-name>/run.jsonl` + `run.pid`。

- 实时看：`results/watch_run.sh results/<run-name>/run.jsonl`
- 停整棵进程树：`kill -TERM -- -$(cat results/<run-name>/run.pid)`
- 不加 `--add-dir ..`；不设 `--max-turns`（让它自己跑到停）。

## 跑完（编排侧，不写进 worker）

1. **一键收尾**：`bash scripts/finish_run.sh <run-name>` —— 归档 transcript（token/曲线权威源）→ `check_handwritten`（用库判无效）→ 快照 `src`/`include`/`logs`/`worker.patch` → `parse_run` 出 `result.csv`/`result_table.md`/`curve.png`。
2. 套 `results/TEMPLATE.md` 写 `results/<run-name>/result.md`，**结论务必与 naive(Opus) 同口径并列**（峰值 TFLOPS / 自停点 / 总 token / cost / 是否撞同款 tensor-pipe 天花板）。结尾 `result` 事件含 usage/duration = 权威总量。
3. LLM 有随机性，多跑几轮报均值/方差 —— `bash scripts/launch_naive_fable.sh naive_fable_cycle2` 再来一轮（独立 worktree + slug + results，零串味）。
4. ⚠️ **不要与 naive 并行跑**：worker auto-memory 按 git 仓库（base-slug）键、跨 worktree 共享 → 并行 = 串味作废（见 memory `concurrent-runs-share-worker-memory`）。串行排队，或每轮 `CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`。
5. ⚠️ **GPU 驱动故障(2026-06-10 起)**：评测 GPU 卡 1155 MHz(NVRM assertion,非软锁,`-rgc` 解不掉)→ 绝对 TFLOPS 带 ~−18% 偏置,跨时钟态只能比「占真天花板 %」。故障态内多 cycle 之间可比;想要无偏置绝对值要等驱动重载/重启修复。见 memory `gpu-1155-driver-fault-not-lgc`。

## 想日后基于本轮 resume 跑 Ralph Loop（任何 naive_fable* run 通用）

**前提**：收尾时带 `--keep-worktree`（否则 worktree 被删、没法 resume，只能用 `worker.patch` 在新 cycle 复现）：
```bash
bash scripts/finish_run.sh <run-name> --keep-worktree    # naive_fable / naive_fable_cycle2 / …
```
它会留下 resume 三件套:`worktrees/<run>`(kernel 现状)+ `results/<run>/session_id.txt` + `results/<run>/worker_memory/`(worker 学到的瓶颈/死路)。

**续跑(一条命令,Fable 5 起 fresh-session 外循环,顶自己的峰值)**：
```bash
bash scripts/resume_naive_fable.sh                       # 默认 resume naive_fable
bash scripts/resume_naive_fable.sh naive_fable_cycle2    # resume cycle2(脚本收 run-name 参数)
# RALPH_MAX_ITERS=5 / RALPH_BUDGET_USD=40 可调,见脚本头注
```
机制 = 复用 `launch_ralph_loop.sh <run> --resume`(给它加了 `RALPH_MODEL` 旋钮把模型钉成 claude-fable-5)：复用该 run 的 worktree(kernel 从盘上续)→ 还原 `worker_memory` 回 base-slug → 轮轮全新 Fable session。**口径坑**(续跑 iters 落 `results/<run>/iters/`、报全量要合并 transcript、worker.patch 用 `base..HEAD`、别并行)全在 `results/naive_fable/RESUME.md`(cycle2 把名字替换即可)。
