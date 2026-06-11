#!/usr/bin/env bash
# 在某次 naive_fable* 跑【的基础上】继续优化,Ralph-Loop 式(每轮全新 session、干净 context、
# kernel 进度 + worker 记忆从磁盘续)。= 复用该 run 保留的 worktree(kernel 已 commit 在里面)
# + 还原它的 worker_memory(GPU 锁 1155 / best-config / ptxas 教训,新 session 不白板重走死路),
# 用 **Fable 5** 起一轮轮 fresh session 去顶它自己的峰值。
#
# 用法:
#   bash scripts/resume_naive_fable.sh                      # 默认 resume run-name = naive_fable
#   bash scripts/resume_naive_fable.sh naive_fable_cycle2   # resume cycle2(或任何 naive_fable* run)
#   RALPH_MAX_ITERS=5 bash scripts/resume_naive_fable.sh naive_fable_cycle2   # 续到总 5 轮
#   RALPH_BUDGET_USD=40 bash scripts/resume_naive_fable.sh ...               # 每轮成本兜底(优雅停)
#
# 前置(必须):目标 run 之前用 `finish_run.sh <run> --keep-worktree` 归过档
#   → worktrees/<run> 还在 + results/<run>/worker_memory 有备份。否则下面的 launch_ralph_loop --resume 会报错并提示。
#
# ⚠️ 模型钉死 claude-fable-5(忠实「在 Fable 跑基础上继续」= 单变量仍是 Fable)。
# ⚠️ 跑这个时【别并行】别的 worker:它会把 worker_memory 还原回 base-slug(共享键),并行 = 串味。
# ⚠️ 口径:续跑的 iters 落 results/<run>/iters/;报「单轮 vs Ralph 续跑」要分开,别混进原单 session 结论。
#    完整归档/坑见 results/naive_fable/RESUME.md(cycle2 同理,把 naive_fable 换成 naive_fable_cycle2)。
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${1:-naive_fable}"          # 要 resume 的 run-name(默认 naive_fable;传 naive_fable_cycle2 等)

# —— 前置校验:给出清楚的失败信息,而不是让底层脚本报错 ——
[ -d "$ROOT/worktrees/$RUN" ] || { echo "❌ worktrees/$RUN 不在 —— 该 run 没 launch 过,或 finish 时没带 --keep-worktree(被删了)。" >&2
  echo "   若 kernel 还在 results/$RUN/worker.patch,只能另起新 cycle 复现,不是本 run 的 resume。" >&2; exit 1; }
[ -d "$ROOT/results/$RUN/worker_memory" ] || echo "  ⚠️ 没找到 results/$RUN/worker_memory 备份 → 续跑新 session 将白板起(kernel 仍从盘上续,可接受)。"

export RALPH_MODEL=claude-fable-5                     # ← 唯一关键:续跑也用 Fable 5
export RALPH_MAX_ITERS="${RALPH_MAX_ITERS:-3}"        # 续跑总轮数上限(原单 session 不在 iters/ 计数,故 ≈ 本次新增轮数)
export RALPH_JUDGE="${RALPH_JUDGE:-0}"               # NEVER-STOP seed 无终止态 → 默认纯无限外循环(到 MAX_ITERS 为止)
export RALPH_BUDGET_USD="${RALPH_BUDGET_USD:-}"      # 可选每轮成本兜底

echo "▶ Ralph-Loop 式续跑 $RUN(worker=Fable 5,续 $RALPH_MAX_ITERS 轮 fresh session)"
echo "  复用 worktrees/$RUN(kernel 从盘上续)+ 还原 results/$RUN/worker_memory"
exec bash "$ROOT/scripts/launch_ralph_loop.sh" "$RUN" --resume
