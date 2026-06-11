#!/usr/bin/env bash
# Ralph Loop（Fable 5）—— 以 naive_fable_cycle2 的单 session 跑作为 **iteration 1 的中间结果**,
# 再用 Fable 5 续跑更多 fresh-session 迭代(每轮全新 session、干净 context、kernel 进度从磁盘续)。
# 默认续到【总】11 轮 = 1 个种子 iter(cycle2) + 10 个新迭代(iter 2..11)。
#
# 用法:
#   bash scripts/launch_ralph_loop_fable.sh                      # 续到总 RALPH_MAX_ITERS(默认 11)
#   RALPH_MAX_ITERS=21 bash scripts/launch_ralph_loop_fable.sh   # 续更多(总 21 = 种子 + 20 新)
#   RALPH_BUDGET_USD=60 bash scripts/launch_ralph_loop_fable.sh  # 每轮 claude -p 成本兜底(优雅停)
#   SRC=naive_fable bash scripts/launch_ralph_loop_fable.sh      # 换 iter-1 种子来源(默认 naive_fable_cycle2)
#
# 首次运行自动 seed(从 SRC 拷:worktree 的 kernel 状态 + iter_01 transcript + run.jsonl 种子 + worker_memory),
# 已 seed 则跳过、直接 resume。机制复用 launch_ralph_loop.sh --resume(RALPH_MODEL 旋钮钉成 Fable 5)。
#
# ⚠️ 必须在【带 CAP_SYS_ADMIN 的容器】跑(ncu)。⚠️ 别与别的 worker 并行(worker_memory 还原回 base-slug=共享键,串味)。
# ⚠️ 口径:每轮独立 transcript 落 results/ralph_loop_fable/iters/iter_NN.transcript.jsonl;报全量要合并所有 iters 再 parse;
#    run.jsonl 是【追加】(种子末尾是 cycle2 的 result 事件,别拿它当总量)。detached;停:kill -TERM -- -<pid>。
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${SRC:-naive_fable_cycle2}"          # iter-1 来源 run(需之前 finish --keep-worktree 过)
RUN=ralph_loop_fable
WT="$ROOT/worktrees/$RUN"
OUT="$ROOT/results/$RUN"
BASE="$ROOT/playground-base"

# —— 一次性 seed(idempotent;已 seed 则跳过)——
if [ ! -d "$WT" ] || [ ! -f "$OUT/iters/iter_01.transcript.jsonl" ]; then
  echo "▶ 首次 seed:把 $SRC 当作 Ralph iter-1 的中间结果"
  [ -d "$ROOT/worktrees/$SRC" ]            || { echo "❌ 源 worktree worktrees/$SRC 不在(需 finish_run $SRC --keep-worktree 保留)" >&2; exit 1; }
  [ -f "$ROOT/results/$SRC/transcript.jsonl" ] || { echo "❌ 源 transcript results/$SRC/transcript.jsonl 不在" >&2; exit 1; }
  C2_HEAD="$(git -C "$ROOT/worktrees/$SRC" rev-parse HEAD)"
  [ -d "$WT" ] || git -C "$BASE" worktree add --detach "$WT" "$C2_HEAD" >/dev/null   # 绝对路径!(-C base 下相对路径会落错地方)
  mkdir -p "$OUT/iters"
  cp "$ROOT/results/$SRC/transcript.jsonl" "$OUT/iters/iter_01.transcript.jsonl"      # cycle2 = iter 1
  [ -f "$OUT/run.jsonl" ]      || cp "$ROOT/results/$SRC/run.jsonl" "$OUT/run.jsonl"  # 种子(Ralph 追加 iter2+)
  [ -d "$OUT/worker_memory" ]  || cp -a "$ROOT/results/$SRC/worker_memory" "$OUT/worker_memory"
  echo "  ✓ seeded: worktree@$(git -C "$WT" rev-parse --short HEAD)(22 kernel)+ iters/iter_01 + run.jsonl 种子 + worker_memory($(ls "$OUT/worker_memory"/*.md 2>/dev/null | wc -l))"
else
  echo "▶ 已 seed(worktrees/$RUN + iters/iter_01 已在)→ 直接 resume"
fi

export RALPH_MODEL=claude-fable-5                 # worker 钉死 Fable 5
export RALPH_MAX_ITERS="${RALPH_MAX_ITERS:-11}"   # 【总】上限(含 iter_01 种子)→ 默认续 10 个新迭代(iter 2..11)
export RALPH_JUDGE="${RALPH_JUDGE:-0}"            # NEVER-STOP seed 无终止态 → 纯无限外循环到上限
export RALPH_BUDGET_USD="${RALPH_BUDGET_USD:-}"

echo "▶ Ralph Loop (Fable 5):$RUN,续到【总】 $RALPH_MAX_ITERS 轮(iter-1=cycle2 种子,从 iter 2 起新跑 → $((RALPH_MAX_ITERS-1)) 个新迭代)"
exec bash "$ROOT/scripts/launch_ralph_loop.sh" "$RUN" --resume
