#!/usr/bin/env bash
# 启动 /goal 方法 worker:薄循环 = 每 turn 自动续 + 独立 evaluator 读 transcript 判停。
# 与 naive 唯一机制差异:停不停由【外部 evaluator 按 condition 判】,而非 naive 的纯 "NEVER STOP" 自驱。
# 用法: bash scripts/launch_goal.sh [run-name]   —— 默认 "goal";多跑传 goal_cycle2 / goal_cycle3 …
# 每次跑在自己的 git worktree(独立 cwd-slug → transcript/memory 隔离);跑完用 finish_run.sh 收尾。
# ⚠️ 必须在【带 CAP_SYS_ADMIN 的容器】里跑(ncu 要用)。⚠️ /goal 是 session 级 Stop hook → 需 workspace 已 trust 且 hooks 未被禁。
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METHOD=goal
RUN_NAME="${1:-}"
source "$ROOT/scripts/_run_common.sh"   # 建 worktree、cp 手册、绑 GPU,导出 RUN/WT/OUTDIR/JSONL/PIDFILE

# —— 两边模型旋钮 ——
# worker(主 session):下方 claude 调用 --model/--effort 钉死。
# evaluator(/goal 判停器)= Claude Code 的 "small fast model" / haiku 槽。
# export ANTHROPIC_DEFAULT_HAIKU_MODEL=claude-opus-4-8   # ← 默认注释:evaluator 用 Haiku(省钱、判停一样准)。要判停器也上 opus 就解开本行。
export CLAUDE_CODE_EFFORT_LEVEL=max                       # env 优先级最高;主要作用于 worker

# condition 一物两用(worker 第一 turn 指令 + evaluator 判停依据)。= 与 naive 完全相同的 seed(共享 seed_gemm.txt)。
# 这份是 "NEVER STOP" 无终止态 → evaluator 永判未达成 → 在 worker 每个【想停】点把它顶回去续。
export CONDITION="$(cat "$ROOT/scripts/seed_gemm.txt")"

cd "$WT"
# 界:默认不设硬界,跑到自然终止/手动 kill。要成本兜底,在下面加 --max-budget-usd <N>(优雅停)。
# 不要用 --max-turns(单位 agentic-turn、达上限 error 退出、污染 run.jsonl)。
setsid bash -c 'echo $$ > "$PIDFILE"; exec claude -p "/goal $CONDITION" \
  --model claude-opus-4-8 --effort max \
  --dangerously-skip-permissions \
  --output-format stream-json --verbose > "$JSONL" 2>&1' &

for _ in $(seq 1 50); do [ -s "$PIDFILE" ] && break; sleep 0.1; done
echo "goal worker launched."
echo "  run-name  : $RUN   (结果落 results/$RUN/)"
echo "  worktree  : $WT"
echo "  group PID : $(cat "$PIDFILE" 2>/dev/null || echo '?')"
echo "  GPU       : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-?}"
echo "  worker    : claude-opus-4-8 / effort max"
echo "  evaluator : Haiku (默认 small-fast; opus 重定向已注释,见脚本)"
echo "  condition : = naive seed(完全一致); 无终止态 → evaluator 在每个想停点顶回去续"
echo "  log       : $JSONL"
echo "  watch     : $ROOT/results/watch_run.sh $JSONL"
echo "  stop tree : kill -TERM -- -\$(cat $PIDFILE)"
echo "  收尾归档  : bash $ROOT/scripts/finish_run.sh $RUN"
