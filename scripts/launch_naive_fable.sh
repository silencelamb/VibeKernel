#!/usr/bin/env bash
# 启动 naive 方法 worker —— 【Fable 5 臂】:与 launch_naive.sh 逐字相同,唯一差异 = 主 worker 模型
#   claude-opus-4-8  →  claude-fable-5(Anthropic 最新模型)。其余全同:同一份 seed_gemm.txt、
#   同 --effort max、同 NEVER STOP 纯自驱、同 worktree/归档流 → 单变量 = 模型,可直接与 naive(Opus) 家族横比。
# 用法: bash scripts/launch_naive_fable.sh [run-name]   —— 默认 "naive_fable";多跑传 naive_fable_cycle2 …
# 每次跑在自己的 git worktree(独立 cwd-slug → transcript/memory 隔离);跑完用 finish_run.sh 收尾。
# ⚠️ 必须在【带 CAP_SYS_ADMIN 的容器】里跑(ncu 要用);先 ./scripts/ncu-doctor.sh 证实。
# detached(setsid),关终端也不死;停用 kill -TERM -- -<pid>。
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METHOD=naive_fable
RUN_NAME="${1:-}"
source "$ROOT/scripts/_run_common.sh"   # 建 worktree、cp 手册、绑 GPU,导出 RUN/WT/OUTDIR/JSONL/PIDFILE

export SEED="$(cat "$ROOT/scripts/seed_gemm.txt")"   # worker 指令(与 naive/Opus、/goal 共用同一份,保证跨方法+跨模型文字一致)

cd "$WT"
# 唯一旋钮:--model claude-fable-5(naive/Opus 臂为 claude-opus-4-8)。effort 仍钉 max → 模型是唯一变量。
setsid bash -c 'echo $$ > "$PIDFILE"; exec claude -p "$SEED" \
  --model claude-fable-5 --effort max \
  --dangerously-skip-permissions \
  --output-format stream-json --verbose > "$JSONL" 2>&1' &

for _ in $(seq 1 50); do [ -s "$PIDFILE" ] && break; sleep 0.1; done
echo "naive (Fable 5) worker launched."
echo "  run-name  : $RUN   (结果落 results/$RUN/)"
echo "  worktree  : $WT"
echo "  group PID : $(cat "$PIDFILE" 2>/dev/null || echo '?')"
echo "  GPU       : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-?}"
echo "  worker    : claude-fable-5 / effort max   (vs naive(Opus): claude-opus-4-8 — 唯一变量)"
echo "  log       : $JSONL"
echo "  watch     : $ROOT/results/watch_run.sh $JSONL"
echo "  stop tree : kill -TERM -- -\$(cat $PIDFILE)"
echo "  收尾归档  : bash $ROOT/scripts/finish_run.sh $RUN"
