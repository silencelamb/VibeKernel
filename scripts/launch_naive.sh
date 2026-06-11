#!/usr/bin/env bash
# 启动 naive 方法 worker:纯 prompt、NEVER STOP、单 session 不间断、零脚手架。
# 用法: bash scripts/launch_naive.sh [run-name]   —— run-name 默认 "naive";多跑传 naive_cycle2 / naive_cycle3 …
# 每次跑在自己的 git worktree(独立 cwd-slug → transcript/memory 隔离);跑完用 finish_run.sh 收尾。
# ⚠️ 必须在【带 CAP_SYS_ADMIN 的容器】里跑(ncu 要用);先 ./scripts/ncu-doctor.sh 证实。
# detached(setsid),关终端也不死;停用 kill -TERM -- -<pid>。
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METHOD=naive
RUN_NAME="${1:-}"
source "$ROOT/scripts/_run_common.sh"   # 建 worktree、cp 手册、绑 GPU,导出 RUN/WT/OUTDIR/JSONL/PIDFILE

export SEED="$(cat "$ROOT/scripts/seed_gemm.txt")"   # worker 指令(与 /goal 共用同一份,保证跨方法文字一致)

cd "$WT"
setsid bash -c 'echo $$ > "$PIDFILE"; exec claude -p "$SEED" \
  --model claude-opus-4-8 --effort max \
  --dangerously-skip-permissions \
  --output-format stream-json --verbose > "$JSONL" 2>&1' &

for _ in $(seq 1 50); do [ -s "$PIDFILE" ] && break; sleep 0.1; done
echo "naive worker launched."
echo "  run-name  : $RUN   (结果落 results/$RUN/)"
echo "  worktree  : $WT"
echo "  group PID : $(cat "$PIDFILE" 2>/dev/null || echo '?')"
echo "  GPU       : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-?}"
echo "  log       : $JSONL"
echo "  watch     : $ROOT/results/watch_run.sh $JSONL"
echo "  stop tree : kill -TERM -- -\$(cat $PIDFILE)"
echo "  收尾归档  : bash $ROOT/scripts/finish_run.sh $RUN"
