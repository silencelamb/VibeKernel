#!/usr/bin/env bash
# 启动 naive_strong 方法 worker:= naive 的【纯 prompt、NEVER STOP、零脚手架、无 Stop-hook】,
#   唯一差别 = seed 换成强 framing 版(scripts/seed_gemm_strong.txt:堵死"宣布完成/等我"等收尾借口、
#   要求想停前先列 ≥5 个未实测方向并逐一 ncu 否决;技法无关、不喂答案)。
# 实验定位:介于 naive(弱 prompt) 与 goal(等价强 prompt + 真 Stop-hook 强制) 之间的【第三臂】——
#   用来拆开"framing"和"Stop-hook 强制"各自的贡献(naive_strong 有 framing、无 hook)。
# 用法: bash scripts/launch_naive_strong.sh [run-name]   —— 默认 "naive_strong";多跑传 naive_strong_cycle2 …
# 每次跑在自己的 git worktree(独立 cwd-slug → transcript/memory 隔离);跑完用 finish_run.sh 收尾。
# ⚠️ 必须在【带 CAP_SYS_ADMIN 的容器】里跑(ncu 要用);先 ./scripts/ncu-doctor.sh 证实。
# detached(setsid),关终端也不死;停用 kill -TERM -- -<pid>。
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METHOD=naive_strong
RUN_NAME="${1:-}"
source "$ROOT/scripts/_run_common.sh"   # 建 worktree、cp 手册、绑 GPU,导出 RUN/WT/OUTDIR/JSONL/PIDFILE

export SEED="$(cat "$ROOT/scripts/seed_gemm_strong.txt")"   # ⭐ 唯一与 naive 的差别:强 framing seed

cd "$WT"
setsid bash -c 'echo $$ > "$PIDFILE"; exec claude -p "$SEED" \
  --model claude-opus-4-8 --effort max \
  --dangerously-skip-permissions \
  --output-format stream-json --verbose > "$JSONL" 2>&1' &

for _ in $(seq 1 50); do [ -s "$PIDFILE" ] && break; sleep 0.1; done
echo "naive_strong worker launched."
echo "  run-name  : $RUN   (结果落 results/$RUN/)"
echo "  worktree  : $WT"
echo "  group PID : $(cat "$PIDFILE" 2>/dev/null || echo '?')"
echo "  GPU       : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-?}"
echo "  log       : $JSONL"
echo "  watch     : $ROOT/results/watch_run.sh $JSONL"
echo "  stop tree : kill -TERM -- -\$(cat $PIDFILE)"
echo "  收尾归档  : bash $ROOT/scripts/finish_run.sh $RUN"
