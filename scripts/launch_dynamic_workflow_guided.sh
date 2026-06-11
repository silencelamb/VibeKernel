#!/usr/bin/env bash
# 启动 dynamic_workflow_guided 方法 worker(dynamic_workflow 的【guided 副臂】,headless、无人值守)。
# = 主臂 launch_dynamic_workflow.sh 的同机制(headless claude -p + 末尾 ultracode 关键字 + --effort max),
#   唯一差别 = seed 换成 scripts/seed_gemm_dynwf_guided.txt:在共享 seed 基础上【规定死搜索策略】——
#   禁贪心/孤立增量、强制 leave-one-out(backward elimination)+ 显式 synergy 组合,堵死 "cooperative blindspot"。
# 实验定位:对照【主臂(spontaneous,自发编排)】测「人引导 workflow 策略 能否破自发版的范式保守、冲过天花板」。
# ⚠️ 它的 seed ≠ 共享 seed_gemm.txt(文字不同)→ 只能和【主臂 dynamic_workflow】比,**不能直接和 naive/goal 比峰值**
#   (跨方法峰值对比的前提是同一份 seed;本臂改了 seed)。这点 result.md 必标。计分口径(task1 100轮)仍恒定、可比。
# 用法: bash scripts/launch_dynamic_workflow_guided.sh [run-name]  —— 默认 "dynamic_workflow_guided";多跑传 ..._cycle2 …
# 每次跑在自己的 git worktree(独立 cwd-slug → transcript/memory 隔离);跑完用 finish_run.sh 收尾。
# ⚠️ 前置同主臂:ultracodeKeywordTrigger=on、worker ≥ v2.1.160、带 CAP_SYS_ADMIN 容器(ncu)。
# 💰 workflow fan-out + leave-one-out(每 ingredient 一个子 agent 重 build/bench)→ 比主臂更烧 token;要兜底加 --max-budget-usd <N>。
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METHOD=dynamic_workflow_guided
RUN_NAME="${1:-}"
source "$ROOT/scripts/_run_common.sh"   # 建 worktree、cp 手册、绑 GPU、清 base-slug worker memory,导出 RUN/WT/OUTDIR/JSONL/PIDFILE

export SEED="$(cat "$ROOT/scripts/seed_gemm_dynwf_guided.txt")"   # ⭐ 与主臂唯一差别:规定死搜索策略的 guided seed
export PROMPT="$SEED"$'\n\nultracode'                              # 末尾追加 ultracode 关键字(触发 workflow、保留 --effort max)

cd "$WT"
setsid bash -c 'echo $$ > "$PIDFILE"; exec claude -p "$PROMPT" \
  --model claude-opus-4-8 --effort max \
  --dangerously-skip-permissions \
  --output-format stream-json --verbose > "$JSONL" 2>&1' &

for _ in $(seq 1 50); do [ -s "$PIDFILE" ] && break; sleep 0.1; done
echo "dynamic_workflow_guided worker launched (headless guided 副臂)."
echo "  run-name  : $RUN   (结果落 results/$RUN/)"
echo "  worktree  : $WT"
echo "  group PID : $(cat "$PIDFILE" 2>/dev/null || echo '?')"
echo "  GPU       : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-?}"
echo "  worker    : claude-opus-4-8 / effort max / + ultracode 关键字 / guided seed(规定 leave-one-out + synergy)"
echo "  对照       : 主臂 dynamic_workflow(spontaneous);⚠️ seed≠共享,只能比主臂、不能比 naive/goal 峰值"
echo "  前置      : ultracodeKeywordTrigger=on(/config 查);worker ≥ v2.1.160"
echo "  log       : $JSONL"
echo "  watch     : $ROOT/results/watch_run.sh $JSONL"
echo "  stop tree : kill -TERM -- -\$(cat $PIDFILE)"
echo "  收尾归档  : bash $ROOT/scripts/finish_run.sh $RUN"
echo "  ⚠️ 跑完查 transcript:workflow 是否真按 leave-one-out 4-phase 跑(非退化成 config sweep);输出有没有那张 load-bearing ingredient 表。"
