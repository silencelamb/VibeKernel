#!/usr/bin/env bash
# 启动 dynamic_workflow 方法 worker(主臂:headless、无人值守、可与 naive/goal 同协议横比)。
# 受测机制 = Opus 4.8 的 Dynamic Workflow 自动多 agent 编排:worker 把每个实质子任务 fan-out 成 workflow
#   (并行探索 / 对抗验证 / 综合),而非单 agent 串行。
# ⭐ 与 naive 的【唯一】差别 = 在共享 seed 末尾追加一个 `ultracode` 关键字(seed 文件本身字节不动、
#   与 naive/goal 完全一致)——该关键字触发 Dynamic Workflow,且【保留】下面传的 --effort max、
#   不会把它压成 xhigh(官方:"run a single task as a workflow WITHOUT changing the session's effort level")。
#   → 于是 model / effort(max) / seed / 协议(无人值守 NEVER STOP) / 计分口径 全部与 naive 相同,
#     单一变量 = 「Dynamic Workflow 开/关」。与 /goal 用 `/goal ` 前缀包同一份 seed 完全对称。
# ⚠️ 前置:headless 下 `ultracode` 关键字要生效,需 config 里 `ultracodeKeywordTrigger` 为 on(默认 on;
#   用 /config 查/开)。worker ≥ v2.1.160(更早版本触发词是 `workflow`)。跑完务必在 transcript 里确认
#   workflow 真触发(见 runbooks/dynamic_workflow.md「确认 workflow 真触发」),否则退化成普通 max 单 agent。
# 💰 workflow fan-out 出几十个子 agent → 比 naive 烧 token 多得多;要成本兜底加 --max-budget-usd <N>(优雅停)。
# 用法: bash scripts/launch_dynamic_workflow.sh [run-name]   —— 默认 "dynamic_workflow";多跑传 dynamic_workflow_cycle2 …
# 每次跑在自己的 git worktree(独立 cwd-slug → transcript/memory 隔离);跑完用 finish_run.sh 收尾。
# ⚠️ 必须在【带 CAP_SYS_ADMIN 的容器】里跑(ncu 要用);先 ./scripts/ncu-doctor.sh 证实。
# detached(setsid),关终端也不死;停用 kill -TERM -- -<pid>。
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METHOD=dynamic_workflow
RUN_NAME="${1:-}"
source "$ROOT/scripts/_run_common.sh"   # 建 worktree、cp 手册、绑 GPU、清 base-slug worker memory,导出 RUN/WT/OUTDIR/JSONL/PIDFILE

export SEED="$(cat "$ROOT/scripts/seed_gemm.txt")"   # 与 naive/goal 同一份(保证跨方法文字一致)
export PROMPT="$SEED"$'\n\nultracode'                 # ⭐ 方法标记:末尾追加 ultracode 关键字(触发 workflow、保留 --effort max)

cd "$WT"
setsid bash -c 'echo $$ > "$PIDFILE"; exec claude -p "$PROMPT" \
  --model claude-opus-4-8 --effort max \
  --dangerously-skip-permissions \
  --output-format stream-json --verbose > "$JSONL" 2>&1' &

for _ in $(seq 1 50); do [ -s "$PIDFILE" ] && break; sleep 0.1; done
echo "dynamic_workflow worker launched (headless 主臂)."
echo "  run-name  : $RUN   (结果落 results/$RUN/)"
echo "  worktree  : $WT"
echo "  group PID : $(cat "$PIDFILE" 2>/dev/null || echo '?')"
echo "  GPU       : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-?}"
echo "  worker    : claude-opus-4-8 / effort max / + ultracode 关键字(单变量=Dynamic Workflow 开)"
echo "  前置      : config 的 ultracodeKeywordTrigger 须 on(/config 查);worker ≥ v2.1.160"
echo "  log       : $JSONL"
echo "  watch     : $ROOT/results/watch_run.sh $JSONL"
echo "  stop tree : kill -TERM -- -\$(cat $PIDFILE)"
echo "  收尾归档  : bash $ROOT/scripts/finish_run.sh $RUN"
echo "  ⚠️ 跑完务必查 transcript 里 Workflow/子 agent fan-out 是否真触发(零 = 退化成普通 max,这跑不算本方法)。"
