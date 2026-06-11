#!/usr/bin/env bash
# 【可选副臂】dynamic_workflow 的交互式忠实预设跑法 —— 主臂是 headless 的 launch_dynamic_workflow.sh
#   (单变量、可与 naive/goal 同协议横比;横比优先用主臂)。本脚本仅在你想【忠实复刻产品
#   `/effort ultracode` 字面预设(= xhigh + 自动 workflow)】时用。
# ⚠️【不 headless 启动】,只搭好 worktree 管道,末尾打印手动几行让你【交互式】起 claude、
#   `/effort ultracode`、对话式驱动 —— 这是产品里那个真预设(xhigh),非主臂的 max。
# ⚠️ 比主臂多三条 caveat(与 naive/goal 的受控协议不同,result.md 必须如实标):
#   1) human-in-loop:你每句对话都是 steering → 测的是「ultracode + 你」,非「ultracode 自驱」(naive/goal 零干预)。
#   2) effort=xhigh(非 max):ultracode 预设锁 xhigh,而其余方法钉 --effort max → 多一个变量,峰值差无法干净归因。
#   3) 无 headless run.jsonl:watch_run 喂不进;但 transcript 照常按 worktree cwd-slug 写 → finish_run 仍能归档+出曲线。
# 用法: bash scripts/setup_dynamic_workflow.sh [run-name]   —— 默认 "dynamic_workflow";多跑传 dynamic_workflow_cycle2 …
# ⚠️ 必须在【带 CAP_SYS_ADMIN 的容器】里跑(ncu 要用);先 ./scripts/ncu-doctor.sh 证实。
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METHOD=dynamic_workflow
RUN_NAME="${1:-}"
source "$ROOT/scripts/_run_common.sh"   # 建 worktree、cp 手册、绑 GPU、清 base-slug worker memory,导出 RUN/WT/OUTDIR/JSONL/PIDFILE
# 注:_run_common 末尾 `: > "$JSONL"` 留了个空 run.jsonl —— 交互式不写它(token/曲线从 transcript 出);留着无害,finish_run 容忍。

echo "dynamic_workflow worktree 就绪 —— ⚠️【交互式 /effort ultracode 流程,本脚本不 headless 启动】。"
echo "  run-name  : $RUN   (结果落 results/$RUN/)"
echo "  worktree  : $WT"
echo "  GPU       : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-?}"
echo "  seed      : $ROOT/scripts/seed_gemm.txt  (与 naive/goal 同一份;粘贴为第一条消息,保证跨方法文字一致)"
echo
echo "── 手动开跑:在你自己的交互终端里复制下面整块 ──"
echo "  source $ROOT/env.sh          # 绑 GPU(CUDA_VISIBLE_DEVICES)"
echo "  export IS_SANDBOX=1          # root@docker 绕 root guard(交互式也要)"
echo "  cd $WT"
echo "                               # ⚠️ 必须在 worktree 内起 claude,finish_run 才按这个 cwd-slug 找得到 transcript"
echo "  claude --model claude-opus-4-8 --dangerously-skip-permissions"
echo
echo "  进 claude 后按顺序:"
echo "    1) /effort ultracode       # 真预设 = xhigh + 自动 Dynamic Workflow 编排(= 本方法的受测机制)"
echo "    2) 把 $ROOT/scripts/seed_gemm.txt 的内容粘贴为第一条消息,回车开跑"
echo "       (seed 无终止态/NEVER STOP;交互式下它早晚停或要你接话 —— 接话即 steering,如实记)"
echo
echo "  收尾归档  : bash $ROOT/scripts/finish_run.sh $RUN   (在 VibeKernel 根跑;归档 transcript+防作弊门+快照+曲线+删 worktree)"
echo "  ⚠️ 交互式无 headless run.jsonl → finish_run 的「总计(权威)」会显示 ?;token/曲线照常从 transcript 出,总 token 看曲线 x 轴末值。"
echo "  ⚠️ caveat(result.md 必标):① human-in-loop(测 ultracode+你)② effort=xhigh 非 max ③ 见 runbooks/dynamic_workflow.md。"
