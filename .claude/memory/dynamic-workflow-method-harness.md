---
name: dynamic-workflow-method-harness
description: "dynamic_workflow=headless主臂(claude -p+ultracode关键字+--effort max,单变量可横比)/交互式xhigh副臂;关键字保留--effort不压xhigh"
metadata: 
  node_type: memory
  type: project
  originSessionId: 56b6e1be-8cd3-4066-9b95-210ebc2d4777
---

dynamic_workflow 方法(2026-06-08 搭好,未跑):受测机制 = Opus 4.8 Dynamic Workflow 自动多 agent 编排(worker 把每个实质子任务 fan-out 成 workflow)。两条臂,横比优先主臂:

**主臂(默认,headless)= `scripts/launch_dynamic_workflow.sh`**:`claude -p "<seed>\n\nultracode" --model claude-opus-4-8 --effort max ...`(setsid、有 run.jsonl、无人值守,与 naive/goal 同协议)。⭐关键事实(官方文档核实):① Dynamic Workflow 在 headless `claude -p` 完全支持;② headless 触发 = prompt 里带 `ultracode` 关键字(config 的 `ultracodeKeywordTrigger` 须 on,默认 on;worker ≥ v2.1.160,更早触发词是 `workflow`);③ **该关键字保留你传的 --effort、不压成 xhigh**("run a single task as a workflow WITHOUT changing the session's effort level")→ 所以能在 max 上跑 workflow。**Why 用 max 不用 xhigh**:与 naive/goal 的 effort=max 严格对齐,单变量=workflow 开/关(与 /goal 用 `/goal ` 前缀包同一份 seed 完全对称;seed 文件字节不动,关键字在 launcher 末尾追加)。⚠️「max 上的 workflow」≠ 字面 ultracode 预设(预设=xhigh)。

**副臂(可选,交互式忠实 xhigh 预设)= `scripts/setup_dynamic_workflow.sh`**:不 headless 启动,只搭 worktree 管道 + 打印手动几行(`cd worktrees/<run> && claude ...` → `/effort ultracode` → 粘贴 seed)。`/effort ultracode` 是产品字面预设(xhigh)。比主臂多 3 confounds(result.md 必标):① human-in-loop(测 ultracode+你);② effort=xhigh 非 max;③ 无 run.jsonl → finish_run 末「总计(权威)」显示 `?`(token/曲线照常从 transcript 出,总 token 看曲线 x 轴末值)。⚠️交互式必须在 worktree 内起 claude,finish_run 才按 cwd-slug 找到 transcript。

**guided 臂(2026-06-08 加,headless)= `scripts/launch_dynamic_workflow_guided.sh` + `scripts/seed_gemm_dynwf_guided.txt`**:主臂同机制(headless+ultracode+max),唯一差别=seed 规定死搜索策略(禁 forward-greedy/standalone-delta;强制 Phase1 全栈→Phase2 leave-one-out 消融每 ingredient 一子 agent→Phase3 synergy 组合→Phase4 排序;破 "cooperative blindspot")。动机=cycle1 发现自发 workflow 搜索深但范式保守+贪心评估。⚠️guided seed≠共享 seed_gemm.txt→**只能比主臂(spontaneous)、不能直接和 naive/goal 比峰值**(峰值横比要同 seed);计分口径(task1 4096³/100)恒定仍可比。seed 已去 cuBLAS/cuBLASLt/CUTLASS(基座无库+check_handwritten 判库无效)、ceiling 改用%of312。

**How to apply**:主臂跑 = `bash scripts/launch_dynamic_workflow.sh [run-name]`(先 /config 确认 ultracodeKeywordTrigger on),`finish_run.sh <run>` 照常归档。**跑完务必查 transcript 里 Workflow/子 agent fan-out 是否真触发(零=退化成普通 max 单 agent,这跑不算本方法)**;子 agent 计数与 [[goal-method-harness]] 同口径横比;token 口径核对子 agent 份额是否进 result 事件/transcript。**串行跑、别并行**(见 [[concurrent-runs-share-worker-memory]])。相关:[[playground-base-clean-fork-topology]] [[naive-token-attribution-limit]]。注:headless 无 `--effort ultracode`(合法值仅 low/medium/high/xhigh/max);用户 2026-06-08 已 /config 设 ultracodeKeywordTrigger=on、worktreeBaseRef=fresh。
