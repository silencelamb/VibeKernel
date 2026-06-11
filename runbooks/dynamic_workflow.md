# Runbook — Dynamic Workflow / ultracode

> **主臂(默认)= headless,真源 [`scripts/launch_dynamic_workflow.sh`](../scripts/launch_dynamic_workflow.sh)**:无人值守、可与 naive/goal **同协议横比**。**副臂(可选)= 交互式忠实 xhigh 预设,真源 [`scripts/setup_dynamic_workflow.sh`](../scripts/setup_dynamic_workflow.sh)**(见文末「附」)。
> **worker 指令(seed)= 共享文件 [`scripts/seed_gemm.txt`](../scripts/seed_gemm.txt)，与 naive / /goal 同读一份、文字完全一致**——改 seed 改那一个文件；主臂的「方法标记」只是在 seed 末尾追加一个 `ultracode` 关键字（在 launcher 里加，seed 文件本身字节不动）。**别在本文件复制 seed/命令**。
> worker session 在**本次 worktree** `worktrees/<run-name>/` 根开**全新** headless 进程，自动加载 worktree 的 `CLAUDE.md`（由脚本从根 `CLAUDE_For_KernelAgent.md` cp 来）。跑完用 `scripts/finish_run.sh <run-name>` 一键归档 + 删 worktree（见 META §5）。
> 本文件不透露这是一次方法对比。

## Dynamic Workflow 是什么（被测的「自动多 agent 编排」）

- **Dynamic Workflow** = Opus 4.8 的能力：worker 对**每个实质子任务**自己规划并跑一个多 agent workflow（fan-out 子 agent 并行探索 / 对抗验证 / 综合），而非单 agent 串行。
- **headless 怎么开**：官方确认「Workflows are available in … non-interactive mode with `claude -p`」。触发方式 = **prompt 里带 `ultracode` 关键字**；且该关键字**保留你传的 `--effort`、不压成 xhigh**——原话「run a single task as a workflow **without changing the session's effort level**」。所以主臂用 `claude -p "<seed>\n\nultracode" --effort max`：**workflow 在 max 上跑**。
- **目标** = 向 A100 fp16 峰值 312 TFLOPS 逼近，**无 cuBLAS 基线**（基座已去 cuBLAS；正确性以 v0 cBLAS 为 ground truth，自写 kernel 从 v1 起）。
- 固定 `--model claude-opus-4-8 --effort max`（跨方法可比）。

## 受控变量：与 naive 的【唯一】差别 = `ultracode` 关键字

主臂把混淆变量压到最干净 —— **model / effort(max) / seed / 协议（无人值守、NEVER STOP）/ 计分口径全部与 naive 相同**，单一变量 = 在 seed 末尾追加的 `ultracode` 关键字（= 开「Dynamic Workflow 自动编排」）。这与 `/goal` 用 `/goal ` 前缀包同一份 seed **完全对称**（那边的方法标记是 `/goal ` 前缀 + Stop hook，这边是 `ultracode` 后缀 + workflow 编排）。

> ⚠️ **「max 上的 Dynamic Workflow」≠ 字面 ultracode 预设**。产品里 `/effort ultracode` 预设 = **xhigh** + 自动 workflow。主臂**故意**用 max（而非 xhigh）跑 workflow，为的是**与 naive/goal 的 effort=max 严格对齐**、把唯一变量留给「workflow 开/关」。要的是**忠实复刻字面预设（xhigh）**就走文末副臂（但那会引入 effort=xhigh≠max 的混淆）。两条臂测的不是同一个问题：主臂测「在固定 max 下加 workflow 编排值多少」，副臂测「产品 ultracode 预设整体值多少」。

- **界/成本**：seed 是 NEVER-STOP 无终止态 → worker 跑到自然终止 / 手动 kill（同 naive）。⚠️ **workflow fan-out 出几十个子 agent → 比 naive 烧 token 多得多**；要成本兜底在 launcher 的 claude 行加 `--max-budget-usd <N>`（优雅停，print 模式专用），**不要用 --max-turns**。

## 启动（在 VibeKernel 项目根）

```bash
bash scripts/launch_dynamic_workflow.sh [run-name]   # 默认 dynamic_workflow；多跑传 dynamic_workflow_cycle2 …
```

脚本依次（经 `_run_common.sh`）：①校验基座 + **防覆盖闸**（worktree 已存在 或 `results/<run-name>` 已有结果 → 拒跑、提示换 run-name）→ ②`git worktree add` + cp `CLAUDE.md` → ③`source env.sh` 绑 GPU → ④清 base-slug worker memory（防跨-run 串味，见 [[concurrent-runs-share-worker-memory]]：**别和别的 run 并行**）→ ⑤`setsid` detached 起 `claude -p "<seed>\n\nultracode" --effort max`，落 `results/<run-name>/run.jsonl` + `run.pid`。

- **前置（headless 关键字触发必需）**：config 里 `ultracodeKeywordTrigger` 须为 **on**（默认 on；`/config` 查/开）。worker ≥ **v2.1.160**（更早版本触发词是 `workflow`）。
- 实时看：`results/watch_run.sh results/<run-name>/run.jsonl`
- 停整棵进程树：`kill -TERM -- -$(cat results/<run-name>/run.pid)`
- **确认 workflow 真触发**（本方法的核心证据）：跑中/跑后看 transcript 里有没有 `Workflow` 工具调用、子 agent fan-out（`Task`/agent 事件）。**若全程零 workflow**，说明关键字没触发 → 排查 `ultracodeKeywordTrigger` 是否 on、worker 版本是否 ≥ v2.1.160；**零 workflow 的跑退化成普通 max 单 agent，不算 Dynamic Workflow 臂**，须在 result.md 如实标。

## 跑完（编排侧，不写进 worker）

1. **一键收尾**：`bash scripts/finish_run.sh <run-name>` —— 归档 transcript（token/曲线权威源）→ `check_handwritten`（用库判无效）→ 快照 `src`/`include`/`logs`/`worker.patch` → `parse_run` 出 `result.csv`/`result_table.md`/`curve.png`（每版最佳点、固定画 cuBLAS~225 与 312 参照线）→ **删 worktree**。结尾 `result` 事件含 usage/duration = 权威总量（headless 有；与 naive/goal 同口径）。
   - ⚠️ **token 口径（本方法特有）**：workflow 的子 agent 也花 token。核对它们是否以独立 `message.id` 进 transcript（会被 parse_run 去重累加进曲线 x 轴）、以及最终 `result` 事件的 `usage.output_tokens` 是否含子 agent 份额，据此报「方法总成本（含 workflow 子 agent）」，写进 result.md。
2. 套 `results/TEMPLATE.md` 写 `results/<run-name>/result.md`：峰值 / token / wall / cost、**Workflow 是否真触发 + fan-out 了多少子 agent**（与 /goal 的 sub-agent 计数同口径，便于横比）、workflow 编排对峰值/收敛速度有没有可见帮助。薄 dispatcher 想图上显示技法写 `results/<run-name>/labels.json`。
3. LLM 有随机性，多跑几轮（`dynamic_workflow_cycle2` 等）报均值/方差 —— 直接 `bash scripts/launch_dynamic_workflow.sh dynamic_workflow_cycle2` 再来（独立 worktree + slug + results，零串味；**务必串行、别和别的 run 并行**）。

## 附：交互式忠实 xhigh 预设臂（可选副臂，`setup_dynamic_workflow.sh`）

只在你想**忠实复刻产品 `/effort ultracode` 字面预设（= xhigh + 自动 workflow）**时用。`bash scripts/setup_dynamic_workflow.sh [run-name]` 只搭 worktree 管道、不 headless 启动，末尾打印手动几行（`cd worktrees/<run> && claude --model claude-opus-4-8 --dangerously-skip-permissions` → 进去 `/effort ultracode` → 粘贴 seed）。它比主臂多**三条混淆变量**，result.md 必标：

1. **human-in-loop**：交互式每句话都是 steering → 测的是「ultracode + 你」，非自驱（与 naive/goal 零干预协议冲突）。
2. **effort = xhigh（非 max）**：相对其余钉 max 的方法多一变量，峰值差无法干净归因。
3. **无 headless `run.jsonl`**：`watch_run` 喂不进；但 transcript 照常按 worktree cwd-slug 写 → `finish_run` 仍能归档+出曲线（token/曲线本就从 transcript 取，见 [[naive-token-attribution-limit]]）；唯一损失 = finish 末「总计(权威)」显示 `?`（无 stream-json result 事件），总 token 改看曲线 x 轴末值。
   - ⚠️ 交互式**必须在 worktree 内起 claude**（`cd worktrees/<run>` 后再 `claude`），finish_run 才按这个 cwd-slug 找得到 transcript。

**横比时优先用主臂**（headless + max，单变量）；副臂仅作「字面预设整体值多少」的补充观测。

## 附:guided 臂（headless，`launch_dynamic_workflow_guided.sh`）—— 规定死搜索策略

cycle1 主臂(spontaneous)发现:worker 自发编排的 workflow **搜索深但范式保守**,且用贪心/孤立增量评估(有 "cooperative blindspot")。guided 臂用一份**规定死搜索策略的 seed**([`scripts/seed_gemm_dynwf_guided.txt`](../scripts/seed_gemm_dynwf_guided.txt))测「人引导 workflow 策略能否破自发版的保守、冲过天花板」:

- **机制 = 主臂同款**(headless `claude -p` + 末尾 `ultracode` 关键字 + `--effort max`),**唯一差别 = seed**:禁 forward-greedy / standalone-delta,强制 **Phase1 全栈搭满 → Phase2 leave-one-out 消融(每 ingredient 一个子 agent,从全栈移除该项重测,移除掉性能才丢)→ Phase3 显式 synergy 组合 variant → Phase4 排序**;附 ingredient 清单 + 已知 synergy 组。seed 只规定**策略形态**,不喂 kernel 实现。
- **跑**:`bash scripts/launch_dynamic_workflow_guided.sh [run-name]`(默认 `dynamic_workflow_guided`)。前置/收尾同主臂。
- ⚠️ **可比性边界**:guided seed **≠ 共享 `seed_gemm.txt`**(文字不同)→ **只能和主臂 `dynamic_workflow`(spontaneous)比**,**不能直接和 naive/goal 比峰值**(跨方法峰值对比要求同一份 seed)。但**计分口径(task1 4096³/100 轮)恒定**,测量仍可比。result.md 必标这条边界。
- **跑完查**:transcript 里 workflow 是否**真按 4-phase leave-one-out 跑**(没退化成普通 config sweep);最终输出有没有那张「standalone 会被丢、leave-one-out 证明 load-bearing」的 ingredient 表(= blindspot 被克服的证据)。
- seed 已去掉一切 cuBLAS/cuBLASLt/CUTLASS 引用(基座无库基线 + `check_handwritten` 会判库用为无效);ceiling 指标改用 **% of 312 peak**。
