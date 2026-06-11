# Runbook — /goal(薄循环 + 外部 evaluator 判停)

> **唯一可执行真源 = [`scripts/launch_goal.sh`](../scripts/launch_goal.sh)**:启动参数、两边模型/effort 旋钮都在那(detached/setsid,落 `results/goal/run.jsonl`)。**worker 指令(condition)= 共享文件 [`scripts/seed_gemm.txt`](../scripts/seed_gemm.txt),与 naive 同读一份、文字完全一致**。改 seed 改那一个文件;改命令改脚本——**别在本文件复制**。
> worker session 在**本次 worktree** `worktrees/<run-name>/` 根开**全新** headless 进程,自动加载 worktree `CLAUDE.md`(启动时由脚本从根 `CLAUDE_For_KernelAgent.md` cp 来)。跑完用 `scripts/finish_run.sh <run-name>` 一键归档 + 删 worktree(见 META §5)。
> 本文件不透露这是一次方法对比。

## /goal 是什么(被测的「最薄外部判停循环」)

- **机制**:`claude -p "/goal <condition>"` —— /goal 是一层 **session 级 Stop hook**。每当 worker **finishes a turn 并准备把控制权交还**(即「想停下」)时,一个**独立 evaluator** 读整段 transcript 判 condition 是否达成;**未达成就强制再续一 turn**,达成才收。
- **与 naive 的唯一受控变量 = 谁决定停**,且 **/goal 只在 worker「想停」的那一刻起作用**:
  - naive:`claude -p "$SEED"`,worker 自己想停就停(这正是 naive 测的——纯 prompt 下自身能撑多久)。
  - /goal:**condition 用与 naive 完全相同的 "NEVER STOP" 指令**(无终止态)→ evaluator 永远判「未达成」→ 在 naive 会死掉的那个点把 worker 顶回去再续。**于是 /goal = "naive + 一个不让它自愿停的看门狗"**;两者在 worker 自愿停的点之前行为相同,从那点开始分叉。
  - ⚠️ 看门狗只挡**自愿停**,挡不了**崩溃**(ECONNRESET / usage 限额 / context 爆 等进程级死亡——Stop hook 救不回已崩的进程)。
- **condition 一物两用**:既是 worker 第一 turn 的任务指令(直接以它开跑,无需另发 prompt),又是 evaluator 的判停依据。evaluator **只读 transcript、不跑工具** → 它能判的只有 worker 已 surface 到对话里的东西。
- **目标** = 向 A100 fp16 峰值 312 TFLOPS 逼近,**无 cuBLAS 基线**;正确性以 v0 为 ground truth,自写 kernel 从 v1 起。
- 固定 `--model claude-opus-4-8 --effort max`(跨方法可比)。

## "turn" 与"界"(为什么不用 --max-turns)

- **一个 turn ≈ 一个「worker 工作到准备交还控制权」的完整循环**,中间可含多次工具调用(改代码 / build / 跑 task1 / ncu / 分析)。在本任务 "NEVER STOP" 指令下,worker 往往**轮间不自愿交还**(报完一轮直接接着下一个工具调用)→ Stop hook(evaluator)其实**很少触发,只在 worker 真的想停的稀疏时刻触发**。
- **一次 `claude -p "/goal ..."` 全程是同一个 session / 同一段对话**:turn 结束 evaluator 判「继续」后,**不是重开新对话,而是带着全部累计上下文接着跑**(它记得 v1…vN、所有 ncu 结果、试过什么)。上下文持续变长 → 最终触发 auto-compaction(摘要后继续)或逼近 context 上限。
- **`--max-turns`**:单位是「agentic turn」(≈每次 assistant 推理,粒度 ≠ 优化轮次),且**达上限会 error 退出**(污染 run.jsonl)。**不适合做这里的界,脚本已不用。**
- **能不能一直跑?能。** 无终止态的 condition ⇒ evaluator 永不判达成 ⇒ 跑到自然终止(context 耗尽 / API 错误 / usage 限额)或手动 kill,与 naive 同形。要给成本兜底用 **`--max-budget-usd <N>`**(优雅停、print 模式专用),**不要用 --max-turns**。
- ⚠️ **成本提醒**:看门狗会把自愿停的点顶回去 → /goal 通常**比 naive 跑得更久、更费钱**;再叠加下面 evaluator 在 opus 上的 O(N²) 重读 → 想无界跑要么盯紧、要么挂 `--max-budget-usd`。

## 模型旋钮(满足「两边都用最强」)

| | 配置 | 说明 |
| --- | --- | --- |
| **worker(主 session)** | `--model claude-opus-4-8 --effort max` | 脚本里直接钉。 |
| **evaluator(判停器)** | 默认 **Haiku**(脚本里 `ANTHROPIC_DEFAULT_HAIKU_MODEL` 那行**已注释**省钱);要判停器也上 opus 就解开该行 → `claude-opus-4-8` | evaluator = Claude Code 的 "small fast model" / haiku 槽。`ANTHROPIC_SMALL_FAST_MODEL` 已废弃。 |

⚠️ **默认 evaluator = Haiku(够用、省钱);若解开那行让判停器也上 opus,这三条要知道**(`尽量`都用最强 ≠ 完全可控):
1. `ANTHROPIC_DEFAULT_HAIKU_MODEL` 换的是**整个 haiku/background 槽**(不止 evaluator);headless 跑里其余 background 调用很少,影响可忽略。
2. **evaluator 没有独立 effort 旋钮**:它是一次性 yes/no 的 Stop-hook 判定。opus 4.8 恒用 adaptive reasoning(至少 high),`max` 是否传到这次判定**官方未文档化**;对 yes/no 判定而言 high↔max 无关紧要。`CLAUDE_CODE_EFFORT_LEVEL=max` 主要作用于 worker。
3. 💰 **成本**:opus evaluator 每次触发都重读**全段且不断变长**的 transcript → **O(N²)** 读量,比 Haiku 贵一个数量级。本任务 condition 很简单(判「是否还该继续」),Haiku 大概率判得一样,且 evaluator 模型**不影响我们测的对象(worker)**——所以**默认就用 Haiku**,opus evaluator 仅在你特别想让判停器也最强时才解开。

## 启动(在 VibeKernel 项目根)

```bash
bash scripts/launch_goal.sh [run-name]   # run-name 默认 goal;多跑传 goal_cycle2 / goal_cycle3 …
```

脚本依次(经 `_run_common.sh`):①校验基座 `playground-base/` 在 + **防覆盖闸**(worktree 已存在 或 结果已归档 → 拒跑提示换 run-name)→ ②`git worktree add worktrees/<run-name>` + cp `CLAUDE.md` → ③`source env.sh` 绑 GPU → ④export effort 环境变量(evaluator→opus 那行默认注释)→ ⑤`setsid` detached 起 `claude -p "/goal $CONDITION"`(condition 读自 `scripts/seed_gemm.txt`),落 `results/<run-name>/run.jsonl` + `run.pid`。

- 实时看:`results/watch_run.sh results/<run-name>/run.jsonl`
- 停整棵进程树:`kill -TERM -- -$(cat results/<run-name>/run.pid)`
- 成本兜底(可选):在脚本 claude 行加 `--max-budget-usd <N>`。不加 = 跑到自然终止(同 naive)。
- **前置**:/goal 要 workspace 已 trust + hooks 未禁(`disableAllHooks` / `allowManagedHooksOnly` 任一会让 /goal 不可用)。`--dangerously-skip-permissions` + `IS_SANDBOX=1` 应满足 trust。
- **确认 evaluator 真在判**:跑中粗看 `run.jsonl` 的 `"Stop hook feedback:"`(每次顶回去);跑完权威看 **transcript 的 `goal_status`**(见「跑完」第 3 步)。**若全程零接入**,两种可能:(a) hooks 被禁,/goal 退化成一次性跑(那就不是 /goal 方法了,排查 hooks/trust);(b) worker 一直没自愿停(看门狗没机会触发,此次≈naive)。两种都要在 result.md 如实记。

## 跑完(编排侧,不写进 worker)

1. **一键收尾**:`bash scripts/finish_run.sh <run-name>` —— 归档 transcript + `check_handwritten` + 快照 src/include/logs/worker.patch 进 `results/<run-name>/` + `parse_run`(出曲线,含 /goal evaluator 接入红线)+ 删 worktree。
   - ⚠️ **token 口径核对(/goal 特有)**:evaluator/background 也花 token,核对它是否以独立 `message.id` 进 transcript(会被 parse_run 去重累加进曲线),据此报「方法总成本(含 evaluator)」还是只 worker 份额,写进 result.md。
2. **数 evaluator 接入(/goal 核心证据)**:权威信号 = **transcript** 的 `type:"attachment"` 行嵌的 **`goal_status`**(run.jsonl **没有**):`met`(true=判达成 / false=顶回去)、`sentinel`(true=**启动占位、不算接入**)。**接入次数 = 非-sentinel 的 goal_status 数**:
   ```bash
   jq -c 'select((.|tostring)|test("goal_status"))
          | {met:((.|tostring)|test("met.:true")), sentinel:((.|tostring)|test("sentinel.:true"))}' \
     results/<run-name>/transcript.jsonl
   ```
   旁证(更糙):user 消息 `"Stop hook feedback:"`(每次顶回去)+ `"Stop hook is now active"`(启动);**优先 goal_status**(一个信号区分 启动/达成/顶回去)。`parse_run.sh` 已自动把它换算成「接入时累计 token」、画**红竖虚线**(met:false 红 / met:true 绿)+ 控制台打 `evaluator 接入 N 次 … token=[…]`。
   - ⚠️ **必查:第一次接入时 worker 已到多少 TFLOPS**(cycle1 血泪:worker 自己就冲到 ~205.5 才第一次想停、evaluator 全程才 3 次、净加 +1.3;**别把"更高峰值"想当然归给 harness**——头次接入前的爬升是 worker 自驱,与 naive 的差多半是路径变异)。
3. 套 `results/TEMPLATE.md` 写 `results/<run-name>/result.md`:峰值 / turn / token / cost、evaluator 接入几次 + 每次 token + **头次接入 TFLOPS**、被顶回去后 worker 是真优化还是"强制再论证"。薄 dispatcher 想图上显示技法写 `results/<run-name>/labels.json`。
4. LLM 有随机性,多跑几轮(`goal_cycle2` 等)报均值/方差 —— 直接 `bash scripts/launch_goal.sh goal_cycle2` 再来。
