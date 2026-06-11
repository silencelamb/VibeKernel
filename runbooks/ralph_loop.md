# Runbook — Ralph Loop(每轮全新 session 的薄外循环)

> **唯一可执行真源 = [`scripts/launch_ralph_loop.sh`](../scripts/launch_ralph_loop.sh)**:外循环、旋钮、judge 都在那(detached/setsid,各轮 stream-json 累加进 `results/<run>/run.jsonl`、外循环标记进 `ralph.log`)。**worker 指令(seed)= 共享文件 [`scripts/seed_gemm.txt`](../scripts/seed_gemm.txt),与 naive/goal 同读一份、文字完全一致**;Ralph 每轮把这同一份重喂 = 忠实 Ralph 的 `cat PROMPT.md`。改 seed 改那一个文件;改命令改脚本——**别在本文件复制**。
> worker 每轮在**本次 worktree** `worktrees/<run-name>/` 根开**全新** headless 进程,自动加载 worktree `CLAUDE.md`(脚本启动时从根 `CLAUDE_For_KernelAgent.md` cp 来)。跑完用 `scripts/finish_run.sh <run-name>` 归档 + 删 worktree(见 META §5,Ralph 的 token 合并见下「跑完」)。
> 本文件不透露这是一次方法对比。出处:Ralph technique = [ghuntley.com/ralph](https://ghuntley.com/ralph/) / [ghuntley.com/loop](https://ghuntley.com/loop)(`while :; do cat PROMPT.md | claude-code ; done`)。

## Ralph Loop 是什么(被测的「fresh-session 外循环」)

- **机制**:一个 bash 外循环,**每轮重起一个全新 `claude -p "$SEED"` session**。kernel 进度靠**磁盘**跨轮持久(worktree 里的 `task-1/src` 累积、worker auto-memory 在 run 内保留),但 **context window 每轮清空** → 模型每轮"满血"重读现状、无 transcript 漂移。
- **与 /goal 的对照(核心区别)**:两者都是「让 agent 一直跑」,差别在**怎么续**:
  - **/goal** = 【单个长 session】内续:worker 想停时外部 LLM-judge(默认 Haiku)读 transcript 把它顶回去,**上下文一路滚大、judge 在同一 session 内接入**(session 级 Stop hook)。
  - **Ralph** = 【跨 session 重启】:每轮全新 session、干净上下文;(可选)轮间用一个**全新 judge session** 判 condition 是否达成(`RALPH_JUDGE=1`)。判停器本身也是"新起的 session" = Ralph 的精髓。
  - 一句话:**/goal 是单-session 续跑 + 同 session 判停;Ralph 是跨-session 重启 + 新 session 判停**。
- **与 naive 的单变量**:naive = 一次 `claude -p`(单 session、自驱到自停/崩);Ralph = 把**同一份 seed** 外面包一个 fresh-session 重启循环。`model / effort(max) / seed / worktree / 计分口径` 全相同 → 单变量 = 「fresh-session 外循环 开/关」。
- ⭐ **附带好处(可在 result.md 量化)**:naive/goal 多个 cycle 毁于 ECONNRESET 半途崩(见各 result.md);Ralph 外循环**天然在崩后立刻重起新 session 续命** → 对进程级死亡(API 崩 / context 爆 / usage 限额)鲁棒,而 /goal 的 Stop hook 救不回已崩的进程。
- **目标** = 向 A100 fp16 峰值 312 TFLOPS 逼近,无 cuBLAS 基线;正确性以 v0 为 ground truth,自写 kernel 从 v1 起。固定 `--model claude-opus-4-8 --effort max`(跨方法可比)。

## 「界」与 judge(为什么默认无限外循环)

- seed 是 "NEVER STOP" **无终止态** → 跟 /goal 同理,judge 永判「未达成」。所以 **`RALPH_JUDGE` 默认 0**:纯无限外循环,正对 `while :; do … done`,跑到 `RALPH_MAX_ITERS`(默认 40,安全上限,防无人值守跑飞)/ 手动 kill。
- `RALPH_JUDGE=1` 才起轮间 judge(全新 session、只读上一轮工作摘要、回 STOP/CONTINUE)——给**有真终止条件**的实验用;NEVER-STOP seed 下没意义。
- **hot-loop 防护**:连续 3 轮 <30s(多半 auth/setup 崩、空烧钱)→ 自动停外循环。
- 成本兜底:`RALPH_BUDGET_USD=<N>` 给**每轮** `claude -p` 挂 `--max-budget-usd`(优雅停)。⚠️ Ralph 每轮一个完整 naive 跑 → **总成本 = 各轮之和**,可能比 naive/goal 高很多,务必用 `RALPH_MAX_ITERS` / `RALPH_BUDGET_USD` 控。

## 模型旋钮

| | 配置 | 说明 |
| --- | --- | --- |
| **worker(每轮主 session)** | `--model claude-opus-4-8 --effort max` | 脚本里直接钉。 |
| **judge(可选,轮间判停)** | 默认 `claude-haiku-4-5-20251001`(= /goal evaluator 的 Haiku 槽对位),`RALPH_JUDGE_MODEL` 可换 opus | 仅 `RALPH_JUDGE=1` 时起;NEVER-STOP seed 默认不起。 |

## 启动(在 VibeKernel 项目根)

```bash
bash scripts/launch_ralph_loop.sh [run-name]   # run-name 默认 ralph_loop;多跑传 ralph_loop_cycle2 …
# 可选旋钮:RALPH_MAX_ITERS=40 RALPH_JUDGE=0 RALPH_BUDGET_USD= bash scripts/launch_ralph_loop.sh ralph_loop
```

脚本经 `_run_common.sh`:①校验基座 + 防覆盖闸 → ②`git worktree add` + cp `CLAUDE.md` → ③`source env.sh` 绑 GPU → ④清 base-slug worker memory(本 run 白板)→ ⑤`setsid` detached 起**外循环**:每轮全新 `claude -p "$SEED"`,各轮 stream-json 累加进 `results/<run>/run.jsonl`,每轮 transcript 单独存 `results/<run>/iters/iter_NN.transcript.jsonl`,外循环标记进 `ralph.log`。

- 实时看:`results/watch_run.sh results/<run-name>/run.jsonl`(各轮 stream-json 累加,纯 JSON、不挂);外循环节奏看 `results/<run-name>/ralph.log`。
- 停整棵进程树:`kill -TERM -- -$(cat results/<run-name>/run.pid)`
- **前置**:容器带 CAP_SYS_ADMIN(ncu);先 `./scripts/ncu-doctor.sh`。`--dangerously-skip-permissions` + `IS_SANDBOX=1`。
- **确认真在「fresh-session」循环**:`ralph.log` 应见多条 `==== RALPH ITER N ====` + 每轮一个不同的 transcript 文件名(`iters/` 下逐轮新增)。若只有 1 轮 = 退化成 naive(查 `RALPH_MAX_ITERS` / 是否秒退触发 hot-loop 防护),要在 result.md 如实记。

## 续跑(`--resume`)—— 崩了 / 想再加几轮

```bash
bash scripts/launch_ralph_loop.sh <run-name> --resume     # --resume 与 run-name 顺序随意
```

- **前置**:该 run 之前用 `finish_run.sh <run-name> --keep-worktree` 归档过 → ① `worktrees/<run-name>/` 还在(kernel 在盘上)② `results/<run-name>/worker_memory/` 有备份。若 worktree 已被删(finish 不带 flag)→ 不能 resume,只能另起新 cycle `<run-name>_cycle2`(kernel 可由 `worker.patch` 在新 worktree 上 `git apply` 复现,但那是新 cycle)。
- resume 与 fresh launch 的差(刻意**不** source `_run_common.sh`):**复用**已存 worktree(不新建、不触发防覆盖闸)→ **还原** `results/<run>/worker_memory/` 回 base-slug(新 session 带着上次学到的瓶颈/死路,不白板重走)→ **不清** base-slug memory → **追加**进同一 `run.jsonl`/`ralph.log`、iter 序号**接着** `iters/` 最大号往下。
- `RALPH_MAX_ITERS` 是**总**上限(把已存 iters 也算进去):cycle1 跑了 5 轮、`RALPH_MAX_ITERS=40` → resume 续 6…40。想多续就调大它。
- ⚠️ resume 会把 worker_memory 还原回 base-slug → 同一刻**别并行**别的 worker(`concurrent-runs-share-worker-memory` 串味)。
- 启动后照样 `watch_run` / `ralph.log` 看;停同样 `kill -TERM -- -$(cat results/<run>/run.pid)`。

## 跑完(编排侧,不写进 worker)

1. **⚠️ Ralph 特有:先合并各轮 transcript 再归档**。每轮是独立 session = 独立 transcript;`finish_run` 默认只归档**最后一轮**那份 → token/曲线会只算最后一轮(欠报)。报**全量**:
   ```bash
   cat results/<run-name>/iters/*.transcript.jsonl > results/<run-name>/transcript.jsonl   # 合并(按 message.id 去重,跨 session id 天然唯一、累计正确)
   bash scripts/finish_run.sh <run-name>                                                    # 归档 + check_handwritten + 快照 src/patch + 删 worktree
   # finish_run 内部会用它【最新一轮】覆盖 transcript.jsonl → 跑完再合并一次并重跑 parse:
   cat results/<run-name>/iters/*.transcript.jsonl > results/<run-name>/transcript.jsonl
   ./results/parse_run.sh <run-name> results/<run-name>/run.jsonl results/<run-name>/transcript.jsonl
   ```
   （与 `goal_cycle4` 两段 transcript 手工合并同性质的口径坑;见 `goal-cycle4-result` memory。）
2. **防作弊门**:`finish_run` 已自动跑 `check_handwritten.sh`(用了 cutlass/cublas 判无效)。
3. 套 `results/TEMPLATE.md` 写 `results/<run-name>/result.md`:峰值 / 总轮数 / 总 token / cost、**每轮自停在多少 TFLOPS**(fresh-session 每轮各自爬升曲线,看跨轮是否单调抬升 or 重复探索)、**有没有靠重启续命过崩溃**(Ralph 的卖点)、判停器是否触发(若 `RALPH_JUDGE=1`)。
4. LLM 有随机性,多跑几轮(`ralph_loop_cycle2` 等)报均值/方差。
