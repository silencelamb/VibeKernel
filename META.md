# META — 项目编排文档

> **本文件 = 完整编排上下文（人读、`@META.md` 带上）。** 跑具体方法的 **worker session 绝不应看到本文件**——它揭示"我们在对比多种方式"，会污染实验。
>
> **文档布局（2026-06-05 重构）：**
> - **根目录不再放 `CLAUDE.md`**。原方法无关的 GEMM 任务手册已改名 **`CLAUDE_For_KernelAgent.md`**（worker 手册母版，建 fork 时 cp 成 `<fork>/CLAUDE.md`）。
> - **编排意识放 git 维护的 `.claude/memory/`**（`orchestration-overview.md` 等），主 session（cwd=VibeKernel 根）自动加载、随项目走（换容器也在，靠 `scripts/link_memory.sh` + SessionStart hook 软链）。
> - **为什么编排上下文不放根 `CLAUDE.md`**：CLAUDE.md 会沿目录树向上继承，fork（worker）在 VibeKernel 子目录下跑会继承根 CLAUDE.md → 若把编排内容当根 CLAUDE.md，worker 就读到"在被对比"、污染实验。memory 按 cwd-slug 隔离，worker slug 不同读不到，故安全。

## 1. 项目目的

用 Claude Code 以不同「方法 / 范式」自动编写 playground 的高性能 GEMM kernel，对比每种方法的最终性能、局限性、特点与**迭代提升曲线**。

- 方法清单的原始出处见 `methods.md`。
- kernel 任务本身见 `CLAUDE_For_KernelAgent.md` 与各 fork 的 `task-1/README.md`（也即该手册描述的那个任务）。
- **性能口径**：追求**极致**——向硬件峰值（A100 fp16 312 TFLOPS）逼近，**不设百分比门槛、无库基线对标**。**2026-06-05 起去掉 cuBLAS 作为目标/参照**：基座 `playground-base` 已彻底移除 cuBLAS/CUTLASS，`v0` cBLAS 仅作正确性 ground truth，方法写的 kernel 从 `v1` 起（见 §5、`playground-base-clean-fork-topology` memory）。以 CUDA kernel 专家的标准要求每个方法。

## 2. 方法阶梯（按"脚手架厚度"排）

> ⚠️ 只有 **naive / `/goal` / Dynamic Workflow / Ralph Loop** 我们能直接定义（naive = 纯 prompt、无外部依赖；`/goal` 与 Dynamic Workflow = Claude Code 自带；Ralph Loop = 我们自己用 bash 把 `claude -p` 包成 fresh-session 外循环、无外部依赖）。其余都是**外部项目，我们只参考其思想与设计、自己实现**——下表「新增机制」列是粗略占位，正式定义前要读各自源码 / 报告，别当成已知。

> **跑法 = git worktree 流（2026-06-07）**：每次跑在 `playground-base` 上开一个独立 worktree（见 §5），**不再每方法建仓**。下表末列「方法仓」名仅作**历史占位 / 标识**，新流程不建 per-method repo（除基座 `playground-base` 本身已转为 submodule 外，无任何 per-method submodule;历史 cycle 的 kernel 都快照在对应 `results/<名>/`）。

| 方法                             | 新增机制                                                                                                                                                                                                                                                                                                                                                                                 | 方法仓                                                                                                                                                           |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **naive**                        | 无脚手架；纯 prompt（NEVER STOP）自驱、零人工干预——测它自己能持续多久 / 冲到多少                                                                                                                                                                                                                                                                                                         | worktree（`launch_naive.sh`；cuBLAS 首跑归档 `results/naive_ncu_cublas/`）                                                                                       |
| **/goal**                        | 自动续 turn + 显式可验证目标 + 外部模型判完成（薄循环，**单 session 内**续 + 同 session 判停）                                                                                                                                                                                                                                                                                            | worktree（`launch_goal.sh`）                                                                                                                                     |
| **Ralph Loop**                   | 同 /goal「让 agent 一直跑」,但**每轮重起全新 session**(`while :; do cat PROMPT.md \| claude -p; done`)——新 session 判 condition、context 每轮清空(vs /goal 单 session 读自己越滚越大的 transcript);kernel 状态靠磁盘跨轮持久、上下文每轮干净;附带天然抗崩(session 崩了下轮重起)。出处 ghuntley.com/ralph + /loop                                                                          | worktree（`launch_ralph_loop.sh`；见 `runbooks/ralph_loop.md`）                                                                                                  |
| KDA (+skill)                     | MIT-HanLab kernel-design-agents skill                                                                                                                                                                                                                                                                                                                                                    | `playground-kda`                                                                                                                                                 |
| AKO (+skill)                     | AKO4ALL / AKO4X                                                                                                                                                                                                                                                                                                                                                                          | `playground-ako`                                                                                                                                                 |
| AI-Infra-Auto-Driven-SKILLS      | BBuf 的推理框架工作流 skill                                                                                                                                                                                                                                                                                                                                                              | `playground-aiinfra`                                                                                                                                             |
| **Dynamic Workflow / ultracode** | 自动多 agent workflow 编排。**主臂**：headless `claude -p "<seed>\n\nultracode" --effort max`，`ultracode` 关键字触发 workflow 且保留 max（单变量=workflow 开/关，可与 naive/goal 横比）。**guided 臂**：同机制、seed 规定死搜索策略(leave-one-out 消融+synergy 组合，破 cooperative blindspot;seed≠共享→只比主臂)。**交互式臂(可选)**：`/effort ultracode` 忠实 xhigh 预设(3 confounds) | worktree（主臂 `launch_dynamic_workflow.sh`；guided `launch_dynamic_workflow_guided.sh`；交互式 `setup_dynamic_workflow.sh`；见 `runbooks/dynamic_workflow.md`） |
| **autoresearch / autokernel**    | 参考其「自治研究式循环」的思想/设计自行搭建（机制待读源后定）                                                                                                                                                                                                                                                                                                                            | `playground-autoresearch`                                                                                                                                        |
| Heuristic Learning               | learning-beyond-gradients 思想                                                                                                                                                                                                                                                                                                                                                           | `playground-heuristic`                                                                                                                                           |
| K-Search                         | UC Berkeley K-Search                                                                                                                                                                                                                                                                                                                                                                     | `playground-ksearch`                                                                                                                                             |


## 3. 实验架构 & 防串味协议（核心）

**布局（2026-06-07 起 git worktree 流）**：本仓库（**silencelamb/VibeKernel**，私有）受 git 管理，追踪 docs / runbooks / `.claude/memory` / 各 `results/<run-name>/` 下的摘要（`result.csv` / `result.md` / `result_table.md` / `curve.png`）**与 kernel 产物快照（`src/`(+`include/`) + `worker.patch`，见 §5/§6）**。**每次跑 = `playground-base` 上一个独立 git worktree `worktrees/<run-name>/`（独立 cwd-slug → transcript / worker-memory 隔离，临时）**——跑完用 `scripts/finish_run.sh <run-name>` 归档进 `results/<run-name>/` 并删 worktree；**基座 `playground-base` 自己永不被改**（只当源）。`env.sh` / 运行日志 / profiler 报告（`*.jsonl` / `*.log` / `*.ncu-rep`）/ `worktrees/` 不入库;**`playground-base/` 2026-06-07 起以 git submodule 纳入**(指向 `silencelamb/playground-base`,公开后 `--recurse-submodules` clone 即得基座)。**复现** = submodule 基座 + `results/<run>/worker.patch`（`git apply` 还原 kernel）。（早期 per-method fork 已退役,旧 cycle 的 kernel 都快照在对应 `results/<名>/`。）

**文档分工**：
- `CLAUDE_For_KernelAgent.md`（根、**方法无关**、worker 手册母版）：只讲"怎么写 / 构建 / profile 这个 GEMM"，含硬性约束（纯手写禁库、task1 100 轮口径、环境已验证免自检）。**性能/版本由编排侧从 task1 工具输出解析，不要求 worker 自报结构化标记。****launcher 启动时 cp 成工作区 `CLAUDE.md`（见 §5）**——根目录无 CLAUDE.md，故 worker 靠工作区内副本，这步由 launcher 自动做。**绝不提"在评估多种方式"。**
- `META.md`（本文件）：全部编排上下文，编排时 `@` 带上。
- `.claude/memory/`（git 维护、主 session 自动加载、worker 读不到）：编排意识 + 工具链/口径速查。

**隔离要点**：
1. 每个方法（乃至每个 cycle）开**全新 session**，别把别的方法的代码 / 技巧带进上下文。
2. worker 只看到 `CLAUDE.md` + 它的 runbook；**不知道自己在被对比**。
3. skill 作用域限**本次 worktree 的 `.claude/skills/`**，**先装、后显式调用**；需 skill 的方法在其 `launch_<方法>.sh` 里（建 worktree 后）装，worktree 用完即删 → 方法间不串 skill。`naive` / `goal` 不装任何 skill。内置 skill（deep-research / code-review 等）在所有方法都在 = 受控常量，不是方法间的混淆变量。
4. **编排 agent ≠ worker agent**：编排带 `META.md`；worker 不带。
5. LLM 有随机性，单跑不算数 —— 每方法多跑几轮，报均值 / 方差。

**隔离现在是结构性保证（2026-06-05 重构后）**：根目录**已无 `CLAUDE.md`**（编排上下文只在 `.claude/memory`，按 cwd-slug 隔离，worker 读不到），所以 worker 不会从父级继承任何编排内容——不再需要"验证有没有串味"。worker 的任务手册**完全靠工作区内自带的 `CLAUDE.md`**（launcher 启动时自动从 `CLAUDE_For_KernelAgent.md` cp，否则 worker 没手册）。

> 启动后可顺手确认工作区的 `CLAUDE.md` 在、且 worker 认得任务（问它"任务是什么"）。这不再是为防串味，纯粹是确认 cp 没漏。

## 4. 编排 → 执行 工作流

1. 开**编排 session**，`@META.md`（按需再 `@methods.md`）。
2. 一起把该方法的 **runbook** 写进 `runbooks/<方法>.md`：worker 的任务 prompt（跨方法复用的 seed）+ 该方法的 harness 指令（如 `/goal` 的 condition、要调用哪个 skill）+ 要装的 skill 清单。**runbook 同样不透露在对比。**
3. 装好该方法所需 skill（放工作区 `playground-base/.claude/skills/`），确认 `/goal` 等可用。
4. 用 `scripts/launch_<方法>.sh [run-name]` 在**独立 worktree** 起全新 headless worker（已钉 `--model claude-opus-4-8 --effort max`，固定模型 + effort 保证跨方法可比）；worker 只读 worktree 内的 `CLAUDE.md` + seed，不知道在被对比。
5. 跑完后**一键 `bash scripts/finish_run.sh <run-name> [--keep-worktree]`**（见 §5）：自动归档 transcript（token/曲线权威源）+ **worker auto-memory（`worker_memory/`）+ `session_id.txt`** + `check_handwritten` + 快照 `src`/`include`/`logs`/`worker.patch` + `parse_run` 出曲线 +（默认）删 worktree。再套 `results/TEMPLATE.md` 写 `result.md`（人读;/goal 还要数 evaluator 接入）。**交互式 run 想日后 `claude --resume <session_id>` 继续人工对话 → 用 `--keep-worktree`**（不删 worktree;resume 需 cwd 还在 + 可从 `worker_memory/` 还原 worker 记忆,session_id 见 `session_id.txt`）。

## 5. 跑一个方法 / cycle（git worktree 流，2026-06-07 起）

> 每次跑 = 在 `playground-base` 上开一个**独立 git worktree** `worktrees/<run-name>/`——独立文件夹名 → **独立 cwd-slug → transcript / worker-memory 天然隔离**（方法 / cycle 间零串味）；**基座 `playground-base` 自己永不被改**（只当 worktree 源，共享 `.git`；它本身以 git submodule 纳入 VibeKernel，但内容永不被改、不另建 per-method repo）。跑完 `finish_run.sh` 一键归档 + 删 worktree（用完即焚）。
> （取代 2026-06-06 的"单工作区原地 reset"流；更早是 per-method-repo + submodule，见 §2 历史。worktree 比"原地 reset"更干净：每跑一棵全新树、且每跑独立 slug，不再共用 `playground-base` slug 把多方法 transcript 混在一个目录。）

**前置（一次性）**：`playground-base/` 在（没有就 `git clone https://github.com/silencelamb/playground-base.git playground-base`）。worker 指令（seed）= 共享 `scripts/seed_gemm.txt`（naive 与 /goal 同读，保证跨方法文字一致）。

**每跑一个方法 / cycle**：

```bash
# 1) 起 worker —— run-name 默认=方法名;多 cycle 传 naive_cycle2 / goal_cycle2 等
#    (决定 worktree 名 / 结果目录 / cwd-slug;launcher 防覆盖:worktree 已存在 或 结果已归档 → 拒跑提示换名)
bash scripts/launch_<方法>.sh [run-name]           # 建 worktrees/<run-name>、cp 手册、绑 GPU、detached 起 worker
#    实时看: results/watch_run.sh results/<run-name>/run.jsonl   停: kill -TERM -- -$(cat results/<run-name>/run.pid)

# 2) 跑完(自然终止 / 手动 kill 后)一键收尾归档
bash scripts/finish_run.sh <run-name>              # 归档 transcript + check_handwritten + 快照 src/include/logs/worker.patch
                                                   # + parse_run 出曲线 + 删 worktree(基座自始至终没动)

# 3) 套 results/TEMPLATE.md 写 results/<run-name>/result.md(人读;/goal 还要数 evaluator 接入,见 runbooks/goal.md)
```

> **多 cycle / 多方法**：换 `run-name` 即可——每跑一个独立 worktree + 独立 cwd-slug（transcript / worker-memory 各自隔离）+ 独立 `results/<run-name>/`。单 GPU 本就串行。
> **复现**：基座 remote + `results/<run>/worker.patch` → 干净 worktree 上 `git apply` 即还原 kernel；`src/`(+dispatcher 的 `include/`)是易读快照。
> **手册同步**：`CLAUDE_For_KernelAgent.md` 改动后无需分发——launcher 每次 cp 最新版进 worktree。
> **基座演进**（改 task 框架）：改 `playground-base` 并 push 其 remote；下次跑的 worktree 从新 HEAD 开。
> **历史**：早期 per-method fork（`playground-naive-clean` 等)+ 每方法 submodule 已退役；现仅基座 `playground-base` 一个 submodule，per-method kernel 都以快照存 `results/`。

## 6. 实验记录

**每个方法一个独立文件夹 `results/<方法>/`**（如 `results/naive/`）：run.jsonl（stream-json 重定向）、**transcript.jsonl（Claude 自存的 session 记录，跑完必 copy 进来——token/墙钟/曲线的权威源）**、result.csv（机读曲线）、result_table.md、curve.png、result.md（人读报告，套 `results/TEMPLATE.md`）、**`src/` + `worker.patch`（kernel 产物快照，由 `finish_run.sh` 写；`src/` 易读、`worker.patch` 可 `git apply` 复现）**、`logs/`（task1 计分原始 log）、**`session_id.txt`（供 `claude --resume`）+ 可选 `worker_memory/`（worker 跑中写的 auto-memory 备份；base-slug 那份下次 launch 会被清，故归档留证 + 供 resume 前还原）**、可选 `labels.json`/`invalid.json`（曲线标注/作废版本）。`results/` 根下只放共用工具（parse_run/watch_run/TEMPLATE.md；`launch_*` / `finish_run.sh` / `_run_common.sh` 在 `scripts/`）。**代码不单独建 submodule——产物以快照形式就放在 `results/<run-name>/` 里（代码=结果，同处），由 `finish_run.sh` 写。** 含 cuBLAS 的旧跑等归档整体放 `results/<名>/`（如 `results/naive_ncu_cublas`、`results/naive_no_ncu`）；历史归档（`results/naive_no_ncu`、`naive-break` 等）同样是 `src/`(+`include/`)+`logs/` 快照，无 submodule。

**跑法**：用 `./scripts/launch_<方法>.sh`（detached，路径由脚本自身位置推导，落 `results/<方法>/run.jsonl`），或手敲 `claude -p "$SEED" --model claude-opus-4-8 --effort max --dangerously-skip-permissions --output-format stream-json --verbose > results/<方法>/run.jsonl`。另开终端 `./results/watch_run.sh results/<方法>/run.jsonl` 实时看。

**跑完先归档 transcript（每次必做）**：`cp ~/.claude/projects/<playground-base-slug>/<session-id>.jsonl results/<方法>/transcript.jsonl`。transcript 是 token/墙钟/曲线的**权威源**（见下表 `tokens` 行），slug = `playground-base` 绝对路径里非字母数字全转 `-`、`<session-id>` 取该目录下最新那个 `.jsonl`；`*.jsonl` 被 .gitignore 留本地（随结果做本地备份，不入库）。

**解析（一条命令出结果）**：`./results/parse_run.sh <方法> results/<方法>/run.jsonl results/<方法>/transcript.jsonl` → 自动写 `results/<方法>/{result.csv, result_table.md, curve.png}`。

**口径过滤（2026-06-06 加固，评估正确性的关键）**：worker 会大量用非计分口径做实验——大 shape（8192²，**因波量化尾部消失而虚高**，naive cycle2 这点冲到 191.9 vs 4096³ 的 178）、小 shape、ncu `-t1`、`-t30/50` 短测；还有口径对但**结果写错/inf** 的无效点。parser 现给每个出分点判两列:`canonical`（`4096³ && iters≥100`，从命令的 `-m/-n/-k/-t` 解析,task1.sh run 默认 4096³/100）与 `scored`（`canonical && 相对误差<0.1`）。**只有 `scored` 点进 best / running-best / 峰值 / 计分**;其余降为曲线上的浅灰叉（仅作存在性提示，不计分）。`result.csv` 带 `canonical`/`scored`/`invalid` 三列可逐点审计（`invalid` = 可选 `results/<方法>/invalid.json`〔ver→原因〕标的作废版本，如偷库/非手写：排除出计分与曲线、图上红叉留痕）。**报峰值/写 result.md 一律用 scored 的 4096³ 数,严禁拿 8192² 之类 off-口径数当成绩**（reward-hack 红线）。

**曲线画法（parser 自动，评估侧约定）**：散点 = 每次 scored 计分（含回归）；**每个版本取「最佳一点」打标**（`vN + 技法`，刷新 running-best 的前沿点标在上方、回归版标在下方、全局峰值附 TFLOPS）。技法描述来源优先级：可选 `results/<方法>/labels.json`（`{"3":"XOR swizzle",...}` ver→短技法，给薄 dispatcher/共享头那种**源文件名不带技法**的版本补；每方法跑完按该轮版本号写一份）→ 版本源文件名后缀（`v3_swizzle`→swizzle）→ RESULT_JSON marker。**两条参照线固定画在每张图上**（`parse_run.sh` 内置，无需每图配置）：`cuBLAS ref ~225`（橙虚线，`CUBLAS_REF` 默认 225——**纯可视化标尺/尺度感，基座已去 cuBLAS、数据与口径里并无此基线**）+ `A100 fp16 peak 312`（灰虚线）。⚠️ `result.md` 正文/图注**不必再解释这两条线**（全局约定，写这里一次即可）。

**每次成功 task1 计分 = 一行**（即曲线上一个点），列：

| 列             | 含义                                | 来源                                                                                                                                      |
| -------------- | ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `cycle`        | 成功 profile 累计序号，从 1 起      | parser                                                                                                                                    |
| `wall_clock`   | 开工到此次的**累计**墙钟（秒）      | transcript 顶层 timestamp                                                                                                                 |
| `tokens`       | 开工到此次的**累计** output token   | transcript 按 message.id 去重累加（**重定向 jsonl 的 per-turn usage 是流式分片，不可加**）                                                |
| `correctness`  | 此次 kernel 平均相对误差            | task1 计分行 `Average Error`                                                                                                              |
| `tflops`       | 此次 task1 100 轮口径 TFLOPS        | task1 计分行 `TFLOPS`                                                                                                                     |
| `version`      | 版本号                              | 调用命令 `task1.sh run --ver N` 的 `--ver`（parser 抽、与紧随计分行配对）                                                                 |
| `方法改进说明` | 本轮改了什么（result.md 列）        | 可选 `results/<方法>/labels.json`（ver→技法）→ 版本源文件名后缀（parser 自动，如 `v18_interleave`→interleave）+ worker 每轮简报（人工补） |
| `瓶颈分析`     | ncu 关键指标 / 卡点（result.md 列） | ncu 输出（人工补；本轮若有 ncu 数据可贴 tensor%/regs/stall）                                                                              |
| `log`          | 该版本日志路径（result.md 列）      | 手填 / fork logs/                                                                                                                         |

- 三源分工：**task1 工具输出**（客观、模型伪造不了：计分行给 tflops/error、调用命令的 `--ver` 给版本号——曲线**不靠 worker 自报**）+ **transcript**（token/墙钟，按 message.id 去重）+ **result 事件**（权威总量）。模型看不到自己 token，别让它自报。
- tflops/error **只认 task1 100 轮口径**，禁 sweep/best-of-N 热峰值（见 `CLAUDE_For_KernelAgent.md`）。
- 防作弊：跑完对工作区跑 `./scripts/check_handwritten.sh playground-base`，用了 cutlass/cublas 判该版本无效。
- LLM 有随机性，每方法多跑几轮、报均值/方差（多个 `<方法>_cycleN`）。
- ✅ `results/` 已纳入 git（**silencelamb/VibeKernel**，私有）；追踪 `*.csv`/`*.md`/`*.png`/`*.json` 摘要 **+ kernel 快照 `src/`/`worker.patch`**，大日志/profiler 报告（`*.jsonl`/`*.log`/`*.ncu-rep`）`.gitignore` 留本地。**基座 `playground-base/` 以 git submodule 纳入（`.gitmodules`，指向 `silencelamb/playground-base`）；除它外无 per-method submodule——所有方法 / 归档的 kernel 都以 `src/`(+`include/`)+`worker.patch` 快照存在各自 `results/<名>/`。**

## 7. `/goal` 用法要点（薄循环方法）

> 完整设计与坑见 `runbooks/goal.md` + `scripts/launch_goal.sh`（真源）+ `goal-method-harness` memory。要点：

- **condition = 与 naive 完全相同的 seed**（共享 `scripts/seed_gemm.txt`，verbatim）——为公平，worker 看到的指令和 naive 一字不差；/goal 的唯一增量是"worker 想停时被外部 evaluator 顶回去续"（看门狗只在自愿停点起作用）。
- **evaluator 默认 Haiku**（= "small fast model" 槽；解开 `launch_goal.sh` 里 `ANTHROPIC_DEFAULT_HAIKU_MODEL=claude-opus-4-8` 可改 opus，但 O(N²) 贵、判停一样准，默认不开）。它**只读 transcript、不跑工具**。
- **界**：condition 是 "NEVER STOP" 无终止态 → evaluator 永判未达成 → 跑到自然终止 / 手动 kill（同 naive）。**不要用 `--max-turns`**（单位 agentic-turn、达上限 error 退出、污染 run.jsonl）；要成本兜底用 `--max-budget-usd <N>`。
- headless：`claude -p "/goal <condition>"`（launcher 已封装）。
- 前置：Claude Code ≥ v2.1.139、workspace 已接受 trust、hooks 未被 `disableAllHooks` / `allowManagedHooksOnly` 关（`/goal` 本质是 Stop hook）；**启动后看 run.jsonl 确认 evaluator 真介入**，否则退化成一次性跑（≠ /goal）。

## 8. Ralph Loop 用法要点（每轮全新 session 的薄外循环）

> 完整设计与坑见 `runbooks/ralph_loop.md` + `scripts/launch_ralph_loop.sh`（真源）。要点：

- **机制**：bash 外循环，每轮重起一个**全新** `claude -p "$SEED"` session；kernel 进度靠**磁盘**跨轮持久（worktree `task-1/src` 累积 + worker auto-memory 在 run 内保留），**context window 每轮清空**。seed 逐轮重喂 = 忠实 Ralph 的 `cat PROMPT.md`。
- **与 /goal 的唯一概念差 = 怎么续**：/goal 单 session 内续（同 session 判停、上下文一路滚大）；Ralph 跨 session 重启（每轮干净上下文，可选轮间起**全新 judge session** 判停）。**seed / model / effort(max) / worktree / 计分口径 全与 naive/goal 同 → 单变量 = 「fresh-session 外循环 开/关」**。
- **界**：seed = NEVER STOP 无终止态 → judge 永判未达成，故 `RALPH_JUDGE` **默认 0**（纯无限外循环，正对 `while :; do … done`），跑到 `RALPH_MAX_ITERS`（默认 40，安全上限）/ 手动 kill；成本兜底 `RALPH_BUDGET_USD`（每轮挂 `--max-budget-usd`）。⚠️ 每轮 = 一个完整 naive 跑 → **总成本 = 各轮之和**，可能远高于 naive/goal，务必控 iters/budget。另有 hot-loop 防护（连续 3 轮 <30s 自动停）。
- ⭐ **附带卖点**：naive/goal 多 cycle 毁于 ECONNRESET 半途崩；Ralph 外循环天然在崩后立刻重起新 session 续命（/goal 的 Stop hook 救不回已崩进程）。
- **归档口径坑（必知）**：每轮独立 session = 独立 transcript；脚本存 `results/<run>/iters/iter_NN.transcript.jsonl`，`finish_run` 默认只归档**最后一轮** → 报全量 token/曲线要先 `cat results/<run>/iters/*.transcript.jsonl > results/<run>/transcript.jsonl` 再 `parse_run`（message.id 跨 session 天然唯一、累计正确；同 `goal_cycle4` 两段手工合并的口径坑）。
- headless：`bash scripts/launch_ralph_loop.sh [run-name]`（launcher 已封装；旋钮 `RALPH_MAX_ITERS` / `RALPH_JUDGE` / `RALPH_BUDGET_USD`）。**跑完查 `ralph.log` 是否真多轮 fresh-session**（只 1 轮 = 退化成 naive，须在 result.md 记）。

## 9. 环境

A100-80GB → 用根 `env.sh` 绑定 GPU(`CUDA_VISIBLE_DEVICES`)。`env.sh` 不入库(机器相关),提供 `env.sh.example` 模板;launcher(`_run_common.sh`)缺 `env.sh` 时回退到 example。CUDA 13.0 / vcpkg `/opt/vcpkg` / CMake ≥ 3.30 + Ninja / C++20 / CUDA20。

- 目标口径：**向 A100 fp16 峰值 312 TFLOPS 逼近，无库基线对标**（已去 cuBLAS）。`v0` cBLAS 仅作正确性 ground truth（f16 误差 ≲ 0.02 即正常）。历史参照：含 cuBLAS 那次 naive 手写顶 v18≈195.5 TFLOPS（`results/naive_ncu_cublas/`），可作新跑的非正式对照，但**不再写进任务/不作目标**。
