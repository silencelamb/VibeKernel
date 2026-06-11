# VibeKernel

> 用不同的 **coding agent 方法 / 范式**自动编写高性能 CUDA GEMM kernel,在**同一个任务、同一套口径**下对比它们的最终性能、迭代曲线与各自的局限。

**[English README →](README.md)**

---

## 1. 这是什么 / 为什么做

同样是「让 Claude Code 写一个 GEMM kernel」,加不加脚手架、用什么范式,差别有多大?业界出现了很多写 kernel 的 agent 玩法(纯 prompt、KDA/AKO 这类 skill、`/goal` 循环、autoresearch 自治循环……),但它们各说各话、口径不一,**很难公平对比**。

VibeKernel 就是为公平对比而建的「裁判台」:

- **任务固定**:同一个 A100 fp16 GEMM 任务、同一份正确性校验、同一套性能口径。
- **变量唯一**:模型(`claude-opus-4-8 --effort max`)、prompt(seed)、评分方式全部钉死,**唯一变量是「方法」**。(唯一刻意例外:受控**模型臂**——同 harness 只把模型换成 `claude-fable-5`,即 `naive_fable` / `ralph_loop_fable`,见 §4。)
- **结果可审计**:每次跑都归档逐版本的 TFLOPS 曲线、token/墙钟成本、kernel 源码快照与可复现 patch。
- **反 reward-hack**:只认固定口径的成绩,防止用「热峰值 / 偷库」刷分(见 §5)。

> 方法清单的原始出处(articles / papers / repos)见 [`methods.md`](methods.md);完整编排设计见 [`META.md`](META.md)。

## 2. 任务:A100 上的高性能 GEMM

- **目标**:在 NVIDIA A100 上手写 fp16(Tensor Core)/ fp32(CUDA Core)GEMM,**不断逼近硬件峰值**(A100 fp16 理论峰值 **312 TFLOPS**)。
- **无库基线**:基座已**彻底移除 cuBLAS / CUTLASS**——不与任何库比较,硬件峰值是唯一标尺。`v0`(cBLAS,CPU)仅作正确性 ground truth,**不算成果**。
- **必须从零手写**:`--ver ≥ 1` 的 kernel 的 tiling / Tensor Core(`mma.sync`/`wmma`)/ `cp.async` / swizzle / 寄存器流水等全部自己实现;**禁止 include 或调用任何现成 GEMM 库**(CUTLASS/CuTe/cuBLAS/cuDNN)。跑完用 [`scripts/check_handwritten.sh`](scripts/check_handwritten.sh) 扫源码,检测到偷库则该版本判无效。
- **评分口径**:只认 task1 自带计分方式——固定 **4096³、10 warmup + 100 轮取平均**的 sustained TFLOPS。**禁止自建 sweep / best-of-N / 热峰值**(短爆发满 boost 时钟会虚高 10~20%,不是真实性能,判无效)。
- worker 的完整任务手册见 [`CLAUDE_For_KernelAgent.md`](CLAUDE_For_KernelAgent.md)(方法无关,启动时拷进每个 worker 工作区当 `CLAUDE.md`)。

## 3. 方法阶梯

按「脚手架厚度」从薄到厚排。⚠️ 目前 **naive、`/goal`、Ralph Loop、Dynamic Workflow 已实现并跑出结果**;其余是外部项目,我们参考其思想自行实现,**尚未完成**(下表为路线图)。方法轴之外另有一条受控**模型臂**:`naive_fable` / `ralph_loop_fable` 用同一套 naive / Ralph harness、**只把模型换成 Claude Fable 5**。

| 方法                                                                                                     | 增量机制                                                                                                                                                                                                                                                                                                                                                                   | 状态                             |
| -------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| **naive**                                                                                                | 无脚手架;纯 prompt(NEVER STOP)自驱、零人工干预——测它自己能持续多久 / 冲到多少                                                                                                                                                                                                                                                                                              | ✅ 已跑                           |
| **[`/goal`](https://code.claude.com/docs/en/goal)**                                                      | naive 同款 seed + 一个外部 LLM-judge(看门狗,默认 Haiku)在**单个长 session 内**、worker 想停时读 transcript 把它顶回去续,直到 `condition` 达成——「让 agent 一直跑」的循环,但全程不离开这个 session                                                                                                                                                                          | ✅ 已跑                           |
| **[Ralph Loop](https://ghuntley.com/ralph/)**                                                            | 与 `/goal` 同样「让 agent 一直跑」,但**每轮都重起一个全新 session**(`while :; do cat PROMPT.md \| claude -p; done`)——新 judge 重读磁盘上的现状、决定是否再循环。kernel 进度靠磁盘持久,而 **context window 每轮清空**(vs `/goal` 单个越滚越大的 transcript)→ 每轮上下文更干净、无漂移,还天然抗崩(session 崩了下一轮直接重起)。出处见 [loop 一文](https://ghuntley.com/loop) | ✅ 已跑(Opus c1 + **Fable 5 臂**) |
| **[Dynamic Workflow](https://claude.com/blog/introducing-dynamic-workflows-in-claude-code) / ultracode** | naive 同款 seed + `ultracode` 关键字,让 worker 对每个实质子任务自动编排 fan-out 子 agent [workflow](https://code.claude.com/docs/en/workflows)(这里:并行编译 + 串行 benchmark 的 config tournament),同样 `--effort max`——测「自组织多 agent 搜索」能带来什么                                                                                                               | ✅ 已跑                           |
| KDA (+skill)                                                                                             | MIT-HanLab [kernel-design-agents](https://github.com/mit-han-lab/kernel-design-agents) skill                                                                                                                                                                                                                                                                               | ⏳ 计划                           |
| AKO (+skill)                                                                                             | [AKO4ALL / AKO4X](https://github.com/TongmingLAIC/AKO4ALL)                                                                                                                                                                                                                                                                                                                 | ⏳ 计划                           |
| AI-Infra-Auto-Driven-SKILLS                                                                              | [BBuf 的推理框架工作流 skill](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS)                                                                                                                                                                                                                                                                                         | ⏳ 计划                           |
| autoresearch / autokernel                                                                                | 参考 [autoresearch](https://github.com/karpathy/autoresearch) / [autokernel](https://github.com/rightnow-ai/autokernel) 的「自治研究式循环」                                                                                                                                                                                                                               | ⏳ 计划                           |
| Heuristic Learning                                                                                       | [learning-beyond-gradients](https://trinkle23897.github.io/learning-beyond-gradients/) 思想                                                                                                                                                                                                                                                                                | ⏳ 计划                           |
| K-Search                                                                                                 | UC Berkeley [K-Search](https://github.com/caoshiyi/K-Search)                                                                                                                                                                                                                                                                                                               | ⏳ 计划                           |

## 4. 当前结果

> 所有数字均为 **4096³ / 100 轮 sustained**。展示两个精度档:**fp32-acc**(误差 ~3e-5,跨方法主档)与 **fp16-acc**(误差 ~0.018,部分 worker 探索的降精度取舍——只与其它 fp16-acc 数比)。naive 3 个干净 cycle、`/goal` 4 个、Dynamic Workflow 1 次(+1 guided)、Ralph Loop 1 次,另有 **Fable 5 模型臂**(naive ×2、Ralph ×1)。**峰值由路径变异主导**——naive 自己的 cycle 就在 154–203 间——单方法差值看趋势别当定论。⚠️ **时钟口径**:Ralph / Fable 各行跑时 **GPU 被驱动故障锁在 1155 MHz**,真天花板 **255.6**,绝对 TFLOPS 较 Opus 老轮(~1410 boost / 312 名义;当时未记录时钟)带 ~−18% 偏置——**跨臂只比「占各自真天花板 %」、别比绝对值**。逐跑报告见各 `results/<方法>/result.md`;跨方法细节见 [`results/SUMMARY.md`](results/SUMMARY.md)。

| 方法                     | 最佳峰值 — fp32-acc / fp16-acc                            | 跑数(峰值区间)                           | 结束方式             | 报告                                                                             |
| ------------------------ | --------------------------------------------------------- | ---------------------------------------- | -------------------- | -------------------------------------------------------------------------------- |
| **naive**                | **203 / 208**(cycle6,崩前)                                | 3 cycles,154–203;干净自停 196.9(c4)      | 自停 / 崩            | [c6](results/naive_cycle6/result.md) · [c4](results/naive_cycle4/result.md)      |
| **`/goal`**              | **206.8** / —                                             | 4 cycles,188.7–206.8(c4 为 f16-acc 档)   | 看门狗 + 自停        | [c1](results/goal/result.md)                                                     |
| **Dynamic Workflow**     | **192.9 / 194.8**                                         | 1 次(+1 guided,172.5)                    | 自停                 | [result](results/dynamic_workflow/result.md)                                     |
| **Ralph Loop** (Opus)    | 196.1 / 196.6                                             | 1 次,2 个实质 iter(iter2 崩)             | 崩 → hot-loop 防护停 | [RESUME](results/ralph_loop/RESUME.md)                                           |
| **naive × Fable 5**      | **219.5**(c2)/ 211.5(c1)                                  | 2 cycles,211.5–219.5                     | 干净自停 ×2          | [c2](results/naive_fable_cycle2/result.md) · [c1](results/naive_fable/result.md) |
| **Ralph Loop × Fable 5** | **227.3** 严格档(放宽 fp32 **229.1**)/ —   **反超cuBLAS** | 1 次,3 iter(iter1 = naive_fable c2 种子) | 每 iter 干净自停     | [result](results/ralph_loop_fable/result.md)                                     |

<p align="center">
  <img src="results/comparison_best.png" width="92%" alt="各范式最佳跑对比:running-best TFLOPS vs 累计 output token">
</p>

<sub>头牌图——六条粗线(naive c6、/goal c1、Dynamic Workflow、naive×Fable 两个 cycle、Ralph×Fable;红★ = 其冠军点,黑线从深青 naive×Fable c2 的自停点接着续顶,因为那一轮就是它的 iter 1),外加同族同色淡细线衬底(各族其余 cycle)。`@1155` 各轮跑在锁频故障态下(真天花板 = 图中 255.6 点划线),跨臂别比绝对值。[完整跨方法图](results/comparison.png)(全部 run 带标注,含 naive_strong、Ralph-Opus、guided 臂)是完整画面。</sub>

**几个诚实的结论(避免过度解读):**

- **naive 自己就很能打**:纯 prompt、零脚手架、全程 ncu 驱动,爬到 ~197–203 后**自行判定已到实际天花板并收尾**。3 个 cycle 峰值在 **154–203** 间摆动——这 49 分的跨度就是其它方法必须超越的路径变异标尺。
- **`/goal` 看门狗是地板抬升器、不是天花板抬升器**(现 4 cycle 坐实):worker 靠自身 NEVER-STOP 冲到大半,看门狗只在它想停处接入。自停越低、看门狗逼出越多——边际增量 **+1.3 / +9.7 / +22.7 / +9.8** 与自停点 **205.5 / 195 / 178.9 / 178.9** 反相关;cycle3 被顶回去 13 次逼出了真正的 barrier-free `mbarrier` 重写(+13%)。但头部峰值仍 ~205,代价是高得多的 token/成本。
- **Dynamic Workflow(ultracode)是结构化搜索器、不是天花板抬升器**:worker 自发 fan out **7** 个「并行编译、串行 benchmark」的 config tournament,把瓶颈干净定位到 mma-latency-bound,~2h 到 192.9——但每个 workflow 都停在「调配置」层,于是落在和 naive 同一堵 ~193 墙。强制规定 leave-one-out 严格搜索(*guided* 臂)被忠实执行、峰值反而更低(172.5);它的回报是一个独有副产物——一张证明 **6 个优化里 4 个单独看毫无价值、合起来却值 +137 TFLOPS** 的表(cooperative-blindspot 证据)。
- **Fable 5 模型臂是最强的 naive 变体**:同一套 harness、只换模型——两轮都干净自停在占真天花板 **82.7% / 85.9%**(家族纪录),cycle2 纯 prompt 把 fp32 推到 **219.5 ≈ 同卡同钟实测 cuBLAS fp32(218.7)**;它还**主动拒绝 f16-acc**(+0.7 TF 换 500× 误差,判不值),并抓修了一个真实的 cp.async commit 竞态。行为签名:可见叙述文字少 ~13× 但 tool_use 更多(不是少干活)、自发 git commit 打里程碑;成本签名:多短 turn → cache-read 重($53–60,naive Opus 是 $13–21)。
- **Ralph Loop(fresh-session 外循环)是 plateau 破壁器——目前在 Fable 臂坐实**:Opus 跑(2 个实质 iter)只在自己 ~196 的 plateau 上原地重探;而 **Ralph × Fable 5**——以 naive_fable cycle2 为种子,后者已自封 219.5 是"源码层天花板"("尾波无经济解")——**第一轮 fresh session 就直接打那条被放弃的尾波瓶颈**(in-kernel last-wave K-split),冲到 **229.1 / 占真天花板 89.7%,同卡同钟反超实测 cuBLAS ~4%**(严格精度档 227.3 vs 218.7)。机制:全新 session 从磁盘继承 kernel + 设计笔记,但**不继承上一轮"这条路已放弃"的结论包袱**。量化结论:在 plateau 上 fresh 重启 > 单 session 续跑(+9.6 TFLOPS / 2 轮 $87)——注意两条 Ralph 臂模型不同,该增益目前只在 Fable 臂成立。
- **天花板分析**:所有手写 fp32 方法都聚在 **~190–207 / 312**,卡在 Tensor-Core 吞吐气泡;本机实测同精度库天花板为 cuBLAS **218.7** / CUTLASS **217.9**,故手写差距 **~5–8%——是流水工艺/结构成熟度差距,非范式或手写 SASS 差距**(CUTLASS 无一行手写汇编也到 218)。**更新——Fable 臂已把这条差距闭合并反超**:naive×Fable c2 贴上 cuBLAS fp32 线,Ralph×Fable 同卡同钟越过它 ~4%;距其 255.6 真天花板剩下的 ~10% 在 SASS 层(worker 手搓 CuAssembler 探针后判 no-go)。完整库-vs-手写拆解见 [`results/SUMMARY.md`](results/SUMMARY.md)。

## 5. 实验设计:怎么保证公平

公平对比的关键不在跑得多,而在**控制变量 + 防作弊**。VibeKernel 的几条硬协议:

1. **worker 不知道自己在被对比**。揭示「我们在比较多种方法」的编排文档(`META.md`)**绝不进 worker 上下文**;worker 只看到任务手册(`CLAUDE.md`)和它的 seed。
2. **结构性隔离(git worktree 流)**。每次跑在干净基座 `playground-base` 上开一个独立 git worktree → 独立工作目录 → transcript / agent memory 天然隔离,方法/cycle 之间零串味。基座自己永不被改,只当源。
3. **钉死模型与 effort**:所有方法统一 `claude-opus-4-8 --effort max`,保证跨方法可比。**模型臂**(`naive_fable`、`ralph_loop_fable`)是受控例外——harness 完全相同、只把模型换成 `claude-fable-5`——且只与本方法的 Opus 跑对照,绝不跨方法比。
4. **共享同一份 seed prompt**([`scripts/seed_gemm.txt`](scripts/seed_gemm.txt)),naive 与 `/goal` 一字不差,确保唯一变量是方法本身。
5. **反 reward-hack 的评分纪律**:① 成绩只认 `4096³ && iters≥100 && 误差<0.1` 的点,off-口径(如 8192² 因波量化尾部虚高)一律降为曲线上的灰叉、不计分;② 偷库版本由 `check_handwritten.sh` 判无效;③ token / 墙钟从 Claude 自存的 transcript 按 `message.id` 去重统计(模型看不到自己的 token,无法自报作弊)。
6. **每方法应多跑几轮报均值/方差**(naive 现 3 个干净 cycle、`/goal` 4 个;新方法——Ralph Loop、Fable 臂——仍需更多跑,见 §4 关于路径变异的说明)。

## 6. 仓库结构

```
VibeKernel/
├── README.md / README.zh.md     # 英文(默认)/ 中文(本文件)
├── META.md                      # 完整编排设计文档(人读;不进 worker 上下文)
├── methods.md                   # 方法清单的原始出处(知乎/论文/repo 链接)
├── CLAUDE_For_KernelAgent.md    # worker 任务手册母版(方法无关;启动时拷成工作区 CLAUDE.md)
├── runbooks/                    # 各方法的 runbook(任务 prompt + harness 指令)
│   ├── naive.md  goal.md  ralph_loop.md
│   └── dynamic_workflow.md  naive_fable.md
├── scripts/                     # 跑实验的 harness
│   ├── seed_gemm.txt            #   共享 seed prompt(跨方法一致)
│   ├── launch_naive.sh          #   起 naive worker(开 worktree、绑 GPU、detached)
│   ├── launch_goal.sh           #   起 /goal worker(含看门狗 evaluator)
│   ├── launch_ralph_loop.sh     #   Ralph fresh-session 外循环(launch_*_fable.sh = Fable 模型臂)
│   ├── finish_run.sh            #   一键收尾:归档 transcript + 快照 src/patch + 出曲线 + 删 worktree
│   ├── check_handwritten.sh     #   防偷库扫描
│   ├── _run_common.sh           #   launcher 公共逻辑
│   └── ...                      #   link_memory.sh / ncu-doctor.sh 等
├── results/                     # 每次跑的归档(每个方法/cycle 一个子目录)
│   ├── parse_run.sh             #   解析 → result.csv / result_table.md / curve.png
│   ├── watch_run.sh             #   实时看 run.jsonl
│   ├── TEMPLATE.md              #   人读报告模板
│   ├── naive/  goal/  ...       #   各跑:result.md/.csv、curve.png、src/ 快照、worker.patch、logs/
│   └── ...
├── playground-base/             # 干净的 GEMM 任务基座(git submodule → silencelamb/playground-base)
└── .claude/
    ├── memory/                  # 编排意识 + 工具链速查(主 session 自动加载,worker 读不到)
    └── settings.json
```

> 不入库的 scratch:`worktrees/`(每跑的临时 worktree)、运行日志/profiler 报告(`*.log`/`*.ncu-rep` 等)、`env.sh`(机器相关的 GPU pin)、各跑的原始 `run.jsonl`/`transcript.jsonl`(留本地;由它们导出的摘要/CSV/曲线已入库)——见 [`.gitignore`](.gitignore)。

## 7. 怎么跑一个方法

**前置**:GPU docker 容器(A100-80GB)、CUDA 13.0、vcpkg、CMake ≥ 3.30 + Ninja、C++20/CUDA20。

```bash
# 带 playground-base submodule(GEMM 任务基座)一起 clone
git clone --recurse-submodules https://github.com/silencelamb/VibeKernel.git
cd VibeKernel
# (已经 clone 但没带 --recurse-submodules?执行:)
git submodule update --init

# 把 harness 指向你的 GPU
cp env.sh.example env.sh        # 然后改 env.sh 里的 CUDA_VISIBLE_DEVICES

# 1) 起 worker —— run-name 默认=方法名;多 cycle 传 naive_cycle2 / goal_cycle2 等
bash scripts/launch_naive.sh [run-name]        # 或 launch_goal.sh / launch_ralph_loop.sh / launch_naive_fable.sh(Fable 模型臂)
#    实时看:  results/watch_run.sh results/<run-name>/run.jsonl
#    停:      kill -TERM -- -$(cat results/<run-name>/run.pid)

# 2) 跑完(自然终止 / 手动 kill 后)一键收尾归档
bash scripts/finish_run.sh <run-name>
#    → 归档 transcript + check_handwritten + 快照 src/include/logs/worker.patch
#      + parse_run 出曲线 + 删 worktree(基座自始至终没动)

# 3) 套 results/TEMPLATE.md 写 results/<run-name>/result.md(人读报告)
```

**复现某次跑的 kernel**(基座已随 submodule 在本地):

```bash
git -C playground-base worktree add ../repro HEAD
git -C repro apply ../results/<run-name>/worker.patch
cd repro && ./task1.sh run --float f16 --ver <N>     # 跑出 TFLOPS / Average Error
```

`results/<run-name>/src/` 也存了易读的 kernel 源码快照。

## 8. 注意 / Caveats

- **路径变异主导**:naive 3 个干净 cycle、`/goal` 4 个、Dynamic Workflow 1 次(+1 guided)、Ralph Loop / Fable 臂各 1–2 次,但单方法峰值仍摆动很大(naive 154–203),**不要把单次数字当定论**;结论以「趋势 + 机制分析」为主,更紧的定量对比待补更多跑。
- **GPU 中途出了时钟故障、数据分成两个时钟档**:NVRM 驱动 assertion 把时钟卡死在 1155 MHz(容器内修不了)。Ralph / Fable 各轮都在故障态下 → 绝对 TFLOPS 带 ~−18% 偏置,跨臂只能比「占 255.6 的 %」;Opus 老轮当时没记录时钟。今后每轮强制 log SM 时钟。
- **单 GPU 串行**:只用一张卡,各方法/cycle 串行跑(并行跑会共享 worker auto-memory 而串味——已经吃过一次亏、作废了一轮)。
- **成本不低**:`/goal` 单跑 ~$87.5、817k output token;Ralph × Fable 在 $60.5 的种子轮之上又花 $86.7;跑前留意额度。
- **公开说明**:`.claude/memory/` 与各 `results/*/result.md` 含较坦诚的内部工作笔记(含自我更正、口径红线讨论),随仓库一并公开——这是有意保留的「实事求是」记录,不是疏漏。

## 9. 致谢与参考

本项目对比的方法思想来自社区诸多工作,出处汇总见 [`methods.md`](methods.md),含 MIT-HanLab KDA、AKO、BBuf 的 AI-Infra-Auto-Driven-SKILLS、Karpathy 的 autoresearch、rightnow-ai 的 autokernel、UC Berkeley K-Search、Ralph loop、OpenAI 研究员的 Heuristic Learning 等。GEMM 任务本身基于内部 playground task-1。

---

<sub>实验编排与文档由人 + Claude Code 协作完成。峰值由路径变异主导,解读以机制分析为准。</sub>
