---
name: vibekernel-result-harness
description: VibeKernel 跑 worker 的最佳实践工具链(parse_run/watch_run/check_handwritten/TEMPLATE.md)+ CLAUDE.md 三条全局约定 + naive_no_ncu 归档位置
metadata: 
  node_type: memory
  type: project
  originSessionId: 7f4ad05a-b435-4e15-b2a5-ac7cab195e28
---

**通用最佳实践(2026-06-05 落地，所有方法共用，故放 worker 手册 = 受控常量不算混淆变量)：**
注:根目录 CLAUDE.md 已改名 `CLAUDE_For_KernelAgent.md`(worker 手册母版,cp 成 `<fork>/CLAUDE.md`);**根目录不再有 CLAUDE.md**,编排上下文改放 git 维护的 `.claude/memory/`(见 [[orchestration-overview]])——防 worker 继承根 CLAUDE.md 串味。
- 跑法：`./scripts/launch_<方法>.sh`(detached,路径由脚本自身位置 BASH_SOURCE/.. 推导,任意 cwd/机器都能跑) 或 `claude -p "$SEED" --model claude-opus-4-8 --effort max --dangerously-skip-permissions --output-format stream-json --verbose > results/<方法>/run.jsonl`。**每个方法一个独立文件夹 `results/<方法>/`**(run.jsonl/result.csv/result_table.md/curve.png/result.md);fork 仍在根部 `playground-<方法>/`(submodule,代码≠结果)。
- 三源分工：**task1 工具输出**=客观成绩(计分行给 tflops/error、调用命令的 `--ver` 给版本号;模型伪造不了,**曲线不靠 worker 自报**)；**transcript**=只有 harness 知道的(token 按 message.id 去重 / wall_clock 按 timestamp)；**result 事件**=权威总量。改进说明取版本文件名后缀、瓶颈分析贴 ncu。模型看不到自己 token,别让它自报 token。
- 解析/展示/建表/画图全自动：`results/parse_run.sh`、`results/watch_run.sh`、模板 `results/TEMPLATE.md`(markdown,列:cycle,wall_clock,tokens,correctness,tflops,version,方法改进说明,**瓶颈分析**,log)。见 [[naive-csv-field-parsing]]。
- **parser 口径过滤(2026-06-06,防 reward-hack 关键)**:worker 会大量用非计分口径做实验(大 shape 8192²——**波尾消失会虚高**、小 shape、ncu `-t1`、`-t30/50`)、还有口径对但**写错/inf** 的无效点。parse_run 现给每个出分点判 `canonical`(`4096³ && iters≥100`,从命令 -m/-n/-k/-t 解析;task1.sh run 默认 4096³/100)与 `scored`(`canonical && 相对误差<0.1`);**只有 scored 进 best/running-best/峰值/计分**,其余降为曲线浅灰叉。CSV 带 `canonical`/`scored` 两列可审计。**报峰值/写 result.md 一律用 scored 的 4096³ 数,严禁拿 8192² 之类当成绩**。详见 META §6。

**`CLAUDE_For_KernelAgent.md`(worker 手册,cp 成 fork 的 CLAUDE.md)已写三条硬约束：**
0. **性能口径只认 task1 100 轮平均**,禁自建计时/sweep/best-of-N 热峰值(会虚高 10~20%)。
1. **环境已验证免自检**:构建/运行/ncu 都实测可跑,worker 别花 turn 自检(修正了旧的"SYS_ADMIN 已具备"错误描述——那是错的,见 [[ncu-permission-gate]])。换容器后人先跑一次 `scripts/ncu-doctor.sh` 证实再启 worker。
2. **防 reward hacking**:ver≥2 必须纯手写,禁 cutlass/cute/cublas/cudnn;必须自己 ncu profile + 查官方 doc。硬门 `scripts/check_handwritten.sh <fork>`(扫 include,违规退出 1,已验证能揪出 v7_cutlass)。

（原第 3 条「RESULT_JSON 上报」已于 2026-06-05 删除：手册/seed/META/TEMPLATE 全部去掉,worker 不再被要求自报结构化标记——性能/版本改由编排侧从 task1 工具输出解析。理由:self-report 长程不可靠(naive 单跑 136 turn 只打了 3 次),且强制上报本身是脚手架、污染 naive"零脚手架"定义。parser 仍能"有则读"作 ncu 注解,无害。见 [[naive-csv-field-parsing]]。）

**归档**:第一次 naive 跑(无 ncu+用了 cutlass,污染)已整理到 `naive_no_ncu/`并 commit&push(a9f426e)。该 fork **已退出 submodule、当普通文件入库**(用户删了它的 .git;我注销 gitlink+删 .gitmodules;`build/` 82M 由 `.gitignore` 排除,`logs/` 56K 与 kernel 源码保留)。`result.md`/`curve_scored.png`/`transcript.jsonl` 齐全。真实成绩:手写顶 v5=168,v7 cutlass=218(无效,reward hack),cuBLAS=220。注:**活跃方法的 fork(根部 playground-*)仍用 submodule**,只有这个归档退出了。
