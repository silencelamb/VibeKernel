---
name: naive-token-attribution-limit
description: 单 session 也能拿到逐 turn / per-version token——从 Claude 自动存的 transcript 按 message.id 去重读 output_tokens（stream-json 重定向里的不可用）
metadata: 
  node_type: memory
  type: project
  originSessionId: 7f4ad05a-b435-4e15-b2a5-ac7cab195e28
---

VibeKernel 方法对比：naive = 单次 `claude -p "$SEED" --output-format stream-json` 不间断跑。**关键修正（之前以为 per-version token 测不到，错了）：单跑一次就能拿到逐 turn / per-version token，不用重跑、不用拆成离散调用。**

**两个 jsonl 的 usage 字段不一样，这是全部关键：**
- `results/naive_cycleN.jsonl`（你 `--output-format stream-json` 的**重定向输出**）：逐事件 usage 是流式分片，**不可加**（实测求和 9122、按 id 取 max 也才 9122，对不上真实 328328）。
- Claude **自动存的 transcript** `/root/.claude/projects/<cwd-slug>/<session-id>.jsonl`（naive worker 在 playground-naive/ 起 → slug 末尾是 `-playground-naive`）：assistant 消息带**真实 usage**。transcript 里同一 message.id 会重复出现多行，**按 message.id 去重取 output_tokens 求和 = 333206 ≈ 权威 328328**（差 1.5%）。162 个 distinct id = num_turns。

**所以能做：** 沿 transcript 顺序累加逐 turn output_tokens → 任意时刻的累计 token；配合 **task1 计分行**(给 tflops/error)+ 调用命令 `--ver`(给版本号) → 画 "性能 vs 累计 token" 曲线。脚本固化在 `results/parse_run.sh`（旧的 `results/curve.sh` 已删，它取 free-text 最高数字会被 sweep 虚高污染）。cycle1 用 task1 计分口径的诚实曲线见 `naive_no_ncu/result.md`：手写顶 168@253k，v7 cutlass(无效)218@291k。

**限制已大半解决(2026-06-05)：** X 轴(token)硬数据可信；Y 轴(TFLOPS)现取自 **task1 工具输出**(客观计分行,按 `--ver` 配版本)，**不再是 free-text 抽数**,故旧的"混进 cuBLAS 基线/sweep 热峰"问题已消除(基线 ver1 单独标、sweep 本就禁)。详见 [[naive-csv-field-parsing]]。`result.usage.iterations` 不是逐 turn 明细（长度 1），别指望它。
