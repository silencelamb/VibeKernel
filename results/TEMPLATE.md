# 方法结果 — `<方法名，形如 harness + 约束，如 naive、naive+ncu、/goal+ncu>`

> 由 `results/parse_run.sh <方法> <jsonl> <transcript>` 自动生成；可手动补充「关键发现」。

## 一句话结论

<最终手写最佳 TFLOPS / 占 cuBLAS 比例 / 花了多久多少 token>

## 环境与口径

| 项 | 值 |
| --- | --- |
| GPU / 形状 | A100-80GB / 4096³ fp16 |
| 计分口径 | task1 二进制 100 轮均值（sustained） |
| 参照 | 无库基线（基座已去 cuBLAS）；`v0` cBLAS 仅 CPU 正确性 ground truth；目标 = A100 fp16 峰值 312 |
| 模型 | claude-opus-4-8, `--effort max` |
| profiler | ncu <可用/不可用> |
| 手写校验 | `scripts/check_handwritten.sh` <通过/违规> |
| 总计 | wall_clock <s>，output_tokens <N>，turns <N> |

## 迭代曲线（每个版本最佳一行；wall_clock / tokens 为累计）

> 性能列（tflops/correctness/version）全自动来自 task1 工具输出；`方法改进说明` 取可选 `results/<方法>/labels.json`（ver→技法）→ 版本文件名后缀（如 `v18_interleave`→interleave）；`瓶颈分析` 贴 ncu 关键指标（tensor%/regs/stall/卡点），手填或本轮有 ncu 数据时补。

| cycle | wall_clock(s) | tokens(累计 out) | correctness(err) | tflops | 方法改进说明 | 瓶颈分析 | log |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | | | | | | | |
| 2 | | | | | | | |

![curve](curve.png)

> 曲线由 `parse_run.sh` 生成：每版「最佳一点」打标（`vN + 技法`，前沿在上 / 回归在下，峰值附 TFLOPS）；两条固定参照线为全局可视化标尺，含义见 META §6（本文不展开）。

## 关键发现

- 

## 复现 / 数据来源

- kernel 源快照：`results/<方法>/src/`（dispatcher 方法另含 `include/`）+ `worker.patch`（`git apply` 复现；无 submodule）
- transcript（token/曲线权威来源）：`results/<方法>/transcript.jsonl`
- stream-json 重定向（仅取最终 result 事件的总 wall/token；per-turn usage 不可用）：`results/<方法>/run.jsonl`
- （可选）曲线/表标注源：`results/<方法>/labels.json`（ver→技法）
