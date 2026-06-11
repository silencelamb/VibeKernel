---
name: naive-csv-field-parsing
description: results/<方法>.csv 的 schema(每次成功 profile 一行、累计口径)与各字段怎么从 jsonl+transcript 解析;新工具 results/parse_run.sh
metadata: 
  node_type: memory
  type: project
  originSessionId: 7f4ad05a-b435-4e15-b2a5-ac7cab195e28
---

`results/<方法>.csv` schema(见 META.md §6)：**每次成功 task1 计分 = 一行**(不是每次启动一行)，`cycle`=checkpoint 累计序号，`wall_clock`/`tokens`=**累计**值，`correctness`=该版本误差，`tflops`=该版本实测。即这张表本身就是"性能 vs token"曲线数据。

**字段来源(2026-06-05 改:task1 工具输出为骨架,RESULT_JSON 退为注解 —— 旧"marker 优先"会漏点):**
- `tflops`/`error`/`version` ← **task1 工具输出**(客观、模型伪造不了、每跑必有):计分行 `TFLOPS: X; Average Error: Y` 给性能,**调用命令 `task1.sh ... run --ver N` 的 `--ver` 给版本号**(jq 抽 `tool_use.input.command`,awk 里用最近一次 `--ver` 配对紧随的计分行)。**不再依赖 worker 自报**。
- `tokens`(累计) ← **transcript** 按 message.id 去重累加 output_tokens(见 [[naive-token-attribution-limit]])。
- `wall_clock`(累计) ← transcript 顶层 `timestamp`(去小数秒后 `fromdateiso8601`，减起点)。
- 总 wall/token/turns/cost ← stream-json 重定向里最终 `result` 事件(权威)。
- `方法改进说明` ← 版本源文件名后缀(`matmul_f16_v18_interleave.cu`→"interleave")。
- `瓶颈分析`(新列,TEMPLATE/表都加了) ← 从 **ncu 输出**贴(tensor%/regs/bank/blk-per-SM/top_stall),**手填**;parser 会"有则读"历史 RESULT_JSON.ncu 当注解(naive cycle1 的 v16/v18 就是这么自动填上的),但今后 worker 不打 marker → 一律手填。

**RESULT_JSON 已于 2026-06-05 全面弃用**(手册母版/launch seed/META/TEMPLATE 全删,见 [[vibekernel-result-harness]]):naive 即便种子明确要求"每版打一行",136 turn 也只打了 3 次 —— 长程自报不可靠;且强制上报本身是脚手架、污染 naive"零脚手架"定义。故 [[orchestration-overview]] 下**曲线只认客观 task1 工具输出**,worker 不再被要求自报。parser 保留"有则读 marker 作 ncu 注解"的向后兼容(无害)。

**输出**:`result.csv`=全量逐次计分(密,带 `version` 列,含 ver1=cuBLAS 基线);`result_table.md`=每版最佳(readable);`curve.png`=worker(ver≥2)逐次散点 + running-best 包络 + cuBLAS/312 参考线。CSV/表/图全在 parse_run.sh 末尾的 python heredoc 里做 join。
**每方法独立文件夹** `results/<方法>/`;`results/` 根只放共用工具。`parse_run.sh <方法> results/<方法>/run.jsonl <transcript>`。实时看 `results/watch_run.sh`(同时高亮绿 RESULT_JSON 与黄 task1 计分行——别把后者误认成前者)。
注意:tflops 只认 task1 100 轮口径,**禁 sweep/best-of-N 热峰**(虚高 10~20%;naive_no_ncu 那次曲线被 sweep 的 267 污染过,真实手写顶 168)。
jq 坑:`.text?//""` 缺空格语法错;`capture(...)?.field` 非法,写 `(capture(...) | .field)?`;scan 多捕获组不稳用 `capture`。matplotlib 无中文字体 → plot 文案用英文免方块。
