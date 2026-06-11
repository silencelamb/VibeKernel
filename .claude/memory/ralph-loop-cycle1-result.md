---
name: ralph-loop-cycle1-result
description: "Ralph Loop 首跑结果(2 实质 iter,峰值~196,没破 plateau)+ 已归档 keep-worktree 供 resume + 口径坑"
metadata: 
  node_type: memory
  type: project
  originSessionId: 7134553b-8933-449d-a5c3-935f51cf35af
---

Ralph Loop **首跑(`results/ralph_loop/`)= cycle1**,2026-06-09 跑。见 [[ralph-loop-method-harness]]。

**实际跑了 2 个实质 iteration**(外循环计 5 次,但 iter3-5 撞 API 故障每次 <10s 秒退,被 **hot-loop 防护(连续3轮<30s)自动停在 iter5**——防护按预期生效):
- iter1 `58f37889…`:全新 session 从零爬到 v7~195、到 v10~193,**干净自停**,~2h/$20.64/291k tok。
- iter2 `25b7c1cd…`:全新 session 读盘上 v0-v10 续做 v11-v18,在 ~196 墙反复,**崩前(is_error)**到 v18,~1.6h/$18.25/245k tok。
- **总 ~$38.89 / ~536k out_tok / ~3.6h**(用户选了 40-iter 无界默认,但 iter2 崩+故障止于 iter5,实际只花 ~$39)。

**峰值(scored 4096³)**:fp32-acc **196.1**(v18,err3.4e-5)/ f16-acc **196.6**(v11,err0.021)。防作弊✅(24文件无库)。**核心发现:第 2 个 fresh session 没突破 iter1 的 ~196 墙、只在同 plateau(192-196.6)重探**——与 worker 自己 auto-memory 记的「best v7~194、瓶颈 cp.async/sync/wave-quant、GPU 锁 1155MHz 真峰值~256 非312」一致。与 naive 干净自停 196.9 同档,低于 /goal ~205。N=1 且 iter2 崩,不足以定论 fresh-session 重启能否破 plateau。

**⚠️ 口径坑(Ralph 特有,已坐实)**:`run.jsonl` 末尾 result 事件 = iter5 崩(0 tok)→ `finish_run`/`parse_run` 默认报「总计 0 tok / no scored rows」**全错**;`finish_run` 还把 transcript 取成 iter5 的 7 行 crash stub。**真值靠合并 iter1+2 transcript 重算**:`cat results/ralph_loop/iters/iter_0{1,2}.transcript.jsonl > transcript.jsonl` 再 `parse_run`(已做,curve/csv 已更正)。session_id.txt 已改写成两个真 session id。

**已为 resume 归档(`finish_run ralph_loop --keep-worktree`)**:worktree **未删**(kernel 累积到 v18 在盘上)+ `worker.patch`/`src/` 快照 + **worker auto-memory 备份到 `results/ralph_loop/worker_memory/`(4文件,= 用户要的「包括 memory」)**。**resume 步骤见 `results/ralph_loop/RESUME.md`**:① 先把 worker_memory 还原回 base-slug(否则新 session 白板重走死路)② 在保留 worktree 上再起 fresh `claude -p "$SEED"`(读盘上 v18 续)。**别用 launch_ralph_loop.sh resume**(防覆盖闸拒跑 + 清 base-slug memory);全新一轮才另起 `ralph_loop_cycle2`。注意 [[concurrent-runs-share-worker-memory]]:resume 前确保无并行 worker。
