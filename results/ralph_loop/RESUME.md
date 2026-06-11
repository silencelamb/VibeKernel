# Ralph Loop — cycle1 归档 & resume 指南

> 本次跑(`results/ralph_loop/`)= Ralph Loop 方法**首跑**。**外循环跑了 5 次,但只有 2 次是实质迭代**;第 2 次崩(ECONNRESET 级)后,iter 3-5 撞上 API 故障(每次 <10s 秒退),被脚本的 **hot-loop 防护(连续 3 轮 <30s)自动停在 iter 5**。worktree 已**保留**(`--keep-worktree`)、worker memory 已备份 —— 为 resume 备好。

## 跑了什么(2 个实质 iteration)

| iter | session_id | 时长 | 结束 | out_tok | cost | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `58f37889-1f6a-417c-b49a-d16329561569` | ~7172s(~2h) | ✅ 干净自停 | 291,183 | $20.64 | 全新 session 从零爬到 v7~195;一路到 v10~193 |
| 2 | `25b7c1cd-dcac-4117-a64e-c5675facbd9c` | ~5918s(~1.6h) | ❌ 崩(is_error) | 245,045 | $18.25 | 全新 session 读盘上 v0-v10 续做 v11-v18,在 ~196 墙反复,崩前到 v18 |
| 3-5 | (crash stubs) | 3-9s | ❌ API 故障秒退 | 0 | $0 | hot-loop 防护在 iter5 停外循环 |

- **总计**:~$38.89 / ~536k output token / ~3.6h 计算(2 个实质 session 之和)。
- **峰值(scored 4096³)**:**fp32-acc 196.1**(v18, err 3.4e-5)/ **f16-acc 196.6**(v11, err 0.021)。running-best 196.6。
- **结论(N=1,且 iter2 崩)**:第 2 个全新 session **没能突破** iter1 的 ~196 墙,只是在同一 plateau 重新探索(v11-v18 全在 192-196.6)。与 worker 自己 memory 记的「best v7~194,瓶颈是 cp.async/sync/wave-quant、GPU 锁 1155MHz 真峰值 ~256 不是 312」一致。和 naive 干净自停 196.9 同档,低于 /goal ~205。
- **防作弊门 ✅ 通过**(24 文件无 cutlass/cublas)。
- ⚠️ **口径坑**:`run.jsonl` 末尾 result 事件是 iter5 崩(0 tok)→ `parse_run` 的「总计(权威)」行报 0,**不可信**;真值用本文件表 / 已重算的 `result.csv`(基于合并后的 iter1+2 transcript)。

## 已保留的状态(resume 用)

| 东西 | 位置 | 说明 |
| --- | --- | --- |
| **worktree(kernel 现状)** | `worktrees/ralph_loop/`(**未删**) | `task-1/src/matmul_f16/` 累积到 **v18**;detached HEAD,可直接当 resume 的 cwd |
| **kernel patch(可复现)** | `results/ralph_loop/worker.patch` | `git apply` 即还原 v1-v18(worktree 万一丢了的兜底) |
| **kernel 源码快照** | `results/ralph_loop/src/` | 18 个 kernel 易读快照 |
| **⭐ worker auto-memory** | `results/ralph_loop/worker_memory/`(4 文件) | iter1-2 学到的:`gemm-f16-bottleneck-analysis`(每个试过/失败的 tuning 轴,别重explore)、`gpu-clock-locked-1155`、`always-check-error-not-just-tflops`。**resume 前要还原回 base-slug,否则新 session 白板重走死路** |
| **逐 iter transcript** | `results/ralph_loop/iters/iter_0{1,2}.transcript.jsonl` | iter3-5 是 7 行 crash stub,已忽略 |
| **合并 transcript / 曲线** | `results/ralph_loop/transcript.jsonl` `result.csv` `curve.png` | 已基于 iter1+2 重算(非 finish_run 默认的 iter5) |

## 怎么 resume(继续往下跑)

> Ralph 的「resume」**不是** `claude --resume <session_id>`(每 iter 本就是独立 session、无单一 session 可续)。resume = **在保留的 worktree 上、带着盘上累积的 kernel + 还原的 worker memory,再起新的 fresh session**。

**一条命令(已内建 `--resume`)**:
```bash
cd /home/daixu/code/github_code/VibeKernel
bash scripts/launch_ralph_loop.sh ralph_loop --resume
```
它会自动:复用 `worktrees/ralph_loop`(kernel 从盘上 v18 续)→ 把 `results/ralph_loop/worker_memory/` 还原回 base-slug(新 session 带着上次学到的瓶颈/死路)→ **不**清 base-slug memory → 追加进同一 `run.jsonl`/`ralph.log`、iter 序号从 **6** 接着往下(cycle1 已有 5 轮)续到 `RALPH_MAX_ITERS`(默认 40 = 总上限,想多续 `RALPH_MAX_ITERS=60 bash … --resume`)。

跑完照常归档(记得带 `--keep-worktree` 以便再 resume,并合并 transcript 重算曲线):
```bash
bash scripts/finish_run.sh ralph_loop --keep-worktree
cat results/ralph_loop/iters/iter_0{1,2}.transcript.jsonl results/ralph_loop/iters/iter_{06,07,...}.transcript.jsonl \
  > results/ralph_loop/transcript.jsonl          # 只挑实质轮(跳过 03-05 crash stub)
./results/parse_run.sh ralph_loop results/ralph_loop/run.jsonl results/ralph_loop/transcript.jsonl
```

> ⚠️ resume 会把 worker_memory 还原回 base-slug → 同一刻**别并行**别的 worker(串味)。
> ⚠️ **不带 `--resume`** 直接 `launch_ralph_loop.sh ralph_loop` 会被防覆盖闸拒跑(且清 memory)。要从零的全新一轮 → 另起 run-name:`launch_ralph_loop.sh ralph_loop_cycle2`。
> **彻底不要本次 worktree 了**:`bash scripts/finish_run.sh ralph_loop`(不带 `--keep-worktree`)删掉它;kernel 仍可由 `worker.patch` 复现。

## 后续(可选)
- 套 `results/TEMPLATE.md` 写完整人读 `result.md`(本文件已含核心数据)。
