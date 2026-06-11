# ralph_loop_fable — Ralph Loop（Fable 5），iter-1 = naive_fable_cycle2

这是 **Ralph Loop 方法 × Fable 5 模型**的一轮跑,但 **iteration 1 不是从零跑的**——它直接**复用了 `naive_fable_cycle2` 的单 session 跑**作为 iter-1 的中间结果(kernel 已到 v22 / fp32-acc 219.5)。然后用 Fable 5 起**新的 fresh-session 迭代**(iter 2 起)继续顶。

## 血缘 / 怎么 seed 的
- **worktree**：`worktrees/ralph_loop_fable` 从 `naive_fable_cycle2` 的 HEAD（v22，2640dfe）分叉 → kernel 从 v22 续。
- **iter-1 种子**：`iters/iter_01.transcript.jsonl` = cycle2 的 transcript；`run.jsonl` = cycle2 的 run.jsonl（Ralph 从 iter 2 起**追加**）；`worker_memory/` = cycle2 学到的（冠军设计 / GPU 锁频 / cp.async 纪律）。
- **iter-1 只读快照**：`iter01_cycle2_snapshot/`（cycle2 的 result.md / curve / csv / worker.patch，供溯源）。
- 完整 cycle2 分析见 `../naive_fable_cycle2/result.md`。

## 怎么跑 / 续
```bash
bash scripts/launch_ralph_loop_fable.sh              # 续到【总】11 轮 = iter-1 种子 + 10 个新迭代(iter 2..11)
# RALPH_MAX_ITERS=21 ...  续更多;  RALPH_BUDGET_USD=60 ...  每轮成本兜底
```
worker = claude-fable-5 / effort max;每轮全新 session（干净 context），kernel 从磁盘续；停整树 `kill -TERM -- -$(cat results/ralph_loop_fable/run.pid)`。

## ⚠️ 口径坑（报结果前必看）
- **每轮独立 transcript** 落 `iters/iter_NN.transcript.jsonl`。报**全量** token/曲线要**合并所有 iters 再 parse**：
  ```bash
  cat results/ralph_loop_fable/iters/iter_*.transcript.jsonl > results/ralph_loop_fable/transcript.jsonl
  ./results/parse_run.sh ralph_loop_fable results/ralph_loop_fable/run.jsonl results/ralph_loop_fable/transcript.jsonl
  ```
- `run.jsonl` 是**追加**的：开头是 cycle2(iter-1)、末尾 result 事件是最后一个新 iter，**别拿单个 result 事件当总量**。
- 收尾：`bash scripts/finish_run.sh ralph_loop_fable --keep-worktree`（worker.patch 已按 merge-base 算，含 cycle2 的 v1-v22 + 新迭代的全量 diff）。
- GPU 仍卡 1155（驱动故障，真天花板 255.6）→ 绝对 TFLOPS 带 −18% 偏置,只比「占真天花板 %」；与 cycle1/cycle2 同卡可比。
- ⚠️ 别与别的 worker 并行（worker_memory 还原回 base-slug = 共享键，串味）。
