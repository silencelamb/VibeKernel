# naive_fable — resume 指南(在本次跑基础上继续再跑几轮)

> 本次跑 = naive 范式的 **Fable 5** 单 session 跑,**已干净自停**(stop_reason=end_turn),峰值 **f16-acc 211.5 / fp32-acc 201.8 TFLOPS**,详见 `result.md` / `method_table.md`。
> 收尾时用了 `--keep-worktree` → **worktree 保留、session_id 已存、worker_memory 已备份**,为 resume 备好。

## 已保留的状态(resume 用)

| 东西 | 位置 | 说明 |
| --- | --- | --- |
| **worktree(kernel 现状)** | `worktrees/naive_fable/`(**未删**) | `task-1/src/matmul_f16/` 累积到 **v13**,且 **worker 已自 commit**(git log 有 4 个里程碑);可直接当 resume 的 cwd |
| **session_id(同 session 续)** | `results/naive_fable/session_id.txt` | `10777a72-f54c-422d-b4f6-f660731b3d10` |
| **⭐ worker auto-memory** | `results/naive_fable/worker_memory/`(4 文件) | `vibekernel-task1-best-config`(v9 设计点 + 每个邻域 config 为何撞墙)、`vibekernel-task1-environment`(**GPU 锁 1155 → 真峰 255.6**)、`vibekernel-task1-ptxas-lessons`(哪些重构会 REG 168→255 spill、完整 rejected 清单)。**resume 前要还原回 base-slug,否则新 session 白板重走死路**(resume 脚本会自动还原) |
| **kernel patch(可复现)** | `results/naive_fable/worker.patch` | 已修正(base..HEAD,4348 行 / 13 文件);worktree 万一丢了的兜底 |
| **kernel 源快照** | `results/naive_fable/src/`(13 版) | 易读快照 |

## 怎么续跑

### ⭐ 方式 A — Ralph-Loop 式续跑(推荐,= 用户要的「类似 Ralph Loop」)
每轮起**全新 Fable 5 session**(干净 context、满血重读盘上 v13 + 还原的 worker 记忆),去顶自己的 211.5。一条命令:
```bash
cd /home/daixu/code/github_code/VibeKernel
bash scripts/resume_naive_fable.sh                 # 默认续 3 轮 fresh session(worker=Fable 5)
# RALPH_MAX_ITERS=5 bash scripts/resume_naive_fable.sh    # 续更多轮
# RALPH_BUDGET_USD=40 bash scripts/resume_naive_fable.sh  # 每轮成本兜底(优雅停)
```
它自动:复用 `worktrees/naive_fable`(kernel 从盘上 v13 续)→ 还原 `worker_memory` 回 base-slug → 用 **claude-fable-5** 起轮轮 fresh session,iters 落 `results/naive_fable/iters/iter_NN.transcript.jsonl`、stream-json 追加进 `run.jsonl`、外循环标记进 `ralph.log`。
> 机制 = 复用了 `launch_ralph_loop.sh --resume`(新加的 `RALPH_MODEL` 旋钮把模型钉成 Fable 5);细节见 `runbooks/ralph_loop.md`。

### 方式 B — 同 session 续(继续 Fable 的原始 context)
最字面的「接着上次」,但 Fable 上次已自判到顶、可能很快又停:
```bash
cd /home/daixu/code/github_code/VibeKernel/worktrees/naive_fable
# 还原 worker 记忆(可选但建议):
cp -a ../../results/naive_fable/worker_memory/. \
   "$HOME/.claude/projects/$(printf '%s' "$PWD" | sed 's/[^[:alnum:]]/-/g')/memory/" 2>/dev/null || true
IS_SANDBOX=1 claude --resume 10777a72-f54c-422d-b4f6-f660731b3d10 \
  --model claude-fable-5 --effort max --dangerously-skip-permissions \
  -p "继续:用 ncu 复核 v9 的 k-step 交接 stall,试着再逼近 255.6 真天花板;仍 NEVER STOP、task1 100 轮计分口径。"
```

## 跑完续跑后的归档(口径坑,务必看)
```bash
bash scripts/finish_run.sh naive_fable --keep-worktree    # 想再续就继续带 --keep-worktree
```
- ⚠️ **worker.patch 坑**:Fable 会自 commit → finish_run 的 `git diff --cached HEAD` 算出空 patch。续跑后请手工重生成:
  ```bash
  git -C worktrees/naive_fable diff $(git -C playground-base rev-parse HEAD)..HEAD > results/naive_fable/worker.patch
  ```
- ⚠️ **token/曲线全量口径**:续跑每轮是独立 session = 独立 transcript。报「单轮 naive_fable」用现在归档的(`transcript.jsonl` = 那一个 10777… session);报「naive_fable + Ralph 续跑全量」要合并:
  ```bash
  cat results/naive_fable/transcript.jsonl results/naive_fable/iters/*.transcript.jsonl > /tmp/all.jsonl
  ./results/parse_run.sh naive_fable results/naive_fable/run.jsonl /tmp/all.jsonl
  python3 results/naive_fable/plot_fable.py     # 重画带真天花板 255.6 的曲线
  ```
  **别**把续跑结果混进原单 session 的「211.5 / 96min / $53」结论里——那是单 session naive 的纯净数;续跑是 Ralph 式,属另一个口径。
- ⚠️ **别并行**:resume 会还原 worker_memory 回 base-slug(共享键)→ 同一刻别跑别的 worker(串味,见 memory `concurrent-runs-share-worker-memory`)。
- ⚠️ **时钟**:GPU 现锁 1155(真峰 255.6)。续跑前/后用 `result.md` 复现节那段高频采样确认时钟没变,否则绝对 TFLOPS 又不可比。
