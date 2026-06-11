---
name: concurrent-runs-share-worker-memory
description: "并行跑两个worktree=串味!worker auto-memory按git仓库键(存base-slug、非cwd-slug),所有playground-base worktree共享同一份;_run_common清-memory只防串行;要并行必须串行排队或每轮CLAUDE_CODE_DISABLE_AUTO_MEMORY=1"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: efbd0bee-80cb-44d8-a7a2-221db97a0819
---

**血泪(2026-06-08):把本该串行的两轮【并行】跑 → worker memory 串味、一轮作废。**

机制:Claude worker auto-memory **按【git 仓库】键**,落在 `~/.claude/projects/<playground-base-slug>/memory/`(**不是**按 cwd-slug;transcript 才按 cwd-slug 隔离)。所有 `playground-base` worktree 的 `.git` 是同一个 → **共享同一份 worker-memory**。`_run_common.sh` 的「每 launch 前 rm base-slug memory」**只防串行**(上一轮写完、下一轮 launch 前清);**并行跑时两个 worker 全程同写同读一个 dir**。

实锤:`naive_cycle5`(session 32bf3829)与 `goal_cycle3`(9ce2dce9)并行,naive 读了 goal 写进共享 memory 的设计笔记(172/178 winning config、sustained-clock、"fp16-acc 1-block 没收益"),在其上做出 f16-acc 218 → **非独立、已 deprecate**(同 `naive_cycle2_deprecated` 的串味)。goal_cycle3 经查只读自己的、且事后把 naive 文件移出共享 dir 保护,暂判干净。

**Why:** 隔离设计默认单 GPU 串行(META "单 GPU 本就串行")。并行跑打破了这个隐含前提,而 worktree 只隔离了 transcript、没隔离 worker-memory。

**How to apply:**
- **要并行跑:每轮 launch 加 `export CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`**(`_run_common.sh` 已有注释行,解开即可)→ 彻底无共享 memory;代价是丢"单 run 内跨 compaction 持久化"(长跑略亏,但换来真隔离)。**或**老老实实串行排队(一个 finish 再 launch 下一个)。
- **诊断串味**:每个 memory 文件 frontmatter 有 `originSessionId`;比对各 run 的 `session_id`(`jq .session_id run.jsonl`)。某 run 的 run.jsonl/transcript 里出现【别的 run 的 memory 文件名】= 它读了别人的 = 串味。
- **保护在跑的干净轮**:把污染轮的 memory 文件从共享 dir 移走 + 改 base-slug `MEMORY.md` 索引(只动污染轮的、别碰干净轮自己的)。证据备份在 `results/<deprecated>/worker_memory_LEAKED/`。
- 值得给 `_run_common.sh` 加个 launch 闸:检测到已有 sibling worktree 在跑就警告/拒绝并行(防再踩)。

关联 [[playground-base-clean-fork-topology]]、[[f16-accumulate-precision-confound]]、[[orchestration-overview]]。
