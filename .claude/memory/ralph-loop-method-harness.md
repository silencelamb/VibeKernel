---
name: ralph-loop-method-harness
description: "Ralph Loop 方法 harness:fresh-session 外循环 launch_ralph_loop.sh,与 /goal 的单变量区别 + 归档口径坑"
metadata: 
  node_type: memory
  type: project
  originSessionId: 7134553b-8933-449d-a5c3-935f51cf35af
---

新增方法 **Ralph Loop**(出处 ghuntley.com/ralph + ghuntley.com/loop,`while :; do cat PROMPT.md | claude-code ; done`)。harness 已就绪、**尚未跑**(README 标 ⏳ script ready / 脚本就绪)。

**与 [[goal-method-harness]] 的唯一概念差 = 怎么续**:/goal 单 session 内续(同 session 判停、上下文一路滚大);Ralph **每轮重起全新 session**(context 每轮清空、kernel 进度靠磁盘 worktree task-1/src 跨轮持久),可选轮间起**全新 judge session** 判停。seed/model(opus-4.8)/effort(max)/worktree/口径全与 naive/goal 同 → **单变量 = fresh-session 外循环开/关**,可与 naive/goal 横比。⭐ 附带卖点:naive/goal 多 cycle 毁于 ECONNRESET,Ralph 外循环天然崩后重起续命。

**真源**:`scripts/launch_ralph_loop.sh`(detached setsid 外循环,各轮 stream-json 累加进 run.jsonl、外循环标记进 ralph.log)+ `runbooks/ralph_loop.md`。旋钮 env:`RALPH_MAX_ITERS`(默认 40,是【总】上限、resume 把已存 iters 算进去)、`RALPH_JUDGE`(默认 0——NEVER-STOP seed 无终止态、judge 永判未达成,纯无限外循环正对 while:;)、`RALPH_JUDGE_MODEL`(默认 haiku-4-5)、`RALPH_BUDGET_USD`(每轮挂 --max-budget-usd);hot-loop 防护=连续 3 轮 <30s 自动停;iter 序号从已存 `iters/` 最大号续。

**`--resume` 模式**(`bash scripts/launch_ralph_loop.sh <run> --resume`,--resume/run 顺序随意):刻意**不** source `_run_common.sh`,改为复用已存 worktree + 把 `results/<run>/worker_memory/` 还原回 base-slug + 不清 memory + 追加进同一 run.jsonl/ralph.log + iter 接着往下。前置=之前 `finish_run --keep-worktree` 归档过(worktree 还在 + worker_memory 有备份);worktree 被删则不能 resume、只能另起 `<run>_cycle2`。⚠️ resume 还原 memory → 别并行别的 worker([[concurrent-runs-share-worker-memory]])。

**⚠️ 归档口径坑(必知)**:每轮独立 session = 独立 transcript;脚本存 `results/<run>/iters/iter_NN.transcript.jsonl`,`finish_run` 默认只归档**最后一轮** → 报全量 token/曲线要先 `cat results/<run>/iters/*.transcript.jsonl > results/<run>/transcript.jsonl` 再 parse_run(message.id 跨 session 唯一、累计正确;同 [[goal-cycle4-result]] 两段手工合并)。跑完查 ralph.log 是否真多轮 fresh-session(只 1 轮=退化成 naive)。

文档已同步:README.md/README.zh.md §3 方法阶梯(goal 行后插 Ralph Loop 行,goal 行改措辞强调"单 session 内")、META.md §2 表 + 新 §8(环境顺延 §9)、methods.md §1.5b。
