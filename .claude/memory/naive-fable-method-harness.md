---
name: naive-fable-method-harness
description: "naive 的 Fable-5 模型对比臂(launch_naive_fable.sh):唯一变量=模型 claude-fable-5 vs claude-opus-4-8,只与 naive(Opus) 家族横比"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3665db98-7374-4e66-ac17-02609d9cc4ce
---

新增 naive 的【模型对比臂】= `scripts/launch_naive_fable.sh` + `runbooks/naive_fable.md`(2026-06-10 建)。**首跑已完成**,结果见 [[naive-fable-cycle1-result]](211.5f16/201.8fp32,但 GPU 锁 1155 时钟混淆)。续跑能力已加:`scripts/resume_naive_fable.sh`(Ralph-Loop 式 Fable 续跑,靠给 launch_ralph_loop.sh 新增的 `RALPH_MODEL` 旋钮)。

**单变量 = 主 worker 模型**:与 `launch_naive.sh` 逐字相同(同 `seed_gemm.txt`、同 `--effort max`、同 NEVER STOP 纯自驱、同 worktree/finish_run 归档流),唯一改动 `--model claude-opus-4-8` → `--model claude-fable-5`(Anthropic 最新)。METHOD=naive_fable → 默认 results/naive_fable/(与 naive 不撞)。已 preflight 验过 `claude -p --model claude-fable-5 --effort max` 可用(返回 OK/exit0)。

**Why**:naive(Opus) 家族(naive/naive_cycle2…6)已坐实 Opus 在纯 prompt 下的峰值/自停点/token-cost;换最新模型 Fable 5 隔离「模型」这一维。

**How to apply**:
- 启动 `bash scripts/launch_naive_fable.sh [run-name]`(默认 naive_fable;多跑 naive_fable_cycle2)→ 收尾 `finish_run.sh`(同 naive)。
- 结论**只与 naive(Opus) 家族并列**(峰值/自停点/总token/cost/是否撞同款 tensor-pipe 天花板),**别**和 /goal(有看门狗)、dynamic_workflow(有 fan-out)混比——那些是范式变量,这是模型变量。
- token 单价/持续力与 Opus 不同是【预期结果】不是 harness 坏了。
- ⚠️ 别与 naive 并行跑(共享 base-slug worker memory 串味,见 [[concurrent-runs-share-worker-memory]])。

参考 naive 家族结果 [[f16-accumulate-precision-confound]]、[[naive-cycle1-econnreset-and-gotchas]];harness 同源 [[playground-base-clean-fork-topology]]、[[vibekernel-result-harness]]。
