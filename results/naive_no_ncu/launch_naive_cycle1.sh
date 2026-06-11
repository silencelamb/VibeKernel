#!/usr/bin/env bash
# Orchestration helper — launch the naive worker exactly per runbooks/naive.md.
# Detached via setsid so it survives the launching shell and forms its own
# process group; the session-leader PID is written to results/naive_cycle1.pid
# so the orchestrator can `kill -- -<pid>` the whole tree later.
# Worker stream-json output -> results/naive_cycle1.jsonl
set -uo pipefail

ROOT=/home/daixu/code/github_code/VibeKernel
export JSONL="$ROOT/results/naive_cycle1.jsonl"
export PIDFILE="$ROOT/results/naive_cycle1.pid"

cd "$ROOT/playground-naive"
cp ../CLAUDE.md ./CLAUDE.md      # refresh fork CLAUDE.md (runbook step)
source ../env.sh                 # bind GPU (CUDA_VISIBLE_DEVICES)
export IS_SANDBOX=1              # root inside a docker container -> bypass the root
                                 # guard that blocks --dangerously-skip-permissions

export SEED="$(cat <<'SEEDEOF'
你是 CUDA kernel 专家。你的任务是在 A100 上做高性能 fp16 GEMM kernel，不断优化性能。
**NEVER STOP**：不断追求更好的性能，追求极致，（A100 fp16 理论峰值 312 TFLOPS）。持续地做、尝试各种底层优化（包括但不限于shared-mem / register tiling、双缓冲 / 多级流水、cp.async、swizzle 避免 bank conflict、提高 L2 命中、warp 排布等）、细致 profiling，绝不停下、不要回头问、可以查找CUDA编程的官方资料（比如ptx：https://docs.nvidia.com/cuda/parallel-thread-execution/ 、cuda https://docs.nvidia.com/cuda/cuda-programming-guide/index.html）。

具体约定：
每轮简报：当前 TFLOPS、Average Error、本轮改了什么、下一步试什么——然后立刻继续下一轮，不要停、不要问我。
SEEDEOF
)"

: > "$JSONL"        # fresh log
setsid bash -c 'echo $$ > "$PIDFILE"; exec claude -p "$SEED" \
  --model claude-opus-4-8 --effort max \
  --dangerously-skip-permissions \
  --output-format stream-json --verbose > "$JSONL" 2>&1' &

for _ in $(seq 1 50); do [ -s "$PIDFILE" ] && break; sleep 0.1; done
echo "naive worker launched."
echo "  group-leader PID : $(cat "$PIDFILE" 2>/dev/null || echo '?')"
echo "  GPU              : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-?}"
echo "  log              : $JSONL"
echo "  stop whole tree  : kill -TERM -- -\$(cat $PIDFILE)"
