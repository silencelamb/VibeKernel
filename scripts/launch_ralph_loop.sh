#!/usr/bin/env bash
# 启动 Ralph Loop 方法 worker:每轮重起【全新 session】的薄外循环(while :; do cat PROMPT.md | claude -p; done)。
#
# 与 /goal 的对照(=本方法被测的核心区别,见 ghuntley.com/ralph + ghuntley.com/loop):
#   /goal  = 【单个长 session】内,worker 想停时外部 LLM-judge(默认 Haiku)读 transcript 把它顶回去续
#            → 上下文一路滚大、judge 在【同一 session】内接入(session 级 Stop hook)。
#   Ralph  = 【每轮全新 session】重起,kernel 进度靠【磁盘】跨轮持久(worktree 里的 task-1/src 累积),
#            但【context window 每轮清空】→ 上下文永远干净、无 transcript 漂移、模型每轮"满血"重读现状;
#            (可选)轮间用一个【全新 judge session】判 condition 是否达成,达成才停(默认关,见下)。
#   两者都是「让 agent 一直跑」,差别 = 跨-session 重启(Ralph) vs 单-session 续跑(/goal)。
#   ⭐ 附带好处:naive/goal 多次毁于 ECONNRESET 半途崩;Ralph 外循环天然在崩后立刻重起新 session 续命。
#
# 与 naive 的单变量:naive = 一次 claude -p(单 session、自驱到自停/崩);
#   Ralph = 把【同一份 seed】外面包一个 fresh-session 重启循环。
#   model / effort(max) / seed(scripts/seed_gemm.txt,逐轮重喂 = 忠实 Ralph 的 cat PROMPT.md) /
#   worktree / 计分口径 全与 naive/goal 相同 → 单变量 = 「fresh-session 外循环 开/关」。
#
# 用法:
#   bash scripts/launch_ralph_loop.sh [run-name]            首跑:默认 "ralph_loop";多 cycle 传 ralph_loop_cycle2 …
#   bash scripts/launch_ralph_loop.sh <run-name> --resume   续跑:复用已存 worktree + 还原 worker memory + 接着 iter 序号往下
#     (--resume 与 run-name 顺序随意)。resume 前置:该 run 之前用 `finish_run.sh <run> --keep-worktree` 归档过
#     (worktree 还在 + results/<run>/worker_memory 有备份);见 results/<run>/RESUME.md 与 runbooks/ralph_loop.md。
#   旋钮(env,可选):
#     RALPH_MAX_ITERS=40          外循环【总】上限(resume 时把已存 iters 也算进去 → 续到这个总数为止)
#     RALPH_JUDGE=0               1=轮间起【全新 judge session】判 condition;NEVER-STOP seed 无终止态 → 默认 0(纯无限外循环,同 while:;)
#     RALPH_JUDGE_MODEL=claude-haiku-4-5-20251001   judge 用的小快模型(= /goal evaluator 的 Haiku 槽对位)
#     RALPH_BUDGET_USD=           每轮 claude -p 的 --max-budget-usd 成本兜底(可空)
#
# 归档口径(Ralph 特有,务必知道):每轮是【独立 session】=【独立 transcript】。本脚本把每轮 transcript
#   单独存到 results/<run>/iters/iter_NN.transcript.jsonl。finish_run 只会归档【最后一轮】那份 →
#   token/曲线会只算最后一轮。报【全量】要先合并:cat results/<run>/iters/*.transcript.jsonl > results/<run>/transcript.jsonl
#   再 parse_run(按 message.id 去重,跨 session 的 id 天然唯一、累计正确)。见 runbooks/ralph_loop.md。
#
# 每次跑在自己的 git worktree(独立 cwd-slug → 与别的方法/cycle 隔离);跑完用 finish_run.sh 收尾。
# ⚠️ 必须在【带 CAP_SYS_ADMIN 的容器】里跑(ncu 要用);先 ./scripts/ncu-doctor.sh 证实。
# ⚠️ resume 会把备份的 worker memory 还原回 base-slug → 同一时刻【别并行】跑别的 worker(见 concurrent-runs-share-worker-memory)。
# detached(setsid),关终端也不死;停用 kill -TERM -- -<pid>(停整组 = 停外循环 + 当前 iter)。
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METHOD=ralph_loop

# —— 参数:[run-name] 与 --resume 任意顺序 ——
RUN_NAME=""; RESUME=0
for a in "$@"; do
  case "$a" in
    --resume) RESUME=1 ;;
    -*) echo "❌ 未知参数: $a   (用法: launch_ralph_loop.sh [run-name] [--resume])" >&2; exit 1 ;;
    *) [ -z "$RUN_NAME" ] && RUN_NAME="$a" || { echo "❌ 多余参数: $a" >&2; exit 1; } ;;
  esac
done

if [ "$RESUME" = "1" ]; then
  # —— resume:复用已存 worktree、还原 worker memory、不清 base-slug、不 truncate run.jsonl/ralph.log ——
  #    （刻意不 source _run_common.sh:那会建新 worktree + 清 base-slug worker memory = 与 resume 相反）
  RUN="${RUN_NAME:-$METHOD}"
  BASE="$ROOT/playground-base"
  WT="$ROOT/worktrees/$RUN"
  OUTDIR="$ROOT/results/$RUN"
  export JSONL="$OUTDIR/run.jsonl"
  export PIDFILE="$OUTDIR/run.pid"
  [ -d "$WT" ] || { echo "❌ resume 需要已存在的 worktree: worktrees/$RUN(没找到)" >&2
    echo "   它可能被 finish_run(不带 --keep-worktree)删了 → 只能另起新 cycle: bash scripts/launch_ralph_loop.sh ${RUN}_cycle2" >&2
    echo "   (kernel 仍可由 results/$RUN/worker.patch 在新 worktree 上 git apply 复现,但那是新 cycle 不是本 run 的 resume)" >&2; exit 1; }
  [ -d "$OUTDIR" ] || { echo "❌ resume 需要已归档的 results/$RUN(没 finish 过?)" >&2; exit 1; }
  if [ -s "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; then
    echo "❌ results/$RUN 似乎还有活着的 loop(run.pid=$(cat "$PIDFILE"))。先 kill -TERM -- -$(cat "$PIDFILE") 再 resume。" >&2; exit 1; fi
  # ♻️ 还原 worker auto-memory(让 resume 的新 session 带着上次学到的瓶颈/死路,不白板重走)
  _BASESLUG="$(printf '%s' "$BASE" | sed 's/[^[:alnum:]]/-/g')"
  if [ -d "$OUTDIR/worker_memory" ] && [ -n "$(ls -A "$OUTDIR/worker_memory" 2>/dev/null)" ]; then
    mkdir -p "$HOME/.claude/projects/$_BASESLUG/memory"
    cp -a "$OUTDIR/worker_memory/." "$HOME/.claude/projects/$_BASESLUG/memory/"
    echo "  ♻️ 已还原 worker auto-memory(results/$RUN/worker_memory → base slug)→ resume 带着上次记忆"
  else echo "  ⚠️ resume:没找到 results/$RUN/worker_memory 备份,新 session 将白板起步(可接受:kernel 仍从盘上续)"; fi
  cp "$ROOT/CLAUDE_For_KernelAgent.md" "$WT/CLAUDE.md"   # 刷新 worker 手册(每次启动都 cp 最新版)
  if   [ -f "$ROOT/env.sh" ];         then source "$ROOT/env.sh"
  elif [ -f "$ROOT/env.sh.example" ]; then source "$ROOT/env.sh.example"
       echo "  ⚠️ 未找到 env.sh,回退 env.sh.example(CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-?});请 cp env.sh.example env.sh"; fi
  export IS_SANDBOX=1
  touch "$JSONL"                                         # resume 追加进同一 run.jsonl(不 truncate)
  echo "  ▶️ resume:复用 worktrees/$RUN(kernel 从盘上续)、iter 序号接 results/$RUN/iters/ 往下、RALPH_MAX_ITERS 是【总】上限"
else
  source "$ROOT/scripts/_run_common.sh"   # 建 worktree、cp 手册、绑 GPU、清 base-slug worker memory,导出 RUN/WT/OUTDIR/JSONL/PIDFILE
fi

export SEED="$(cat "$ROOT/scripts/seed_gemm.txt")"         # 与 naive/goal 同一份;Ralph 每轮重喂这同一份
export RALPH_MODEL="${RALPH_MODEL:-claude-opus-4-8}"       # worker 模型(默认 Opus = 不改原行为);Fable 续跑设 RALPH_MODEL=claude-fable-5
export RALPH_MAX_ITERS="${RALPH_MAX_ITERS:-40}"            # 外循环【总】上限(resume 把已存 iters 算进去)
export RALPH_JUDGE="${RALPH_JUDGE:-0}"                     # 轮间 fresh-session 判停;NEVER-STOP seed 默认关
export RALPH_JUDGE_MODEL="${RALPH_JUDGE_MODEL:-claude-haiku-4-5-20251001}"
export RALPH_BUDGET_USD="${RALPH_BUDGET_USD:-}"
export ITERDIR="$OUTDIR/iters"                            # 每轮 transcript 单独存(全量 token 要合 iters/)
export RLOG="$OUTDIR/ralph.log"                           # 外循环的人读标记(不污染 run.jsonl 的 stream-json)
mkdir -p "$ITERDIR"
[ "$RESUME" = "1" ] || : > "$RLOG"                        # fresh 才清 ralph.log;resume 追加保留 cycle 历史

cd "$WT"
# 外循环整体 detached;循环的 echo 标记进 ralph.log,每轮 claude 的 stream-json 进 run.jsonl(保持纯 JSON、watch_run 不挂)。
setsid bash -c '
  echo $$ > "$PIDFILE"
  BUDGET_ARG=(); [ -n "$RALPH_BUDGET_USD" ] && BUDGET_ARG=(--max-budget-usd "$RALPH_BUDGET_USD")
  # iter 序号接着已存 iters/ 往下(fresh=0;resume=上次最大号)——RALPH_MAX_ITERS 是【总】上限,不是【本次新增】。
  iter=$(ls "$ITERDIR"/iter_*.transcript.jsonl 2>/dev/null | sed -E "s/.*iter_0*([0-9]+).*/\1/" | sort -n | tail -1); iter=${iter:-0}
  [ "$iter" -gt 0 ] && echo "==== RALPH RESUME: 已有 $iter 轮,从 iter $((iter+1)) 续到上限 $RALPH_MAX_ITERS ===="
  short=0
  while [ "$iter" -lt "$RALPH_MAX_ITERS" ]; do
    iter=$((iter+1))
    echo "==== RALPH ITER $iter / $RALPH_MAX_ITERS  (全新 session,clean context;kernel 进度从磁盘续) ===="
    t0=$SECONDS
    # ⭐ 核心:每轮一个【全新 claude -p session】(干净 context),同一 worktree(kernel 状态从磁盘续)。
    claude -p "$SEED" \
      --model "$RALPH_MODEL" --effort max \
      --dangerously-skip-permissions \
      --output-format stream-json --verbose "${BUDGET_ARG[@]}" >> "$JSONL" 2>&1
    rc=$?
    dt=$((SECONDS - t0))
    # 把这一轮的 transcript(全新 session = worktree-slug 下最新那个 .jsonl)单独存一份。
    SLUG=$(printf "%s" "$PWD" | sed "s/[^[:alnum:]]/-/g")
    LATEST=$(ls -t "$HOME/.claude/projects/$SLUG/"*.jsonl 2>/dev/null | head -1)
    [ -n "$LATEST" ] && cp "$LATEST" "$ITERDIR/iter_$(printf %02d "$iter").transcript.jsonl"
    echo "---- iter $iter done: rc=$rc, ${dt}s, transcript=$(basename "${LATEST:-none}") ----"
    # 防 hot-loop:连续 3 轮 <30s(多半是 auth/setup 崩、空烧钱)→ 停外循环。
    if [ "$dt" -lt 30 ]; then short=$((short+1)); else short=0; fi
    [ "$short" -ge 3 ] && { echo "⚠️ 连续 3 轮秒退(疑似 setup/auth 崩),停外循环。"; break; }
    # (可选)fresh judge session 判 condition;NEVER-STOP seed 无终止态 → 默认关。
    #   体现 Ralph 的「每次都新起一个 session 判是否达成」——judge 也是全新 session、只读上一轮工作摘要。
    if [ "$RALPH_JUDGE" = "1" ] && [ -n "${LATEST:-}" ]; then
      SUMMARY=$(jq -rc "select(.type==\"assistant\") | .message.content[]? | select(.type==\"text\") | .text" "$LATEST" 2>/dev/null | tail -40)
      VERDICT=$(claude -p "你是判停器(全新 session)。任务:不断优化 A100 fp16 GEMM、逼近 312 TFLOPS、NEVER STOP。下面是 worker 上一轮的工作记录。仅当任务已【彻底无可优化、确实该停】时回 STOP,否则回 CONTINUE。只回一个词。

$SUMMARY" --model "$RALPH_JUDGE_MODEL" --dangerously-skip-permissions 2>/dev/null | tr -d "[:space:]")
      echo "---- judge(全新 session, $RALPH_JUDGE_MODEL): ${VERDICT:-?} ----"
      [ "$VERDICT" = "STOP" ] && { echo "==== judge 判 STOP,收外循环 ===="; break; }
    fi
  done
  echo "==== RALPH LOOP ended at iter $iter (上限 $RALPH_MAX_ITERS) ===="
' >> "$RLOG" 2>&1 &

for _ in $(seq 1 50); do [ -s "$PIDFILE" ] && break; sleep 0.1; done
echo "ralph_loop worker launched (fresh-session 外循环$([ "$RESUME" = "1" ] && echo ' / RESUME'))."
echo "  run-name  : $RUN   (结果落 results/$RUN/)"
echo "  mode      : $([ "$RESUME" = "1" ] && echo 'resume(复用 worktree + 还原 worker memory + 接 iter 序号)' || echo 'fresh(新 worktree + 清 base-slug memory 白板起)')"
echo "  worktree  : $WT"
echo "  group PID : $(cat "$PIDFILE" 2>/dev/null || echo '?')"
echo "  GPU       : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-?}"
echo "  worker    : $RALPH_MODEL / effort max / 每轮全新 session(单变量=fresh-session 外循环开)"
echo "  loop      : max_iters=$RALPH_MAX_ITERS(总上限)  judge=$RALPH_JUDGE(默认关,NEVER-STOP seed 无终止态)  budget_usd=${RALPH_BUDGET_USD:-(none)}"
echo "  seed      : scripts/seed_gemm.txt(逐轮重喂 = 忠实 Ralph 的 cat PROMPT.md;与 naive/goal 同一份)"
echo "  log       : $JSONL   (各轮 stream-json 累加;外循环标记看 $RLOG)"
echo "  iters     : $ITERDIR/iter_NN.transcript.jsonl(每轮独立 transcript;全量 token 要合并,见脚本头注)"
echo "  watch     : $ROOT/results/watch_run.sh $JSONL"
echo "  stop tree : kill -TERM -- -\$(cat $PIDFILE)"
echo "  收尾归档  : bash $ROOT/scripts/finish_run.sh $RUN [--keep-worktree]   (想日后再 resume 就带 --keep-worktree)"
echo "  ⚠️ 报全量 token/曲线前先合并各轮 transcript:cat $ITERDIR/*.transcript.jsonl > $OUTDIR/transcript.jsonl 再 parse_run(见 runbooks/ralph_loop.md)。"
