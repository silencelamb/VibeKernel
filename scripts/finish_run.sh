#!/usr/bin/env bash
# 收尾一次跑(worktree 流):归档 transcript + worker auto-memory + 防作弊门 + 快照 src/include/logs/worker.patch + 解析曲线 +(默认)删 worktree。
# 用法: bash scripts/finish_run.sh <run-name> [--keep-worktree]
#   <run-name>      = launch 时用的那个(如 naive / goal / naive_cycle2)
#   --keep-worktree = 归档但【不删 worktree】。交互式 run 想日后 `claude --resume <session_id>` 继续人工对话时用:
#                     resume 需要 worktree 当 cwd 还在(+ 可从 results/<run>/worker_memory 还原 worker 记忆);
#                     session_id 见 results/<run>/session_id.txt。日后真不要了再 `finish_run <run>`(不带 flag)删掉。
# 跑完 results/<run-name>/ 就齐了(只差人写 result.md);基座 playground-base 自始至终没被碰。
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${1:?用法: finish_run.sh <run-name> [--keep-worktree]  (如 naive / goal / naive_cycle2)}"
KEEP_WT=0; [ "${2:-}" = "--keep-worktree" ] && KEEP_WT=1
BASE="$ROOT/playground-base"
WT="$ROOT/worktrees/$RUN"
OUT="$ROOT/results/$RUN"
[ -d "$WT" ] || { echo "❌ worktree 不在: worktrees/$RUN(没 launch 过?还是已 finish?)" >&2; exit 1; }
mkdir -p "$OUT"

echo "=== ① 归档 transcript(本次 worktree 的独立 cwd-slug,唯一、不用猜)==="
SLUG="$(printf '%s' "$WT" | sed 's/[^[:alnum:]]/-/g')"        # cwd 路径非字母数字全转 -
TDIR="$HOME/.claude/projects/$SLUG"
[ -d "$TDIR" ] || TDIR="$(ls -dt "$HOME/.claude/projects/"*worktrees* 2>/dev/null | head -1)"   # 兜底:最近的 worktree slug 目录
LATEST="$(ls -t "$TDIR"/*.jsonl 2>/dev/null | head -1)"
if [ -n "$LATEST" ]; then cp "$LATEST" "$OUT/transcript.jsonl"; echo "  ✅ ← $TDIR/$(basename "$LATEST") ($(wc -l < "$OUT/transcript.jsonl") 行)"
else echo "  ⚠️ 没找到 transcript($TDIR);token/曲线会缺权威源"; fi

echo "=== ② 备份 worker auto-memory + 记 session_id(resume 用)==="
SESSION_ID="$(basename "${LATEST:-}" .jsonl 2>/dev/null)"
if [ -n "$SESSION_ID" ]; then printf '%s\n' "$SESSION_ID" > "$OUT/session_id.txt"
  echo "  session_id=$SESSION_ID → session_id.txt  (resume: cd worktrees/$RUN && claude --resume $SESSION_ID)"; fi
# worker 的 auto-memory 按【git 仓库】键 = base-slug 那份(run 内 worker 写/读、跨 context 压缩持久化,worktree-slug 里没有)。
# 下次 launch 会被 _run_common 清掉 → 这里先备份进 results/<run>/worker_memory/:留证 + 给 resume 用(resume 前可手动还原回去)。
_BASESLUG="$(printf '%s' "$BASE" | sed 's/[^[:alnum:]]/-/g')"
WMEM="$HOME/.claude/projects/$_BASESLUG/memory"
if [ -d "$WMEM" ] && [ -n "$(ls -A "$WMEM" 2>/dev/null)" ]; then
  rm -rf "$OUT/worker_memory"; cp -a "$WMEM" "$OUT/worker_memory"
  echo "  ✅ worker auto-memory → worker_memory/($(ls "$OUT/worker_memory" 2>/dev/null | wc -l) 文件)"
else echo "  (worker 本次没写 auto-memory,跳过)"; fi

echo "=== ③ 防作弊门(用了 cutlass/cublas 判该跑无效)==="
bash "$ROOT/scripts/check_handwritten.sh" "$WT" 2>&1 | tail -2 | sed 's/^/  /' \
  || echo "  ⚠️⚠️ 防作弊门未过——该跑结果无效(用了现成 GEMM 库),但仍快照存证;result.md 须标明"

echo "=== ④ 快照 kernel 产物(从 worktree)==="
# ⚠️ worker 可能自己 git commit(如 Fable)→ HEAD 已含其提交,`diff --cached HEAD` 会算出空 patch。
#    用 worktree 与基座 HEAD 的 merge-base(= fork 点)当基准 → 不管 worker 提交没提交,都拿到相对干净基座的【全量】diff。
#    (worker 没提交时 merge-base == HEAD,行为与旧版完全一致;基座始终没动,merge-base 就是当初 worktree add 的那个 HEAD。)
BASE_REF="$(git -C "$WT" merge-base HEAD "$(git -C "$BASE" rev-parse HEAD)" 2>/dev/null || git -C "$BASE" rev-parse HEAD)"
git -C "$WT" add -A
git -C "$WT" diff --cached "$BASE_REF"             > "$OUT/worker.patch"      # 相对干净基座的全部改动(git apply 可复现;含 worker 自提交)
git -C "$WT" diff --cached --name-status "$BASE_REF" > "$OUT/worker.files.txt"  # 改了哪些文件(自提交时 status --short 会空,故用相对 base 的 name-status)
_NCMT="$(git -C "$WT" rev-list --count "$BASE_REF"..HEAD 2>/dev/null || echo 0)"
[ "${_NCMT:-0}" -gt 0 ] && echo "  ℹ️ worker 自提交 $_NCMT 个 commit → patch 已按 merge-base 算(非空)"
rm -rf "$OUT/src"; mkdir -p "$OUT/src"
cp -a "$WT/task-1/src/." "$OUT/src/"
# dispatcher 方法真 kernel 在 include/(如 mma_gemm.cuh)→ worker 动过 include 才快照(src-based 不留多余 base 头)
if git -C "$WT" diff --cached "$BASE_REF" --name-only | grep -q 'task-1/include/'; then
  rm -rf "$OUT/include"; mkdir -p "$OUT/include"; cp -a "$WT/task-1/include/." "$OUT/include/"; echo "  + include/(dispatcher 真 kernel)"
fi
mkdir -p "$OUT/logs"
find "$WT" -maxdepth 3 \( -name '*.ncu-rep' -o -name '*.nsys-rep' -o -name '*.log' \) -not -path '*/build/*' -exec cp -a {} "$OUT/logs/" \; 2>/dev/null || true
rmdir "$OUT/logs" 2>/dev/null
echo "  ✅ src/($(ls "$OUT/src/matmul_f16/"*.cu 2>/dev/null | wc -l) kernel) + worker.patch($(wc -l < "$OUT/worker.patch") 行) + logs/($(ls "$OUT/logs" 2>/dev/null | wc -l))"

echo "=== ⑤ 解析曲线(canonical/scored/invalid + /goal evaluator 接入红线)==="
if [ -s "$OUT/transcript.jsonl" ]; then
  "$ROOT/results/parse_run.sh" "$RUN" "$OUT/run.jsonl" "$OUT/transcript.jsonl" 2>&1 | grep -vi 'glyph\|UserWarning' | tail -6 | sed 's/^/  /'
else echo "  跳过(无 transcript)"; fi

rm -f "$OUT/run.pid"
if [ "$KEEP_WT" = "1" ]; then
  echo "=== ⑥ 保留 worktree(--keep-worktree)==="
  echo "  ⏭️ worktrees/$RUN 未删 —— 可 resume: cd worktrees/$RUN && claude --resume ${SESSION_ID:-<id>}"
  echo "     日后真不要了再: bash scripts/finish_run.sh $RUN  (不带 flag,删 worktree)"
else
  echo "=== ⑥ 删 worktree(基座 playground-base 一直没动)==="
  git -C "$BASE" worktree remove --force "$WT" && echo "  ✅ worktrees/$RUN 已删"
  git -C "$BASE" worktree prune
fi
echo
echo "完成。results/$RUN/ 已就绪 —— 只差套 results/TEMPLATE.md 写 result.md。"
echo "  /goal 跑别忘:数 evaluator 接入(transcript 的 goal_status,见 runbooks/goal.md 第3步)+ 查头一次接入时的 TFLOPS。"
