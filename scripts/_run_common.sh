# 被 launch_<方法>.sh source(非独立执行)。调用前 launcher 须设好:ROOT、METHOD、RUN_NAME(可空)。
# 作用:为本次跑建一个独立的 git worktree —— 独立文件夹名 → 独立 cwd-slug → transcript / worker-memory
#       天然隔离(方法/cycle 之间不串味);基座 playground-base 自己永不被改(只当 worktree 源)。
# 导出给 launcher:RUN(=结果目录名/worktree名)、WT(worktree 路径)、OUTDIR、JSONL、PIDFILE。
# worker 的 claude 调用由各 launcher 在 `cd "$WT"` 后自己写(naive 用 SEED、/goal 用 /goal+CONDITION)。
BASE="$ROOT/playground-base"
RUN="${RUN_NAME:-$METHOD}"                 # run-name:默认=方法名;多 cycle 传 naive_cycle2 / goal_cycle2 等
WT="$ROOT/worktrees/$RUN"
OUTDIR="$ROOT/results/$RUN"
export JSONL="$OUTDIR/run.jsonl"
export PIDFILE="$OUTDIR/run.pid"

[ -d "$BASE/.git" ] || { echo "❌ 干净基座不在: $BASE" >&2
  echo "   克隆: git clone https://github.com/silencelamb/playground-base.git $BASE" >&2; exit 1; }
# 防覆盖:① worktree 已存在(上次没 finish)② 结果已归档(别冲掉)→ 拒跑,提示换 cycle 名
[ -e "$WT" ] && { echo "❌ worktree 已存在: worktrees/$RUN(上次没 finish?)" >&2
  echo "   先 bash $ROOT/scripts/finish_run.sh $RUN,或换个 run-name 跑。" >&2; exit 1; }
[ -f "$OUTDIR/result.csv" ] && { echo "❌ results/$RUN 已有结果。换个 run-name,如: bash scripts/launch_${METHOD}.sh ${METHOD}_cycle2" >&2; exit 1; }

mkdir -p "$OUTDIR" "$ROOT/worktrees"
# CLAUDE.md(worker 手册)不进 worker.patch:写共享 exclude(对所有 worktree 生效),idempotent
grep -qxF CLAUDE.md "$BASE/.git/info/exclude" 2>/dev/null || echo CLAUDE.md >> "$BASE/.git/info/exclude"
# ⚠️ worktree = 基座 HEAD 的快照,只含【已提交】内容。基座若留了未提交改动(改框架头 / 加文件没 commit),
#    这些不会进 worktree → 可能"上个 run 能编、这个缺文件编不过"的假差异。软警告(不拦):基座该改动须 commit 进 playground-base。
_DIRTY="$(git -C "$BASE" status --porcelain 2>/dev/null)"
[ -n "$_DIRTY" ] && { echo "⚠️ playground-base 有未提交改动(worktree 用 HEAD、不含它们;build 若依赖请先 commit 进基座):" >&2
  echo "$_DIRTY" | head -5 | sed 's/^/     /' >&2; }
# 🧹 主修(防 worker 跨-run 串味):每 launch 前清掉 base-slug 的 worker auto-memory → 每 run 白板起步。
#    背景:Claude auto-memory 按【git 仓库】键 → 所有 worktree 与 base 共享 base slug 那份 worker-memory(上个 worker 写、下个读 = 抄答案)。
#    这样既挡跨-run 抄答案,又保留 worker 在【本 run 内】写/读 memory(跨 context 压缩持久化,见下 IS_SANDBOX 后注释)。
#    ⚠️ 只清 worker 的(base slug),【不动】编排 memory(VibeKernel slug = .claude/memory)。
_BASESLUG="$(printf '%s' "$BASE" | sed 's/[^[:alnum:]]/-/g')"
[ -e "$HOME/.claude/projects/$_BASESLUG/memory" ] && rm -rf "$HOME/.claude/projects/$_BASESLUG/memory" && echo "  🧹 已清 worker auto-memory(base slug)→ 本 run 白板"
git -C "$BASE" worktree add --detach "$WT" HEAD >/dev/null || { echo "❌ worktree add 失败" >&2; exit 1; }
cp "$ROOT/CLAUDE_For_KernelAgent.md" "$WT/CLAUDE.md"   # 刷新 worker 手册(每次启动都 cp 最新版)
if   [ -f "$ROOT/env.sh" ];         then source "$ROOT/env.sh"          # 本地 GPU 绑定(不入库,优先)
elif [ -f "$ROOT/env.sh.example" ]; then source "$ROOT/env.sh.example"  # 回退:别人 clone 后没建 env.sh 也能跑
     echo "  ⚠️ 未找到 env.sh,回退用 env.sh.example(CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-?});请 cp env.sh.example env.sh 并改成你的 GPU"
fi
export IS_SANDBOX=1                                    # root@docker 绕过 root guard
# auto-memory:【保持开启】——让 worker 在单个【长 session 跨 context 压缩】时把关键发现记进 memory、压缩后取回(run 内持久化,长跑很有用)。
#   跨-run 串味由上面那条 rm(每 launch 前清 base-slug memory)挡:每 run 白板起步、跑中写自己的、下个 run launch 前再清掉。两头都要。
#   (若哪天想连 run 内 memory 也彻底关=每步纯无状态,解开下一行;代价是丢压缩后的发现持久化,慎用。验于 bundle 2.1.167。)
# export CLAUDE_CODE_DISABLE_AUTO_MEMORY=1
: > "$JSONL"
