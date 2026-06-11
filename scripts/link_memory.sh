#!/usr/bin/env bash
# link_memory — 让 Claude 的自动记忆随本仓库走(git 维护、换容器也在)。
# 把 repo/.claude/memory 软链到 Claude 实际读取的 ~/.claude/projects/<cwd-slug>/memory。
# 由 .claude/settings.json 的 SessionStart hook 自动调用;也可手动跑(换容器后跑一次最稳)。
set -uo pipefail
REPO="${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC="$REPO/.claude/memory"
[ -d "$SRC" ] || exit 0                          # 本项目没有 git 维护的 memory → no-op(对 worker fork 安全)

SLUG=$(printf '%s' "$REPO" | sed 's#[^a-zA-Z0-9]#-#g')  # 非字母数字全转 -(含 / _ .),与 Claude slug 一致
DST="$HOME/.claude/projects/$SLUG/memory"
mkdir -p "$HOME/.claude/projects/$SLUG"

[ -L "$DST" ] && exit 0                           # 已是软链,完事
if [ -e "$DST" ]; then                            # 是真目录:把其中新文件并回仓库,再替换成软链
  cp -an "$DST"/. "$SRC"/ 2>/dev/null || true
  rm -rf "$DST"
fi
ln -s "$SRC" "$DST"
echo "linked $DST -> $SRC"
