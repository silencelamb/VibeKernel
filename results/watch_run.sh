#!/usr/bin/env bash
# watch_run — 实时跟看一个 headless worker 的进展(在另一个终端跑)。
# 渲染 assistant 正文 / 工具调用,并高亮 RESULT_JSON 标记与 task1 计分行。
# 用法: ./results/watch_run.sh results/<方法>.jsonl
set -uo pipefail
JSONL="${1:?用法: watch_run.sh <stream-json重定向文件>}"
tail -n +1 -f "$JSONL" | jq -r --unbuffered '
  if .type=="assistant" then
    ( .message.content[]? |
        if .type=="text" then
          ( if (.text|test("RESULT_JSON")) then "[1;32m"+.text+"[0m" else .text end )
        elif .type=="tool_use" then "[36m🔧 "+.name+"[0m  "+((.input|tostring)[0:200])
        else empty end )
  elif .type=="user" then
    ( .message.content[]? | select(.type=="tool_result") | (.content|tostring)
      | if test("\\[Playground\\] Result") then "[1;33m   ↳ "+(capture("(?<l>\\[Playground\\] Result[^\\n]*)").l)+"[0m"
        else "   ↳ "+(.[0:160]) end )
  elif .type=="result" then
    "\n[1;35m===== END  "+(if .is_error then "❌ ERROR "+((.result//"")|tostring|.[0:80]) else "✅ ok" end)+"  (subtype="+(.subtype//"")+" stop="+(.stop_reason//"?")+")  "+((.duration_ms//0|tostring))+"ms  out_tok="+((.usage.output_tokens//0|tostring))+" =====[0m"
  else empty end' 2>/dev/null
