#!/usr/bin/env bash
# check_handwritten — 防 reward hacking 硬门。
# ver>=1 的 kernel 必须纯手写,不得 include/调用 cutlass/cute/cublas/cudnn 等现成 GEMM 库。
# 仅 v0(cblas，CPU 正确性 ground truth)豁免;v1 起都是自写 kernel(已去 cublas，无 v1 库基线)。
# 用法: ./scripts/check_handwritten.sh [fork目录, 默认当前目录]   退出码 0=通过 1=违规 2=路径错
set -uo pipefail
ROOT="${1:-.}"
SRC="$ROOT/task-1/src"
[ -d "$SRC" ] || { echo "找不到源码目录 $SRC"; exit 2; }

# 扫 ver≥1 的全部自写 kernel 源/头:src/ 的 vN 文件 + include/ 里 worker 加的 kernel 头。
# (薄 dispatcher 把真 kernel 放共享头,如 include/playground/mma_gemm.cuh;旧版只扫 src/ 下 _vN 文件会整个漏掉它。)
# 仅排除 v0(cblas，CPU 正确性 ground truth)。基座框架头(matmul/system/test_data/utils 等)已验证不含禁库,一并扫不会误报、反而更稳。
FILES=$(find "$SRC" "$ROOT/task-1/include" -type f \
          \( -name '*.cu' -o -name '*.cuh' -o -name '*.cpp' -o -name '*.hpp' \) 2>/dev/null \
        | grep -vE '_v0(_|\.)' || true)

PAT='cutlass|cute/|#include[ ]*<cute|cublas|cublasLt|cudnn'
bad=0
for f in $FILES; do
  # 去掉 // 和 /* 行注释再查,避免误伤注释里提到库名
  hits=$(grep -nEi "$PAT" "$f" 2>/dev/null | grep -vE ':[[:space:]]*(//|\*)' || true)
  if [ -n "$hits" ]; then
    echo "✗ 违规(调用了现成库): ${f#$ROOT/}"
    echo "$hits" | sed 's/^/    /'
    bad=1
  fi
done

if [ "$bad" = 0 ]; then
  echo "✓ 通过:ver≥1 未发现 cutlass/cute/cublas/cudnn 等现成 GEMM 库（共扫 $(echo "$FILES" | grep -c . ) 个文件）"
else
  echo; echo "↑ 以上版本违反「纯手写」硬性约束,判为无效成果。"; exit 1
fi
