#!/usr/bin/env bash
# ncu-doctor — 诊断 Nsight Compute 能否用,定位 ERR_NVGPUCTRPERM 的两道闸门。
# 用法: ./scripts/ncu-doctor.sh [可选:一个能跑的 task1 二进制做真实测试]
#   例: ./scripts/ncu-doctor.sh ./build/src/task1_float16_v5
set -uo pipefail

echo "== 闸门1: 进程有没有 CAP_SYS_ADMIN =="
CE=$(grep CapEff /proc/self/status | awk '{print $2}')
if (( 0x$CE & 0x200000 )); then echo "  [PASS] 有 cap_sys_admin"; CAP=1
else echo "  [FAIL] 无 cap_sys_admin  (CapEff=0x$CE = docker 默认)"; CAP=0; fi

echo "== 闸门2: 驱动 RmProfilingAdminOnly =="
AO=$(grep -i RmProfilingAdminOnly /proc/driver/nvidia/params 2>/dev/null | grep -oE '[0-9]+$')
echo "  RmProfilingAdminOnly = ${AO:-未知}"
if [ "${AO:-1}" = "0" ]; then echo "  [PASS] 驱动对所有用户放开 profiling"; else echo "  [限制] 仅 admin 可 profiling"; fi

echo "== 判定 =="
if [ "${AO:-1}" = "0" ] || [ "$CAP" = "1" ]; then
  echo "  ✓ 闸门满足其一,ncu 应当可用"
else
  echo "  ✗ 两道全关 → 必 ERR_NVGPUCTRPERM"
  echo "    修法A: docker run 加 --cap-add=SYS_ADMIN 重启容器"
  echo "    修法B: host 设 NVreg_RestrictProfilingToAdminUsers=0 后重载模块/重启"
fi

BIN="${1:-}"
if [ -n "$BIN" ] && [ -x "$BIN" ]; then
  echo "== 实测: ncu 对 $BIN 取 1 个指标 (-t 1) =="
  ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \
      --launch-count 1 "$BIN" -t 1 2>&1 \
    | grep -iE 'ERR_NVGPUCTRPERM|permission|sm__throughput' | head -5 \
    || echo "  (无匹配输出,完整跑一遍看错误)"
else
  echo "== 跳过实测 (传一个 task1 二进制路径可做真实 ncu 测试) =="
fi
