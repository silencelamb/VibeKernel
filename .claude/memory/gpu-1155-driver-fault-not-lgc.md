---
name: gpu-1155-driver-fault-not-lgc
description: "GPU 卡 1155 的真因是 NVRM 驱动 assertion(SM throttle ctrl NULL)非 -lgc 软锁,nvidia-smi 解不掉,只能 GPU reset/驱动重载"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2e810630-0fad-4470-a9c5-1c2e554af914
---

2026-06-10 实测确认:评测 GPU 满载死钉 **1155 MHz** 的真正原因**不是** `-lgc` 软锁(之前 [[naive-fable-cycle1-result]] / [[ncu-base-clock-vs-bench-boost]] 都误以为是可解的锁频),而是**驱动级故障**。

**证据链:**
- 满载 100%util/364W 仍死钉 1155,完全不 boost。
- `nvidia-smi -rgc`(reset locked clocks)、`-rac`、甚至强制 `-lgc 1410,1410` 全部"All done"但**无效**,满载照样 1155 → 时钟控制平面坏了,nvidia-smi 运行时命令盖不掉。
- 满载时**所有 throttle reason 全 Not Active**(bitmask 0x0)→ 不是 throttle,是 control-plane 初始化失败。
- ECC 全 0、无 remapped rows、无 repair pending → 不是显存/硅片降级。
- **smoking gun**:dmesg 刷屏 `NVRM: nvAssertFailedNoLog: Assertion failed: pStaticInfo->pSmIssueThrottleCtrl != NULL @ kernel_graphics.c:3369`,出现 **408 次**。SM Issue Throttle Control 的 GSP static info 是 NULL → 驱动无法管理 SM 时钟 → 钳在默认 app clock 1155。

**含义(改写历史结论):** 故障态跑出的绝对 TFLOPS 比 1410 boost 低 ~18%,且这是**故障态不是可控的 base-clock**;凡是在故障态下跑过的 run(naive_fable cycle1 等)绝对值都带这个 -18% 病态偏置,只能比"占自身天花板的%"。

**修复:** 只能清 GSP/驱动 wedged 状态。2026-06-10 实测 `nvidia-smi -r` 返回 **"Not Supported"**,在线 reset 不可用,只能驱动重载(rmmod nvidia*)/ reboot(对 wedged GSP 最稳)。在那之前修不了,只能按 255.6 真天花板折算带病跑。vbios 92.00.94.00.04,driver 580.65.06,CUDA 13.0。
