---
name: vibekernel-task1-environment
description: "VibeKernel task-1 GPU environment facts — GPU clock-locked at 1155MHz, effective fp16 ceiling 255.6 TFLOPS, scoring quirks"
metadata: 
  node_type: memory
  type: project
  originSessionId: 10777a72-f54c-422d-b4f6-f660731b3d10
---

VibeKernel task-1 (A100 fp16 GEMM) environment facts, measured 2026-06-10:

- Assigned GPU (A100-SXM4-80GB) is **clock-locked at 1155 MHz** (likely `nvidia-smi -lgc` by operators): under 99% util / 311W load, `clocks_event_reasons.active = 0x0` (no throttle flag) yet SM stays 1155; never boosts to 1410 even after 12s sustained. Power limit is the full 400W.
- Effective fp16 Tensor Core ceiling on this card = 312 × 1155/1410 = **255.6 TFLOPS**. All TFLOPS scores must be judged against 255.6, not 312.
- Neighboring GPUs (2-7, other users) run at 1380-1410 under load — do NOT touch their clocks or use them.
- Test harness: `[Playground]` binary does CPU cBLAS ground truth (~4-8 s) before the GPU burst; when sampling nvidia-smi during a run, the GPU burst starts ~4-5 s in and a 100-round burst lasts only ~80 ms.
- fp16-accumulator GEMM error metric fluctuates 0.017–0.033 across random seeds (mean rel err, small-|GT| entries inflate it); README blesses "~1e-2 magnitude, ≲0.02 normal". fp32-acc gives ~3.5e-05 but costs ~3.5% perf.

Related: [[vibekernel-task1-ptxas-lessons]], [[vibekernel-task1-best-config]]
