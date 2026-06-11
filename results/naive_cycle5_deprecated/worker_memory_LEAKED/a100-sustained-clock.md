---
name: a100-sustained-clock
description: "A100 in this container runs sustained at 1155 MHz, not the 1410 MHz boost — real fp16 TC peak is ~256 TFLOPS"
metadata: 
  node_type: memory
  type: project
  originSessionId: 9ce2dce9-fcc9-4bef-a082-2f0acb6820fc
---

The A100-SXM4-80GB in this playground container runs at **1155 MHz SM clock during sustained kernels** (the task1 100-round average), not the 1410 MHz max boost. Verified via `nvidia-smi --query-gpu=clocks.sm` sampled during a 100-round run: steady 1155 MHz at ~315W / 35°C (not thermal/power throttled — likely a fixed application clock).

Implication: the nominal "312 TFLOPS at 1410 MHz" peak is unreachable at the sustained clock. The **real ceiling is 312 × 1155/1410 ≈ 256 TFLOPS**. So a kernel reporting e.g. 172 TFLOPS is at ~67% of the achievable sustained peak (matches the ~69% tensor-pipe-active that ncu reports), even though it's only ~55% of the nominal 312.

Do NOT change clocks (would game the sustained benchmark; against task spirit). Optimize tensor-pipe utilization instead. Short ncu/burst runs may show 1410 MHz (inflated) — always trust the task1 100-round number. See [[fp16-gemm-design-findings]].
