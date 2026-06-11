---
name: always-check-error-not-just-tflops
description: "When sweeping GEMM kernel variants, always read BOTH TFLOPS and Average Error — a faster number may be a correctness bug"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 25b7c1cd-dcac-4117-a64e-c5675facbd9c
---

When benchmarking GEMM kernel variants, ALWAYS check the Average Error alongside TFLOPS. A "faster" config can be a silent correctness bug.

**Why:** Concretely got fooled by `cp_async_wait<STAGES-1>` (allowing 3 cp.async groups in flight): it grepped 195 TFLOPS (vs 194 baseline) and looked like a free win — but it was a race (reads a smem stage before its cp.async completed), Average Error 0.11-0.47 (garbage). The correct threshold is `cp_async_wait<STAGES-2>`.

**How to apply:** In sweep scripts, print TFLOPS AND err on the same line; reject any result with err ≳ 0.02 (f16-acc) or ≳ 1e-3 (f32-acc) as broken, no matter how fast. The f32-accumulate kernels should land at ~3e-5. See [[gemm-f16-bottleneck-analysis]].
