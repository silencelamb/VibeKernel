---
name: gpu-clock-locked-1155
description: A100 runs at fixed 1155MHz base clock (not 1410 boost) — true fp16 peak is ~256 TFLOPS not 312
metadata: 
  node_type: memory
  type: project
  originSessionId: 25b7c1cd-dcac-4117-a64e-c5675facbd9c
---

The task runs on an A100-SXM-80GB. Under sustained load the SM clock stays pinned at **1155 MHz (base)**, NOT the 1410 MHz boost the 312-TFLOPS target assumes. `nvidia-smi -lgc 1410`/`-ac` report "All done" but DO NOT take effect (admin/cluster policy caps at base; power was 345W<400W limit, temp ~42°C, throttle reason 0x0 — so it's a policy cap, not power/thermal).

**Consequences:**
- Real achievable fp16 peak = 312 × 1155/1410 = **~256 TFLOPS**. So a kernel at 194 TFLOPS is **76% of achievable peak**, not 62.5% of 312.
- The 312 target is unreachable here (would need 1410). Optimize for % of the 1155-peak; that efficiency is clock-independent so the best kernel wins regardless.
- Old log variance (v7 ranged 173-197) is mostly clock/contention noise, NOT kernel differences.

**For clean comparison during dev**: `nvidia-smi -lgc 1155,1155` pins it → stable ±0.5% measurements. Reset with `nvidia-smi -rgc` at the end so grading isn't capped by my lock. See [[gemm-f16-bottleneck-analysis]].
