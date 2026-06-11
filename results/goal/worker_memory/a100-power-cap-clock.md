---
name: a100-power-cap-clock
description: "The task GPU is power-capped to ~1155MHz, so real fp16 peak is ~256 TFLOPS not 312"
metadata: 
  node_type: memory
  type: project
  originSessionId: 5a95c572-9465-4ad5-9a57-4a886883fdab
---

The bound GPU has a **400W power limit**. A hard fp16 tensor kernel draws ~376-407W and triggers **SW Power Cap throttling** (nvidia-smi throttle reason `0x4`), holding the SM clock down. **The throttled clock VARIES by session/thermal state: observed 1155 MHz in one session, ~1230 MHz in a later one (2026-06 naive_cycle2 run, power pinned at 400-407W).** So the real fp16 peak is ~256 TFLOPS @1155 or ~272 @1230 — measure the actual clock each session via a long sustained run (`-t 4000`) sampling `nvidia-smi -i $CUDA_VISIBLE_DEVICES`, NOT a short run (which is dominated by the CPU ground-truth phase where the GPU idles at 210MHz).

**Consequence:** the 312 TFLOPS "peak" (defined at 1410 MHz) is **physically unreachable** under sustained load. The real ceiling is **312 × 1155/1410 ≈ 256 TFLOPS**. So the best fp16 kernel (v17, 205.6 TFLOPS) is at **~80% of the achievable hardware ceiling**, not 66% of 312. The throttle is a *consequence of high tensor utilization* — lighter kernels stay at 1410 MHz but do less work.

**How to apply:** (1) When sampling clocks/power, query the bound GPU (`-i $CUDA_VISIBLE_DEVICES`). (2) Judge kernel efficiency against ~256 TFLOPS, not 312. (3) Reducing the mma-latency "wait" yields less than expected because more tensor activity → more power → lower clock (a power feedback loop). The cleaner wins are energy-per-flop reductions (less memory traffic) and recovering the ~5% wave-quantization tail (stream-K). Do NOT change the power limit (no permission; would alter the eval environment). See [[fp16-gemm-best-kernel]].
