---
name: gemm-f16-tuning-deadends
description: A100 fp16 GEMM optimizations that were tried and did NOT beat the best
metadata: 
  node_type: memory
  type: project
  originSessionId: 7e2a74fb-3baf-4b2f-b58b-4cc8b4757a93
---

For task-1 fp16 GEMM (see [[gemm-f16-best-kernel]], best ≈179 TFLOPS). Things
tried that did NOT help (so future sessions don't repeat):

- **3 blocks/SM**: impossible for 64×64 tile — 64-reg f16 accumulator forces ≤2
  blocks; forcing ≤85 regs (launch_bounds ,3) spills the accumulator → ~79 TFLOPS.
- **Smaller warp tiles (64×32, 32×64) for more occupancy**: memory-bound — their
  mma:ldmatrix ratio (2.67) vs 64×64's (4.0) means 1.5× smem/L2 traffic; 3-4
  blocks worth of occupancy can't overcome it (~150 TFLOPS).
- **1 block/SM with full operand double-buffer** (k16, 160 regs): 154 — too few
  warps (8) to hide latency despite a tight pipeline.
- **128×128 block, 4 warps, 4 blocks/SM** (good wave-quant): 143-147 — the forced
  2-stage pipeline + thin 4-warp blocks lose more than quantization gains.
- **mma-before-sync reorder** (overlap barrier): 165-168 — breaks the more
  valuable prefetch→mma overlap. Prefetching the next ldmatrix during the current
  mma matters more than overlapping the barrier.
- **threadblock swizzle (L2)**: neutral — kernel is latency-bound, not L2-bound
  (L2 ~63%, DRAM ~13%, cycles_active 93.5% so wave-quant is a non-issue).
- **cp.async.ca (L1 cache)**: hurts (~164) — streamed GEMM data shouldn't cache in L1; use .cg.
- **lean/incremental addressing, register-trim**: neutral (shifts IMAD↔LEA);
  kernel is latency-bound not issue-bound, so cutting address math barely helps.
- **split-K / persistent / stream-K**: not worth it. Wave-quant on 4096³ is ~4.5-6%
  (confirmed: v25 hits 188 at 4096³ but 197-200 at clean-2-wave sizes like
  3456×4096). But it's FUNDAMENTAL — power-of-2 tiles can't give a clean-wave
  count (mult of 216=8×27) on 108 SMs. Stream-K (the only fix) nets only ~+1%:
  the ~189/512 split-tile reduction overhead (~3.5%) nearly cancels the wave gain.
  Plain split-K is worse (each half-block is half-time → same makespan + reduction
  overhead; atomic split-K=2 measured 147, -22%).
- **cp.async L2 prefetch hints (.L2::128B/.L2::256B)**: neutral (A-load already 89% L2-hit).
- **B-operand XOR smem swizzle**: -4.5% (179.5) — padding is strictly better; the
  swizzle adds addressing overhead and the 64% wavefront efficiency is from row-
  spacing (8 K-rows far apart), not bank conflicts, so swizzle doesn't fix it.

Compute ceiling (pure-HMMA probe, no smem/barrier) ≈230 TFLOPS at this 2-block /
16-warp / 512-chain occupancy; best kernel is at 78% of it. Remaining gap is
barrier (~1.2 cyc/inst) + HMMA-latency "wait" (~2.5) — both fundamental at 16 warps.
