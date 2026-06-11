---
name: gemm-f16-best-kernel
description: Best A100 fp16 GEMM kernel config and the key m16n8k8 register trick
metadata: 
  node_type: memory
  type: project
  originSessionId: 7e2a74fb-3baf-4b2f-b58b-4cc8b4757a93
---

Task-1 fp16 GEMM on A100 (task1 100-round scoring). Best kernel so far:
**v25 (`matmul_f16_v25_cspad.cu`) ≈ 188 TFLOPS, error ~0.02** (60% of 312 peak).
v25 = v24 + padded Cs staging stride (BN+8) to kill the 8-way bank conflict on
the epilogue staging write. v24 = v23 + coalesced epilogue (stage BM×BN output
through smem, write global C as 128-bit float4 rows; C-store 50%→100% sector eff).
Epilogue work total +5% (179→188), found via `ncu --set full`. KEY LESSON:
reducing memory traffic helps even when "compute-bound" — the inefficient store
was contending for bandwidth with the next wave's cp.async loads.

Winning config: block 128×256, warp 64×64, BK=32, 8 warps (256 threads),
3-stage `cp.async` pipeline, f16 accumulate, 2 blocks/SM (~25% occ), padded smem
(APAD=BPAD=8; padding alone is conflict-free for these strides — swizzle gave nothing).

**Key non-obvious unlock — use `mma.m16n8k8` not `m16n8k16`:** k8 needs HALF the
operand registers (A=2 regs, B=1 reg per tile vs 4,2). That's what lets a full
cross-stage operand double-buffer fit in the 128-reg / 2-block budget. With k16
the 64-reg accumulator + 64-reg double-buffered operands = 160 regs → only 1
block (spills if forced to 128). k8 → ~112 regs → 2 blocks WITH the pipeline.
This broke a hard 160-TFLOPS plateau (every k16 variant landed 137-160).

Other findings: kernel is latency-bound, not memory-bound (L2 ~63%, DRAM ~12%);
occupancy is hard-capped at 2 blocks by the 64-reg accumulator (3 blocks needs
≤85 regs, impossible). Compute ceiling at this occupancy ≈231 (pure-HMMA probe).
Smaller tiles (more occ) lose to smem traffic; 1-block full-pipeline loses to too
few warps; threadblock-swizzle and cp.async.ca both hurt/neutral. See [[gemm-f16-tuning-deadends]].
