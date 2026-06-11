---
name: gemm-f16-bottleneck-analysis
description: What limits the fp16 GEMM kernel on A100 and which optimization axes are already saturated
metadata: 
  node_type: memory
  type: project
  originSessionId: 25b7c1cd-dcac-4117-a64e-c5675facbd9c
---

ncu analysis of the best kernel (v7-style: mma m16n8k16 + 4-stage cp.async + register double-buffered frags, 128×256×32 block, 64×64 warp tile, 8 warps). At ~194 TFLOPS (76% of the 1155MHz peak, see [[gpu-clock-locked-1155]]):

**Co-limited by Compute AND L2** (ncu SOL: Compute 77%, L2 throughput 71%, "well-balanced"). DRAM only 16% (NOT memory-bw bound). Shared-mem bank conflicts = 0 (the +8 SKEW padding works). No register spills (238 regs/thread, 1 block/SM, occupancy 12.5%, 8 warps). Dominant stall = `stall_wait` 39% (waiting on fixed-latency tensor pipe); `warps_eligible` only 0.40/cyc → too few warps to fully hide HMMA latency.

**Axes already at their sweet spot (don't re-explore):**
- L2→SM traffic = 2·M·N·K·(1/BM+1/BN). Minimized at BM·BN=32768 (=512 blocks) with BM≈BN → **128×256 is L2-optimal**.
- Wave quantization: 512 blocks/108 SMs = 4.74 waves (~5.5% tail). Bigger tiles (256×256) cut L2 33% but drop to 2.37 waves → worse net (measured 170 vs 195). Smaller tiles (128×128) same wave-quant but +33% L2 → 151.
- BK=32/STAGES=4 is peak; BK=64 worse (181), STAGES 3 or 5 worse (188).
- More warps (16) WORSE (170): smaller warp tile kills ILP/intensity. More ILP per warp (4×2=64 mma/substep) = same as 2×4 (195) → 32-way ILP already saturates latency hiding.
- Fewer LDSM via ldmatrix.x4.trans for B (12→8/substep): NO help (192) — issue bandwidth isn't the limit (issue_active 0.28/cyc).
- f16-accumulate (v11): 197 but error 0.027 (>0.02 guideline), marginal, risky. Bigger f16-acc tiles don't help (wave quant).
- Threadblock swizzle/rasterization (v9): WORSE (172) — only helps L2 hit-rate/DRAM, but we're not DRAM-bound; doesn't cut L2→SM bandwidth.

**Microbench ceiling (diagnostic, /tmp/puremma.cu, /tmp/mma_ldsm.cu):** pure-mma loop (frags from regs, no mem) = **246 TFLOPS = 96% of peak@1155**. Adding 12 LDSM/substep (ldmatrix from prefilled smem, no cp.async/sync) = **250** — LDSM is FREE (LSU runs parallel to tensor pipe). So the 21% real-kernel gap (250→194) is the cp.async + syncthreads + wave-quant memory pipeline, NOT LDSM or tensor issue-rate. 8 warps / 64×64 tile DO saturate the tensor pipe.

**Things tried for the 21% gap that FAILED (don't retry):**
- single-buffer frags (v16): 130 — WAR hazard serializes loadFrag→mma; register double-buffer is ESSENTIAL.
- BK=64 double-buffer (v13): 181 — KSTEP=4 unroll pushes loadTile to 255 regs (ceiling) → starved schedule.
- macro-tile sync / sync every TPS=2 tiles at BK=32 (v17): 168-176 — slower; v7's single per-tile sync is already well-pipelined (cheap), the macro structure + deeper STAGES add more overhead than the sync savings.
- Diagnostic: adding a redundant 2nd __syncthreads/tile to v7 → 158 (-18%). So a BADLY-PLACED sync costs ~18% (disrupts the SW pipeline), but v7's existing sync is overlapped/cheap — reducing sync COUNT doesn't help.

**Conclusion:** v7 (~194, 76%) is at the practical ceiling for the mma + multistage-cp.async + reg-double-buffer approach on this clock-capped A100. The residual gap is distributed memory-pipeline overhead, not one fixable bottleneck. Bigger tiles (less L2) always lose to wave-quant; only Stream-K could decouple them (deferred — complex, marginal net for square 4096³).
