---
name: vibekernel-task1-best-config
description: "Best fp16 GEMM config so far (v9, 211 TFLOPS) and the proven-optimal design point for A100 sm80"
metadata: 
  node_type: memory
  type: project
  originSessionId: 10777a72-f54c-422d-b4f6-f660731b3d10
---

VibeKernel task-1 best kernel as of 2026-06-10: **v9 (`matmul_f16_v9_tune.cu`) ≈ 211.2 TFLOPS** (10 runs: 210.7-211.9), error 1.7-3.2e-2 (fp16 acc), 82.6% of the 255.6 ceiling. TC-pipe-active 89.07%, REG 168, zero bank conflicts, zero spills. Insurance variant v13 (fp32 acc, same machinery): 201.0 @ 3.4e-05 error.

Per-SASS-instruction stall attribution (ncu SourceCounters): top stalls are the first B-LDSM of each k-step (2.7k samples) and the first 1-2 HMMAs consuming it (4.3k) — the k-step handoff. fr[4] (4-deep frag buffers, compile-time indices) was tried to fix it: REG 168→198, perf 211→199 — extra register pressure beats the WAR relief; fr[2] ping-pong is optimal. Remaining ~5-7%% in-loop gap needs SASS-level scheduling (no assembler in env).

Proven-optimal design point (every neighbor measured worse):
- CTA 128×256, BK=64 (4 mma k-steps per barrier), 3 stages × 48KB = 144KB dynamic smem, 1 CTA/SM, 8 warps, warp tile 64×64.
- `mma.m16n8k16` fp16-accumulate (fp32 acc = −3.5% perf, use only if error budget tightens).
- cp.async.cg 16B + L2::128B hint, XOR-swizzled smem (A: row=1 seg of 8 chunks `c^(r&7)`; B: 4 segs of 8, `(c&24)|((c&7)^(r&7))`).
- Register frag double-buffer, cross-tile handoff, ONE `__syncthreads` per K-tile at ks3 (wait_group(1) before it).
- cp.async sliced A@ks0/B@ks1/B@ks2, commit at ks2.
- Rotating stage pointers; B-fragments loaded before A everywhere (first HMMA needs b[0]).
- Epilogue staged through padded smem (CSTRIDE=BN+8), 16B coalesced stores.
- Grouped raster GROUP_M=16 (L2 hit ~93%).

Why the walls are where they are: more warps ⇒ smaller warp tiles ⇒ smem-BW wall (128B/cy); bigger warp tiles ⇒ register wall (255); more stages ⇒ 164KB smem wall; 2 CTA/SM impossible at 144KB. Remaining gap is ptxas scheduling quality (~11% in-loop) + end-of-kernel straggle (~5%) + launch gaps (~1%); cuBLAS-class SASS would be ≈225-230 here.

Related: [[vibekernel-task1-environment]], [[vibekernel-task1-ptxas-lessons]]
