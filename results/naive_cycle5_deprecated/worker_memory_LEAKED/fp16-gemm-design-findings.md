---
name: fp16-gemm-design-findings
description: "What works/doesn't for the A100 fp16 Tensor-Core GEMM kernel (task-1 matmul_f16)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 9ce2dce9-fcc9-4bef-a082-2f0acb6820fc
---

A100 fp16 GEMM (`task-1/src/matmul_f16/`, mma.sync hand-written). Progression of the 100-round task1 score:
- v1 WMMA baseline: 64 TFLOPS
- v2 mma.sync m16n8k16 + ldmatrix + 2-stage double buffer: 96
- v3 multi-stage cp.async pipeline (wait_group<STAGES-2>): 119
- v4 bigger warp tile 64×64 (block 128×256, 8 warps): 137
- v5 register double-buffer of fragments (prefetch 1 substep ahead): 157
- v6/v7 BK=64 + ldmatrix.x4.trans for B (2 n-tiles/instr): 171
- v9/v11 precompute ldmatrix smem addresses: ~172
- **v15 COALESCED smem-staged epilogue: ~178 (BEST), err ~4e-5**

**Winning config: block 128×256, warp tile 64×64, 8 warps (256 thr, 1 block/SM), BK=64, STAGES=3 cp.async, register double-buffer, fp32 accumulate, PAD=8 smem (0 bank conflicts), + coalesced epilogue. ~178 TFLOPS.**

**Epilogue lesson (big, non-obvious):** the naive per-thread C store (scattered 4-byte half2 writes from the mma D-fragment layout) ran at ~150 GB/s and cost ~5% of runtime. Fix: stage each warp's 64×64 tile through shared mem ([64][WARP_N+8] padded, conflict-free), `__syncwarp`, then write C as coalesced 128B/row (each lane writes 1 float = 2 cols/row). +3% (173→178). Needs `cp.async.wait_group 0` + `__syncthreads` before reusing the A/B smem for staging. WARNING: a diagnostic that skips stores OVER-reports epilogue cost — the compiler dead-code-eliminates the now-unused mma, so it looks like ~22% when the real store cost was ~5%.

What did NOT help (measured worse):
- More occupancy: 16 warps via warp 64×32 (v8=151) or 2 blocks/SM (v10=128) or fp16-acc 2-block (v13=163). This kernel is NOT occupancy-limited; register-DB at 8 warps wins.
- fp16 accumulate (v12=160, err 0.017): same mma rate on A100 but slower here + worse accuracy. Only frees regs (128→64 C regs) but no payoff.
- Manual ldmatrix/mma interleave: 160 (compiler schedules better).
- Address precompute: neutral (compiler already strength-reduces).
- 256×128 instead of 128×256: 164 (slightly worse).
- Threadblock rasterization/L2 swizzle (v14): 169 — not memory-bound, no help.
- **Stream-K (v16, hybrid: DP blocks + K-split leftover tiles + half2 atomicAdd into pre-zeroed C): 165, WORSE.** The wave tail is only ~5.5% (512 blocks / 108 SMs = 4.74 waves) but the stream-K machinery (partial-tile prologue/epilogue per split, atomic reduction, extra syncs) costs MORE than the tail saving — the 15.6% leftover mma takes >15.6% of time. Coalesced-atomic epilogue didn't rescue it. Conclusion: tail not worth fixing for this shape. ncu flags "wave 20%" but real waste is ~5%.
- Deeper register pipeline (v17, fp16-acc + 3-deep frag ring, prefetch 2 substeps ahead): FAILED (NaN bug + tensor pipe collapsed to 19% — extra syncs/register-shuffle serialize). Deeper pipelining is not the lever.
- L2 persistence (access policy window pinning A in L2): neutral (not memory-bound, long_scoreboard only 0.31).
- Compiler flags: `-rdc=false`, ptxas `--allow-expensive-optimizations` — all neutral. tile-loop `#pragma unroll 2`: 167 (worse).

**SASS confirms ptxas scheduled the inner loop near-optimally:** `HMMA.16816.F32 R4, R132.reuse, R136, R4` — the A operand carries `.reuse` (operand-reuse-cache, shared across the 8 nj-mma), LDSM/address ops interspersed 1-2 per HMMA (hidden in the HMMA shadow). Hand-PTX wouldn't beat it. **Pipe metrics: every pipe <20% issue rate** → pure latency exposure at register-locked 8-warp / 1-block / 12.5%-occupancy. The wall is fundamental: warp-64×64 fp32 needs 128 C-regs, every occupancy-increasing perturbation (smaller tile / fp16-acc / 2-block) trades away more than the latency-hiding gains. **178 TFLOPS (70% of the 1155 MHz sustained peak) is the validated hand-optimization optimum.**

**The wall:** warp 64×64 needs 128 fp32 C-regs → with register-DB the kernel sits at 254/255 regs, 1 block/SM, 12.5% occupancy. ncu: tensor-pipe 69%, bound by `wait` (HMMA fixed-latency dependency, 3.3 cyc/issue). Can't add warps (reg-locked) and can't deepen the register pipeline (no reg headroom). Power-of-2 tiles force warp 64×64 as the max (C=128 regs). At 1155 MHz sustained ([[a100-sustained-clock]]) 172 ≈ 67% of real peak.

Build: edit a `.cu` in matmul_f16/, `./task1.sh run --float f16 --ver N`. All .cu in the dir are GLOB-compiled together, so every version must stay compilable. Sanitize new kernels: `compute-sanitizer --tool memcheck ./build/src/task1_float16_vN -m 512 -n 512 -k 512 -t 1`.
