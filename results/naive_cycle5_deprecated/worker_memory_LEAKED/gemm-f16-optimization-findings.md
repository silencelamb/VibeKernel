---
name: gemm-f16-optimization-findings
description: BREAKTHROUGH config for A100 fp16 GEMM — 218 TFLOPS (f16-acc 2-block + shared epilogue)
metadata: 
  node_type: memory
  type: project
  originSessionId: 32bf3829-27d6-49f9-b1ad-e9eb3f51e108
---

**Best: ~218 TFLOPS, f16 accumulate, err ~0.018** (task1 100-round, 4096³) = `matmul_f16_v8_epi.cu`. Far beats the prior 172 ([[fp16-gemm-design-findings]]). 85% of the 256 sustained peak ([[a100-sustained-clock]]). Safe f32 alternative: ~188 TFLOPS, err 4e-5 = `matmul_f16_v10_f32burst.cu` (also has the shared epilogue).

**Shared-memory epilogue (+4%, 209→218 / f32 178→188):** the mma D-fragment layout makes direct global C stores scattered (~16 B, ~50% bus eff). After the mainloop As/Bs smem is free — reuse it: each warp writes its 64×64 acc to `Cs[BM][BN+8]` (pad→conflict-free) by the mma layout, `__syncthreads`, then ALL threads copy Cs→C as coalesced float4 (128 B). 2 extra one-time barriers, big win on the write phase. v8 = v7 + this.

**TWO earlier key insights (each ~+25 TFLOPS):**

1. **cp.async BURST, not interleaved.** Issue the next tile's cp.async as a single burst AFTER the k-step mma loop (single-barrier multistage), NOT spread across the mma loop. Interleaving steals scheduler slots from the dense HMMA stream; burst lets HMMA issue back-to-back and the cp.async *latency* is hidden by the next iteration (it's async). f32 151→180.

2. **f16-accumulate to reach 2 blocks/SM with a 64×64 warp tile.** The wall for f32 is `wait` (HMMA accumulate latency) at 1 block/SM — a 64×64 warp needs 128 fp32 C-regs so only 1 block fits. f16-acc halves C-regs to 64 → the kernel fits 128 regs → with BK=32 (shared 81KB) TWO blocks/SM fit, which HIDES the wait stall (tensor 62%→74%). f16 154→209. (Occupancy DOES matter here — the prior session's "not occupancy-limited" was true only for their 1-block f32.)

**Winning v7 config:** block 128×256, warp 64×64, 8 warps/block, **2 blocks/SM** (`__launch_bounds__(NTHREADS,2)`, 128 regs, no spill), **BK=32** (shared 81KB = exactly 2 blocks), STAGES=3, **f16 accumulate**, register-DB (frag prefetch 1 k-step ahead), single-barrier burst multistage (prologue loads STAGES-1; loop: `cp_wait<STAGES-2>; __syncthreads(); reg-DB mma over KSTEP; burst-prefetch tile kt+STAGES-1 into buffer (kt-1)%STAGES`), PAD=8, carveout=MaxShared. Race/memcheck clean.

**Dead ends (all re-confirmed worse):** interleaved cp.async; accumulator splitting; triple-buffer frags; precompute ldmatrix addrs (neutral); 64×32 warp at 2-3 blocks (mio-bound, ~155); 128×128 at 3 blocks (L2/stall-bound, 110); BK=16 (256 barriers, 93); BK=64 at 2 blocks (shared won't fit). 64×64 warp + 2 blocks is THE sweet spot; bigger BK or bigger tile drops you to 1 block.

**Remaining wall (best v8 ~218):** `wait` 3.4 (HMMA latency) + `barrier` 2.75 @ 2 blocks/SM, tensor 77%, L2 64% (not yet saturated, has headroom), sustained 1155MHz verified during the run. Can't get 3 blocks (needs ≤85 regs AND ≤54KB shared — incompatible with 64×64+128×256). f32 path tops out at 188 (1-block wait-bound). isinf in ~1/8 scoring runs is a TEST artifact (GT exact-0 → /0), affects all kernels.

**mbarrier analysis (don't bother without warp-spec):** replacing `cp_wait+__syncthreads` with cp.async.mbarrier won't cut the `barrier` stall here — every warp needs the freshly-loaded tile immediately after the barrier (no independent work to overlap, no producer/consumer split), so the cross-warp wait is identical. mbarrier only pays off WITH warp specialization (dedicate warps to cp.async vs mma), which is a big rewrite with modest Ampere ROI (cp.async is already async). That + persistent/tile-stationary L2 reuse are the only remaining structural levers to chase >85% peak.
