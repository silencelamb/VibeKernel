---
name: vibekernel-task1-ptxas-lessons
description: "Hard-won ptxas/codegen lessons for the A100 fp16 GEMM kernels (what breaks register allocation, what was measured and rejected)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 10777a72-f54c-422d-b4f6-f660731b3d10
---

ptxas (CUDA 13.0, sm_80) codegen lessons from VibeKernel task-1 GEMM tuning:

**Why:** the 128-HMMA mainloop sits at a register-allocation cliff; tiny source restructures flip between clean 168-reg code and 255-reg + stack-spill disasters (−25% perf).

**How to apply:**
- NEVER split the mma loop (`domma`) into halves interleaved with ldmatrix calls — every variant (runtime `mi0` arg, template lambda, lo/hi lambdas) caused REG 230-255 + spills or slow scheduling (v5 174, v8 175-194 vs baseline 209-211).
- NEVER unroll the K-tile loop by STAGES with template-lambda instantiations (REG 255 + STACK 392, 57 TFLOPS).
- Two kernel-quality instantiations of the mainloop in ONE kernel (template `<bool SK>` dispatch) poison scheduling globally: TC-active dropped 88→67% even for the path that never executes the loop. Separate `__global__` kernels are the only safe split.
- Runtime loop bounds with `mi0` parameters block unrolling → dynamic `acc[]` indexing → local-memory spill. Compile-time template args fix it.
- ptxas ALREADY does barrier-straddling (rotates the loop so ~32 HMMA precede BAR.SYNC and LDSM follow it). Hand-moving mma before the barrier in source made it WORSE (193.9 vs 211).
- Rotating stage pointers (rotate 3 base addresses at loop end) beats `(kt%3)*STAGE` chains: ptxas lowers them to uniform-register UMOVs, post-barrier LDSM issues immediately. +1.7 TFLOPS.
- Measured and rejected: L2 prefetch of stage+3 (−16), `st.global.cs` epilogue (−9), ni-major mma order (−8), zigzag mma (−2), serpentine raster (−2), bulk loads at ks0 (−5), GROUP_M 8/32 (≈−1), BK=32×6-stage (−13), CUDA-graph launch (±0), stream-K any flavor (−3 to −50: GPU greedy scheduler already smooths the 4.74-wave tail; v7 dual-launch proved the "tail loss" is mostly mirage).

Related: [[vibekernel-task1-environment]], [[vibekernel-task1-best-config]]
