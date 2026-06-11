---
name: harness-inf-nearzero-gt
description: "task1 fp16 error check intermittently aborts with inf — harness artifact, not a kernel bug"
metadata: 
  node_type: memory
  type: project
  originSessionId: 5a95c572-9465-4ad5-9a57-4a886883fdab
---

The task-1 fp16 benchmark intermittently aborts with `Check failed ... !isinf(errSum)` (test_data.hpp:155). This is a **harness artifact, NOT a kernel bug**: the error metric is `mean(|GT-C|/|GT|)`, and the ground truth `GT` is stored as fp16 (half). Each run uses fresh `std::random_device` data, so occasionally some GT[i] rounds to exactly 0 in half → division by zero → inf → abort.

**Verified:** v1 (trivially-correct WMMA) and the optimized pipelined kernels fail at the SAME rate (~1/12 runs each). So an occasional inf does NOT mean your kernel is wrong.

**How to apply:** When validating a kernel, run it several times; judge correctness by the Average Error on SUCCESSFUL runs (a correct f32-accumulate kernel gives ~3-5e-5). Don't chase a "race" when you see an intermittent inf. Distinguish from a real bug: a real layout bug gives garbage error (~1.0) on EVERY run, or inf on most runs. Related: [[fp16-gemm-best-kernel]].
