---
name: gpu-locked-1155mhz
description: 本环境 GPU SM 时钟锁定在 1155MHz，fp16 实际峰值 255.5 TFLOPS 而非 312
metadata: 
  node_type: memory
  type: project
  originSessionId: 295962ba-8f43-4538-a648-f2f000790f3a
---

VibeKernel task-1 (A100 fp16 GEMM) 的评测 GPU 的 application clock 被锁在 **1155 MHz**（空载/满载均 1155，sw_power_cap Not Active）。因此本卡 fp16 Tensor Core 理论峰值 = 2048 FLOP/cyc/SM × 108 SM × 1.155 GHz ≈ **255.5 TFLOPS**，不是 312。

**Why:** 评分用 task1 100 轮均值，时钟锁定直接决定可达上限；不要按 312 校准利用率（ncu 的 TC% × 255.5 才等于实测 TFLOPS）。

**How to apply:** 优化时以 TC 利用率（`sm__pipe_tensor_op_hmma_cycles_active`）为主指标；不要修改共享 GPU 的时钟设置（影响他人/评测环境）。验证时钟：跑长 kernel 时 `nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv`（注意二进制先跑 ~20-40s 的 CPU GT 阶段，要等 kernel 阶段再采样）。
