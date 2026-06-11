---
name: task1-f16-champion-design
description: task1 fp16 GEMM 冠军 kernel (v19) 的设计要点与已探明的优化边界
metadata: 
  node_type: memory
  type: project
  originSessionId: 295962ba-8f43-4538-a648-f2f000790f3a
---

截至 2026-06-10，task1 fp16 4096³ 冠军 = `matmul_f16_v19_spread.cu`：**~218 TFLOPS 中位（官方口径 217.8，波动 217.4-219.6）**，Error ~3.5e-5；8192×8192×4096 时 235 TF（=锁频峰值 92%，证明 4096³ 差距主要是 4.74-wave 尾波量化 ~5%）。

设计：BM256×BN128×BK64、3-stage cp.async（144KB smem）、8 warp（4×2）各 64×64、mma.m16n8k16 **f32 累加**、XOR swizzle（bank conflict=0）、寄存器双缓冲 frag、持久化 block 跨 tile + epilogue 期间预取下一 tile 的 stage 0+1、smem 中转 epilogue（B2 槽 8×32 行）、m-fast GW=8 panel 序（并发工作集 29.5MB）、commit-每迭代一次纪律（见 [[cpasync-commit-discipline]]）。

**已探明的负面/边界**（别再试）：f16 累加误差 0.0225 超线只 +0.7TF；2-CTA/SM（寄存器墙逼小 warp tile→LSU 爆）；mbarrier 在 sm80 是软件路径（167/85TF）；stream-K/K-split 尾部修复的 merge+边界成本 ≥ 收益（kernel B 类启动 long_sb 延迟泄漏 ~100µs/item）；j-outer mma 序、GW=4/16、.cs C 写、4+4+4 发射分布 均为负或噪声。剩余理论空间：尾波 5%（无经济解）、barrier 3%、HMMA cadence ~3%。相关 [[gpu-locked-1155mhz]]
