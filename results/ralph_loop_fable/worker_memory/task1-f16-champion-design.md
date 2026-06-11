---
name: task1-f16-champion-design
description: task1 fp16 GEMM 冠军 kernel (v42) 的设计要点与已探明的优化边界
metadata: 
  node_type: memory
  type: project
  originSessionId: 295962ba-8f43-4538-a648-f2f000790f3a
---

截至 2026-06-10 深夜，task1 fp16 4096³ 冠军 = `matmul_f16_v42_shfl2.cu`：**5 样本官方口径 229.71/228.83/228.76/228.74/229.51 → 均值 229.10 ±0.5**（= 锁频峰值 255.5 的 89.7%），Error ~1.4-2.7e-4。当日血统：v27 227.0 → v39 228.3 → v41 228.7 → v42 229.1。

设计 = v27 全套（BM256×BN128×BK64、3-stage cp.async 144KB、8 warp 64×64、mma.16816 f32 acc、XOR swizzle、寄存器双缓冲、持久化 block、GW=8、每迭代恰一次 commit [[cpasync-commit-discipline]]、末波 K-split KS=50/BIAS=3）+ 四层新机制：
1. **形状特化**：模板 <4096,4096,4096,108> 全套 tile/split 标量编译期化（<0,0,0,0> 通用兜底，已验证 2048³ 等形状正确）。省 ~12 live regs + 部分地址算术进 uniform datapath（UIADD3/ULEA 不占主发射口，k-loop 每波 -1µs）。特化实例 REG 254 / STACK 0 —— **一切 epilogue 重构能活的前提**。
2. **寄存器 shuffle 转置 epilogue**（零 smem、零 barrier）：每 (i,half,group) 做 quad 内 4×4 转置 = XOR-rename SEL + 2×SHFL.BFLY + un-permute SEL → 1×STG.128。8 warp 独立流式写 C。SHFL.SYNC 的收敛簿记在无特化余量时会触发 spill（v35 教训）。
3. **acc 分相边界**：stage-0 预取最先（v41，+0.4TF）→ EPI(2)+EPI(3)（64 regs 死亡）→ stage-1 预取 → EPI(0)+EPI(1)；helper dump 同样分相、flag 在全部 dump 后发布；owner spin 一次前置。
4. **epilogue 内禁 __syncwarp**：直线收敛代码用 `asm volatile("" ::: "memory")`；WARPSYNC 单独就能把 ptxas 推成热循环每迭代 STL/LDL（bisection 证明）。

**~597µs 预算（v45 wavedbg）**：537.8 理想 + ~5µs/波 in-loop（bar 收敛抖动，**结构性闭合**：split-barrier 对称负载下代数等价单 bar；4-stage 需 192KB>164KB；BM192 不整除）+ 边界 3.4µs/tile（phase A 2.2 + phase C 1.2，已贴近 64KB L2-store 下限 1.8µs + 发射开销）+ 首波冷启动 ~8.5µs + 尾部 ~6µs。热循环 SASS = 630 instr/3iter（384 HMMA, 96 LDSM, 36 LDGSTS），pcsamp 无热点。

**negative 全集（数据闭合）**：entry-skew；epi-defer 热循环 drain 注入（-8~-17TF）；蛇形 MMA；大块 epi 砍 bar；KS_BIAS=2 三次验证负（faster-epi 后 margin 仍被抖动吃掉）；跨轮 L2 乒乓（v44 中性）；旧负面（f16 acc/2-CTA/mbarrier/kernel-B/j-outer/GW4/16/.cs 写/热循环分支/热交接）。

**新增 negative**：v46 prologue stage-2 预热（228.56×2，中性偏负）；v48 pf1 后置（aSrc/bSrc 活跃域延长 → STACK 120 再爆，213TF——边界顺序只有 v42 这一个可行点）。v47 no-bar 探针（结果故意错、只看时钟）：砍 bar 仅回收 1.3µs/波，**其余 3.7µs/波 = HMMA 操作数 cadence**（v42 TC 91.11%）。

剩余可打的唯一大项 = cadence 3.7µs/波 ≈ +7TF 理论上限，需 SASS 级重排。**CuAssembler 探针已做（2026-06-10）：no-go**。过程：git clone 可用（容器有网）、pyelftools 从 PyPI 拉 sdist 进 PYTHONPATH、5 个 CUDA-13 格式补丁（e_flags SM 字段移位到 bits 8-15、e_version 字符串化、.tkinfo no-op、.string 指令、.nv.prototype 的 index@/str_index@ fixup 用原 cubin 字面值 0x3/0x1 回填）后**反汇编→cuasm 成功**；重汇编被三重阻塞：① `@!UP1 ULEA` 等 uniform 谓词变体编码基不足（least-squares 缺样本，需合成语料数日）；② **LDGSTS/LDG/STG 文本欠定**（InputCode≠AsmCode，如 0x1d62 vs 0x0d7f——nvdisasm 隐藏 cache-policy/descriptor 位）→ 改动任何调度都需逐行原字节回填机制；③ EIATTR 0x4c02/0x5f03 未知、驱动加载未验证。工具状态留在 /tmp/CuAssembler（含补丁）+ /tmp/repos_sm80_cuda13.txt。现实收益估 +1-3TF vs 1-2 天投入。CUTLASS 级 sm80 实际包络 88-92%，v42 的 89.7% 已贴边。其它全部 <+0.5TF 或低于 ±0.5 噪声。相关 [[gpu-locked-1155mhz]]
