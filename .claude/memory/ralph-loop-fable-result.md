---
name: ralph-loop-fable-result
description: "Ralph Loop×Fable5(iter1=cycle2种子):fresh session突破cycle2自封的219.5尾波天花板→v42冠军229.1(89.7%真峰)/严格3e-5档227.3,反超cuBLAS;坐实fresh-restart>单session续"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3665db98-7374-4e66-ac17-02609d9cc4ce
---

`results/ralph_loop_fable/`(2026-06-10)= **Ralph Loop × Fable 5**,**iter-1 = naive_fable_cycle2 单 session 种子**(v22/219.5),再 2 轮 fresh-session 续(iter2/iter3 均干净 end_turn)。Ralph 新增 = **$86.70 / 2.7h / 2 轮**(iter1=cycle2 $60 另计)。防作弊过(54 文件纯手写)。

**结果**:冠军 **v42_shfl2 = 229.1 均值**(5 样本,err ~1.4-2.7e-4)= 占 1155 真天花板 255.6 的 **89.7%**;**严格 fp32 档(err 3e-5,与 cycle2 同档)= 227.3(v24/v25)**。**两档都反超本机实测 cuBLAS**(f16 219.8 / fp32 218.7)→ 纯 prompt 手写超 cuBLAS ~4%。

**⭐ 核心 = 坐实 Ralph 假设(fresh-restart > 单 session 续)**:cycle2 那个长单 session 在 219.5 **自封"4.74-wave 尾波无经济解、源码到顶"主动收尾**;Ralph 的全新 session **不继承那条放弃记忆**(只继承 kernel+设计 memory),iter2 第一刀就打尾波 → **v23 wavesplit(in-kernel last-wave K-split)225.5、+6 一举越 cuBLAS**;iter3 精修到 229.1。量化:219.5→229.1(+4.4%/+9.6TF),代价 2 轮 $87。

**冠军 v42 怎么做到**:= cycle2 v27 全套 + iter3 加的 4 层:①形状特化(模板编译期常量化 tile/split,省 12 reg、地址进 uniform datapath,特化实例 REG254/STACK0)②寄存器 shuffle 转置 epilogue(零 smem/零 barrier,quad 内 4×4 转置 = SEL+2×SHFL.BFLY→STG.128)③acc 分相边界预取 ④epilogue 内禁 __syncwarp(单个 WARPSYNC 就逼 ptxas 每迭代 STL/LDL spill)。

**瓶颈(数据闭合)**:唯一大项 = HMMA operand cadence **3.7µs/波≈+7TF**,需 SASS 重排;worker **手搓 CuAssembler 汇编器探针→no-go**(uniform 谓词编码基不足/LDGSTS文本欠定/未知EIATTR),判 +1-3TF vs 1-2 天不值。89.7% = CUTLASS-class sm80 包络 88-92% 上沿,已贴边。

**精度两档(报数必标)**:严格 3e-5 = 227.3;放宽 ~2e-4(v26 relax 起,仍≪0.02 门)= 229.1。

**口径坑**:每轮独立 transcript(iters/iter_0{1,2,3}),全量须合并 transcript.jsonl 再 parse;run.jsonl 末尾单 result 事件只是 iter3(299k/$59)非全量,全量=三轮 result 之和。worktree 已 --keep-worktree,`bash scripts/launch_ralph_loop_fable.sh` 从 iter4 续。

对照 [[naive-fable-cycle2-result]](iter1 种子)、[[ralph-loop-cycle1-result]](Opus Ralph 首跑没破 plateau)、[[library-ceilings-a100-gemm]](cuBLAS 219.8/218.7)、[[gpu-1155-driver-fault-not-lgc]];harness [[ralph-loop-method-harness]]。
