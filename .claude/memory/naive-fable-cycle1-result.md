---
name: naive-fable-cycle1-result
description: naive_fable 首跑(Fable 5 naive 臂):峰值211.5f16/201.8fp32但GPU锁1155时钟混淆=只能比占真天花板%;Fable输出文字少~13×+自发git commit
metadata: 
  node_type: memory
  type: project
  originSessionId: 3665db98-7374-4e66-ac17-02609d9cc4ce
---

naive 范式 **Fable 5** 臂首跑(`results/naive_fable/`,2026-06-10,N=1)。干净自停(end_turn),13 版,0 sub-agent,145 turns,96min,**$53.14**,309,897 out_tok。防作弊门过(纯手写)。

**性能**:f16-acc 峰 **211.5**(v9,err0.018)/ fp32-acc 峰 **201.8**(v4,3.6e-5)、终版 v13 201.0(3.4e-5)。轨迹:122→150(big tile)→193(cp.async 流水,关键)→202(fp32 封顶)→转 f16-acc→211.5(v9 全调优:CTA128×256 BK64 3-stage 旋转 stage 指针 frag 双缓冲 XOR-swizzle)。v6-8/v10-12 系统化消融逐一否决(stream-K/dual-launch/手动 straddle/BK32/CUDA-graph/256×128)。

**⚠️ 头号坑=时钟混淆**:本轮评测 GPU **硬锁 1155MHz**(实测复现:v9 跑 211.7 时 SM 全程钉 1155、321W、不 boost),真天花板=255.6 不是 312 → 211.5=**占真峰 82.7%**(naive 家族最高 %)。但老 Opus 跑(naive_cycle3 引用 1410)在更高时钟 → **绝对 TFLOPS 跨模型不可直接比**;公平指标=占真天花板 %:Fable 82.7% vs cycle3 49%(可 pin 的两轮);cycle6(208.3)/goal(206.8)时钟没记录→无法定论(若 1410 则 Fable 完胜,若也 1155 则 ~打平 81%)。**教训:每轮必须 log SM 时钟**。详见 [[ncu-base-clock-vs-bench-boost]](已更新 GPU 锁 1155)。

**📉 输出文字确实少很多(用户问的)**:Fable 可见叙述 **1,684 字符/11 块** vs Opus naive_cycle3 **22,510/54 块** = **~13× 少**;但 tool_use 反更多(144 vs 91)、总 out_tok 相近(310k vs 360k)→ 少掉的全是散文简报(seed 明确要求每轮简报它基本跳过、直接动手)。**Fable 签名=极简叙述+动作密集**。(thinking 内容两边都加密、不计入文字对比。)

**🔧 Fable 自发 git commit**(4 个带决策 message,Opus 不提交)→ **harness 坑**:finish_run 的 worker.patch 用 `git diff --cached HEAD` 假设未提交 → 对自提交算出**空 patch(0 行)**;已手工 `git diff base..HEAD` 重生成(4348 行/13 文件)。**待修 finish_run**(自提交 run 要 fallback 到 base..HEAD)。

**💰 贵但 token 少**:$53 vs cycle6 $12.6/cycle3 $20.7,却 out_tok 更少;因 145 turns(多短 turn)→ cache_read 32.8M(cycle3 17.8M)撑高成本=极简多轮风格的代价。

**resume(用户要的 Ralph-Loop 式继续)已备好**:finish 用了 `--keep-worktree`(worktree+session_id+worker_memory 全留)。一键续:`bash scripts/resume_naive_fable.sh`(给 launch_ralph_loop.sh 加了 `RALPH_MODEL` 旋钮→续跑也用 claude-fable-5,复用 naive_fable worktree+还原 worker_memory)。口径/坑见 results/naive_fable/RESUME.md。harness 见 [[naive-fable-method-harness]];精度档比对先看 [[f16-accumulate-precision-confound]];别并行见 [[concurrent-runs-share-worker-memory]]。
