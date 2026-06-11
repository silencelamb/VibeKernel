---
name: f16-accumulate-precision-confound
description: "f16-accumulate技法:降累加精度→2-block/SM→掩盖HMMA wait→手写更高TFLOPS但err0.02-0.03;⚠️cycle5首测(并行串味)218作废;✅naive_cycle6(干净串行)独立走到f16-acc 208.3→'naive自发'坐实(无需抄goal),framing那半待验;注:cuBLAS f16≈fp32只差1,f16优势是手写1-block-register-wall产物非本质(见library-ceilings)"
metadata: 
  node_type: memory
  type: project
  originSessionId: efbd0bee-80cb-44d8-a7a2-221db97a0819
---

**精度档 = 跨方法对比里的混淆变量**(`naive_cycle5` 揭示,2026-06-08)。⚠️ **首次观测的那轮 naive_cycle5 已污染作废**(与 goal_cycle3 并行串味,见 [[concurrent-runs-share-worker-memory]]):**218/188 两个数都作废**,下面"naive 自发探到 / framing 影响"的解读因污染**不成立**;但 **f16-acc 这条技法本身、以及 ERRMAX 门太松这个口径问题,与污染无关、仍成立**。**✅ 已由 `naive_cycle6`(干净串行、blank memory、优化 base)回答:naive 能【自发】走到 f16-acc(独立做到 208.3,err.018),无需抄 goal**——cycle5 的串味只是给了它更高跳板(218)。⚠️ 但 **cuBLAS f16(219.85)≈ fp32(218.74)只差 1**,说明 f16-acc 的"优势"是手写 fp32 卡 1-block register-wall 的产物、非本质(见 [[library-ceilings-a100-gemm]])。framing 是否影响"擦精度门"那半仍待更多 N。

- naive_cycle5(weak prompt)首次拥抱 **f16-accumulate**:裸峰值 **218.6 TFLOPS 但 err 0.02–0.03**(v8);**fp32-accumulate(err ~3e-5)峰值只有 188.1**(v10),反而 < naive_cycle4 的 196.9。
- **用户裁定(2026-06-08):218.6 算 headline = 全场最高,f16-acc 当 fp16 GEMM 任务的合法优化、可与 goal 206.8 同台比——但精度档不同必须始终点名**(218=f16-acc err 0.03;goal/cycle4=fp32-acc err 3e-5)。218 靠降累加精度换吞吐(accumulator 寄存器减半 → 64×64 warp 塞 2 block/SM → 多 warp 掩盖 HMMA wait 气泡)。**另问"同精度档谁最强"则取 fp32 子峰 188.1**(反而 < cycle4 196.9)。
- **两种读法都要摆**:headline =「裸峰值 218(naive c5,f16)> goal 206.8 > cycle4 196.9」;同精度档 =「goal 205–207 > naive-fp32 154–197 > naive_strong 151–185」。`parse_run.sh` ERRMAX=0.1 收 f16-acc(0.03)进 scored 是有意的(headline 口径);`plot_comparison.py` 对 cycle5 画全曲线到 218、另空心点标 fp32 子峰 188。
- worker **透明双报**(v8=218/f16 "max perf" vs v10=188/f32 "max precision",自荐 grader 严就用 v10)。**有意思的行为对照**:`naive_strong_cycle2`(强 framing)实测 f16-acc err 0.027 后**主动弃用**、`naive`(弱 prompt)**追了** → 提示 **framing 影响"是否走降精度捷径"**,N=1、值得单独设计实验。

应用:报峰值/写 result.md/SUMMARY 时**始终标精度档**(headline 用裸峰值但点名 f16-acc/fp32)。详见 `results/naive_cycle5/result.md`、`results/SUMMARY.md` §核心结论 1/6。关联 [[vibekernel-result-harness]]、[[naive-csv-field-parsing]]。
