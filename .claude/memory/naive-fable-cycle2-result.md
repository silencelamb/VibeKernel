---
name: naive-fable-cycle2-result
description: "Fable5 naive第2跑:fp32-acc 219.5(85.9%真峰,naive家族最高)比cycle1更强更精确;拒f16-acc;抓到并修cp.async commit竞态;冠军v19设计"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3665db98-7374-4e66-ac17-02609d9cc4ce
---

naive 范式 **Fable 5** 臂第 2 跑(`results/naive_fable_cycle2/`,2026-06-10,与 cycle1 同卡/seed/harness → 干净重复性对比)。干净自停(end_turn),22 版,0 sub-agent,149 turns,134min,**$60.46**,397,616 out_tok。防作弊门过。

**性能(关键差异)**:冠军 v19 = **fp32-accumulate 219.5 TFLOPS**(err ~3e-5,全精度)= 占 1155 真天花板 255.6 的 **85.9%**(naive 家族最高)。**≈ 本机实测 cuBLAS fp32(218.74,同 1155 时钟可直接比,源 [[library-ceilings-a100-gemm]]+_baseline_cublas_f16.log)→ 纯 prompt 手写基本追平库 fp32 天花板**(名义上压线略高)。⚠️ 我 plot 早先画的 "cuBLAS-class ~227" 是 cycle1 worker 的 ncu 估计、估高 ~8,已更正为实测 219.85(f16)/218.74(fp32)两线;cycle1 的 f16 211.5 = cuBLAS f16 的 96%。**比 cycle1 更强且更精确**:cycle1 靠 f16-acc 才到 211.5、fp32 仅 201.8;cycle2 直接 fp32 冲 219.5。**v14 试过 f16-acc(217.9)但只 +0.7TF、err 跳 500× → 主动否决**(与 cycle1 拥抱 f16-acc 相反 = 路径变异,fp32 路径其实更优)。

**轨迹 7 阶段**:v1 WMMA(126)→v2 裸 mma+cp.async 流水(194,最大跃迁)→v3-8 加 stage/swizzle/smem-epilogue(210)→**v9 stream-K 抓到 cp.async commit 竞态**(空 commit 让 wait_group 错位一档→读在飞数据,间歇错;立"每迭代一次 commit"纪律+110 次连跑验证,弃 stream-K)→v12-13 persistent+epilogue 预取下一 tile(217,首冠军)→v18 镜像 256×128+v19 m-fast GW=8 panel(219.5 冠军)→v20-22 mbarrier/bundle/tail2 受控负结果钉边界。

**冠军 v19 怎么做到**:BM256×BN128 BK64、3-stage cp.async(144KB smem)、8warp各64×64、mma.m16n8k16 **f32 累加**、cp.async.cg L2::128B、XOR swizzle(bank conflict=0)、frag 双缓冲、persistent block+epilogue 期预取、smem 中转 epilogue、m-fast GW=8 panel(L2 局部性)。8192² 跑 235=锁频峰 92%→证 4096³ 只 85.9% 是 **4.74-wave 尾波量化(~5%)** 非 kernel 低效。残差:尾波 5%+barrier 3%+HMMA cadence 3%,再上需手写 SASS。

**verbosity 签名稳**:可见文字 2,087 字符/15 块(cycle1 1,684/11),依旧极简;0 sub-agent。

**harness**:✅ 我修的 finish_run merge-base 逻辑**生产验证通过**(自动检出 5 commit、worker.patch 7514 行非空)。已 `--keep-worktree`,可 `bash scripts/resume_naive_fable.sh naive_fable_cycle2` 跑 Ralph Loop。

对照 [[naive-fable-cycle1-result]];GPU 故障 [[gpu-1155-driver-fault-not-lgc]];harness [[naive-fable-method-harness]];精度档 [[f16-accumulate-precision-confound]]。
