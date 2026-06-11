---
name: naive-cycle1-econnreset-and-gotchas
description: "naive cycle1 死于 API ECONNRESET 作废(峰167.2);cycle2 已重跑、自停有效、4096³ 峰值 178(8192²=192 但 off-口径);跑完先验 result.is_error/stop_reason;inf=误差除零\"数据彩票\"非bug;naive 撞 tensor-pipe barrier 气泡天花板、对范式级重写保守"
metadata: 
  node_type: memory
  type: project
  originSessionId: 988936b7-d4da-4545-91a7-31f078c8baf0
---

**cycle1(2026-06-05,作废)**:fork `playground-naive-clean`、去 cuBLAS 基座,手写 mma.sync+cp.async+XOR-swizzle,峰值 167.2,~1.75h/121k/42 turns——但末尾 `result` 事件 `is_error:true`/`ECONNRESET`(`api_retry` 10/10),**是 API 中断非自停**,对「内在持续力」无效。已归档 `results/naive-break/`(fork 去 .git 作普通文件)。

**cycle2(2026-06-06,有效、正式结论)**:重 clone 干净 fork 重跑,**自然终止**(`is_error:false`/`stop:end_turn`/`completed`),128 turns/370k out-tokens/~1.68h,做到 v60。**4096³ 计分峰值 178 TFLOPS(v56,≈57%)**;8192² 冲到 191.9(61.5%,但**波尾消失=off-口径,不作成绩**)。手写门通过。结果见 `results/naive/result.md`。自停时它判定"只剩 stream-K",**漏看 async-barrier 这条更大的杠杆**(瓶颈分析见 result.md)。

**How to apply(每个方法跑完都查):**
- 归档后**先验最终 `result` 事件的 `is_error` / `stop_reason` / `num_turns`**,辨清「自停」vs「限流/ECONNRESET 等 infra 停」。infra 停的 wall/turn/token **不算** naive 持续力,要重跑。
- **inf 是 harness 的「数据彩票」非 kernel bug**:误差度量 `|GT−C|/|GT|`,某随机种子下个别 `GT[i]` 舍入到 f16 的 0→除零,约 0.6%/run、影响所有版本(含 v1)。别因偶发 inf 判 kernel 无效,也别为刷分回避。所有方法通用。
- **occupancy 不是杠杆(cycle2 确证、推翻 cycle1 直觉)**:高算术强度 64×64 warp tile(128 fp32 acc 寄存器 ≈65K regs + 153KB shared)各自都把 occupancy 逼到 1 block/SM(12.5%),但**比高 occupancy 更快**;真瓶颈 = tensor-pipe 在块级 `__syncthreads` barrier 的气泡(74% HMMA-active 封顶,~62% of peak)。fp16 累加(省 58 寄存器零提速)/ XOR-swizzle(cp.async+ldmatrix 已 0 bank conflict)/ 堆 warp 全证伪。突破要 **async-barrier / warp-specialized 流水**(stream-K 只修波尾)。详见 `results/naive/result.md`「瓶颈与后续优化方向」。

**归档:** 这轮(cycle1,ECONNRESET)已被用户重命名挪到 `results/naive-break/`(fork `playground-naive-clean` 也挪进该文件夹内),给即将开始的 naive 重跑腾出干净的 `results/naive/`。网络问题已修。

**工具已随这轮修(见 [[vibekernel-result-harness]]):** `results/parse_run.sh` 去掉旧 cuBLAS 拓扑——不再把 v1 误当 cuBLAS 基线从曲线剔除;cuBLAS 改为**一条固定参照线** `CUBLAS_REF`(默认225,纯可视化标尺,数据/口径里无此基线);SRCDIR 自适应 `playground-<m>*` 后缀如 -clean。**曲线标注(2026-06-06 再改,从「只标里程碑」→ 信息更密):每个版本取最佳一点标注**(按 tflops 取最高、自动跳过 k=64 调试小分),含回归版;前沿(刷新 running-best)标点上方蓝色、回归版标下方灰色,交替偏移防叠字,峰值附 TFLOPS。**技法描述优先级(表 result_table.md + 图共用同一份):可选 `results/<方法>/labels.json`(`{"3":"XOR swizzle",...}` ver→短技法)→ 文件名后缀(v3_swizzle)→ marker.technique/changed → 仅 vN**。labels.json 专为薄 dispatcher/共享头(源文件名不带技法,如 v8.cu)补描述;无此文件全自动不报错。注:每方法重跑后版本号会变,labels.json 要按新跑重写。`scripts/check_handwritten.sh` 改扫 `src/`+`include/` 全部非 v0 源/头——因薄 dispatcher 把真 kernel 放共享头 `include/playground/mma_gemm.cuh`,旧版只扫 `src/_vN` 会整个漏掉。(以上工具改动已 commit。)**2026-06-06 parser 再加口径过滤**:只计 canonical(`4096³ && iters≥100`)且正确性过关(err<0.1)的点进 best/曲线/峰值,off-口径(大/小 shape、ncu -t1、-t30/50、8192²)与写错/inf 降灰不计分,CSV 加 `canonical`/`scored` 列——**防 8192² 之类虚高 peak**(见 [[vibekernel-result-harness]]、META §6)。cycle2 fork kernel(v1/v2/v3/v56)已 push 到 silencelamb/playground-naive-clean 并注册 submodule。详见 [[playground-base-clean-fork-topology]]。
</content>
