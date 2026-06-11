---
name: ncu-base-clock-vs-bench-boost
description: "ncu profile 锁 base~1140MHz、裸跑 benchmark boost 到 1410(撞400W掉1365-1380);两档差~24%,ncu 的 duration 不能反算裸跑 TFLOPS"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 59888ed5-8aee-42a5-897a-ce3477939e8e
---

⚠️ **2026-06-10 状态变更**:评测 GPU 现卡死 1155 MHz,裸跑也**不再 boost 到 1410**。**真因 = NVRM 驱动 assertion 故障(`pSmIssueThrottleCtrl NULL`),非 `-lgc` 软锁、`-rgc/-lgc` 解不掉**,详见 [[gpu-1155-driver-fault-not-lgc]]。实测复现:跑 naive_fable 的 v9,SM 全程钉 **1155**(20ms 采样 172 点 min=max=1155)、功耗冲 321W 仍不动。**→ 有效天花板现 = 312×1155/1410 = 255.6 TFLOPS,不是 312;故障态下的 run 绝对值带 ~−18% 偏置、只能比 %。** 后果:**老 Opus 跑(naive_cycle3 引用 1410、cycle6/goal 未记录)与现在的跑绝对 TFLOPS 不可直接比**;跨跑只能比「占当轮真天花板 %」,且**每轮必须 log 实测 SM 时钟**(见 results/naive_fable/result.md 复现节的高频采样段)。下方原描述是**变更前** boost-1410 的状态,留作历史(解释 Opus 跑所处时钟)。相关 [[naive-fable-cycle1-result]]。

A100-SXM4-80GB(driver 580.65,root)上"跑任务"分两种频率档,常被混淆:

- **裸跑 benchmark(报 TFLOPS 的那次)**:GPU auto-boost。满载 = **1410 MHz**(=312T 定义频率:108SM×4TC×256FMA×2×1410e6=311.9T)。但此机 dense fp16 近峰值会顶到 400W 功耗墙 → 掉到 **1365–1380**,nvidia-smi `clocks_event_reasons.active=0x4`(SW Power Cap)。这就是 goal_cycle3 等反复出现的"功耗混淆"。
- **ncu profile(分析瓶颈那次)**:ncu 默认 `--clock-control base`(=Lock GPU clocks to base),把时钟钉死在 base ≈ **1140 MHz**。实测 naive_strong v1=1.15、v7/v12/v15=1.14、cycle2 v6=1.14,全是 1.14-1.15 GHz。
- **1155 MHz = 空闲档**(`0x1` GpuIdle,~71W),跑任务时绝不会停在这。用户最初以为是 1410 或 1155 都不对。

**陷阱**:别拿 ncu 的 Duration 反算裸跑报告的 TFLOPS——ncu 在 1140、裸跑在 1410,频率差 1410/1140≈1.24,会低估 ~20%+。ncu 的定性瓶颈分析(occupancy/stall/pipe 利用率)因全版本同在 1140 base 反而是干净对比、可信;有混淆的是裸跑性能曲线那一侧。

**消混淆**:性能曲线侧锁频(root):`nvidia-smi -i <idx> -lgc <f>,<f>`,`-rgc` 撤销;但锁频不抬 400W 上限(此机 power.limit=enforced=400 已是该 SKU 上限,-pl 无余量),硬件 hw_slowdown 仍可压低锁定值。最干净是锁一个任何 kernel 都不触发 400W 的可持续频率(如 1290)做跨版本对比,另跑一次不锁的 1410 作头条。想让 ncu 跟裸跑同频用 `--clock-control none`(牺牲可复现,一般不建议)。

相关:[[naive-csv-field-parsing]](profile 一行 vs benchmark 口径要分开的底层原因)、[[ncu-permission-gate]]、[[goal-cycle3-result]]、[[library-ceilings-a100-gemm]]。
