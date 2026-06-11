---
name: goal-cycle4-result
description: "/goal 第4跑:分两段(part1崩401→resume→part2崩ECONNRESET),峰值188.7(goal族最低);看门狗5接入,首接入178.9处逼出epilogue合并+9.8→188.7(part1),part2三接入净0;resume实操+两段日志口径坑"
metadata:
  node_type: memory
  type: project
  originSessionId: 7e2a74fb-3baf-4b2f-b58b-4cc8b4757a93
---

**/goal 第4跑(2026-06-09):峰值 188.7 TFLOPS(v25,f16-acc err 0.017,= goal 族最低)。** 分两段跑、两次 API 崩:part1 跑 ~188min 到 v25=188.7 后**崩于 401 Invalid auth**(用户重登)→ 我 **resume 续上**(`claude -p "/goal $CONDITION" --resume 7e2a74fb…`,session_id 不变、transcript 同文件续写、worker 完整记得 v1..v25,part2 第一句准确接上崩前的"下一步")→ part2 跑 ~99min **崩于 ECONNRESET**。合计 ~287min/4.78h、223 turns(184+39)、862.8k out-tok、**$79.69**($50.73+$28.96)、**0 sub-agent**、手写校验过。归档:`results/goal_cycle4/`。

**看门狗(evaluator)= 地板抬升器,本轮首接入就逼出真增益、part2 接入全是净 0 长尾(N=1 两阶段证据):**
- 5 次接入全 `met:false`,token 位 `[436783,522301,742047,781975,862819]`(2 在 part1≤601018、3 在 part2)。
- **首接入 @178.9(v23)**:worker 在 179 plateau 上自封"179 是 occupancy 天花板"(写满"78% of 230 探针、64-reg f16 累加器卡 2 blocks/SM"论证)想停 → 被顶 → **才做出 epilogue 访存合并(v24 coalesce C-store + v25 Cs 消 bank 冲突)→ 179→188.7 = +9.8(~5.5%)**。**净看门狗增量全在 part1**。
- part2 三次接入(187-188)全在 v25 真收敛后 → v26-v29 全回归 → **净加 0**,$29 烧在反复确认。
- **★诡异:cycle4 首接入 178.9 == [[goal-cycle3-result]] 自停点 178.9**——两轮 worker 独立爬到同一 v23 级 ~179 plateau、用同一套机理自封顶,但突破路径不同(c3 被顶 13 次→mbarrier +22.7→201.6;c4 被顶 2 次→epilogue +9.8→188.7 即收敛)= **净增量由随机探索路径主导,非看门狗剂量**。goal 四轮峰值 `{206.8,204.7,201.6,188.7}`,c4 最低(这条路径早收敛、没摸到 200+ 的 mbarrier/深流水)。

**收敛在 188 的机理(worker pure-HMMA 探针实测):** 2-block/16-warp/512-累加链 occupancy 的算力天花板 ≈230(不是 312),v25=该探针 82%;occupancy 被 64-reg f16 累加器硬卡 2 blocks/SM(强行 3→spill→79)。破 160 plateau 靠 **m16n8k8(半操作数寄存器,自驱、首接入前)**,破 179 靠 epilogue 合并(被顶后)。wave-quant ~6%(4096³=188 vs clean-wave 3456×4096=197-200,off-口径已排除)。详见 [[library-ceilings-a100-gemm]] 同档对标。

**两个方法学坑(务必记):**
1. **231 是假的**:pure-HMMA 探针(去 smem/barrier 的裸发射,err 223/147=不算真 GEMM,canonical 形状但失正确性→已排除),不是性能结果。seed 也明令"不报 best-of-N 热峰"。真 deliverable = task1 计分的 188.7。
2. **两段日志口径**:`parse_run` 的"总计(权威)"只读传入的那个 `run.jsonl` 末 result 事件 = **单段**(part1 = 11282s/601018)。真总计必须**手工把 `run.jsonl` + `run.part2.jsonl` 两段 result 事件相加**。曲线/接入点不受影响(走 transcript,同 session 连续含两段)。resume 前用 haiku one-shot 验 auth 已恢复再跑(别白烧)。

相关:[[goal-cycle3-result]] [[goal-cycle2-result]] [[goal-cycle1-result]] [[goal-method-harness]] [[naive-cycle1-econnreset-and-gotchas]] [[f16-accumulate-precision-confound]] [[concurrent-runs-share-worker-memory]]
