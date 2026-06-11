---
name: goal-cycle1-result
description: /goal cycle1:worker 靠自己冲到~205.5(evaluator全程没介入),第一次想停才被顶,看门狗真实增量仅205.5→206.8(+1.3,L2persist)+2.5h复验;峰值206.8=naive家族最高但178-vs-205是路径变异非看门狗;N=1
metadata: 
  node_type: memory
  type: project
  originSessionId: 6a0f753c-03d7-4ba8-bbe3-2d966510d17f
---

**`/goal` cycle1(2026-06-06,results/goal/)。** condition = 与 naive **完全相同**的 seed,evaluator = Haiku,worker = opus-4-8/max,ncu 可用,playground-base 单工作区。纯手写 fp16 GEMM 峰值 **v21 206.8 TFLOPS**(scored 4096³/100)= naive 家族最高;4.03h/284turn/817k out-tok/$87.5;防作弊门通过、零库。

**⚠️ 看门狗的真实贡献被初稿严重高估(已更正)——重要教训:** transcript 时间线实锤:
- worker 靠自己的 NEVER-STOP 动力把 **60→~205.5**(v17/v18,~1.4h)一路冲完,**全程 evaluator 没接入一次**(它在进步、没想停、就没触发 Stop hook)。
- **第一次想停时已经 205.5** → evaluator 才第 1 次接入。**evaluator 全程只接入 3 次**(transcript 656/916/1122 行,都在 205–206)。
- 看门狗净把它从 **205.5 推到 206.8 = +1.3(~0.6%,靠 L2 persistence)** + 逼出 ~2.5h 复验。**so /goal 机制真实增量 ≈ +1.3,不是 178→206.8 的 +16%。**
- **178(naive cycle2 自停)vs 205.5(本轮自停点)的差是两个 N=1 跑的路径变异,不能记到看门狗头上。** 初稿误把 178→206.8 全归看门狗——**跑完对照别想当然把"更高峰值"归给 harness,先查 evaluator 头一次接入时已经到多少(grep 'Stop hook feedback:')。**

**机制 / 行为:**
- evaluator 接入信号 = transcript 的 **`goal_status`**(attachment 行,有 `met`/`sentinel` 字段;run.jsonl 没有;详见 [[goal-method-harness]])。**4 个 goal_status 全 met:false**(1 sentinel 启动 + 3 接入,接入时累计 token≈430k/605k/752k)——**evaluator 从没判 met:true**;3 次顶回去后 worker 继续 re-verify、run 在 ~817k token 收尾(clean success,**不是被判达成而停**,确切机制日志没说;早于我设的 2h 计时器)。worker 自己写了 46 个 "Round brief"(它**自己**的每轮简报节奏)≠ evaluator 接入(只 3 次)。
- worker 自发 spawn **129 个 sub-agent(Task 工具)** 做并行 config 微实验(test cp.async.ca vs .cg / missProp=Normal / mma order / serpentine 等)→ 比 naive 明显更 agentic,放大 token/cost(Task 是受控常量,用不用模型自选)。
- **被顶回去后的"功耗/频率"工作 = 真实但低价值的强制再论证**:ncu SOL roofline(compute-bound 81% tensor SOL)、f16-acc 2-CTA occupancy 受控实验(证 wait 是 occupancy-bound)、power 复查(throttle 0x4 power-cap ~1200MHz、tensor 峰~250-265 非312)。测量都真、逻辑自洽(register-walled occupancy + power-cap 是真 A100 GEMM 天花板),但本质是"被逼着继续、没新招 → 从多个角度反复证明自己到顶"——用户直觉"优化不上去找理由"方向对(是强制再确认,不是造假)。

**⚠️ 2026-06-07:naive_cycle2 对照作废 → 已修隔离并重跑 naive_cycle3(干净对照已补)。** cycle2 因 worktree 共享 worker auto-memory(按 **git 仓库**键)读到 goal 的 `fp16-gemm-best-kernel.md` 抄答案作废(存 `results/naive_cycle2_deprecated/`)。**修复(`_run_common.sh` 每 launch 前清 base-slug worker-memory)已经 naive_cycle3 实测验证:泄漏标记(fp16-gemm-best-kernel/206.7/206.8/L2-persist/PAD=8)全 0、worker 自写自己的 `gemm-f16-best-config.md`=白板起步坐实。**

**🎯 干净对照结论(naive_cycle3,results/naive_cycle3/):** 无看门狗、纯 naive 自停于 **fp32 级 ~142(v3)/ f16 级 154(v15,err 0.016)**;1.53h/360k tok/$20.7/**0 sub-agent**/92 turn/自然终止。**这印证"看门狗机制增量仅 +1.3"**:goal worker 是【靠自己】到 205.5 的,看门狗 205.5 后才接入 → **205.5 vs 142 的大 gap 出现在看门狗接入【之前】,机制上不能归给 evaluator**。该 gap 更像**探索强度差**(goal 129 sub-agent vs cycle3 **0**)/ `/goal` framing / 路径变异,**三者本数据分不开、未定论**。⚠️ 口径坑:goal 206.8 是 **fp32 级(err 4e-5)**,naive_cycle3 同口径只 142;154 是放宽到 **f16 累加**换的——**公平对照取 142**。

**🆕 naive_cycle4(干净,results/naive_cycle4/):canonical t=100 峰值 196.9(v33,fp32 级),自停、0 sub-agent 但自迭代到 v40+重构 dispatcher 头。** ⚠️**口径坑(差点误判)**:transcript 有个 206.8,但那是 worker 跑 `-t2000` **暖机长跑**(GPU 久跑时钟 boost 虚高),它**自己**把结果定为"~197"(原话 Surpassing ~197…)。**parser 旧 `iters≥100` 误收 → 已修 `iters==100`**(暖机长跑/8192²/-t1 全 off-口径)。c4=206.8 与 goal 206.8 数值撞车是巧合(都贴暖机功耗墙),**只有 goal 的是 t=100 实打实**。隔离同样干净(0 startup 读、0 越界、fp16-gemm-best-kernel=0、206.7=它自己暖机实测)。**parser 修复连带影响**:results/naive(178)/goal(206.8)/cycle3(154) 峰值都是 t=100、不变;但 **naive_no_ncu 旧 headline 168 是 t=300 暖机虚高、真 t=100 只 118**(待重 parse,该跑本就 cutlass 污染存疑)。

**🆕 naive_strong(强 framing prompt、无 Stop-hook;results/naive_strong/)= 🟡resume 瑕疵版**:首跑 turn77/168k **死于 ECONNRESET(is_error:True,同 cycle1)**→ 我重建 worktree+apply worker.patch+`claude -p --resume <sid>` 续跑(注入了句"继续按硬规则"nudge,协议外输入)到自停。**t=100 峰:f16 185.2(v15)/ fp32 168(v8);合计 ~410k/$32.7/161turn/0 sub-agent。** 行为上 framing **明显起作用**(worker 明说"我没有宣布完成",逐一 ncu 否决 ~27 个方向,还撞到真硬件墙:mbarrier.try_wait.parity 需 sm_90、A100 不支持),**但没把 TFLOPS 顶高**(185 仍 < cycle4 196.9),**且它仍然会停**(prompt-only 无强制力,只是把"practical ceiling"换成"撞遍每条路")。⚠️瑕疵+N=1 不能下结论;干净版 = **naive_strong_cycle2**(在跑)。隔离干净(泄漏标记0;"L2 persist"2 次=worker 自己试 persistence→158 否决,非抄)。**resume 可行性记一笔**:finish_run 用 cp 不 mv→原 session transcript 留在 slug,`--resume` 能找回;但需手动重建 worktree+git apply worker.patch 还原内核(build/ 没了会重编)。

**N(干净自停,t=100):naive {142/154(c3),178(c2),196.9(c4)} / naive_strong {185f16/168fp32(resume瑕疵), cycle2 在跑} / goal {205.5 自达→206.8(+1.3 看门狗), cycle2 待跑}。** naive 方差极大(142→197)。**看门狗/framing 贡献仍未决**:强 framing 改变的是**行为(更穷尽)不是结果(TFLOPS)**,且 prompt-only 框架**没有真正的不许停强制力**(那是 goal Stop-hook 独有)。**队列:naive_strong_cycle2(干净版)→ goal_cycle2,跑完看分布。** 相关:[[goal-method-harness]] [[naive-cycle1-econnreset-and-gotchas]] [[playground-base-clean-fork-topology]]
