---
name: dynamic-workflow-guided-result
description: guided臂忠实执行leave-one-out但172.5<主臂192.9;规定死搜索=峰值反降、价值在独有的standalone-vs-LOO表(4/6 ingredient贪心会丢却贡献+137)
metadata: 
  node_type: memory
  type: project
  originSessionId: 56b6e1be-8cd3-4066-9b95-210ebc2d4777
---

dynamic_workflow_guided cycle1(2026-06-09,headless+guided seed 规定死 leave-one-out;**首跑死ECONNRESET于orientation作废→清理重跑**得本结果)。防作弊门✅通过。N=1。**seed≠共享→只比主臂dynamic_workflow(192.9),不比naive/goal峰值**。

**结果**:fp32 **172.5**(55.3%of312,err4.4e-05,v10=128×128×64 2×2warp S2 2blocks/SM)。4 workflow(phase1-tile-sweep+phase1b-occupancy-sweep+**phase2-3-ablation**+phase2b-standalone)。$18.2/~212k token/~1.4h(比主臂$37.5/349k/2h便宜、更早收)。**自停**(交付Final Report后收)。⚠️多result事件(turns35+15):总token用transcript末值~212k、cost用末result累计$18.2(parse_run总计tail-1=末段60134少报)。

**✅严格执行策略的证据链**:① workflow脚本phase2-3-ablation的meta原文"Phase2-LOO: remove exactly one ingredient from the full stack; remeasure"+"Phase3-Synergy: both-off corners";② 先Phase1 sweep建全栈参照(锁128×128×64)再Phase2逐项移除;③ 判定反转"rejected only if removing from full stack doesn't cost TFLOPS, no forward-greedy/standalone-delta";④ 产出规定的standalone-vs-LOO表+标4个RESCUED;⑤ flock串行计时+正确性gate。脚本留证results/dynamic_workflow_guided/workflow_scripts/(主臂7脚本也补留results/dynamic_workflow/workflow_scripts/)。**没退化回自发config-tournament**(早期Phase1用sweep是seed允许的建参照步)。

**⭐核心交付物=那张表(主臂/naive/goal都给不出)**:6 ingredient里4个standalone≈0或负(cp.async+0.3/pipeline+2.3/ldmatrix−0.2/vectorize+2.5)→贪心全丢,但LOO Δ分别+40.8/+10.6/+32.0/+53.9、合贡献+137/172→只有backward elimination救得回=cooperative blindspot被量化坐实。synergy四角:cp.async⊗pipeline(无cp.async加pipeline−9.2、有则+10.6,互为前提);−swizzle崩到56.9(BK=64连续shared bank conflict 104K→235M)。

**⭐为何规定死策略反而制约(峰值172.5<主臂192.9低~20)**:① 预算被"方法论严谨"(搭可切换全栈+6项LOO+3组synergy四角+6项standalone=十几个证明实验)吃掉,非主臂全砸激进调tile/STAGES;② "先锁全栈再消融"钉死在Phase1局部最优,**LOO只优化"该不该留某ingredient"、不优化"tile选得对不对"(峰值主要由后者定)**;③ seed给了可完成的"交付排序表"目标→达到即自停(对照主臂纯NEVER-STOP无终点只能继续磨config)。**一句话:强加严谨方法论≠更高峰值,但产出naive/goal/主臂都给不出的"哪些优化只协同才生效"知识。与goal看门狗"逼出突破"是两种人为干预:看门狗抬峰值、guided抬知识严谨度。** 数据:results/dynamic_workflow_guided/{result.md,result.csv,curve.png,workflow_scripts/,transcript.jsonl}。SUMMARY结论#8+comparison.png已更新。相关:[[dynamic-workflow-cycle1-result]] [[dynamic-workflow-method-harness]] [[goal-cycle3-result]]
