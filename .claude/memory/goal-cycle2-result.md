---
name: goal-cycle2-result
description: "/goal第2跑(死于ECONNRESET,204.7崩前地板):看门狗=地板抬升器非天花板;边际+9.7(c2自停195)vs+1.3(c1自停205.5)全由worker自愿停点决定;0 sub-agent(c1=129)峰值照样~205→129是路径变异"
metadata: 
  node_type: memory
  type: project
  originSessionId: efbd0bee-80cb-44d8-a7a2-221db97a0819
---

`/goal` 第 2 跑(`results/goal_cycle2`)**死于 ECONNRESET**(`is_error:true`,186 turn/478k tok/$41.4)——看门狗只挡自愿停、挡不了进程级崩溃。**204.7 是崩前地板,非收敛终值**;当部分轮,干净 /goal 仍以 cycle1(206.8)为准。已 finish_run 归档 + 写 result.md(curve 带 round 区间)。

**两轮合看,坐实 cycle1 的核心结论(见 [[goal-cycle1-result]]):看门狗 = 高方差的「地板抬升器」,不是「天花板抬升器」。**
- c1:worker 自驱到 **205.5** 才首次想停 → evaluator 接入 3 次(@430k/606k/753k tok)仅 +1.3 到 206.8。
- c2:worker 自驱到 **195** 就想停 → evaluator 接入 2 次(@403k/439k)崩前 +9.7 到 204.7。
- 两轮峰值都 ~205,但「worker 自愿停点」差 10(195 vs 205.5)→ **看门狗边际(+1.3~+9.7)几乎全由 worker 在哪想停决定**(救过早想停的轮、顶回 ~205;不救则不抬)。

**0 sub-agent(c2)vs 129(c1),峰值都 ~205 → c1 的 129 sub-agent 是路径变异,不是 /goal 效应**(cycle1 曾存疑,c2 解了)。

口径细节:parse_run 数 evaluator 接入 = transcript 非-sentinel `goal_status`(c2 共 3 个 goal_status:1 sentinel + 2 met:false);**3 个 goal_status 别全当接入**。round 区间 = parse_run 自动画(红竖虚线 = not-met 接入,`round i` 箭头 = 第 i-1→i 条线;round 1 永远是头次接入前的自驱大段)。

**跨方法总览已落 `results/SUMMARY.md`(三方法文字说明 + 对比表 + `results/comparison.png` 综合曲线);对比图脚本 `results/plot_comparison.py`**(读各 run result.csv scored 点 + goal 族 transcript 接入 token)。naive_strong_cycle2 也补了 result.md。详见 [[vibekernel-result-harness]]。
