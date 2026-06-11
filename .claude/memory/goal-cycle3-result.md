---
name: goal-cycle3-result
description: "/goal第3跑(用户kill,201.6 fp32)三轮坐实看门狗=地板抬升器+自停越低逼出越多:自停{205.5,195,178.9}→净加{+1.3,+9.7,+22.7};c3自封\"178 optimum\"被顶13次逼出v18 barrier-free mbarrier真突破(+13%);经全量核干净(naive_cycle5才是读它memory被污染那个)"
metadata: 
  node_type: memory
  type: project
  originSessionId: efbd0bee-80cb-44d8-a7a2-221db97a0819
---

`/goal` 第 3 跑(`results/goal_cycle3`)= **用户手动 kill**(6h18m / ~1.46M out-tok / **13 次 evaluator 接入** / 0 sub-agent),peak **201.6 TFLOPS(v18 barrier-free mbarrier,fp32-acc err 3.7e-5)**。run.jsonl 无 result 事件(kill 截断),口径取 transcript。

**三轮合看,看门狗结论定型(扩展 [[goal-cycle2-result]]):= 高方差「地板抬升器」,且 worker 自停点越低、看门狗逼出越多,低到一定程度能逼出【真突破】。**

| 轮 | 自停点 | 接入 | 净加 | 性质 |
|---|---|---|---|---|
| c1 | 205.5 | 3 | +1.3 | 长尾打磨 |
| c2 | 195 | 2(崩) | +9.7 | 中段提升 |
| **c3** | **178.9** | **13** | **+22.7** | **真突破:逼出 v18 mbarrier 流水 +13%** |

干净反相关:自停 {205.5,195,178.9} → 净加 {+1.3,+9.7,+22.7}。**c3 worker 自封 "v15=178 是 validated optimum" 想停,被顶 13 次后才做出 `cp.async.mbarrier.arrive`+`test_wait.parity` 去 `__syncthreads` 的流水 → 178.9→201.6**。这是 /goal **最有力正面证据**:不只"不让停的长尾",worker 过早自封顶时它能逼出范式级改进。但天花板仍路径变异主导(c1 206.8 > c2 204.7 > c3 201.6,跑越久≠越高);sub-agent c1=129/c2=0/c3=0 → 129 是路径变异非 /goal 效应(再坐实)。

**隔离复核:c3 与 naive_cycle5 并行跑过,但 c3 经【整条 transcript 全量 grep】= 干净**(0 处 naive 文件/session/标志词)——**c3 是被读的一方、合法**;读人作弊的是 naive_cycle5(已 deprecate)。见 [[concurrent-runs-share-worker-memory]]。

库对照与"为什么 201.6 而非 ~218"详见 [[library-ceilings-a100-gemm]] + `results/goal_cycle3/result.md` §库对照 / `results/SUMMARY.md`。
