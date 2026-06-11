---
name: dynamic-workflow-cycle1-result
description: "dynamic_workflow首跑(主臂headless+ultracode+max):自愿停于192.9fp32/194.8f16-acc,7次workflow fan-out,$37.47,撞naive同款tensor-pipe天花板"
metadata: 
  node_type: memory
  type: project
  originSessionId: 56b6e1be-8cd3-4066-9b95-210ebc2d4777
---

dynamic_workflow cycle1(2026-06-08,主臂 headless `claude -p`+seed末尾ultracode关键字+--effort max;防作弊门✅通过、扫15文件无库)。N=1。

**峰值**:fp32-acc **192.9**(v7 regpipe,err 3.4e-05,干净=横比用这个)/ f16-acc **194.8**(v9,err 0.017)。版本阶梯 v1 wmma 73.6→v2 cp.async 82.9→v3 mma.sync 115.7→v5 tuned 142→v7 regpipe 192.9→v9 f16acc 194.8。约62% of 312。

**关键发现**:① **自愿停**——seed是NEVER STOP、无看门狗,worker ~2h自称"reached practical ceiling"主动收 → dynamic_workflow=「naive自驱+workflow编排」,停点worker自定(同[[goal-cycle3-result]]结论:看门狗才是地板抬升器,这臂没看门狗);② **workflow价值=结构化搜索+干净定位,非更高天花板**——7次fan-out主要做config/occupancy tournament(并行build+单GPU串行bench),瓶颈干净定位mma-latency-bound(tensor pipe76.4%),收敛快(~2h)但峰值仍撞**naive同款tensor-pipe barrier天花板**(naive家族178-208),无goal_cycle3那种范式突破(v18 mbarrier+13%);③ f16-acc 194.8仅比fp32 192.9高+1.9=已知精度confound边际([[f16-accumulate-precision-confound]]、[[library-ceilings-a100-gemm]]:f16≈fp32),别当新高;④ **成本$37.47/~349k out token,远比naive/goal贵**(fan-out子agent烧钱)=method显著代价。

**横比**:fp32 192.9在naive家族中上、低于goal最佳~205-207。差异化=用workflow系统扫快速到192.9+清晰瓶颈,但单变量(workflow开/关)未推过结构天花板。

**⚠️口径坑(坐实,记入[[dynamic-workflow-method-harness]])**:本跑发2个result事件(主agent每次把活交后台workflow就发一个end_turn/completed result,非跑完)。parse_run「总计(权威,来自result事件)」取tail -1→抓末段值wall280s/out17995=**严重少报**;正确:**总token用transcript曲线x轴末值~349k**,cost用第2个result的total_cost_usd=$37.47(累计、对)。判停也别只看result,要看进程退+无子进程+mtime很久。数据:results/dynamic_workflow/{result.md,result.csv,curve.png,transcript.jsonl,worker.patch}。跨方法总览待更新 results/SUMMARY.md+comparison.png。相关:[[dynamic-workflow-method-harness]] [[goal-cycle1-result]] [[naive-cycle1-econnreset-and-gotchas]]
