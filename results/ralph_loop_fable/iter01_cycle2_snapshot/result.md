# 方法结果 — `naive_fable_cycle2`(naive harness + Claude **Fable 5**,第 2 跑)

> naive 范式 Fable 5 臂第 2 跑(与 cycle1 同卡、同 seed、同 harness → cycle1↔cycle2 是干净的**重复性/方差**对比)。
> 逐版表见自动生成的 [`result_table.md`](result_table.md);本文重点 = **一步步做了什么 / 每步解决了什么 / 剩什么瓶颈 / 冠军 v19 怎么做到的**。

## 一句话结论

Fable 5 干净自停(end_turn)于 **fp32-accumulate 219.5 TFLOPS**(冠军 v19,err ~3e-5,**全精度**),22 版,0 sub-agent,134min / 397,616 out_tok / 149 turns / **$60.46**。⚠️ GPU 驱动故障卡 1155 MHz(有效天花板 255.6)→ 219.5 = **占真峰 85.9%**(naive 家族最高)。**比 cycle1 更强且更精确**:cycle1 靠 f16-acc(降精度)才到 211.5、其 fp32 仅 201.8;cycle2 **直接在 fp32 精度下冲到 219.5**。防作弊门通过(28 文件纯手写、零库)。

## 环境与口径

| 项 | 值 |
| --- | --- |
| GPU / 形状 | A100-80GB / 4096³ fp16 |
| 计分口径 | task1 二进制 100 轮均值(sustained);4096³ canonical |
| **时钟(关键)** | **GPU 卡 1155 MHz = NVRM 驱动故障**(`pSmIssueThrottleCtrl` assertion,非 `-lgc`,解不掉;满载也死钉 1155 不 boost)→ 有效天花板 = 312×1155/1410 = **255.6 TFLOPS**。**绝对 TFLOPS 带 ~−18% 偏置,跨时钟态只能比「占真天花板 %」**。见 memory `gpu-1155-driver-fault-not-lgc` |
| 模型 | **claude-fable-5**, `--effort max` |
| 手写校验 | `scripts/check_handwritten.sh` 通过(扫 28 文件无 cutlass/cublas/cudnn) |
| 总计(权威 result 事件) | wall 8037s(134min)、out_tok 397,616、turns 149、cost $60.46、0 sub-agent |

## 一步步做了什么 / 每步解决了什么(22 版,分 7 个阶段)

逐版数字见 [`result_table.md`](result_table.md) / [`curve.png`](curve.png)。下面按**它实际的推进逻辑**分段讲(版本号 = worker 自己的命名,worker 还自 commit 了 5 个里程碑,见 `worker.patch` 来源的 git log)。

### 阶段 1 — 立骨架:WMMA → 裸 mma + 软件流水(v1→v2,**126→194,+54%**)
- **v1 `wmma`(126.5)**:用 WMMA C++ API 起手 —— 能跑但慢。问题:WMMA 经 nvcc 自动 codegen、无法控制软件流水与寄存器布局。
- **v2 `mma_pipeline`(194.0,fp32-acc)**:换成**裸 PTX `mma.sync.m16n8k16` + `cp.async` 多级软件流水**。**这是全程最大单步跃迁**(126→194):异步拷贝把 global→smem 的延迟藏进计算,显式 mma 拿回寄存器/调度控制权。**自此精度档锁定 fp32 累加(err 3e-5)**。

### 阶段 2 — 喂饱流水线 + 修内存路径(v3→v8,194→210.5)
- **v3 `bk64`(190)/ v4 `stage4`(196)**:调 BK=64、加到 4 级 stage —— 更深流水进一步藏 global 延迟(v3≈噪声,v4 站稳 196)。
- **v6 `swizzle`(196)**:smem 加 **XOR swizzle** 布局,**bank conflict 清零**(A:`c^(row&7)`;B:`(c&8)|((c&7)^(row&7))`)。
- **v7 `2cta`(200.1)**:试 2-CTA/SM + 顺手试 f16-acc,破 200,但靠降精度(err 0.016)、且 2-CTA 后撞寄存器墙 → 这条线后面被放弃。
- **v8 `smemepi`(210.5,回到 fp32-acc)**:**epilogue 经 smem 中转再合并写回 global**(B2 槽 8×32 行)→ 写带宽对齐,200→210,且回到全精度。**埋了一个隐蔽 bug(见阶段 3)**。

### 阶段 3 — 抓到并修掉一个真竞态(v9,**最有含金量的一步**)
- **v9 `streamk`(142,间歇错!err 0.0017)**:上 stream-K 修波尾,结果**性能暴跌且结果间歇性错**。worker 诊断出根因:**`cp.async` 的 commit/wait 计数纪律被破坏**——循环顶部无条件 `commit_group()` 而序言已 commit 完所有 stage,注入了一个**空 group**,使 `wait_group(N)` 的保证整体**错位一档** → 下一 tile 的 `ldmatrix` 可能读到**仍在飞行**的 cp.async 数据(v8 侥幸没炸,v9 stream-K 尾段炸出来)。
- **解决**:立下纪律「**每个主循环迭代恰好一次 commit、位置固定**」,并用对拍工具**连跑 110 次** launch 验证(间歇竞态单跑测不出)。stream-K 本身因 merge/边界成本 > 收益被弃。**这步没涨分,但把一个会偶发出错的正确性地雷拆了**——是这轮工程质量的关键证据。(已写进 worker memory `cpasync-commit-discipline`。)

### 阶段 4 — 摊薄开销 + 藏波尾:persistent + prefetch(v10→v13,210→217.2)
- **v10 `stage6`(210.6)/ v11 `bk64`(214)**:6 级 stage(逼近 144KB smem 墙)、重配 BK64。
- **v12 `persist`(216)**:**持久化 block** —— 一个 block 连做多个 output tile,摊薄 launch + 序言(填流水线)开销。
- **v13 `prefetch2`(217.2,fp32-acc)**:**在写 epilogue 的同时,预取下一个 tile 的 stage 0+1** → 把 tile 之间的"排空再填充"空窗藏掉。**首个冠军**。

### 阶段 5 — 一个反向决策:拒绝 f16-acc(v14,与 cycle1 相反)
- **v14 `f16acc`(217.9,err 0.0225)**:试把累加从 fp32 换成 f16。结果**只 +0.7 TF**,但 err 跳 ~500×(3e-5→0.0225)。**判定:不值,保持 fp32 全精度**。→ 这是 cycle2 与 cycle1 最大的路径差异:cycle1 拥抱了 f16-acc(冲 211.5),cycle2 算过账后**主动放弃**,坚持精确档反而走得更高(219.5)。

### 阶段 6 — 临门一脚:镜像 tile 形状 + panel 铺排(v15→v19,217→**219.5**)
- **v15 `tailsplit`(163,回退)/ v16 `tune`(216)/ v17 `2cta_bk64`(212)**:尾段拆分、邻域微调、2-CTA 重试 —— 都落在 v13 噪声内或更差。
- **v18 `mirror`(219.0)**:把 tile 形状从 v13 的 128×256 **镜像成 256×128**(同 BK64/3-stage/8-warp)→ 改善 raster/L2 局部性,**首次越过 217 平台**。
- **v19 `spread`(219.5,fp32-acc)= 冠军**:在 v18 镜像基础上加 **m-fast GW=8 panel 铺排序**(见下节)。**全精度下 219.5 = 占 1155 真天花板 85.9%**。

### 阶段 7 — 把天花板探明:一串受控负结果(v20→v22)
- **v20 `mbar`(168/86)**:mbarrier —— 在 sm80 上是**软件路径**,暴跌,弃。
- **v21 `bundle`(215)**:发射 bundle 重排,噪声内/略负。
- **v22 `tail2`(154)**:转置槽尾段拆分 —— **kernel-B 类启动有 `long_sb` 延迟泄漏 ~100µs/item**,负。
- 这三步不涨分,但**逐一钉死了"还能往哪试"的边界**,佐证 v19 已近源码层极限 → worker 主动收尾。

## 冠军 v19 具体怎么做到的(`matmul_f16_v19_spread.cu`,实测 217.8 中位 / 219.5 峰)

一句话:**一个塞满 A100 sm80 全部已知 GEMM 杠杆、且每个杠杆都经实测校准过的单 kernel**。要点(源码坐实):

1. **tile / 占用**:CTA **BM256×BN128×BK64**,8 warp(4×2)各算 **64×64** warp-tile;3-stage cp.async 流水 = `3×(32KB A + 16KB B)` = **144KB 动态 smem**(贴着 A100 上限,1 CTA/SM)。
2. **算核**:`mma.sync.aligned.m16n8k16.row.col.**f32**.f16.f16.f32` —— f16 输入、**f32 累加**(全精度);累加器 `float acc[4][8][4]` 全驻寄存器;A/B 片段用 `ldmatrix.x4`(B 用 `.trans`)。
3. **喂数**:`cp.async.cg.shared.global.**L2::128B** [...],16`(16B 向量、带 L2 提示);smem **XOR swizzle** 让 bank conflict=0;寄存器 **frag 双缓冲**(算当前 k-step 时载下一 k-step)。
4. **流水纪律**:**每迭代恰好一次 `commit_group`**(阶段 3 的教训)、`wait_group` 不变量可证 → 无空 group、无读到在飞数据。
5. **藏开销(v12+v13 的精华)**:**持久化 block** 跨多个 tile;**在 epilogue 期间预取下一 tile 的 stage 0+1**,把 tile 间空窗填上。
6. **写回**:epilogue 经 **smem 中转**(`EPI_STRIDE=BN+8` 消 padding 冲突)再 16B 合并写 global。
7. **铺排(v18→v19 的关键)**:**镜像 256×128** + **m-fast GW=8 panel 序**(panel 内 m 走最快、n 在 panel 内推进,tN=32),并发工作集 ~29.5MB → L2 命中最大化。

**为什么这就到顶了**:同一 kernel 在 **8192×8192×4096 跑出 235 TF = 锁频峰值的 92%**。这证明 4096³ 只到 85.9% 不是 kernel 低效,而是 **4.74-wave 尾波量化**(4096³ 下 grid 不能整除 wave、最后一波只占 ~74%)——属问题尺寸的结构损耗,非源码可救。

## 还有什么瓶颈(worker ncu 实测,已探明无经济解)

| 残差来源 | 量级 | 说明 |
| --- | --- | --- |
| 4.74-wave 尾波量化 | ~5% | 4096³ 下最后一波 SM 没喂满;8192² 几乎消失(→235/92%)。无经济解。 |
| barrier / k-step 交接 | ~3% | 每 K-tile 一次 `__syncthreads`,跨 stage 交接的 stall。 |
| HMMA cadence | ~3% | ptxas 调度的指令节拍空隙;再榨需**手写 SASS**,本环境无汇编器。 |

→ 结论与 cycle1 / Opus naive 一致:**源码层 ~86% 真峰已到顶,再上需 SASS 级手排**,worker 据此主动停。cuBLAS-class SASS 在此卡估 ~225-230。

## cycle1 vs cycle2 vs Opus(同 naive harness)

| run(模型) | 冠军精度档 | 峰值 TFLOPS | 占真天花板 % | 版本数 | turns | out_tok | cost | wall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **naive_fable_cycle2(Fable5)** | **fp32-acc(3e-5)** | **219.5** | **85.9%**(/255.6) | 22 | 149 | 397,616 | $60.46 | 134min |
| naive_fable(Fable5,cycle1) | f16-acc(0.018) | 211.5 | 82.7%(/255.6) | 13 | 145 | 309,897 | $53.14 | 96min |
| naive_cycle6(Opus,最佳naive) | fp32 203 / f16 208.3 | 208.3 | ~67% @1410 | 22 | 72 | 233,790 | $12.55 | 93min |
| goal(Opus,有看门狗) | fp32 206.8 | 206.8 | ~66% @1410 | — | 284 | 817,132 | $87.52 | 242min |

> ⚠️ Opus 跑在 ~1410(真峰 312)、Fable 两跑在 1155(真峰 255.6)→ **绝对 TFLOPS 不可直接比**;`占真天花板 %` 是唯一公平的跨时钟态指标。cycle1/cycle2 同在 1155、可直接比。

## 关键发现

1. **重复性好 + fp32 路径更优**:两跑都干净自停、都到 ~83-86% 真峰、都撞同款波尾/barrier 墙;但 cycle2 **走的是全精度 fp32 路径**(拒了 f16-acc)还更高(219.5 > 211.5)。说明 Fable 在 naive 下稳定逼近源码层极限,f16-acc 不是必需(cycle1 的降精度是路径变异,非更优解)。
2. **会抓真 bug**:阶段 3 的 cp.async commit 竞态是个**会偶发出错的正确性地雷**,worker 自己用 110 次连跑诊断+修复+写 memory —— 不是只调性能,有正确性工程纪律。
3. **极简叙述签名稳定**:可见叙述文字 **2,087 字符 / 15 块**(cycle1 1,684 / 11),依旧极简;tool_use 148(109 Bash),0 sub-agent。Fable 把 token 花在 thinking + 动手,不写散文简报。
4. **成本签名稳定**:$60.46、149 turns、cache_read 34M —— 与 cycle1 同型(多短 turn → cache-read 重)。

## 复现 / 数据来源

- kernel 源快照:`src/`(22 版)+ **`worker.patch`(7514 行,merge-base 算,含 worker 自提交的 5 个 commit)**;冠军 = `src/matmul_f16/matmul_f16_v19_spread.cu`。
- worker 自写 auto-memory(冠军设计 / GPU 锁频 / cp.async 纪律,极有价值):`worker_memory/`。
- transcript(token/曲线/文字对比权威源):`transcript.jsonl`;曲线脚本 `plot_fable.py`(画真天花板 255.6);标注源 `labels.json`。
- **时钟复现**:见 cycle1 `results/naive_fable/result.md` 复现节(高频采样;SM 全程 1155)。
- **resume(后面要跑 Ralph Loop)**:已 `--keep-worktree`,worktree + session_id(`295962ba-…`)+ worker_memory 全留。一条命令续:`bash scripts/resume_naive_fable.sh naive_fable_cycle2`(Fable 5 fresh-session 外循环;口径/坑见 `results/naive_fable/RESUME.md`,把名字换成 cycle2)。
