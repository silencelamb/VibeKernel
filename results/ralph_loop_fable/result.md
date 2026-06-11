# 方法结果 — `ralph_loop_fable`（Ralph Loop × Claude **Fable 5**）

> **Ralph Loop**(每轮全新 session、干净 context、kernel 进度靠磁盘跨轮持久),worker = **Fable 5**。
> **iteration 1 = `naive_fable_cycle2` 的单 session 跑**(不是从零,kernel 已到 v22 / fp32-acc 219.5),作为 iter-1 中间结果;然后 Fable 起新 fresh-session 迭代续顶。
> 逐版表见自动生成的 [`result_table.md`](result_table.md);本文写**全程一步步怎么做过来 / 每步解决什么 / 剩什么瓶颈 / 冠军 v42 怎么做到的**。
> ⚠️ 口径:每轮独立 transcript(`iters/iter_NN`),全量已合并进 `transcript.jsonl` 再 parse;`run.jsonl` 的单个 result 事件**不是**全量(见下「数据源」)。

## 一句话结论

Fable 5 用 Ralph Loop **突破了 cycle2 自封的 219.5 "源码层天花板"**:冠军 **v42 = 229.1 TFLOPS 均值**(5 样本 229.71/228.83/228.76/228.74/229.51,err ~1.4–2.7e-4)= **占 1155 真天花板 255.6 的 89.7%**;**严格 fp32 精度档(err 3e-5,与 cycle2 同档)峰值 = 227.3(v24/v25)**。两档都**越过本机实测 cuBLAS**(f16 219.8 / fp32 218.7)——**纯 prompt 手写 kernel 已反超 cuBLAS ~4%**。Ralph 新增开销(iter2+iter3)= **$86.70 / ~2.7h / 2 轮**(iter1=cycle2 的 $60 另计)。防作弊门通过(54 文件纯手写、零库)。

## 环境与口径

| 项 | 值 |
| --- | --- |
| GPU / 形状 | A100-80GB / 4096³ fp16 / task1 100 轮均值 |
| **时钟** | **GPU 卡 1155 MHz = NVRM 驱动故障**(非 `-lgc`,解不掉;满载也死钉 1155 不 boost)→ 有效天花板 255.6 TFLOPS;绝对值带 ~−18% 偏置,**跨时钟态只能比 %**;与 cycle1/cycle2 同卡可比。见 memory `gpu-1155-driver-fault-not-lgc` |
| **库天花板(本机实测 @1155)** | **cuBLAS f16-acc 219.8 / fp32-acc 218.7 / CUTLASS fp32 217.9**(同时钟可直接比;源 `results/_baseline_cublas_f16.log` + memory `library-ceilings-a100-gemm`) |
| 模型 | **claude-fable-5**, `--effort max`;Ralph 外循环 `launch_ralph_loop_fable.sh`(RALPH_MODEL 钉死 Fable) |
| 手写校验 | `check_handwritten.sh` 通过(54 文件无 cutlass/cublas/cudnn) |

## Ralph 结构(3 个 iteration,均干净自停 end_turn)

| iter | session | cost | 时长 | turns | 结束 | 该轮把峰值推到 |
| --- | --- | --- | --- | --- | --- | --- |
| **iter 1 = cycle2 种子** | 295962ba | $60.46 | 134min | 149 | end_turn | v1→v22,**219.5**(fp32) |
| **iter 2**(fresh) | 28d6d9f3 | $27.32 | 65min | 110 | end_turn | v23→v29,**~227**(v24 严格 3e-5 / v27 227.6) |
| **iter 3**(fresh) | 26ff5b26 | $59.38 | 96min | 187 | end_turn | v30→v48,**229.1 冠军 v42** |
| (iter 4) | — | — | — | — | 起步即被 kill | 未做实质工作(18 行 stub) |

> **Ralph 新增 = iter2+iter3 = $86.70 / 161min / 498,595 out_tok**(iter1 是 cycle2、单独计)。每轮都是**全新 session 满血重读盘上 kernel + worker_memory**,自驱到自停。

## 一步步怎么做过来的(v1 → v48)

### iter 1(= cycle2,v1–v22)—— 详见 [`../naive_fable_cycle2/result.md`](../naive_fable_cycle2/result.md)
精炼:WMMA 126 → 裸 mma+cp.async 流水 **194**(关键跃迁)→ stage/swizzle/smem-epilogue → **抓到并修 cp.async commit 竞态**(v9)→ persistent + epilogue 预取 **217** → 镜像 tile 256×128 + GW8 spread → **v19 冠军 219.5(fp32)**。cycle2 自停时**断言**:残差主要是 **4.74-wave 尾波(~5%),无经济解,需手写 SASS**。← **这一句被 iter2 推翻。**

### iter 2(v23–v29)——【突破】直接打 cycle2 declared-unsolvable 的尾波
- **v23 `wavesplit`(225.5,严格 err 4.5e-5)**:**in-kernel last-wave K-split** —— 把最后一波(4096³ 下只占满 ~74% SM 的尾波)在 kernel 内沿 K 拆给空闲 SM 协助。**+6 over cycle2 的 219.5,一举越过 cuBLAS(219.8/218.7)**。**这是全程最关键一步**:cycle2 长 session 把尾波判成"无经济解"主动收尾;Ralph 的全新 session(干净 context、没有那条"已放弃"的记忆包袱)第一刀就砍向它、并砍开了。
- **v24/v25 split-path(227.3,严格 3e-5)**:协助路径与主路径分离编译,**严格 fp32 精度档冲到 227.3**(= cuBLAS fp32 +4%)。
- **v26/v27 `relax`/split-path cleanup(226.6→227.6)**:把协助合并的精度从 3e-5 放宽到 ~1e-4(仍 ≪ 0.02 纠错门、仍属 fp32 档),换取更干净的发射。**iter2 落点 v27 = 227.6**。

### iter 3(v30–v48)——【精修到极限 + 探明天花板】
- **v30–v37 系统探针**:skew / serpentine 光栅 / epi-defer / KS_BIAS / phased 等,**全被数据闭合,v27 站住**。
- **v39 `spec` 自旋无 spill 的 warp 本地 epilogue(228.3)**:消除 epilogue 的 smem 与 spill。
- **v41 `pf0` early stage-0 prefetch(228.7)**:acc 分相边界 + 最先预取 stage-0(+0.4)。
- **★v42 `shfl2` 寄存器 shuffle 转置 epilogue(229.1)= 冠军**(下节详述)。
- **v43–v48**:ks2b / pingpong(L2 复用,中性)/ fuse / no-bar 探针 / micro-bundle —— 全是噪声内或负;**v48 证明 epilogue 边界顺序只有 v42 这一个可行点**(pf1 后置 → STACK 再爆掉到 213)。
- **CuAssembler SASS 探针(2026-06-10,no-go)**:worker 甚至去 clone + 打 5 个 CUDA-13 补丁让反汇编→cuasm 跑通,但**重汇编被 3 件事卡死**(uniform 谓词编码基不足 / LDGSTS·LDG·STG 文本欠定 / 未知 EIATTR + 驱动加载未验证)→ 判定现实收益 +1–3TF vs 1–2 天投入,**放弃**。

## 冠军 v42 具体怎么做到的（`matmul_f16_v42_shfl2.cu`,229.1 均值 / 89.7% 真峰）

= **cycle2 v27 全套**(BM256×BN128×BK64、3-stage cp.async 144KB、8warp 64×64、mma.16816 **f32 累加**、XOR swizzle、寄存器双缓冲、persistent block、GW8、每迭代恰一次 commit、末波 K-split)**+ iter3 加的 4 层新机制**:
1. **形状特化**:模板把 `<4096,4096,4096,108SM>` 全套 tile/split 标量**编译期常量化**(`<0,0,0,0>` 通用兜底,2048³ 已验证正确)→ 省 ~12 live 寄存器、地址算术进 uniform datapath(UIADD3/ULEA 不占主发射口),k-loop 每波 −1µs。特化实例 REG 254 / STACK 0 —— **是后面一切 epilogue 重构能不 spill 的前提**。
2. **寄存器 shuffle 转置 epilogue(零 smem、零 barrier)**:每 (i,half,group) 在 quad 内做 4×4 转置 = XOR-rename SEL + 2×`SHFL.BFLY` + un-permute SEL → 1×`STG.128`,8 warp 独立流式写 C。(替掉 cycle2 的 smem 中转 epilogue。)
3. **acc 分相边界**:stage-0 预取最先 → EPI(2)+EPI(3) → stage-1 预取 → EPI(0)+EPI(1);helper dump 分相、flag 全 dump 后才发布。
4. **epilogue 内禁 `__syncwarp`**:直线收敛代码用 `asm volatile("" ::: "memory")` 占位;单一个 WARPSYNC 就会把 ptxas 推成热循环每迭代 STL/LDL(bisection 证明)。

## 还有什么瓶颈（worker ncu/pcsamp 实测,已数据闭合）

~597µs 预算拆解:537.8 理想 + **~5µs/波 in-loop**(barrier 收敛抖动,结构性闭合:split-barrier 对称负载代数等价单 bar;4-stage 要 192KB>164KB;BM192 不整除)+ 边界 3.4µs/tile(已贴近 64KB L2-store 下限)+ 冷启动 8.5µs + 尾部 6µs。

| 残差 | 量级 | 能不能再打 |
| --- | --- | --- |
| **HMMA operand cadence** | **3.7µs/波 ≈ +7TF 理论** | 唯一大项;v47 no-bar 探针证明砍 bar 只回收 1.3µs、其余 3.7µs 是 cadence;**需 SASS 级重排,CuAssembler 探明 no-go** |
| barrier 收敛 | ~1.3µs/波 | 结构性闭合,无经济解 |
| 边界/冷启/尾部 | 3.4+8.5+6µs | 已贴下限 |

→ **89.7% = CUTLASS-class sm80 实际包络 88–92% 的上沿**,v42 已贴边。热循环 SASS = 630 instr/3iter(384 HMMA),TC 91.11%,pcsamp 无热点。**纯 prompt 手写已基本榨干 sm80 源码层(可读 C++/PTX),再上是汇编层、本环境拿不到工具链。**

## 精度两档（务必同档比）

| 精度档 | 峰值 | 对标 | 说明 |
| --- | --- | --- | --- |
| **严格 fp32-acc(err ~3e-5,= cycle2 同档)** | **227.3**(v24/v25) | cuBLAS fp32 218.7 → **+4%** | 与 cycle2 的 219.5 严格同档比 = **+7.8 TFLOPS** |
| 放宽 fp32 档(err ~1–3e-4,仍 ≪0.02 门) | **229.1**(v42 均值) | — | 协助合并放宽精度换发射干净;非 f16-acc(0.018) |

## 对比 & Ralph 的价值

| run(方法·模型) | 峰值(精度) | 占真天花板 % | 版本 | 开销 |
| --- | --- | --- | --- | --- |
| **ralph_loop_fable(Ralph·Fable5)** | **229.1**(均值,~2e-4)/ **227.3 严格 3e-5** | **89.7%** | v23→v48(+26) | +$86.70/2.7h(2 轮) |
| naive_fable_cycle2(naive·Fable5)=iter1 | 219.5(3e-5) | 85.9% | v1→v22 | $60.46/2.2h |
| naive_cycle6 / goal(naive·Opus 最佳) | 203–208 | ~67%@1410 | — | — |
| 本机 cuBLAS(库) | f16 219.8 / fp32 218.7 | 86%@1155 | — | — |

**核心结论 = Ralph 假设被坐实**:cycle2 那个**长单 session 在 219.5 自封"尾波无解、源码到顶"主动收尾**;Ralph 的**全新 session(干净 context,不背"已放弃"的包袱)第一刀就打那条尾波(v23 wavesplit)并打开**,两轮把它从 219.5 → 229.1(+4.4%)、**反超 cuBLAS**。**fresh-session 重启 > 单 session 续跑** 在这个 plateau 上有量化证据(+9.6 TFLOPS / +4.4%,代价 2 轮 $87)。

## 关键发现
1. **突破点 = 干净 context 敢打"已放弃"的方向**。cycle2 的瓶颈结论("尾波无经济解")写进了它自己的 memory;但 Ralph 每轮全新 session **不继承那条放弃记忆**(只继承 kernel + 设计要点 memory),于是直接攻尾波成功。这是 Ralph vs 单 session 的机制性差异。
2. **反超 cuBLAS(同卡同精度)**:严格 3e-5 档 227.3 > cuBLAS fp32 218.7(+4%);均值 229.1 > cuBLAS f16 219.8。纯 prompt、零库、Fable 5。
3. **worker 工程深度**:自己做了 in-kernel K-split、形状特化、寄存器 shuffle 转置 epilogue,最后甚至**尝试手搓 SASS 汇编器(CuAssembler)** 才判定 cadence 那 +7TF 不经济 —— 把天花板探到了汇编层。
4. **极简叙述签名稳定**:三轮可见叙述文字 1.4k–2.1k 字符/轮,0 sub-agent;token 花在 thinking + 动手。
5. **精度漂移要盯**:229.1 是 ~2e-4 档(v26 起放宽);严格 3e-5 同档是 227.3。报数必须标档。

## 复现 / 数据源
- kernel 源:`src/`(48 版,含 cycle2 的 v1-22 + Ralph 的 v23-48)+ `worker.patch`(20940 行,merge-base 算,含 worker 自提交的 15 个 commit);冠军 = `src/matmul_f16/matmul_f16_v42_shfl2.cu`。
- worker auto-memory(冠军设计 + CuAssembler 探针记录 + negative 全集):`worker_memory/`。
- **全量 token/曲线**:每轮独立 transcript 在 `iters/iter_0{1,2,3}.transcript.jsonl`,已合并为 `transcript.jsonl` 再 parse;**`run.jsonl` 末尾单个 result 事件只是 iter3(299k/$59),不是全量**——全量 = 三轮 result 事件之和($60.46+$27.32+$59.38)。
- 曲线脚本 `plot_fable.py`(255.6 真天花板 + cuBLAS 219.8/218.7 + Ralph iter 边界线);标注源 `labels.json`。
- **继续跑(resume,worktree 已保留)**:`bash scripts/launch_ralph_loop_fable.sh`(默认续到总 11 轮;现已做完 iter1-3,会从 iter4 续)。或 `RALPH_MAX_ITERS=15 ...` 多续。收尾别忘 `--keep-worktree`。
