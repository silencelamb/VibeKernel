# method_table — `naive_fable`(Claude **Fable 5**,naive harness)

> naive 范式的**模型对比臂**:与 `launch_naive.sh` 逐字相同,唯一变量 = 主 worker 模型 `claude-opus-4-8` → `claude-fable-5`。
> ⚠️ **跨模型横比的头号坑**:本轮评测 GPU 被**硬锁在 1155 MHz**(实测 v9 跑出 211.7 TFLOPS 时 SM 全程钉 1155、321W、不 boost),真天花板 = 312×1155/1410 = **255.6 TFLOPS**;而 Opus 老跑(naive_cycle3 引用 1410)在更高时钟下。**绝对 TFLOPS 跨模型不可直接比**——下表同时给【绝对值】与【占当轮时钟真天花板的 %】,后者才是公平指标。

---

## A. 每个版本的改进与瓶颈(naive_fable 内部轨迹,13 版)

| 版本 | 改进(改了什么) | 实测 best TFLOPS(精度档) | 瓶颈 / 为何止步或回退 | 去留 |
| --- | --- | --- | --- | --- |
| v1 `mma` | `mma.m16n8k16` HMMA 基线,单 warp-tile | 122.0(fp32-acc, 3.7e-5) | mma:ldmatrix 比低、global 延迟没藏住 | 起点 |
| v2 `bigtile` | 放大 CTA tile 到 128×256,smem 复用↑ | 150.2(fp32-acc) | 仍无软件流水、global 延迟暴露 | ✅进 |
| v3 `pipe` | **cp.async 3-stage 软件流水**(关键跃迁) | 192.5–193.1(fp32-acc) | fp32 累加 warp 数受限 → 封在 ~193 | ✅进 |
| v4 `sched` | warp-tile 64×64 + 指令调度,藏 mma 延迟 | **201.8(fp32-acc, 3.6e-5)** ← fp32 峰 | fp32 累加封顶 ~202;再上必须降精度 | ✅进 |
| v5 `unroll` | 按 K-tile 展开实验;**转 f16 累加** | 209.1(f16-acc, 0.020) | 按 STAGES 展开→REG 255+STACK 溢出**崩到 57**;改 f16-acc(err 跳 ~500×)才换回 209 | ⚠️半进(留 f16-acc、弃展开) |
| v6 `streamk` | stream-K 分块抢占,想救 4.74-wave 尾 | 191.3(f16-acc) | GPU 贪心调度**已抹平尾巴**;stream-K 任何变体 −3~−50 | ❌回退 |
| v7 `2launch` | dual-launch 双 kernel 处理波尾 | 207.0(f16-acc) | **证实"尾部损失"多为幻觉**;拆两核没赚 | ❌回退 |
| v8 `straddle` | 手动把 mma 搬到 barrier 前(straddle) | 193.9(f16-acc) | **ptxas 早已自动跨 barrier 调度**;手搬反更慢 | ❌回退 |
| **v9 `tune`** | **全栈调优 champion**:CTA128×256 BK64、3-stage×48KB、旋转 stage 指针、寄存器 frag 双缓冲、XOR-swizzle、GROUP_M=16、B-frag 先载 | **211.5(f16-acc, 0.018)** ← **总峰** | **82.7% of 1155-锁定真天花板 255.6**;残差 = k-step 交接 stall(首 B-LDSM + 首 1–2 HMMA),~5–7% in-loop gap **需手写 SASS**(环境无汇编器);TC-pipe-active 89%、REG 168、零 bank conflict、零 spill | ✅**冠军** |
| v10 `bk32` | BK=32 × 6-stage | 197.8(f16-acc) | −13;6 段 smem 压力 + k-step 变多得不偿失 | ❌回退 |
| v11 `graph` | CUDA-graph launch | 210.9(f16-acc) | ±0;launch 开销本就 ~1%,图化无收益 | ❌回退 |
| v12 `256x128` | 256×128(CUTLASS 形状) | 199.0(f16-acc) | 邻域 config 全更差(register/smem 墙) | ❌回退 |
| v13 `f32acc` | = v9 机器,换回 fp32 累加(保险版) | 201.0(**fp32-acc, 3.4e-5**) | −5% 性能换回精确精度;**双交付的精确档** | ✅留(精确档) |

**轨迹小结**:122 → 150(大 tile)→ 193(cp.async 流水,关键)→ 202(fp32 封顶)→ 转 f16-acc 突破 → 211.5(v9 全调优)。v6–v8、v10–v12 是**系统化消融**:stream-K / dual-launch / 手动 straddle / BK32 / CUDA-graph / 256×128 **逐一实测后否决**(理由全在上表)——这是高质量工程纪律的体现,不是失败。最终瓶颈一句话:**源码层已到顶(~83% 真峰),剩下的全是 ptxas 调度质量 + 波尾 + launch gap,只有手写 SASS 能再榨,而本环境无汇编器** → worker 自停。

---

## B. 跨方法 / 跨模型对比(同 naive harness,同一份 seed)

| run(模型) | fp32-acc 峰(err) | f16-acc 峰(err) | 时钟(真天花板) | **占真天花板 %** | cost | turns | out_tok | wall | sub-agent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **naive_fable(Fable 5)** | 201.8 / 201.0终 (3.4e-5) | **211.5**(0.018) | **1155 锁定✅实测**(255.6) | **fp32 79% / f16 82.7%** | $53.14 | 145 | 309,897 | 96min | 0 |
| naive_cycle3(Opus) | 142(3e-5) | 154.2(0.016) | ~1410(引用,312) | fp32 45% / f16 49% | $20.70 | 92 | 360,368 | 92min | 0 |
| naive_cycle6(Opus) | 203.0(3.1e-5) | 208.3(0.018) | Opus 期(未记录) | ≈67% @1410 / ≈81% @1155 | $12.55 | 72 | 233,790 | 93min | 0 |
| goal(Opus,有看门狗) | 206.8(3.0e-5) | — | Opus 期(未记录) | ≈66% @1410 / ≈81% @1155 | $87.52 | 284 | 817,132 | 242min | 129 |

**怎么读这张表(关键)**:
- **绝对值**:Fable 211.5(f16)/201.8(fp32)≈ 略高于 Opus 家族最佳(208.3/203),且**远高于**用户指定的对照 naive_cycle3(154/142)。
- **但绝对值跨模型不可信**:Fable 在 **1155 锁定**(真峰 255.6)、naive_cycle3 在 ~1410(真峰 312)。把 Fable 的 211.5 线性折算回 1410 ≈ **258 TFLOPS-等效**,会盖过所有 Opus 跑——但这依赖完美时钟线性,仅作参考。
- **公平指标 = 占真天花板 %**:Fable **82.7%** vs naive_cycle3 **49%**(同为可 pin 时钟的两轮)→ Fable 的 kernel 效率显著更高。对 cycle6/goal,**它们的时钟没记录**,若也在 1410 则 Fable 完胜(82.7% vs 67%),若它们其实也在 1155 则三者打平(~81%)——**无法定论,因为历史时钟没存**。
- ⚠️ **效率比在低时钟略被高估**:1155 下算力峰降到 255.6 但 HBM 带宽不变 → 访存相对变便宜、易冲高 %。但 v9 ncu 显示 TC-pipe-active 89%(算力绑定),高估幅度有限。

**最大方法学教训**:GPU 时钟状态在两次跑之间**变了且没被控制 / 记录** → 整个 naive 家族的绝对 TFLOPS 横比都受污染。**今后每轮必须 log 实测 SM 时钟**(见 `result.md` 复现节)。

---

## C. 模型行为差异(Fable 5 vs Opus,非性能)

| 维度 | Fable 5(naive_fable) | Opus(naive_cycle3) | 量化 |
| --- | --- | --- | --- |
| 可见叙述文字 | 极简 | 啰嗦 | **1,684 字符 / 11 块** vs **22,510 字符 / 54 块**(~13× 少) |
| 动作密度 | 高 | 低 | tool_use **144**(123 Bash)vs 91 |
| 自发 git commit | ✅ 4 次带详细 message | ❌ 不提交 | 见 worktree git log(harness 因此踩坑,见 result.md) |
| 环境判断 | ✅ 自己测出 1155 锁定、改算 255.6 真峰 | ❌ 默认 1410/312 | Fable 态势感知更准 |
| 成本/token | 贵但 token 略少 | 便宜 | $53 / 310k vs $21 / 360k → **多 turn(145 vs 92)→ cache_read 32.8M vs 17.8M** 撑高成本 |
| sub-agent | 0 | 0 | (goal 才 129) |

详见 `result.md` 的「输出文字对比」与「关键发现」。
