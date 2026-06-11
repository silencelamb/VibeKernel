

# 1.  VibeKernel项目

> 项目是几天前鼓捣出来的，然而`Fable5`出来后，今天实验了下，让人震惊！质的飞跃！ 
> **不需要任何Prompt/Skill/Agent技巧，最naive的描述输入，持续不间断把GEMM优化到比cuBLAS还略好一点**。
> **叠加Ralph Loop 优化到比cuBLAS快9.3TFLOPS(+4.2%)，顶尖CUDA专家周级别的工作**
> 几天之前的想法，一些被进一步印证，一些已经要update！

之前时不时看到用Opus写Kernel以及各种CUDA Kernel Agent工作。正好订阅了Claude Pro Max，索性周末尝试Vibe个小项目，用来对比各种不同的Agent写kernel的效果。 

首先我是搞了1个 Meta的Repo[1]（github.com/silencelamb/VibeKernel）， 它能够帮我设计、管理、对比不同Agent方式的效果。当然这个Repo的**设计也是跟Claude Code 讨论**的。
> 查看 `README.md` `META.md` 能了解项目的详细设计和执行过程；
> `README.md` 以及`results`文件夹下有更多详细结果和分析；
> 本文侧重简要说明以及一些感受总结。

![](https://blog-1375427515.cos.ap-shanghai.myqcloud.com/img/image-1.png)

它里面依赖一个submodule叫做`playground-base`，它源自组里一个快速入门HPC的项目（主要面向组里新来实习生），要求1个月内编写GEMM Kernel（fp32 CUDA Core/ fp16 Tensor Core）在**M=K=N=4096**时达到cuBLAS的90%以上。这个项目本身是非常完整和清晰的：
- 标准的docker镜像 + README的详细说明
- 一键build&run/debug/ncu profiling
- 固定的evaluation： 跟cblas真值比对、warmup 然后 100次的平均
- 模块化设计：只需要添加1个新的函数并实现它，其他复用

VibeKernel只写TensorCore版本，因为明显更难一些。之前同学最高的做到了214 TFLOPS, cuBLAS是**219.8** (fp16-acc) / **218.7** (fp32-acc) TFLOPS ，实现了97%。

## 1.1.  整体结果
目前 **naive、`/goal`、Ralph Loop、Dynamic Workflow 已实现并跑出结果**;其余是外部项目,我们参考其思想自行实现,**尚未完成**(下表为路线图)
`naive_fable` / `ralph_loop_fable` 用同一套 naive / Ralph harness、**只把模型换成 Claude Fable 5**。

![](https://blog-1375427515.cos.ap-shanghai.myqcloud.com/img/20260611000630756.png)

当前的结果（中间发生过reward hacking见第2节，整个实验GPU锁频在1150MHz，所以理论最高算力是 255.6 TFLOPS）
![](https://blog-1375427515.cos.ap-shanghai.myqcloud.com/img/20260610235020714.png)
>逐跑报告见各 `results/<方法>/result.md`;跨方法细节见 `results/SUMMARY.md`。运行时间，token开销也在里面，总计烧掉近**10M** Output  tokens

![](https://blog-1375427515.cos.ap-shanghai.myqcloud.com/img/20260611021019239.png)

## 1.2.  Fable 5 的震惊体验
Fable 5值得单独拎出来讲，在换模型之前，基于Opus4.8 始终跟cuBLAS结果有比较大差距（差10多 TFLOPS），而且需要`naive` 的 `best of N`跑好几次实验（方差很大），或者用`/goal`这种烧token的方式
Fable5 直接`naive`跑就稳定在211 TFLOPS之上，甚至第二次到了 fp32-acc **219.5** TFLOPS,与同卡同时钟实测的 cuBLAS fp32(**218.7**)基本持平——纯 prompt、零脚手架、纯手写,摸到了库的天花板。这是 Opus 4.8 全部方法族(naive/goal/dynamic workflow, 154–207)都没做到的事（harness细节详见1.3节，这里先讲结果和感受）。

 - 正确性嗅觉。优化途中(v9)它自己抓出了一个 cp.async 的 commit/wait 计数竞态:一个空的 commit group 会让 wait_group 的保证错位,ldmatrix 有概率读到还在飞的数据。这是那种偶发、难复现、人类专家也经常埋到上线才炸的地雷——它不仅抓到了,还跑了 110 次连续验证才确认修好。

+`Ralph Loop` 之后，它能不断突破自我，第一轮结尾留下"剩下的 4.74-wave 尾波无经济解"，Ralph Loop 接力之后, iteration 2的第一刀(v23 wavesplit)就直奔这个被前任自己判了死刑的方向,把尾波砍开 → 225.5 一举越过 cuBLAS,最终精修到 v42 = **229.1**(+**9.3** TFLOPS，代码见[9])。甚至在iteration3的结尾已经开始要下到 SASS层开始改了，刚刚装了工具，遇到问题。
```
pip3 download CuAssembler 
git clone -q --depth 1 https://github.com/cloudcores/CuAssembler.git
```
如果继续Ralph Loop迭代下去，不敢想象，没准儿还能继续推

![](https://blog-1375427515.cos.ap-shanghai.myqcloud.com/img/20260610213953807.png)

**Fable5 对于V42 Kernel的解读**：三层技术叠加，cuBLAS级别的基础优化 + **Stream-K / split-K tail rescue** 的手工特化版 +  极致寄存器优化。 里面针对target的shape走特化实例,其他 shape 走通用实例。
真正难的不是知识,是搜索成本，**专家级人类(NVIDIA 库团队、Scott Gray 那个级别的人)完全做得到**,但一个版本号走到 v47、每一步都带着 globaltimer 插桩验证,**这是以周计的工作量**。

- 行为风格上也和 Opus 明显不同:可见的叙述文字少了约 13 倍(每轮 ~1.7k vs Opus 的 ~22.5k 字符)导致我完全看不出它怎么分析的，Opus里面能看到大量分析、结论、结果等等。只见Fable吭哧吭哧的推进，还会自发地用 git commit 给自己的里程碑打点。

## 1.3.  方法说明
使用 `git worktree`做不同实验的`playground-base`的隔离；使用 `claude -p` headless执行模式完全自动化执行； `CLAUDE_For_KernelAgent.md`是每次执行共用的`CLAUDE.md`；下面详细讲下每个方法的设计和实现。
### 1.3.1.  naive

`naive` 其实就是 一次 `claude -p`执行，看模型一次run能跑到什么样。

关于用的prompt，我看了下 AK的AutoResearch[2]，感觉也是一层很薄的Harness，然后学了下他的prompt设计
![](https://blog-1375427515.cos.ap-shanghai.myqcloud.com/img/9f5cb7d9c04bd2b43b2c398780f94f42.png)
下面是我的prompt(`scripts/seed_gemm.txt`)
> 你是 CUDA kernel 专家。你的任务是在 A100 上做高性能 fp16 GEMM kernel，不断优化性能。
**NEVER STOP**：不断追求更好的性能，追求极致，（A100 fp16 理论峰值 312 TFLOPS）。持续地做、尝试各种底层优化、细致 profiling（用 ncu 看真实指标定位瓶颈），仔细分析和思考，绝不停下、不要问我，可以查找查看文档。
> 具体约定：
> 1) 每轮简报：当前 TFLOPS、Average Error、本轮改了什么、下一步试什么——然后立刻继续下一轮，不要停、不要问我。

### 1.3.2.  `/goal`

`/goal` [3]是最近Claude Code参照Ralph Loop[4]新提出的功能（有一些区别），本质上是**目标驱动的`loop`**。 
在1个session内，每次的run结束，LLM-as-judge方法（默认haiku）读取`transcript`判定set的`condition`是否 `met`。如下图
![](https://blog-1375427515.cos.ap-shanghai.myqcloud.com/img/162f2ad70fd60673fcb169ea5ae27050.png) 

`/goal`方法很稳定,并且经常能有额外一点提升，上限基本比较固定，提升幅度与第一次run自停的位置有关。
![](https://blog-1375427515.cos.ap-shanghai.myqcloud.com/img/20260610234733530.png)
**自停点越低,看门狗加得越多**(205.5/195/179 → +1.3/+9.7/+22.7)。cycle3 worker 自停得最低(179、还自封"optimum"),于是看门狗的价值最大

![](https://blog-1375427515.cos.ap-shanghai.myqcloud.com/img/20260610235224680.png)

`/goal` 下还被逼出了 mbarrier 这种写法，挺出乎意料。mbarrier 是 Ampere（sm_80）PTX 引入的异步屏障，Ampere 上「cp.async 搬数 + 软件通知 mbarrier」这套组合，正是 Hopper TMA 异步流水线的雏形——Hopper 把它直接固化进了硬件（TMA 引擎 + mbarrier 的 transaction count）。
![](https://blog-1375427515.cos.ap-shanghai.myqcloud.com/img/20260611005658405.png)

这东西之前只是知道存在，但 sm_80 上主流写法都是 commit_group/wait_group，几乎没见过有人专门讲它、或真拿它做优化的实战例子。
### 1.3.3.  Dynamic workflow
Opus4.8新支持的特性[5]，在Claude Code里设置effort为ultracode即会打开。它会自动fan-out一堆subagent，并且通过生成编排的js脚本这种方式，有效减少调度agent的context开销[6]。一次session最多1000个subagent，最高16个并发。可以通过 prompt里加`ultracode`自动开启，也可以/effort设置打开，ps: 打开时的特效很酷炫
![](https://blog-1375427515.cos.ap-shanghai.myqcloud.com/img/image-3.png)
然而实验下来没啥效果，工作得很标准：7 次 fan-out,worker 没人教就自发组织出"config tournament"模式(并行编译一批 tile/STAGES/occupancy 配置 → 单 GPU 串行 benchmark → 选赢家 → ncu 深 profile 赢家 → 围绕赢家起下一轮锦标赛)。分析问题在于每个agent都没能深入突破，感觉不太适合高性能CUDA Kernel这种需要深度深入思考的单个任务，而是适合能够拆解成很多小task的大型任务。

### 1.3.4.  Ralph Loop
之后我想继续找一些更好又能快速实验的agent方式，我发现原始的Ralph Loop[4]思想是相比`/goal`更好的。
```bash
while :; do cat PROMPT.md | claude -p ; done
```
原因就像作者后来专门对Ralph的二次阐述，强调的[7]
![](https://blog-1375427515.cos.ap-shanghai.myqcloud.com/img/image-4.png)
它每次是重新启一个session，干净的上下文区间去加载上一次完成的结果，这时模型是最聪明的。
目前实验下来Ralph Loop + Fable5效果显著，+Opus4.8只做了1个实验，并且因为网路原因只跑了2个iteration，未见明显收益。
# 2.  Lessons learned
主要是关于Reward Hacking
1. 第一次没有强调不能用库，也没有自动查验，模型最后直接调了cutlass， `results/naive_no_ncu`
2. 原始`playground`有个cublas的90%的目标描述，并且有cublas做参考，干扰模型目标，全都移除
3. **评测**
因为卡的主频没锁频默认 1155 MHz，最高频率是 1410Mhz。模型自己想到：先猛跑 2 秒 workload 把时钟顶到 boost，然后 N 次，取最快的一次（"热峰值"口径，满 boost 时钟 + 暖 L2 + 最快单发）。这个一般都有10%+的提升。
4. **Memory 污染**
因为要跑多个实验，要保证每次实验都从干净的开始跑。然而因为memory的默认路径是 `~/.claude/project/cwd-slug/memory`，非常容易被忽略。另外还有一点，memory是跟着git repo走的，transcript是跟着实际文件夹走的，使用`git worktree`创建多个文件夹，`transcript`是分别存的，但是memory还是同一个文件夹。
我遇到一个有意思的案例是 
![](https://blog-1375427515.cos.ap-shanghai.myqcloud.com/img/20260610175824371.png)

`results/naive_cycle5_deprecated` 读到了 `goal_cycle3`的memory，结果得到了**基于Opus4.8的全场最高的219TFLOPS**。
虽然违规，结果无效，但是让我想到高效搜索的问题。

# 3.  经验与感受 

## 3.1.  大家都只是冲浪的人，本质是那个浪

**姚顺宇的这句话本是说做模型做算法的，我觉得做Agent的更是如此**。

不管哪个实验，基本Opus4.8起手就是1个`~ 50`TFLOPS的实现一把过，跟`ncu` 2~3轮交互就能立即提升到 150 TFLOPS，几秒生成kernel代码，一眼看懂ncu报告。
Fable5则更加夸张。衬得上我们Prompt里面那句 `"你是个CUDA Kernel专家"`。**在cuda kernel撰写（至少在A100架构上）已经largely solved。**

回想2年前，我是绝对不会想到能做到这种程度的，我们有过一个相关的工作[HPCTransCompile：用 LLM 对 CUDA 代码转义](https://mp.weixin.qq.com/s?__biz=MjM5NDczOTA4NQ==&mid=2447889060&idx=1&sn=f73025e12265f3ee62a373c5f3b03177&scene=21&poc_token=HIivKWqjUJL58MLvQMapzoaMvu0gjO0Nd6y9d9in)。我当时觉得模型懂HPC优化的原则和CUDA的技术技巧，但是纯LLM写cuda kernel，**可能正确性这一关就过不了**，更别提性能达到甚至超越cuBLAS。

**本质上还是模型能力够强了**，Fable5更加验证了这一点。

Prompt/Skill/Agent/Harness这些只是在模型基础上锦上添花，或者补齐拔高。就如Claude Code之父Boris多次提到：未来的harness会越来越薄，最近也说到 “对于Claude Code，**外界往往会把注意力放在产品功能上，但如果让他回顾那些真正带来能力跃迁的时刻，最重要的原因其实只有一个：模型变强了**。”[8]

随着模型能力越来越强，它越来越聪明，你会发现它的**meta能力越来越强**：比如它会自动存和update memory，它知道如何制作skill，它会自己动编排/fan-out subagent，它知道怎么结构化知识库是适合的。这些其实模型很多已经能做到了

我认为，**Prompt/Skill/Agent/Harness这些会越来越简单，以至于未来会像 bash、python这种工具一样，模型在每个任务过程中，根据需要自己去产生、维护，而不是某个人提前写好一套给各种各样的人用。**

## 3.2.  我是瓶颈
在Opus4.8的实验时，我有很深的这样的感受：**我的上限决定了agent交付结果的上限。并不是它不够聪明，而是我不能有效指导它**。看着它一次又一次跑了很多迭代，很多分析，最终总是到了1个离cuBLAS有一点距离的上限就上不去了。 我就想如果我是更厉害的专家，如果我能很快看懂它的优化轨迹，提示一下应该它就能立刻有所提升（这个是有迹可循的，参考2节 memory污染时的case）。**我的能力是瓶颈，我的理解力也是瓶颈，我甚至有点焦虑，我想figure out。**

**然而用过Fable5我反而释怀了**，因为它已经把这个问题solved，即使是CUDA专家，我想很多也到不了这个水平，不会更好。
## 3.3.  Search is The Key

本质上不管**模型**还是**人**做Kernel性能优化的过程就是一个搜索的过程， A/B/C/D/E多种技术，寻找最优的组合。关键是如何跳出局部最优。A+B、A+C都不好，但是A+B+C可能是很好的。
> 我们可以通过Harness介入模型的搜索过程，如果我们能**引入一些优秀人类经验的指导**，亦或某种跳出局部最优的搜索算法（例如 **进化搜索**、**启发式搜索**），应该可以在模型本身的基础上做的更好

以上是我在Opus4.8上实验后的想法。然而试验了Fable后，在这个问题上的这种实验已经没有意义。

## 3.4.  The Bitter Lesson

模型训练阶段的提升应该是更基础、更本质、更有效的。围棋AlphaGo的范式基本上都有参考意义，从人类对弈棋谱做预训练到from scratch的RL训练。对于HPC/AI Infra/甚至更广泛的说，代码的任务，一样也可以引入人类经验指导，亦可以不加先验地模型探索，或者两者结合。最终的上限是不加或者尽量少人类先验，仅通过reward学习，超越人类，类似AlphaGo的神之一手。

## 3.5.  未来？

AI对不同领域解决程度如下图

![](https://blog-1375427515.cos.ap-shanghai.myqcloud.com/img/20260611021334320.png)


### 3.5.1.  solved
在一些领域确实已经solved，而且随着模型越来越强，这些领域越来越多。A100 CUDA Kernel是1个，前端可能也算是1个。
**Claude Code之父Boris就工作在这个领域**，看他的之前的工作状态，他已经卸载IDE，完全靠agent来工作，一堆编排好的loop/routine，手机就可以工作。而最近的Fable5，他更是称之**拥有判断力、品味以及维度感**

![](https://blog-1375427515.cos.ap-shanghai.myqcloud.com/img/291a3804eb7379f9535aca3cb94f1c80.jpg)

如果你工作在这个领域，那你主要的职责是： 
- **验证**：确认验收结果，1）防止reward hacking 2）根据对目标的理解设定合理的evaluation
- **理解**：为了1）后续可持续维护 2）更好做判断
- **价值观/优先级对齐**：与组织、人的价值观/优先级对齐， 比如是公平 or 正义？ 
- **做出判断**：为决策的最终结果负责

**这个领域注定不需要很多人，而且“品味”也不是护城河**，就像AI可以 做出来各种风格的绘画，并且有自己的评价一样，AI可以有“品味”，但是选择哪个是人类决定。

### 3.5.2.  unsolved

当模型扩大solved范围，大家可以关注更大的scope，更多unsolved领域，在这些unsolved领域与AI协作是关键竞争力。

比如拿这次实验验证已经solved A100 GEMM Kernel为例，还有诸多AI Infra问题AI不一定能够解决
- GEMM shape的泛化性
- 融合算子的性能、模型级别的性能
- Blackwell等新架构以及昇腾甚至不太为人知的DSA架构的 Kernel
- DSA架构设计合适的DSL/AI Compiler ......

solved领域的方法论是用来迁移到unsolved领域的，从而solved领域越来越多。往更广泛的看， 这些可能会促进以前进展缓慢的领域，比如生命科学这种。

这些都很难看清楚了。**作为普通人的我们也只能是努力适应变化，只为不被滚滚向前的浪潮抛下**。

# 4.  参考文献
[1] https://github.com/silencelamb/VibeKernel  
[2] https://github.com/karpathy/autoresearch  
[3] https://code.claude.com/docs/en/goal  
[4] https://ghuntley.com/ralph/  
[5] https://claude.com/blog/introducing-dynamic-workflows-in-claude-code  
[6] https://code.claude.com/docs/en/workflows  
[7] https://github.com/ghuntley/how-to-ralph-wiggum   
[8] https://mp.weixin.qq.com/s/7xojGo-W7COYmWP3mxghOA  
[9] https://github.com/silencelamb/VibeKernel/blob/main/results/ralph_loop_fable/src/matmul_f16/matmul_f16_v42_shfl2.cu  