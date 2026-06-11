我现在想做一个事情是用大模型coding agent做我们的cuda playground任务，它是一个高性能gemm编写任务，在A100 GPU上。本身playground是有严格标准的：统一的 correctness 校验、固定的shape、性能口径。有几个可能的范式:

## 1.1.  claude code naive 不加什么东西
如何用Opus 4.6写出一个比开源社区更快的gpu算子 - hello王先生的文章 - 知乎 https://zhuanlan.zhihu.com/p/2017212755590005622
如何让Claude Opus 4.6写一个100% CUBLAS性能的GEMM算子 - 椎名深雪的文章 - 知乎 https://zhuanlan.zhihu.com/p/2028849708638979935

**模型对比臂（同 naive 范式，换模型）**：`scripts/launch_naive_fable.sh` —— 与 `launch_naive.sh` 逐字相同，唯一变量 = 主 worker 模型 `claude-opus-4-8`（Opus 4.8）→ `claude-fable-5`（Fable 5，Anthropic 最新）。同 seed、同 `--effort max`、同 NEVER STOP 纯自驱 → 单变量隔离「换最新模型在 naive 下能冲多高 / 持续多久 / 花多少」。runbook: `runbooks/naive_fable.md`。
## 1.2.  claude  code + skill
### 1.2.1.  MIT-HanLab
KDA:让 Agent 自己优化 CUDA kernel，并在 MLSys 2026 FlashInfer Full-Agent Track 拿下前三 - Lyken的文章 - 知乎 https://zhuanlan.zhihu.com/p/2044459666327999866
code： https://github.com/mit-han-lab/kernel-design-agents

### 1.2.2.  AKO

https://tongminglaic.github.io/AKO/
https://tongminglaic.github.io/AKO/assets/ako-tech-report.pdf
https://github.com/TongmingLAIC/AKO4ALL
https://github.com/TongmingLAIC/AKO4X

## 1.3.  AI-Infra-Auto-Driven-SKILLS
https://zhuanlan.zhihu.com/p/2041180794375377333  SGLang SOTA Humanize Loop：让 Codex 自动追推理性能到 SOTA
https://zhuanlan.zhihu.com/p/2042740770457772060  AI-Infra-Auto-Driven-SKILLS v0.1.0：给 Codex / Claude Code 的推理框架工作流

github： https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS

## 1.4.  Claude Code + Dynamic Workflow

https://agentpedia.codes/blog/claude-opus-4-8-claude-code-workflows
https://claudefa.st/blog/guide/development/dynamic-workflows
这个是 Opus4.8 最新加入的。 
Claude Code设置effort为 ultracode就是使用xhigh + Dynamic Workflow

## 1.5.  Claude Code +/goal
https://mp.weixin.qq.com/s/VwsCC3WNHatLbGBPDw7VOw
Ralph loop 的思想（让 agent 一直跑而不是停下），提高完成情况。Ralph loop 出处：https://ghuntley.com/ralph/ 与 https://ghuntley.com/loop
/goal https://code.claude.com/docs/en/goal
（`/goal` = Ralph 思想的「单 session 内」实现：同一 session 续跑 + LLM-as-judge 同 session 判停。）

## 1.5b.  Claude Code + Ralph Loop（原汁原味的外循环）
Ralph technique：https://ghuntley.com/ralph/ ；loop 延伸：https://ghuntley.com/loop
`while :; do cat PROMPT.md | claude-code ; done` —— 与 `/goal` 的区别：**每轮重起全新 session**（context 每轮清空、靠磁盘持久 kernel 进度），新 session 判是否达成条件。我们自己用 bash 包 `claude -p` 实现（`scripts/launch_ralph_loop.sh`），无外部依赖。


## 1.6.  claude code + auto research 
Andrej Karpathy的 auto research https://github.com/karpathy/autoresearch ，参考auto research的思路

autokernel 即是参考了 auto research   https://github.com/rightnow-ai/autokernel
![](https://blog-1375427515.cos.ap-shanghai.myqcloud.com/img/20260604161042489.png)


## 1.7.  claude code 参考 OpenAI研究员最新这篇工作
Heuristic Learning  https://trinkle23897.github.io/learning-beyond-gradients/  
参考这个思想
## 1.8.  一些 专门的kernel agent方案

### 1.8.1.  K-Search
UC Berkeley， 有论文和代码： https://github.com/caoshiyi/K-Search 
https://arxiv.org/pdf/2602.19128v1