---
name: orchestration-overview
description: VibeKernel 是"用不同 coding-agent 方法做 A100 GEMM"的编排对比主仓;主 session 的编排意识入口
metadata:
  type: project
---

本仓库(silencelamb/VibeKernel)是**编排/对比主仓**:用 Claude Code 的不同方法/范式(naive、/goal、KDA、AKO、dynamic workflow…见 `methods.md`)自动写高性能 A100 fp16 GEMM,对比各方法的最终性能、局限、**迭代提升曲线**。**当前 session(cwd=VibeKernel 根)就是编排 session。**

**关键文档(此 memory 是编排意识的入口,详情 `@` 对应文件):**
- `CLAUDE_For_KernelAgent.md` = kernel agent(worker)的任务手册母版,方法无关;建 fork 时 cp 成 `<fork>/CLAUDE.md`。**根目录不再放 CLAUDE.md**(见下"防串味")。
- `META.md` = 完整编排上下文(方法阶梯、防串味协议、建 fork、实验记录口径)。讨论编排时 `@META.md`。
- `methods.md` = 方法清单原始出处。
- 结果工具链与全局约定见 [[vibekernel-result-harness]];曲线/字段解析见 [[naive-csv-field-parsing]] / [[naive-token-attribution-limit]];ncu 见 [[ncu-permission-gate]]。

**防串味(为什么编排上下文放这里而不是根 CLAUDE.md):** Claude 的 CLAUDE.md 会沿目录树向上继承;fork(worker)在 VibeKernel 子目录下跑,会继承根 CLAUDE.md。若把编排文档当根 CLAUDE.md,worker 就会读到"在被对比"→污染实验(违反 META §3 隔离)。所以编排意识放**本 memory**(按 cwd-slug 隔离,worker 的 slug 不同、读不到)+ META.md,根目录不放 CLAUDE.md。

**记忆持久化:** 本 memory 已 git 维护于 `repo/.claude/memory/`,`scripts/link_memory.sh`(`.claude/settings.json` 的 SessionStart hook 自动调用)软链到 `~/.claude/projects/<slug>/memory`,换容器也在。换容器后第一次手动跑一次最稳。

**Claude project slug 规则(踩过坑):** `~/.claude/projects/<slug>/` 的 slug = 绝对 cwd 路径里**所有非字母数字字符都转 `-`**(不只是 `/`,`_`/`.` 也转)。例:`/home/daixu/code/github_code/VibeKernel` → `-home-daixu-code-github-code-VibeKernel`(`github_code` 的 `_` 也变 `-`)。找 transcript / 软链 memory 时按这个算,`sed 's#[^a-zA-Z0-9]#-#g'`。
