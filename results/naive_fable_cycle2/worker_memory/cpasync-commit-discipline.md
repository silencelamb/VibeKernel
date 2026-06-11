---
name: cpasync-commit-discipline
description: cp.async 多级流水的 commit/wait 计数纪律——空 commit 会让 wait_group 永久错位一档（v8/v9 踩过的竞态）
metadata: 
  node_type: memory
  type: project
  originSessionId: 295962ba-8f43-4538-a648-f2f000790f3a
---

多级 cp.async 流水中，`cp.async.wait_group(N)` 只按"已 commit 的 group 数"计数。如果循环顶部无条件 `commit_group()` 而序言已把所有 stage commit 完，会注入一个**空 group**，使后续所有 wait 的保证从 stage kt+1 退化为 stage kt——下一 tile 的 ldmatrix 可能读到仍在飞行的 cp.async 数据。该 bug 时序敏感（v8 侥幸不炸，v9 stream-K 尾部段间歇性出错，坏值差量约一个 k-tile 的量级）。

**Why:** group 完成按提交顺序排队，wait_group 数的是"未完成 group 数 ≤ N"，空 group 也占一个名额。

**How to apply:** 纪律 = **每个主循环迭代恰好一次 commit、位置固定**（含 guard 关闭时的空 commit 用于垫尾），`wait_group(STAGES-3)`（commit 在 wait 之后时）或重新推导不变量：进入迭代 kt 时未完成 group 集合必须可证明 ⊇ 等待目标。验证方法：对拍工具跑 110 次连续 launch（间歇性竞态单次跑不出来）。相关 [[gpu-locked-1155mhz]]
