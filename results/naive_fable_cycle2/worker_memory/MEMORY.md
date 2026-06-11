# Memory Index

- [GPU 锁频 1155MHz](gpu-locked-1155mhz.md) — 评测卡 fp16 实际峰值 255.5 TFLOPS 而非 312，用 TC% 校准
- [cp.async commit 纪律](cpasync-commit-discipline.md) — 空 commit 让 wait_group 错位的隐蔽竞态；每迭代恰好一次 commit
- [task1 f16 冠军设计](task1-f16-champion-design.md) — v19 ~218TF 的设计要点 + 已排除的优化死路清单
