 为什么 sweep 能到 263/267？——不是 kernel 更强，是测法不同（口径作弊）

  sweep 小程序和 task1 的计时方式根本不是一回事，证据在源码里：

  - sweep：// hard warmup ~2s to stabilize clocks 先猛跑 2 秒把时钟顶到 boost，然后 int iters=80; float best=1e30f —— 取 80 次里最快的一次（min 
  time）。这是"热峰值"口径：满 boost 时钟 + 暖 L2 + 最快单发。
  - task1（计分口径）：10 warmup + 100 轮取平均。连跑 100 轮 GPU 因功耗/温度降频，sustained 时钟掉下来 → 平均自然低。

  铁证：在 sweep 自己的口径下，连 cuBLAS 都读出 242.3 / 221.5 TFLOPS（=== S=4096 cuBLAS=242.3 TFLOPS (warm reference) ===），而 task1 里 cuBLAS
  只有 220。同一个 cuBLAS、同样 4096³，换个测法就 +10%。所以那些 255/263/267 的 CUTLASS 配置，就是 task1 里跑 218 的同一族 kernel，只是用"热峰值 
  best-of-80"量出来的虚高，不是真的多快。

  根因：时钟锁不住（见下），所以频率浮动——短爆发能 boost 到 267，持续 100 轮就 throttle 到 218。模型最后的结论"312 要不降频的
  1410MHz、这机器锁不了时钟"就是指这个。

  模型用 ncu 了吗？——试了，被权限挡死

  它跑过 2 条 ncu --metrics ...（想看 tensor-pipe 利用率、dram 吞吐、shared bank conflict、long-scoreboard stall、寄存器数、occupancy
  上限），但全部返回：

  ERR_NVGPUCTRPERM - 无权限访问 GPU 性能计数器
  无权限修改 GPU 时钟
