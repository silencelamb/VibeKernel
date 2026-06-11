| cycle | wall_clock(s) | tokens | correctness | tflops | 方法改进说明 | 瓶颈分析 | log |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4963 | 330451 | 3.570567e-05 | 142.8 | v1: WMMA baseline (70, f32) |  | |
| 2 | 395 | 28826 | 5.1797324e-05 | 139.4 | v2: cp.async 3-stage pipe (139) | 70->139 翻倍,~45% peak | |
| 3 | 639 | 48278 | 4.4305387e-05 | 142.1 | v3: mma.sync (142, f32 peak) | fp32 累加 ~142 封顶(warp 数少) | |
| 4 | 807 | 59480 | 3.6785576e-05 | 124.1 | v4: 64x64 warp tile, 256x128 | mma/ldmatrix 比 2.7->4,藏延迟 | |
| 5 | 1060 | 78566 | 3.9118557e-05 | 133.2 | v5: 128x128 + register double-buf | double-buf 反伤(phase-lock 主导) | |
| 6 | 1396 | 102755 | 3.6986836e-05 | 121.1 | v6: config regression |  | |
| 7 | 2773 | 193798 | 0.0186733 | 147.2 | v13: f16-acc turn (147) | 转 f16 累加,err 跳 ~500x(0.018) | |
| 8 | 4328 | 292803 | 0.016016863 | 154.2 | v15: occupancy: 24-warp high-occ (154 peak) | 高占用(24warp,f16)胜低占用完美流水;err~0.02;champion;同范式离库上限仍 ~30% | |
| 9 | 3699 | 256330 | 0.018409988 | 150.2 | v19: config variant |  | |
