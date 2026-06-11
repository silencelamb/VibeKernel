| cycle | wall_clock(s) | tokens | correctness | tflops | 方法改进说明 | 瓶颈分析 | log |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 136 | 9225 | 0.020516157 | 114.0 | v1: WMMA 128x128 baseline | WMMA 开销大、无流水 | |
| 2 | 289 | 18911 | 0.01779154 | 140.9 | v2: cp.async double-buf + WMMA |  | |
| 3 | 439 | 30174 | 4.4293207e-05 | 103.4 | v3: mma.sync f32-acc (regress 103) | warp/scheduler 太少、mma 管线吃不满 | |
| 4 | 664 | 45279 | 0.000108578446 | 103.8 | v4: 256x128 16warp f32-acc | 算术强度/寄存器未拉满,仍 103 | |
| 5 | 933 | 64890 | 0.017687991 | 142.3 | v5: ldmatrix.x4 + 64x64 warp + f16-acc |  | |
| 6 | 4883 | 281343 | 3.208354e-05 | 196.9 | v33: + L2-swizzle rasterization (196.9 peak) | tensor-pipe 气泡;occupancy 锁 1 blk/SM(寄存器墙)——非范式极限,同范式参考实现可达 ~214 | |
| 7 | 3354 | 191632 | 3.8543127e-05 | 196.6 | v36: config variant |  | |
| 8 | 3722 | 209838 | 3.87398e-05 | 195.4 | v40: config variant (regress) |  | |
