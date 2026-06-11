| cycle | wall_clock(s) | tokens | correctness | tflops | 方法改进说明 | 瓶颈分析 | log |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 630 | 10934 | 3.1933712e-05 | 41.9 | v2: WMMA |  | |
| 2 | 1824 | 44189 | 3.6326535e-05 | 65.1 | v3: mma+ldmatrix |  | |
| 3 | 2517 | 62170 | 3.30855e-05 | 118.5 | v4: +cp.async |  | |
| 4 | 5759 | 107336 | 3.885773e-05 | 105.7 | v5: XOR-swizzle (HW ceiling) |  | |
