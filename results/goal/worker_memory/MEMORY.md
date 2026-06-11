# Memory Index

- [Best fp16 GEMM kernel](fp16-gemm-best-kernel.md) — v21 ≈206.7 TFLOPS (v17 kernel + L2 persistence on B) + the register-file ceiling on A100
- [Harness inf artifact](harness-inf-nearzero-gt.md) — intermittent inf in fp16 error check is harness near-zero GT, not a kernel bug
- [A100 power cap](a100-power-cap-clock.md) — GPU throttled to ~1155MHz/256 TFLOPS real peak; query bound GPU index for clocks
