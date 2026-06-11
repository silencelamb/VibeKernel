#!/usr/bin/env python3
# Run-specific honest curve for naive_fable_cycle2: the GPU is stuck at 1155 MHz
# (NVRM driver assertion fault, not -lgc; verified for cycle1). Real Tensor-Core
# ceiling = 312*1155/1410 = 255.6 TFLOPS, NOT 312. We draw 255.6 (this run's real
# ceiling) + 312 (theoretical @1410, the clock the Opus runs saw) so the confound
# is visible. NOTE: cycle2's champion (v19) is fp32-accumulate (err ~3e-5).
import csv, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.path.dirname(__file__)
rows = list(csv.DictReader(open(os.path.join(D, "result.csv"))))
CEIL_1155, PEAK_1410 = 255.6, 312.0
# MEASURED cuBLAS on THIS card @1155 (results/_baseline_cublas_f16.log = 220.5;
# library-ceilings memory: f16-acc 219.85 / fp32-acc 218.74). Same clock as this run
# → directly comparable, no estimate. cycle2 champion is fp32-acc, so fp32 is the match.
CUBLAS_F32, CUBLAS_F16 = 218.74, 219.85

# English short labels for annotations (table keeps the Chinese labels.json).
EN = {
    "1": "WMMA baseline (126)", "2": "raw mma + cp.async pipe (194)",
    "3": "BK64 (190)", "4": "4-stage (196)", "5": "unroll3 (reverted)",
    "6": "XOR swizzle (196)", "7": "2-CTA + f16-acc (200)",
    "8": "smem epilogue (210.5, fp32)", "9": "stream-K (142, race!)",
    "10": "6-stage (210.6)", "11": "BK64 reuse (214)", "12": "persistent block (216)",
    "13": "prefetch2 (217.2, fp32)", "14": "f16-acc (217.9, rejected)",
    "15": "tail-split (163, rev)", "16": "tune (216)", "17": "2-CTA+BK64 (212)",
    "18": "mirror tile shape (219.0)", "19": "v19 spread champion (219.5, fp32)",
    "20": "mbarrier (168, rev)", "21": "micro-bundle (215)", "22": "tail2 (154, neg)",
}

sc = [r for r in rows if r.get("scored") == "1"]
xs = [float(r["tokens"]) / 1000 for r in sc]
ys = [float(r["tflops"]) for r in sc]

rb, best, ann, seen = [], 0, [], set()
for r in sc:
    t = float(r["tflops"]); v = r["version"]
    if t > best + 0.05:
        best = t
        if v not in seen:
            seen.add(v); ann.append((float(r["tokens"]) / 1000, t, f"v{v}: {EN.get(v,'')}"))
    rb.append(best)

plt.figure(figsize=(11, 6))
plt.scatter(xs, ys, s=20, alpha=.30, color="tab:blue", zorder=2,
            label="scored 4096³/100-iter (incl. regressions)")
plt.plot(xs, rb, "-o", lw=2, ms=3.5, color="tab:blue", zorder=3, label="running best")
plt.axhline(CEIL_1155, ls="-", color="tab:red", lw=1.6,
            label=f"real ceiling @1155-stuck = {CEIL_1155:.1f}  (this run)")
plt.axhline(PEAK_1410, ls="--", color="grey", lw=1.2,
            label=f"theoretical peak @1410 = {PEAK_1410:.0f}  (Opus-era clock)")
plt.axhline(CUBLAS_F32, ls=":", color="tab:orange", lw=1.4,
            label=f"cuBLAS fp32-acc measured @1155 = {CUBLAS_F32:.1f}  (same precision as champion)")
plt.axhline(CUBLAS_F16, ls=":", color="tab:green", lw=1.0, alpha=.7,
            label=f"cuBLAS f16-acc measured @1155 = {CUBLAS_F16:.1f}")
for i, (x, y, txt) in enumerate(ann):
    plt.annotate(txt, (x, y), textcoords="offset points",
                 xytext=(4, 9 if i % 2 == 0 else -16), fontsize=7.4, color="tab:blue")

peak = max(ys)
plt.text(0.015, 0.965,
         f"Claude Fable 5 · naive cycle2 (NEVER-STOP, pure prompt)\n"
         f"peak {peak:.1f} TFLOPS @ fp32-acc (err ~3e-5) = {peak/CEIL_1155*100:.0f}% of 255.6 ceiling\n"
         f"   ≈ {peak/CUBLAS_F32*100:.0f}% of measured cuBLAS fp32 ({CUBLAS_F32:.1f}) — essentially AT the library ceiling\n"
         f"⚠ GPU stuck 1155 MHz (driver fault) — abs. TFLOPS NOT comparable to\n"
         f"   Opus runs (~1410); compare % of clock-real ceiling instead",
         transform=plt.gca().transAxes, fontsize=8.2, va="top", ha="left",
         bbox=dict(boxstyle="round", fc="#fffbe6", ec="tab:red", alpha=.9))
plt.xlabel("cumulative output tokens (k)")
plt.ylabel("TFLOPS (task1 4096³ 100-iter sustained)")
plt.ylim(0, 330)
plt.title("naive_fable_cycle2 (Claude Fable 5): A100 fp16 GEMM — TFLOPS vs token  @ GPU stuck 1155 MHz")
plt.grid(alpha=.3)
plt.legend(fontsize=8, loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(D, "curve.png"), dpi=140)
print("saved curve.png  peak=%.1f (%.0f%% of %.1f)" % (peak, peak / CEIL_1155 * 100, CEIL_1155))
