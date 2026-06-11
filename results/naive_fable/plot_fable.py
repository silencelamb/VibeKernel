#!/usr/bin/env python3
# Run-specific honest curve for naive_fable: the GPU was HARD-LOCKED at 1155 MHz this run
# (empirically verified: 211.7 TFLOPS measured while SM pinned 1155 @ 321W, no boost).
# So the real achievable Tensor-Core ceiling is 312*1155/1410 = 255.6 TFLOPS, NOT 312.
# We draw 255.6 (this run's real ceiling) + 312 (theoretical @1410, the clock the Opus
# runs saw) so the confound is visible at a glance.
import csv, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.path.dirname(__file__)
rows = list(csv.DictReader(open(os.path.join(D, "result.csv"))))
# English short labels for plot annotations (table keeps the Chinese labels.json);
# matplotlib has no CJK font here, so annotate in ASCII to stay glyph-clean.
EN = {
    "1": "mma baseline (122, fp32)", "2": "big CTA 128x256 (150)",
    "3": "cp.async 3-stage pipe (193)", "4": "warp-tile 64x64 (201.8 fp32 peak)",
    "5": "unroll + f16-acc turn (209)", "6": "stream-K (191, reverted)",
    "7": "dual-launch (207)", "8": "barrier-straddle (194, reverted)",
    "9": "full-tune champion (211.5, f16)", "10": "BK32 x6-stage (reverted)",
    "11": "CUDA-graph (reverted)", "12": "256x128 shape (199, reverted)",
    "13": "fp32-acc insurance (201.0)",
}

CEIL_1155 = 255.6      # 312 * 1155/1410 — real ceiling at the locked clock
PEAK_1410 = 312.0      # theoretical fp16 TC peak at full 1410 (Opus-era clock)
# MEASURED cuBLAS on THIS card @1155 (results/_baseline_cublas_f16.log = 220.5;
# library-ceilings memory: f16-acc 219.85 / fp32-acc 218.74). Same clock → directly
# comparable. cycle1 champion (v9) is f16-acc, so f16 is the matching reference.
CUBLAS_F16, CUBLAS_F32 = 219.85, 218.74

sc = [r for r in rows if r.get("scored") == "1"]      # scored 4096^3 / 100-iter
xs = [float(r["tokens"]) / 1000 for r in sc]
ys = [float(r["tflops"]) for r in sc]

# running best + first-time-best annotations
rb, best, ann, seen_v = [], 0, [], set()
for r in sc:
    t = float(r["tflops"]); v = r["version"]
    if t > best + 0.05:
        best = t
        if v not in seen_v:           # one annotation per version (v9 sets best many times)
            seen_v.add(v)
            ann.append((float(r["tokens"]) / 1000, t, f"v{v}: {EN.get(v, '')}"))
    rb.append(best)

plt.figure(figsize=(11, 6))
plt.scatter(xs, ys, s=20, alpha=.30, color="tab:blue", zorder=2,
            label="scored 4096³/100-iter (incl. regressions)")
plt.plot(xs, rb, "-o", lw=2, ms=3.5, color="tab:blue", zorder=3, label="running best")

# ceiling lines
plt.axhline(CEIL_1155, ls="-", color="tab:red", lw=1.6,
            label=f"real ceiling @1155-locked = {CEIL_1155:.1f}  (this run)")
plt.axhline(PEAK_1410, ls="--", color="grey", lw=1.2,
            label=f"theoretical peak @1410 = {PEAK_1410:.0f}  (Opus-era clock)")
plt.axhline(CUBLAS_F16, ls=":", color="tab:orange", lw=1.4,
            label=f"cuBLAS f16-acc measured @1155 = {CUBLAS_F16:.1f}  (same precision as champion)")
plt.axhline(CUBLAS_F32, ls=":", color="tab:green", lw=1.0, alpha=.7,
            label=f"cuBLAS fp32-acc measured @1155 = {CUBLAS_F32:.1f}")

# version annotations (alternate offset to reduce overlap)
for i, (x, y, txt) in enumerate(ann):
    dy = 9 if i % 2 == 0 else -16
    plt.annotate(txt, (x, y), textcoords="offset points", xytext=(4, dy),
                 fontsize=7.6, color="tab:blue")

peak = max(ys)
plt.text(0.015, 0.965,
         f"Claude Fable 5 · naive (NEVER-STOP, pure prompt)\n"
         f"peak {peak:.1f} TFLOPS = {peak/CEIL_1155*100:.0f}% of 255.6 real ceiling (f16-acc)\n"
         f"⚠ GPU hard-locked 1155 MHz this run — abs. TFLOPS NOT comparable to\n"
         f"   Opus runs (which ran ~1410); compare % of clock-real ceiling instead",
         transform=plt.gca().transAxes, fontsize=8.2, va="top", ha="left",
         bbox=dict(boxstyle="round", fc="#fffbe6", ec="tab:red", alpha=.9))

plt.xlabel("cumulative output tokens (k)")
plt.ylabel("TFLOPS (task1 4096³ 100-iter sustained)")
plt.ylim(0, 330)
plt.title("naive_fable (Claude Fable 5): A100 fp16 GEMM — TFLOPS vs token  @ GPU locked 1155 MHz")
plt.grid(alpha=.3)
plt.legend(fontsize=8, loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(D, "curve.png"), dpi=140)
print("saved curve.png  peak=%.1f (%.0f%% of %.1f)" % (peak, peak / CEIL_1155 * 100, CEIL_1155))
