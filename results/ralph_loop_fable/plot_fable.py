#!/usr/bin/env python3
# ralph_loop_fable curve: Ralph Loop (Fable 5), iter-1 = naive_fable_cycle2 seed (v1-v22),
# then 2 fresh-session Ralph iters (iter2 v23-v29, iter3 v30-v48). GPU stuck 1155 MHz
# (driver fault) → real Tensor-Core ceiling 255.6, NOT 312.
# cuBLAS measured on THIS card @1155 (user-confirmed): f16-acc 219.8 / fp32-acc 218.7.
import csv, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.path.dirname(__file__)
rows = list(csv.DictReader(open(os.path.join(D, "result.csv"))))
CEIL_1155, PEAK_1410 = 255.6, 312.0
CUBLAS_F16, CUBLAS_F32 = 219.8, 218.7      # measured on this card @1155 (user-confirmed)

# milestone annotations (running-best setters); rest fall back silently
EN = {
    "1": "WMMA (126)", "2": "raw mma + cp.async pipe (194)", "8": "smem epilogue (210)",
    "13": "prefetch champ (217)", "19": "cycle2 champ: mirror+spread (219.5)",
    "23": "wavesplit: in-kernel last-wave K-split (225.5)",
    "27": "relax/split cleanup (227.6)",
}  # v24 & v42 are highlighted with red stars below, not blue labels

sc = [r for r in rows if r.get("scored") == "1"]
xs = [float(r["tokens"]) / 1000 for r in sc]
ys = [float(r["tflops"]) for r in sc]

rb, best, ann, seen = [], 0, [], set()
for r in sc:
    t = float(r["tflops"]); v = r["version"]
    if t > best + 0.05:
        best = t
        if v in EN and v not in seen:
            seen.add(v); ann.append((float(r["tokens"]) / 1000, t, f"v{v}: {EN[v]}"))
    rb.append(best)

# Ralph iter boundaries (cumulative tokens at first version of each fresh session)
def first_tok(vmin):
    for r in rows:
        try:
            if int(r["version"]) >= vmin: return float(r["tokens"]) / 1000
        except: pass
    return None
it2, it3 = first_tok(23), first_tok(30)

plt.figure(figsize=(12, 6.5))
plt.scatter(xs, ys, s=18, alpha=.28, color="tab:blue", zorder=2, label="scored 4096³/100-iter (incl. regressions)")
plt.plot(xs, rb, "-o", lw=2, ms=3, color="tab:blue", zorder=3, label="running best")

# iteration boundaries (Ralph story: fresh session each restart) — vertical guides + lifted <-> span arrows
xmax = max(xs)
for x in (it2, it3):
    if x: plt.axvline(x, ls="-", color="darkorange", lw=1.0, alpha=.55, zorder=1.5)
Y_AR = 112
for a, b, lab, col in [(min(xs), it2, "iter 1 = cycle2 seed (v1–v22)", "0.45"),
                       (it2, it3, "Ralph iter 2  (fresh session)", "darkorange"),
                       (it3, xmax, "Ralph iter 3  (fresh session)", "darkorange")]:
    if a is None or b is None: continue
    plt.annotate("", xy=(b, Y_AR), xytext=(a, Y_AR),
                 arrowprops=dict(arrowstyle="<->", color=col, lw=1.4, alpha=.9))
    plt.text((a + b) / 2, Y_AR + 6, lab, color=col, fontsize=8, ha="center", va="bottom", fontweight="bold")

# reference ceilings
plt.axhline(CEIL_1155, ls="--", color="lightcoral", lw=1.2, alpha=.8, label=f"real ceiling @1155-stuck = {CEIL_1155:.1f}")
plt.axhline(PEAK_1410, ls="--", color="grey", lw=1.2, label=f"theoretical peak @1410 = {PEAK_1410:.0f} (Opus-era clock)")
# single cuBLAS reference line (f16 & fp32 differ by ~1, so one line; both numbers in the label)
plt.axhline(CUBLAS_F16, ls="--", color="magenta", lw=2.1, zorder=2.6,
            label=f"cuBLAS measured @1155 (f16 {CUBLAS_F16} / fp32 {CUBLAS_F32})")
plt.text(min(xs) + (xmax - min(xs)) * 0.02, CUBLAS_F16 + 2.0,
         f"cuBLAS   f16 {CUBLAS_F16}  /  fp32 {CUBLAS_F32}",
         color="magenta", fontsize=9.5, fontweight="bold", va="bottom", ha="left")

for i, (x, y, txt) in enumerate(ann):
    plt.annotate(txt, (x, y), textcoords="offset points", xytext=(4, 9 if i % 2 == 0 else -15),
                 fontsize=7.3, color="tab:blue")

# ★ highlight the two headline points: v24 (strict-3e-5 peak) and v42 (champion)
for ver, lab in [("24", "v24  227.3\n(strict fp32, 3e-5)"), ("42", "v42  229.1\n(champion)")]:
    pts = [(float(r["tokens"]) / 1000, float(r["tflops"])) for r in sc if r["version"] == ver]
    if not pts: continue
    px, py = max(pts, key=lambda p: p[1])
    plt.scatter([px], [py], marker="*", s=520, color="red", edgecolor="black", linewidths=0.7, zorder=7)
    plt.annotate(lab, (px, py), textcoords="offset points", xytext=(0, 14),
                 fontsize=12, color="red", fontweight="bold", ha="center", zorder=7)

peak = max(ys)

plt.xlabel("cumulative output tokens (k)  — across iter-1 (cycle2 seed) + 2 Ralph iters")
plt.ylabel("TFLOPS (task1 4096³ 100-iter sustained)")
plt.ylim(0, 330)
plt.title("ralph_loop_fable (Claude Fable 5): A100 fp16 GEMM — Ralph Loop continued from cycle2")
plt.grid(alpha=.3); plt.legend(fontsize=7.6, loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(D, "curve.png"), dpi=140)
print("saved curve.png  peak=%.1f (%.0f%% of %.1f)  iter2@%.0fk iter3@%.0fk" % (peak, peak/CEIL_1155*100, CEIL_1155, it2 or 0, it3 or 0))
