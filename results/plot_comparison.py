#!/usr/bin/env python3
# 跨方法对比图:各代表性 run 的 running-best(TFLOPS vs 累计 output token)叠一张图,
# 按【方法族】着色 + 曲线末端直接标方法文字;goal 族标 evaluator 接入(round 边界,竖虚线);
# 顶部画【实测库天花板】(cuBLAS f16/fp32、CUTLASS fp32)。
# 数据源:每 run 的 results/<run>/result.csv(scored 点)+ goal 族 transcript(goal_status 接入 token)。
# ⚠️ 时钟混淆(硬口径):Fable / Ralph 各轮跑在驱动故障时钟态(NVRM 锁 1155 MHz,真天花板
#    312×1155/1410 = 255.6)→ 这些 run 的绝对 TFLOPS 带 ~−18% 偏置,与 Opus 老轮(~1410/312,
#    历史未记录时钟)不可直接比;图上以「@1155」标注 + 255.6 点划线呈现。详见 SUMMARY.md。
# 用法: /opt/torch/bin/python results/plot_comparison.py
import csv, os, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES  = os.path.join(ROOT, "results")

# 实测库天花板(本箱 4096³ / task1 100 轮口径;@1155 复测 cuBLAS f16 = 220.5,与下值一致):
#   cuBLAS f16-acc 219.85(err .018) / cuBLAS fp32-acc 218.74(err 3e-5) / CUTLASS fp32-acc 217.9(err 3e-5)
CUBLAS_F16, CUBLAS_F32, CUTLASS_F32 = 219.85, 218.74, 217.9
CEIL_1155 = 255.6   # 驱动故障锁 1155 MHz 的真天花板(Fable/Ralph 各轮)

# (run-dir, 图例名, 颜色, 线型, goal族?, errmax[精度档过滤], 曲线末端文字)
# ⚠️ naive_cycle5 已污染作废(与 goal_cycle3 并行串味、读了对方 memory)→ 不进图;留档 results/naive_cycle5_deprecated/。
# naive_cycle6 errmax=0.005 → 画 fp32-acc 曲线(203,与各 fp32 同档可比);其 f16-acc 峰 208 单列★。
# naive_fable c1 不过滤(冠军是 f16-acc 211.5);c2 / ralph 过滤 err<0.005 = fp32 档(ralph 的 229.7 是放宽
# fp32 ~1e-4–3e-4 档,仍 ≪ 0.02;其严格 3e-5 档峰 = 227.3,见 SUMMARY)。
RUNS = [
    ("naive_cycle3",        "naive c3 (self-stop)",         "tab:blue",    "-",  False, None,  "naive c3"),
    ("naive_cycle4",        "naive c4 (self-stop)",         "navy",        "-",  False, None,  "naive c4"),
    ("naive_cycle6",        "naive c6 (fp32; crashed*)",    "deepskyblue", "--", False, 0.005, "naive c6"),
    ("naive_strong_cycle2", "naive_strong c2 (self-stop)",  "tab:green",   "-",  False, None,  "n_strong c2"),
    ("naive_strong",        "naive_strong c1 (resume*)",    "seagreen",    "--", False, None,  "n_strong c1"),
    ("goal",                "goal c1 (self-stop)",          "tab:red",     "-",  True,  None,  "goal c1"),
    ("goal_cycle2",         "goal c2 (ECONNRESET*)",        "darkorange",  "--", True,  None,  "goal c2"),
    ("goal_cycle3",         "goal c3 (mbarrier; killed*)",  "crimson",     "-",  True,  None,  "goal c3"),
    ("goal_cycle4",         "goal c4 (f16-acc; 401→resume→ECONNRESET*)", "tomato", "--", True, None, "goal c4"),
    ("dynamic_workflow",    "dynamic_workflow (self-stop; 7 wf)", "darkviolet", "-", False, 0.005, "dynwf"),
    ("dynamic_workflow_guided", "dynwf_guided (leave-one-out; 4 wf)", "magenta", "--", False, 0.005, "dynwf_guided"),
    ("ralph_loop",          "ralph_loop c1 (Opus; iter2 crash*) @1155",          "tab:brown",     "--", False, 0.005, "ralph c1"),
    ("naive_fable",         "naive_fable c1 (Fable 5; f16-acc) @1155",           "darkturquoise", "-",  False, None,  "fable c1"),
    ("naive_fable_cycle2",  "naive_fable c2 (Fable 5; fp32) @1155",              "darkcyan",      "-",  False, 0.005, "fable c2"),
    ("ralph_loop_fable",    "ralph_loop_fable (3 iters; iter1=fable c2) @1155",  "black",         "-",  False, 0.005, "ralph×fable"),
]
# 个别末端文字摆位微调:tag → (dx pt, dy pt, ha);默认 (6, 0, "left")
FULL_POS = {"fable c2": (6, -13, "left"), "fable c1": (6, -8, "left"), "naive c6": (-6, 5, "right")}

def scored_curve(run, errmax=None):
    """读 result.csv 的 scored 点(可选 err<errmax 精度过滤),按累计 token 排序 → (xs[k tok], running-best ys, peak)。"""
    p = os.path.join(RES, run, "result.csv")
    if not os.path.exists(p): return [], [], 0.0
    pts = []
    with open(p) as f:
        for r in csv.DictReader(f):
            if r.get("scored") == "1" and (errmax is None or float(r["correctness"]) < errmax):
                pts.append((int(r["tokens"]), float(r["tflops"])))
    pts.sort()
    xs = [t/1000 for t, _ in pts]; ys = []; mx = 0.0
    for _, y in pts: mx = max(mx, y); ys.append(mx)
    return xs, ys, (max(mx, 0.0))

def goal_interventions(run):
    """从 transcript 抽 evaluator 非-sentinel goal_status 的【接入时累计 output token】(k)。"""
    p = os.path.join(RES, run, "transcript.jsonl")
    if not os.path.exists(p): return []
    cum = 0; seen = {}; hits = []
    for line in open(p, encoding="utf-8", errors="replace"):
        s = line.strip()
        if not s: continue
        try: o = json.loads(s)
        except Exception: continue
        t = o.get("type")
        if t == "assistant":
            m = o.get("message", {}) or {}
            mid = m.get("id", "x"); ot = (m.get("usage", {}) or {}).get("output_tokens", 0) or 0
            if ot > seen.get(mid, 0): cum += ot - seen.get(mid, 0); seen[mid] = ot
        elif t == "attachment":
            blob = json.dumps(o)
            if "goal_status" in blob and '"sentinel":true' not in blob.replace(" ", ""):
                hits.append(cum/1000)
    return hits

plt.figure(figsize=(12.5, 6.8))
for run, name, color, ls, is_goal, errmax, tag in RUNS:
    xs, ys, pk = scored_curve(run, errmax)
    if not xs: continue
    plt.plot(xs, ys, ls, color=color, lw=2.0, ms=3, marker=("o" if ls == "-" else None),
             alpha=0.95, label=f"{name} — peak {pk:.0f}")
    # 曲线末端直接标【方法文字 + 峰值】(= "curve 图上的方法文字")
    dx, dy, ha = FULL_POS.get(tag, (6, 0, "left"))
    plt.annotate(f"{tag} · {pk:.0f}", (xs[-1], ys[-1]), textcoords="offset points",
                 xytext=(dx, dy), ha=ha, fontsize=8.5, color=color, va="center", fontweight="bold")
    if is_goal:
        for gx in goal_interventions(run):
            plt.axvline(gx, ls=":", color=color, lw=1.0, alpha=0.45, zorder=1)

# naive_cycle6 的 f16-acc 裸峰(208,精度档不同)单列★ + 文字
fx, fy, c6_f16 = scored_curve("naive_cycle6", None)
if c6_f16 > 0:
    cx = fx[fy.index(c6_f16)] if c6_f16 in fy else fx[-1]
    plt.scatter([cx], [c6_f16], marker="*", s=200, color="deepskyblue", edgecolor="black", lw=0.7, zorder=6)
    plt.annotate(f"naive c6 f16-acc · {c6_f16:.0f}", (cx, c6_f16), textcoords="offset points",
                 xytext=(-8, 12), ha="right", fontsize=8, color="deepskyblue", va="bottom", fontweight="bold")

# round 概念图例(goal 族:竖点线 = evaluator 接入 = round 边界)
plt.plot([], [], ls=":", color="gray", lw=1.0, label="goal evaluator intervention (round boundary)")
# 实测库天花板(本箱)+ 两档时钟天花板:
plt.axhline(312, ls="--", color="grey", lw=1.1, alpha=.8, label="A100 fp16 nominal peak 312 (@1410 boost)")
plt.axhline(CEIL_1155, ls="-.", color="firebrick", lw=1.3, alpha=.8,
            label=f"real ceiling of @1155 runs (clock-stuck) = {CEIL_1155:.1f}")
plt.axhline(CUBLAS_F16, ls="--", color="tab:orange", lw=1.3, alpha=.8,
            label=f"cuBLAS f16-acc {CUBLAS_F16:.0f} (err .018)")
plt.axhline(CUBLAS_F32, ls="--", color="purple", lw=1.3, alpha=.8,
            label=f"fp32 library ceiling ~{CUBLAS_F32:.0f} (cuBLAS {CUBLAS_F32:.0f} / CUTLASS {CUTLASS_F32:.0f}, err 3e-5)")

plt.xlabel("cumulative output tokens (k)")
plt.ylabel("running-best TFLOPS (task1 4096³ 100-iter sustained)")
plt.title("Method comparison — coding-agent paradigms on A100 fp16 GEMM\n"
          "(naive=weak prompt · n_strong=strong framing · goal=watchdog · dynwf=ultracode workflow · "
          "ralph=fresh-session loop · fable=Fable-5 arm @1155)", fontsize=10)
plt.ylim(0, 330); plt.xlim(0, 1580); plt.grid(alpha=.3)
plt.legend(fontsize=7.3, loc="lower right", ncol=2)
plt.tight_layout()
out = os.path.join(RES, "comparison.png")
plt.savefig(out, dpi=140)
print("saved", out)
for run, name, color, ls, is_goal, errmax, tag in RUNS:
    xs, ys, pk = scored_curve(run, errmax)
    gi = goal_interventions(run) if "goal" in run else []
    extra = f"  interventions@{[round(g) for g in gi]}k" if gi else ""
    if errmax is not None:
        _, _, raw = scored_curve(run, None); extra += f"  (raw/all-tier peak {raw:.0f})"
    print(f"  {name:50s} peak={pk:6.1f}  pts={len(xs)}{extra}")

# ── 第二张图:6 条头牌 run + 同族淡细线装饰 → comparison_best.png ──
#   naive c6 · goal c1 · dynwf · naive×Fable c1/c2 · ralph×Fable(229.7,红★);
#   细线 = 同族其余 cycle(同色、无标签、纯衬底;goal c3 长尾按主曲线范围裁掉)。
#   @1155 时钟混淆的完整说明见 comparison.png 与 SUMMARY.md;此图只留 255.6 真天花板线提示。
BEST = [   # (run, 图例名, 颜色, errmax, 末端文字)
    ("naive_cycle6",       "naive — best: cycle6 (fp32)",               "deepskyblue",   0.005, "naive c6"),
    ("goal",               "/goal — best: cycle1",                      "tab:red",       None,  "goal c1"),
    ("dynamic_workflow",   "Dynamic Workflow",                          "darkviolet",    0.005, "dynwf"),
    ("naive_fable",        "naive × Fable 5 — cycle1 (f16-acc) @1155",  "darkturquoise", None,  "fable c1"),
    ("naive_fable_cycle2", "naive × Fable 5 — cycle2 (fp32) @1155",     "darkcyan",      0.005, "fable c2"),
    ("ralph_loop_fable",   "Ralph Loop × Fable 5 @1155",                "black",         0.005, "ralph×fable"),
]
THIN = [   # 装饰用同族细线:(run, 族色, errmax)
    ("naive_cycle3",            "deepskyblue", None),
    ("naive_cycle4",            "deepskyblue", None),
    ("goal_cycle2",             "tab:red",     None),
    ("goal_cycle3",             "tab:red",     None),
    ("goal_cycle4",             "tab:red",     None),
    ("dynamic_workflow_guided", "darkviolet",  0.005),
]
# 末端文字摆位微调:tag → (dx pt, dy pt, ha);默认 (6, 0, "left")
END_POS = {"naive c6": (6, -5, "left"), "fable c2": (-6, 8, "right")}
# ralph×fable 的 csv 含 iter1(= fable c2 整轮)→ 前 397k 与 fable c2 完全重合;
# 把 fable c2 画在黑线之上(zorder 4),视觉即"青线到 219.5 自停、黑线(Ralph)接着续顶"。
Z = {"fable c2": 4}

plt.figure(figsize=(12.5, 6.8))
XB = max(scored_curve(r, e)[0][-1] for r, _, _, e, _ in BEST if scored_curve(r, e)[0])  # 主曲线最远端
for run, color, errmax in THIN:   # 细线先画(zorder 低),裁到主曲线范围内
    xs, ys, _ = scored_curve(run, errmax)
    if not xs: continue
    cx = [x for x in xs if x <= XB]; cy = ys[:len(cx)]
    plt.plot(cx, cy, "-", color=color, lw=0.8, alpha=0.35, zorder=1.5)
plt.plot([], [], "-", color="grey", lw=0.8, alpha=0.6, label="same family, other cycles (thin)")
for run, name, color, errmax, tag in BEST:
    xs, ys, pk = scored_curve(run, errmax)
    if not xs: continue
    plt.plot(xs, ys, "-", color=color, lw=2.6, ms=3.5, marker="o", alpha=0.95, zorder=Z.get(tag, 3),
             label=f"{name} — peak {pk:.1f}")
    dx, dy, ha = END_POS.get(tag, (6, 0, "left"))
    plt.annotate(f"{tag} · {pk:.1f}", (xs[-1], ys[-1]), textcoords="offset points",
                 xytext=(dx, dy), ha=ha, fontsize=9, color=color, va="center", fontweight="bold")
# ralph×fable 冠军点(running-best 首达峰值处)红★
rx, ry, rpk = scored_curve("ralph_loop_fable", 0.005)
if rpk > 0:
    sx = rx[ry.index(rpk)]
    plt.scatter([sx], [rpk], marker="*", s=300, color="red", edgecolor="black", lw=0.7, zorder=6)
plt.axhline(312, ls="--", color="grey", lw=1.1, alpha=.8, label="A100 fp16 nominal peak 312 (@1410 boost)")
plt.axhline(CEIL_1155, ls="-.", color="firebrick", lw=1.4, alpha=.85,
            label=f"real ceiling of @1155 runs = {CEIL_1155:.1f}")
plt.axhline(CUBLAS_F32, ls="--", color="purple", lw=1.3, alpha=.8,
            label=f"fp32 library ceiling ~{CUBLAS_F32:.0f} (cuBLAS {CUBLAS_F32:.0f} / CUTLASS {CUTLASS_F32:.0f}, err 3e-5)")
plt.text(XB * 0.012, CUBLAS_F32 + 2.5, f"cuBLAS fp32 {CUBLAS_F32:.1f}",
         color="purple", fontsize=9, fontweight="bold", va="bottom", ha="left")
plt.xlabel("cumulative output tokens (k)")
plt.ylabel("running-best TFLOPS (task1 4096³ 100-iter sustained)")
plt.title("Best runs — A100 fp16 GEMM\n"
          "naive (cycle6) · /goal (cycle1) · Dynamic Workflow · naive × Fable 5 (cycle1/2) · Ralph Loop × Fable 5",
          fontsize=12)
plt.ylim(0, 330); plt.xlim(0, XB * 1.12); plt.grid(alpha=.3); plt.legend(fontsize=8.5, loc="lower right")
plt.tight_layout()
out2 = os.path.join(RES, "comparison_best.png")
plt.savefig(out2, dpi=140)
print("saved", out2)
