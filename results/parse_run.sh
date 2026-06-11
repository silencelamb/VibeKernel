#!/usr/bin/env bash
# parse_run — 把一次 headless 跑解析成结果:逐版本曲线表(csv)+ markdown 表 + 曲线图。
#
# 取数原则(2026-06-05 改:task1 工具输出为骨架,RESULT_JSON 仅作注解):
#   tflops / error / 版本号 ← task1 工具输出(计分行 "TFLOPS:..;Average Error:.." + 调用命令的 --ver N)
#                             —— 客观、模型伪造不了、每跑必有,无需 worker 自报。
#   token / 曲线 x 轴        ← transcript(按 message.id 去重的逐turn output_tokens 累计)
#   wall_clock              ← transcript 顶层 timestamp(每个计分点相对起点的累计秒)
#   总 wall/token           ← stream-json 重定向里的最终 result 事件(权威)
#   方法改进说明            ← 版本源文件名后缀(v18_interleave→interleave);有 RESULT_JSON 则用其 changed
#   瓶颈分析(新)           ← 仅当该版本有 RESULT_JSON.ncu/technique 时自动填,否则留空(手填)
#
# 注:RESULT_JSON 不再决定曲线点数(旧逻辑 marker 优先 → 漏点);worker 不打 marker 也能出完整曲线。
#
# 用法: ./results/parse_run.sh <方法名> <stream-json重定向.jsonl> <transcript.jsonl>
set -uo pipefail
METHOD="${1:?用法: parse_run.sh <方法名> <重定向.jsonl> <transcript.jsonl>}"
JSONL="${2:?需要 stream-json 重定向(取最终 result 事件)}"
T="${3:?需要 transcript jsonl}"
OUTDIR="results/${METHOD}"; mkdir -p "$OUTDIR"   # 每个方法独立文件夹
OUTCSV="$OUTDIR/result.csv"
# 版本源文件目录(取文件名后缀当改进说明兜底)。fork 命名可能带后缀(如 playground-naive-clean),自动探测。
FORKDIR="playground-${METHOD}"
[ -d "$FORKDIR" ] || FORKDIR="$(ls -d playground-${METHOD}* 2>/dev/null | head -1)"
SRCDIR="${FORKDIR:-playground-${METHOD}}/task-1/src/matmul_f16"
ROWS=$(mktemp)

# ── 从 transcript 抽四类行 ────────────────────────────────────────────────
#   A: assistant 消息的 (message.id, 累计output_tokens, ts) → 累计 token / 起点
#   P: 一次内核运行命令解析出的 (ver, m, n, k, iters) → 给紧随其后的计分行定版本+口径。
#      task1.sh run = 固定 4096³/100;直接调 binary(或 ncu 包裹)取其 -m/-n/-k/-t,缺省 4096³/100。
#      ⚠️ 计分口径只认 4096³ 且 iters==100(task1 标准计分轮数);worker 常用大 shape(8192²)/小 shape/
#         -t1(ncu)/-t50 做实验,**也包括 -t2000 这类"跑久了 GPU 暖机→时钟 boost→虚高"的热峰长跑**,
#         这些 off-口径 数若混进曲线会虚高 peak(naive_cycle4 实测:t=100 是 196.9、t=2000 暖机 206.8)——故 R 行带 canon 标志。
#   R: 计分行 TFLOPS/Error(继承最近 P 的口径,canon = 4096³ && iters==100;≠100 一律 off-口径)
#   M: RESULT_JSON 标记(整条 JSON,作注解)
jq -rc '
  if .type=="assistant" then
    ((.timestamp // "" | sub("\\.[0-9]+";"") | fromdateiso8601? // 0)) as $ts
    | ( "A\t\(.message.id // "x")\t\(.message.usage.output_tokens // 0)\t\($ts)" ),
      ( .message.content[]? | select(.type=="tool_use") | (.input.command // "")
        | select(test("task1\\.sh +run|task1_float16_v[0-9]"))
        | . as $c
        | ((($c|capture("--ver[ ]+(?<v>[0-9]+)")).v)? // (($c|capture("task1_float16_v(?<v>[0-9]+)")).v)? // "?") as $v
        | ((($c|capture("-m[ ]+(?<x>[0-9]+)")).x)? // "4096") as $m
        | ((($c|capture("-n[ ]+(?<x>[0-9]+)")).x)? // "4096") as $n
        | ((($c|capture("-k[ ]+(?<x>[0-9]+)")).x)? // "4096") as $k
        | ((($c|capture("-t[ ]+(?<x>[0-9]+)")).x)? // "100") as $t
        | "P\t\($ts)\t\($v)\t\($m)\t\($n)\t\($k)\t\($t)" ),
      ( ([.message.content[]? | select(.type=="text") | .text] | join("\n")) as $txt
        | if ($txt|test("RESULT_JSON"))
          then ($txt | (capture("RESULT_JSON[ ]*(?<j>\\{[^\\n]*\\})") | .j)?) as $j
               | "M\t\($ts)\t\($j|gsub("\t";" "))"
          else empty end )
  elif .type=="user" then
    ((.timestamp // "" | sub("\\.[0-9]+";"") | fromdateiso8601? // 0)) as $ts
    | ([.message.content[]? | select(.type=="tool_result") | (.content|tostring)] | join("\n")) as $tc
    | if ($tc|test("TFLOPS: [0-9.eE+-]+; *Average Error:"))
      then ($tc | capture("TFLOPS: (?<tf>[0-9.eE+-]+); *Average Error: (?<err>[0-9.eE+-]+)")) as $m
           | "R\t\($ts)\t\($m.tf)\t\($m.err)"
      else empty end
  elif .type=="attachment" then
    ( (.|tostring) as $s          # /goal evaluator 判定 = transcript 的 attachment 里 "goal_status";sentinel:true 是启动占位(排除),met:false=顶回去续、met:true=判达成
      | if ($s|test("goal_status")) and (($s|test("sentinel.:true"))|not)
        then ( "G\t" + (if ($s|test("met.:true")) then "met" else "notmet" end) )
        else empty end )
  else empty end
' "$T" \
| awk -F'\t' '
  $1=="A"{ d=$3-h[$2]; if(d>0){cum+=d;h[$2]=$3}; if(first==0&&$4>0)first=$4; next }
  $1=="P"{ pv=$3; pm=$4+0; pn=$5+0; pk=$6+0; pt=$7+0; next }
  $1=="R"{ canon=(pm==4096&&pn==4096&&pk==4096&&pt==100)?1:0;
           printf "R\t%d\t%d\t%s\t%s\t%s\t%d\t%dx%dx%d\t%d\n", cum,(first>0?$2-first:0),$3,$4,pv,canon,pm,pn,pk,pt; next }
  $1=="M"{ printf "M\t%d\t%d\t%s\n",         cum,(first>0?$2-first:0),$3; next }
  $1=="G"{ printf "G\t%d\t%s\n", cum, $2; next }   # evaluator 接入:cum=接入时累计 token(=曲线 x),$2=met/notmet
' > "$ROWS"

read -r TOTW TOTT < <(jq -r 'select(.type=="result") | "\((.duration_ms/1000)|floor) \(.usage.output_tokens)"' "$JSONL" 2>/dev/null | tail -1)

# ── CSV(逐次计分,密)+ 每版最佳 markdown 表 + 曲线图,全在 python 里做 join ──
CUBLAS_REF="${CUBLAS_REF:-219}"   # 曲线上的 cuBLAS 参照线 = 本机实测(@1155:f16-acc 219.8 / fp32-acc 218.7,见 memory library-ceilings-a100-gemm + results/_baseline_cublas_f16.log);取 ~219。改值: CUBLAS_REF=240 ./results/parse_run.sh ...
/opt/torch/bin/python - "$ROWS" "$OUTDIR" "$METHOD" "$SRCDIR" "$CUBLAS_REF" <<'PY' || { echo "(python/matplotlib 缺失,装: /opt/uv/uv pip install --python /opt/torch/bin/python matplotlib)"; exit 1; }
import csv, sys, json, glob, re, os
rows_path, outdir, method, srcdir, cublas_ref = sys.argv[1:6]; cublas_ref=float(cublas_ref)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R=[]; M={}; GG=[]
for line in open(rows_path):
    p=line.rstrip("\n").split("\t")
    if p[0]=="R" and len(p)>=9:
        R.append(dict(cum=int(p[1]), wall=int(p[2]), tflops=float(p[3]), error=float(p[4]),
                      ver=(p[5] if p[5] else "?"), canon=(p[6]=="1"), shape=p[7], iters=int(p[8])))
    elif p[0]=="M" and len(p)>=4:
        try:
            mk=json.loads(p[3]); v=str(mk.get("variant","")).lstrip("v")
            if v: M[v]=mk          # 同版本多次 marker:后者覆盖
        except Exception: pass
    elif p[0]=="G" and len(p)>=3:  # /goal evaluator 接入(goal_status,非 sentinel):cum=接入时累计 token,met=判定
        GG.append(dict(cum=int(p[1]), met=(p[2]=="met")))

LBL={}                             # 可选人工标签 override(每方法一份 results/<方法>/labels.json,ver→短技法);供表+图共用
                                   # 值可为字符串(=技法,向后兼容)或 {"t":"技法","b":"瓶颈"}(后者额外填表的「瓶颈分析」列)
try:
    _p=os.path.join(outdir,"labels.json")
    if os.path.exists(_p): LBL={str(k):v for k,v in json.load(open(_p)).items()}
except Exception: pass
def _lblt(v): return (v.get("t","") if isinstance(v,dict) else str(v))   # labels 值→技法
def _lblb(v): return (v.get("b","") if isinstance(v,dict) else "")        # labels 值→瓶颈(仅 dict 形)

INV={}                             # 可选作废版本表 results/<方法>/invalid.json(ver→原因,如偷库/非手写):列出的版本不计分、不进曲线,图上单列红叉审计留痕
try:
    _ip=os.path.join(outdir,"invalid.json")
    if os.path.exists(_ip): INV={str(k):str(v) for k,v in json.load(open(_ip)).items()}
except Exception: pass

def fnlabel(ver):                  # 版本源文件名后缀当改进说明兜底
    for g in glob.glob(os.path.join(srcdir, f"matmul_f16_v{ver}_*.cu")):
        m=re.match(rf"matmul_f16_v{ver}_(.+)\.cu", os.path.basename(g));
        if m: return m.group(1)
    return ""

def improve(ver):
    if ver in LBL: return f"v{ver}: {_lblt(LBL[ver])}"
    mk=M.get(ver)
    if mk and mk.get("changed"): return f"v{ver}: {mk['changed']}"
    lbl=fnlabel(ver);  return f"v{ver}: {lbl}" if lbl else f"v{ver}"

def bottleneck(ver):               # 优先 labels.json 的人工 "b";否则 marker 有 ncu/technique 时自动给
    if ver in LBL and _lblb(LBL[ver]): return _lblb(LBL[ver])
    mk=M.get(ver)
    if not mk: return ""
    n=mk.get("ncu",{}); parts=[]
    if "tensor_pct" in n:
        s=f"tensor {n['tensor_pct']}%"
        if "tensor_pct_tailfree" in n: s+=f"(尾段{n['tensor_pct_tailfree']})"
        parts.append(s)
    if "sm_pct" in n: parts.append(f"SM {n['sm_pct']}%")
    if "regs" in n: parts.append(f"regs {n['regs']}")
    if n.get("bank_conflicts")==0: parts.append("0 bank冲突")
    elif "bank_conflicts" in n: parts.append(f"bank冲突 {n['bank_conflicts']}")
    if "blocks_per_sm" in n: parts.append(f"{n['blocks_per_sm']} blk/SM")
    st=n.get("top_stall") or n.get("wait_stall")
    if st: parts.append(f"top stall: {st}")
    return " · ".join(str(x) for x in parts)

def vkey(v):
    try: return int(v)
    except: return 9999

# ── 有效计分判定 ──
#   scored = canonical(4096³ 且 iters≥100,awk 给)且 正确性过关(相对误差 < ERRMAX);二者皆满足才进 best/曲线/计分。
#   排除两类:① off-口径(大/小 shape、ncu -t1、-t50 等);② 口径对但结果无效(首版写错/数据彩票 inf → 误差爆大)。
ERRMAX=0.1                          # f16 正常 ≲0.02(fp16 累加 ~0.017 也算对);误差 >0.1(或 inf——上游正则已滤)= 该次结果无效,不计分
for r in R:
    r["invalid"]=(r["ver"] in INV)          # invalid.json 标的作废版本(偷库/非手写):排除出计分/曲线
    r["scored"]=bool(r["canon"] and (r["error"]==r["error"]) and r["error"]<ERRMAX and not r["invalid"])
Rc=[r for r in R if r["scored"]]; ndrop=len(R)-len(Rc)

# ── 密 CSV:每次计分一行;canonical=口径(4096³/100),scored=口径且正确性过关(= 是否计入曲线)──
with open(os.path.join(outdir,"result.csv"),"w",newline="") as f:
    w=csv.writer(f); w.writerow(["cycle","wall_clock","tokens","correctness","tflops","version","shape","iters","canonical","scored","invalid"])
    for i,r in enumerate(R,1):
        w.writerow([i, r["wall"], r["cum"], r["error"], f"{r['tflops']:.1f}", r["ver"], r["shape"], r["iters"], int(r["canon"]), int(r["scored"]), int(r["invalid"])])

# ── 每版最佳(scored)→ markdown 表 ──
best={}
for r in Rc:
    v=r["ver"]
    if v=="?": continue            # 版本号没解析出的点不进每版最佳表(仍在 csv 与曲线散点)
    if v not in best or r["tflops"]>best[v]["tflops"]: best[v]=r
order=sorted(best, key=vkey)
md=["| cycle | wall_clock(s) | tokens | correctness | tflops | 方法改进说明 | 瓶颈分析 | log |",
    "| --- | --- | --- | --- | --- | --- | --- | --- |"]
for i,v in enumerate(order,1):
    r=best[v]
    md.append(f"| {i} | {r['wall']} | {r['cum']} | {r['error']} | {r['tflops']:.1f} | {improve(v)} | {bottleneck(v)} | |")
md="\n".join(md)+"\n"
open(os.path.join(outdir,"result_table.md"),"w").write(md)

# ── 曲线:worker 全部手写版本逐次散点 + running-best 包络 ──
#   点标注:每个版本取「最佳一次」标一个点(按 tflops 取最高,自动跳过 k=64 调试小分),含回归版本
#           → 尽量多展示方法演进。前沿(刷新 running-best)的标在点上方、回归版标下方,交替偏移防叠字。
#   描述文字来源(优先级):可选人工 override `results/<方法>/labels.json`({"3":"XOR swizzle",...})
#                          → 版本源文件名后缀(v3_swizzle→swizzle) → RESULT_JSON marker 的 technique/changed → 仅 vN。
#   labels.json 给薄 dispatcher/共享头那种「源文件名不带技法」的方法补描述用;无此文件则全自动、不报错。
#   (LBL 已在上方表格段加载,表+图共用同一份。)
#   参照线:固定 cuBLAS 标尺(可视化用)+ 312 硬件峰值。
def techlbl(ver):                  # 该版本的「技法」短描述(可空)
    if ver in LBL: return _lblt(LBL[ver])
    lbl=fnlabel(ver)
    if lbl: return lbl.replace("_"," ")
    mk=M.get(ver)
    if mk:
        t=mk.get("technique") or mk.get("changed") or ""
        if t: return str(t)[:20]
    return ""
def shortlbl(ver):                 # 标签:vN + 技法(无技法则仅 vN)
    t=techlbl(ver); return f"v{ver} {t}" if t else f"v{ver}"
worker=sorted(Rc, key=lambda r:r["cum"])         # 仅 canonical(4096³/100 轮)进曲线
xs=[r["cum"]/1000 for r in worker]; ys=[r["tflops"] for r in worker]
rb=[]; mx=0
for y in ys: mx=max(mx,y); rb.append(mx)
bestpt={}                          # 每版最佳一点,按 token 排序后逐个标注
for r in worker:
    v=r["ver"]
    if v=="?": continue            # 未知版本不打标(点仍在散点 + running-best)
    if v not in bestpt or r["tflops"]>bestpt[v]["tflops"]: bestpt[v]=r
labels=sorted(bestpt.values(), key=lambda r:r["cum"])
plt.figure(figsize=(10,5.5))
# /goal evaluator 接入(goal_status,非 sentinel):竖虚线标在「接入时累计 token」处。met:false=未达成顶回去(红);met:true=判达成(绿)
for g in GG:
    plt.axvline(g["cum"]/1000, ls="--", color=("tab:green" if g["met"] else "tab:red"), lw=1.1, alpha=.55, zorder=1.4)
_nnm=sum(1 for g in GG if not g["met"]); _nm=sum(1 for g in GG if g["met"])
if _nnm: plt.plot([],[],ls="--",color="tab:red",lw=1.1,label=f"/goal evaluator: not met ×{_nnm}")
if _nm:  plt.plot([],[],ls="--",color="tab:green",lw=1.1,label=f"/goal evaluator: met ×{_nm}")
# 未计分点(off-口径:大/小 shape、ncu -t1、-t50;或口径对但结果无效:写错/inf):浅灰叉,仅作存在性提示
ncx=[r["cum"]/1000 for r in R if not r["scored"] and not r["invalid"]]; ncy=[r["tflops"] for r in R if not r["scored"] and not r["invalid"]]
if ncx: plt.scatter(ncx,ncy,s=13,alpha=.35,color="lightgray",marker="x",zorder=1,
                    label=f"not scored: off-spec ({len(ncx)})")
# invalid.json 标的作废版本(偷库/非手写):红叉单列、标出来但不进计分/曲线 —— 防作弊审计留痕,不静默丢弃
inv=[r for r in R if r["invalid"] and r["canon"]]
if inv:
    plt.scatter([r["cum"]/1000 for r in inv],[r["tflops"] for r in inv],s=42,color="tab:red",
                marker="x",lw=1.6,zorder=4,label=f"excluded: non-handwritten/library ({len(inv)})")
    for r in inv:
        plt.annotate(shortlbl(r["ver"]),(r["cum"]/1000,r["tflops"]),textcoords="offset points",
                     xytext=(4,6),fontsize=7,color="tab:red",va="bottom")
if xs:
    pk=max(ys)
    plt.scatter(xs,ys,s=16,alpha=.30,color="tab:blue",zorder=2,label="canonical 4096³/100-iter (incl. regressions)")
    plt.plot(xs,rb,"-o",lw=2,ms=3.5,color="tab:blue",zorder=3,label="running best")
    seen=0; ai=0; bi=0
    for r in labels:
        y=r["tflops"]; x=r["cum"]/1000
        frontier=(y>=seen-1e-6); seen=max(seen,y)
        txt=shortlbl(r["ver"])
        if abs(y-pk)<1e-6: txt+=f" · {y:.0f}"     # 全局峰值(canonical)附 TFLOPS
        if frontier:
            dy=10 if ai%2==0 else 24; ai+=1; col="tab:blue"; va="bottom"
        else:
            dy=-(12 if bi%2==0 else 26); bi+=1; col="dimgray"; va="top"
        plt.annotate(txt,(x,y),textcoords="offset points",xytext=(3,dy),
                     fontsize=7.5,color=col,va=va,
                     arrowprops=dict(arrowstyle="-",lw=.4,color="gray",alpha=.45))
# /goal "轮"区间箭头:红线 = 该轮末 worker 想停、evaluator 判 not met(=一轮的右边界)。
# round1 = 起点→第1条红线(worker 自驱爬升那一大段);round i = 第 i-1 条 → 第 i 条红线。
if GG and xs:
    gx=sorted(g["cum"]/1000 for g in GG)
    bnd=[min(xs)]+gx
    yr=255
    for i in range(len(gx)):
        plt.annotate("",xy=(bnd[i+1],yr),xytext=(bnd[i],yr),
                     arrowprops=dict(arrowstyle="<->",color="tab:red",lw=1.3,alpha=.85,shrinkA=0,shrinkB=0))
        plt.text((bnd[i]+bnd[i+1])/2,yr+7,f"round {i+1}",ha="center",va="bottom",
                 fontsize=8.5,color="tab:red",fontweight="bold")
plt.axhline(cublas_ref,ls="--",color="tab:orange",lw=1.2,label=f"cuBLAS ref ~{cublas_ref:.0f}")
plt.axhline(312,ls="--",color="grey",lw=1.2,label="A100 fp16 peak 312")
plt.xlabel("cumulative output tokens (k)"); plt.ylabel("TFLOPS (task1 4096³ 100-iter sustained)")
plt.ylim(0,330)
plt.title(f"{method}: TFLOPS vs token"); plt.grid(alpha=.3); plt.legend(fontsize=8,loc="lower right")
plt.tight_layout(); plt.savefig(os.path.join(outdir,"curve.png"),dpi=140)

ninv=sum(1 for r in R if r["invalid"])
print(f"== result.csv:{len(R)} 行(scored {len(Rc)} / 未计分 {ndrop},含作废 {ninv}),{len(order)} 个计分版本;markers {len(M)} ==")
print(md)
print(f"peak(scored 手写 4096³)={max(ys):.1f}" if ys else "no scored rows")
if ndrop: print(f"   (排除 {ndrop} 个未计分点[off-口径/写错/inf{('/'+str(ninv)+'个作废非手写') if ninv else ''}]出计分/running-best;明细见 csv 的 canonical/scored/invalid 列)")
if GG: print(f"   /goal evaluator 接入 {len(GG)} 次(not met {_nnm} / met {_nm});接入时累计 token = {[g['cum'] for g in GG]}(曲线红色竖虚线)")
PY

echo "== 总计(权威,来自 result 事件): wall_clock=${TOTW:-?}s  output_tokens=${TOTT:-?} =="
echo "saved $OUTDIR/{result.csv,result_table.md,curve.png}"
rm -f "$ROWS"
