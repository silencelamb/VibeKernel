export const meta = {
  name: 'gemm-phase2-3-ablation',
  description: 'Phase 2 leave-one-out + Phase 3 synergy ablation on the locked FULL config (128x128x64 2x2 S2)',
  phases: [
    { title: 'Phase2-LOO', detail: 'remove exactly one ingredient from the full stack; remeasure (100-round)' },
    { title: 'Phase3-Synergy', detail: 'both-off corners for the 3 hypothesized synergy pairs' },
  ],
}

// FULL = 128,128,64, warps 2x2, STAGES=2, cp.async+swizzle+16B+ldmatrix. Each row:
// [N, label, BM,BN,BK,WPM,WPN,STAGES, CP_ASYNC, SWIZZLE, COPY_BYTES, LDMATRIX]
const LOO = [
  [60, 'FULL',            128,128,64,2,2,2, 'true','true',16,'true'],
  [61, '-cp.async',       128,128,64,2,2,2, 'false','true',16,'true'],
  [62, '-pipeline(S1)',   128,128,64,2,2,1, 'true','true',16,'true'],
  [63, '-swizzle',        128,128,64,2,2,2, 'true','false',16,'true'],
  [64, '-vectorize(4B)',  128,128,64,2,2,2, 'true','true',4,'true'],
  [65, '-ldmatrix',       128,128,64,2,2,2, 'true','true',16,'false'],
  [66, '-regblock(32x32)',128,128,64,4,4,2, 'true','true',16,'true'],
]
const SYN = [
  [70, 'syn:-cpasync-pipeline',          128,128,64,2,2,1, 'false','true',16,'true'],
  [71, 'syn:-cpasync-pipeline-swizzle',  128,128,64,2,2,1, 'false','false',16,'true'],
  [72, 'syn:-regblock-pipeline',         128,128,64,4,4,1, 'true','true',16,'true'],
]

const SCH = {
  type: 'object',
  properties: {
    tflops: { type: 'number' }, err: { type: 'string' },
    status: { type: 'string', enum: ['ok', 'build_fail', 'run_fail'] },
    reason: { type: 'string' },
  },
  required: ['tflops', 'status'],
}

const runCfg = (row, phaseName) => () => {
  const [N, label, BM, BN, BK, WPM, WPN, ST, CA, SW, CB, LD] = row
  const gen = `bash scripts/gen_variant.sh ${N} ${BM} ${BN} ${BK} ${WPM} ${WPN} ${ST} ${CA} ${SW} ${CB} ${LD}`
  const prompt =
`Benchmark ONE GEMM ablation variant. Run EXACTLY these two commands and nothing else. Do NOT edit source files or change the config.

${gen}
bash scripts/bench.sh ${N}

Prints ONE line: "V${N} RESULT TFLOPS=<t> ERR=<e>".
Return: tflops = numeric TFLOPS (0 if BUILD_FAIL/RUN_FAIL), err = the ERR string, status="ok" if numeric else "build_fail"/"run_fail", reason = short failure line (empty if ok).
If ERR >= 0.05, the kernel is numerically wrong: set status="run_fail", reason="bad_err=<e>".`
  return agent(prompt, { label: `v${N} ${label}`, phase: phaseName, schema: SCH })
    .then((r) => (r ? { N, label, ...r } : { N, label, tflops: 0, status: 'run_fail', reason: 'agent null' }))
}

phase('Phase2-LOO')
const loo = await parallel(LOO.map((row) => runCfg(row, 'Phase2-LOO'))).then((rs) => rs.filter(Boolean))

phase('Phase3-Synergy')
const syn = await parallel(SYN.map((row) => runCfg(row, 'Phase3-Synergy'))).then((rs) => rs.filter(Boolean))

const all = [...loo, ...syn]
const get = (n) => { const r = all.find((x) => x.N === n); return r && r.status === 'ok' ? Number(r.tflops) : null }
const FULL = get(60)
log(`FULL=${FULL}`)
if (FULL) {
  for (const r of loo.filter((r) => r.N !== 60)) {
    const t = r.status === 'ok' ? Number(r.tflops) : null
    log(`LOO ${r.label}: ${t == null ? r.reason : t.toFixed(1)}  drop=${t == null ? 'NA' : (FULL - t).toFixed(1)}`)
  }
}
return { full: FULL, loo, syn, all }
