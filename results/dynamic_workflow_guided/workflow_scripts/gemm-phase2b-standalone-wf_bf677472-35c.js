export const meta = {
  name: 'gemm-phase2b-standalone',
  description: 'Standalone (forward-greedy) deltas: each ingredient added ALONE to a minimal baseline, to expose the cooperative blindspot vs leave-one-out',
  phases: [{ title: 'Standalone', detail: 'minimal baseline + one ingredient each; remeasure (100-round)' }],
}

// MINIMAL = 128x128x64, warps 4x4 (32x32 warp tile = low register blocking), STAGES=1,
// cp.async OFF, swizzle OFF, manual frag (no ldmatrix). Then add exactly one ingredient.
// [N, label, BM,BN,BK,WPM,WPN,STAGES, CP_ASYNC, SWIZZLE, COPY_BYTES, LDMATRIX]
const ST = [
  [80, 'MINIMAL(none)',     128,128,64,4,4,1, 'false','false',16,'false'],
  [81, '+cp.async',         128,128,64,4,4,1, 'true','false',16,'false'],
  [82, '+pipeline',         128,128,64,4,4,2, 'false','false',16,'false'],
  [83, '+swizzle',          128,128,64,4,4,1, 'false','true',16,'false'],
  [85, '+regblock(64x64)',  128,128,64,2,2,1, 'false','false',16,'false'],
  [86, '+ldmatrix',         128,128,64,4,4,1, 'false','false',16,'true'],
  [87, 'cpasync+4B(vecbase)',128,128,64,4,4,1,'true','false',4,'false'],
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

phase('Standalone')
const res = await parallel(ST.map((row) => () => {
  const [N, label, BM, BN, BK, WPM, WPN, S, CA, SW, CB, LD] = row
  const gen = `bash scripts/gen_variant.sh ${N} ${BM} ${BN} ${BK} ${WPM} ${WPN} ${S} ${CA} ${SW} ${CB} ${LD}`
  const prompt =
`Benchmark ONE GEMM variant. Run EXACTLY these two commands and nothing else. Do NOT edit source files or change the config.

${gen}
bash scripts/bench.sh ${N}

Prints ONE line: "V${N} RESULT TFLOPS=<t> ERR=<e>".
Return tflops = numeric TFLOPS (0 if BUILD_FAIL/RUN_FAIL), err = ERR string, status = "ok" if numeric else "build_fail"/"run_fail", reason = short failure line (empty if ok). If ERR>=0.05 set status="run_fail", reason="bad_err=<e>".`
  return agent(prompt, { label: `v${N} ${label}`, schema: SCH })
    .then((r) => (r ? { N, label, ...r } : { N, label, tflops: 0, status: 'run_fail', reason: 'agent null' }))
})).then((rs) => rs.filter(Boolean))

const get = (n) => { const r = res.find((x) => x.N === n); return r && r.status === 'ok' ? Number(r.tflops) : null }
const M = get(80)
log(`MINIMAL=${M}`)
if (M != null) {
  const pairs = [['cp.async', get(81), M], ['pipeline', get(82), M], ['swizzle', get(83), M],
                 ['regblock', get(85), M], ['ldmatrix', get(86), M], ['vectorize', get(81), get(87)]]
  for (const [name, a, b] of pairs)
    log(`standalone ${name}: ${a == null || b == null ? 'NA' : (a - b).toFixed(1)} (added=${a==null?'NA':a.toFixed(1)} base=${b==null?'NA':b.toFixed(1)})`)
}
return { minimal: M, res }
