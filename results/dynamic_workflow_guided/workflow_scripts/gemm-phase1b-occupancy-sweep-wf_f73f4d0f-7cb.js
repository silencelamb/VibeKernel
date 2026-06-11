export const meta = {
  name: 'gemm-phase1b-occupancy-sweep',
  description: 'Phase 1b: second tuning sweep targeting occupancy (2 blocks/SM), stage depth, BK=64 conflict-free configs',
  phases: [{ title: 'Sweep2', detail: 'occupancy/stage/tile configs; gen(FP=false)+build+bench under GPU lock' }],
}

// [N, BM, BN, BK, WARPS_M, WARPS_N, STAGES]; FP=false (min registers). cp.async+swizzle+16B+ldmatrix ON.
const CFGS = [
  [200,128,128,64,2,4,2],  // BK64 S2 -> 64KB smem, 2 blocks?  warp 64x32
  [201, 64,128,64,1,4,3],  // 73KB -> 2 blocks; warp 64x32
  [202, 64,128,64,2,2,3],  // warp 32x64
  [203,128, 64,64,2,2,3],  // warp 64x32
  [204,128,128,64,2,2,2],  // S2 warp 64x64, 64KB -> 2 blocks
  [205,256,128,64,4,2,2],  // S2 of the winner (98KB->1 block)
  [206, 64,256,64,1,4,3],  // warp 64x64
  [207,128,256,64,2,4,2],  // S2 wide
  [208,128,128,32,2,4,5],  // deep pipeline BK32, 80KB -> 2 blocks
  [209,128,128,32,2,4,4],  // S4 BK32 control (was v32=142.7)
  [210,256,128,64,4,2,3],  // = best v40 control, now FP=false
  [211,128,128,64,4,2,2],  // S2 warp 32x64
  [212, 64,128,64,1,4,4],  // S4 small block
  [213,256, 64,64,4,1,3],  // warp 64x64, 4 warps
  [214,128,128,64,2,4,3],  // = v31 control, FP=false (warp 64x32)
  [215, 64, 64,64,1,2,4],  // small block, many blocks/SM, S4
  [216, 64,128,32,1,4,4],  // v47 cfg deepened to S4 (v47 S3=154)
  [217,128,128,32,2,4,6],  // very deep pipeline BK32 (96KB->1 block)
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

phase('Sweep2')
const results = await parallel(CFGS.map((c) => () => {
  const [N, BM, BN, BK, WM, WN, ST] = c
  const gen = `bash scripts/gen_variant.sh ${N} ${BM} ${BN} ${BK} ${WM} ${WN} ${ST} true true 16 true false`
  const prompt =
`Benchmark ONE GEMM config. Run EXACTLY these two commands and nothing else. Do NOT edit any source files or change the config.

${gen}
bash scripts/bench.sh ${N}

The bench prints ONE line: "V${N} RESULT TFLOPS=<t> ERR=<e>" (t may be BUILD_FAIL/RUN_FAIL).
Return: tflops = numeric TFLOPS (0 if fail), err = the ERR string, status = "ok" if numeric else "build_fail"/"run_fail", reason = one short failure line (empty if ok).
IMPORTANT correctness note: if ERR >= 0.05 treat it as a wrong kernel — still report the numbers but set status to "run_fail" and reason "bad_err=<e>".`
  return agent(prompt, { label: `v${N}:${BM}x${BN}x${BK}/${WM}x${WN}/S${ST}`, schema: SCH })
    .then((r) => (r ? { N, BM, BN, BK, WM, WN, ST, ...r } : { N, BM, BN, BK, WM, WN, ST, tflops: 0, status: 'run_fail', reason: 'agent null' }))
})).then((rs) => rs.filter(Boolean))

const ok = results.filter((r) => r.status === 'ok' && Number(r.tflops) > 0).sort((a, b) => b.tflops - a.tflops)
const bad = results.filter((r) => !(r.status === 'ok' && Number(r.tflops) > 0))
log(`Phase1b done. ${ok.length} ok, ${bad.length} failed/bad.`)
log('TOP6: ' + ok.slice(0, 6).map((r) => `v${r.N}(${r.BM}x${r.BN}x${r.BK}/${r.WM}x${r.WN}/S${r.ST})=${Number(r.tflops).toFixed(1)}`).join('  |  '))
if (bad.length) log('FAILED/BAD: ' + bad.map((r) => `v${r.N}(${r.reason || r.status})`).join('  '))
return { ranked: ok, failed: bad }
