export const meta = {
  name: 'gemm-phase1-tile-sweep',
  description: 'Phase 1: sweep full-stack fp16 GEMM tile/BK/stage configs to pick the best reference kernel',
  phases: [{ title: 'Sweep', detail: 'one agent per tile config: gen+build+bench (100-round avg) under GPU lock' }],
}

// [N, BM, BN, BK, WARPS_M, WARPS_N, STAGES] — all with cp.async+swizzle+16B+ldmatrix ON
const CFGS = [
  [30,128,128,32,2,4,3],  // baseline control (= v3, 137.8)
  [31,128,128,64,2,4,3],  // BK64 (fix A bank conflict)
  [32,128,128,32,2,4,4],  // 4 stages
  [33,128,128,64,2,4,4],  // BK64 + 4 stages
  [34,128,256,32,2,4,3],  // wider N, WN=64
  [35,256,128,32,4,2,3],  // WM=64 WN=64
  [36,256,128,32,2,4,3],  // WM=128 WN=32
  [37,128,256,32,4,2,3],  // WM=32 WN=128
  [38,128,128,32,2,2,3],  // 4 warps, WM=64 WN=64
  [39,128,128,64,2,2,3],  // 4 warps, BK64
  [40,256,128,64,4,2,3],  // BK64 big
  [41,128,256,64,2,4,3],  // BK64 wide N
  [42,256,256,32,4,4,3],  // big 16-warp
  [43,128,128,16,2,4,3],  // BK16
  [44,64,256,32,1,4,3],   // tall-N
  [45,256,64,32,4,2,3],   // tall-M
  [46,128,64,32,2,2,3],   // small
  [47,64,128,32,1,4,3],   // small M
  [48,128,128,64,4,2,3],  // BK64 WM=32 WN=64
  [49,256,128,32,4,4,3],  // 16 warps WM=64 WN=32
  [50,128,256,64,4,2,3],  // BK64 WM=32 WN=128
]

const SCH = {
  type: 'object',
  properties: {
    tflops: { type: 'number', description: 'numeric TFLOPS, or 0 if build/run failed' },
    err: { type: 'string', description: 'the ERR value reported, or NA' },
    status: { type: 'string', enum: ['ok', 'build_fail', 'run_fail'] },
    reason: { type: 'string', description: 'one-line failure reason if not ok, else empty' },
  },
  required: ['tflops', 'status'],
}

phase('Sweep')
const results = await parallel(CFGS.map((c) => () => {
  const [N, BM, BN, BK, WM, WN, ST] = c
  const gen = `bash scripts/gen_variant.sh ${N} ${BM} ${BN} ${BK} ${WM} ${WN} ${ST} true true 16 true`
  const prompt =
`Benchmark ONE GEMM config. Run EXACTLY these two commands and nothing else. Do NOT edit any source files, do NOT try to optimize or change the config.

${gen}
bash scripts/bench.sh ${N}

The bench prints ONE line: "V${N} RESULT TFLOPS=<t> ERR=<e>"  (t may be BUILD_FAIL or RUN_FAIL).
Return: tflops = numeric TFLOPS (0 if BUILD_FAIL/RUN_FAIL), err = the ERR string, status = "ok" if numeric else "build_fail"/"run_fail", reason = one short line of the failure cause from the output (empty if ok).`
  return agent(prompt, { label: `v${N}:${BM}x${BN}x${BK}/${WM}x${WN}/S${ST}`, schema: SCH })
    .then((r) => (r ? { N, BM, BN, BK, WM, WN, ST, ...r } : { N, BM, BN, BK, WM, WN, ST, tflops: 0, status: 'run_fail', reason: 'agent null' }))
})).then((rs) => rs.filter(Boolean))

const ok = results.filter((r) => r.status === 'ok').sort((a, b) => b.tflops - a.tflops)
const bad = results.filter((r) => r.status !== 'ok')
log(`Phase1 done. ${ok.length} ok, ${bad.length} failed.`)
log('TOP5: ' + ok.slice(0, 5).map((r) => `v${r.N}(${r.BM}x${r.BN}x${r.BK}/${r.WM}x${r.WN}/S${r.ST})=${Number(r.tflops).toFixed(1)}`).join('  |  '))
if (bad.length) log('FAILED: ' + bad.map((r) => `v${r.N}(${r.reason || r.status})`).join('  '))
return { ranked: ok, failed: bad }
