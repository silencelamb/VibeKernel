export const meta = {
  name: 'gemm-64x64-refine',
  description: 'Refine around 64x64 warp sweet spot: single-buffer (v6) vs double-buffer (v7), STAGES, block size, MIN_BLOCKS. Build parallel, bench serial.',
  phases: [
    { title: 'Build', detail: 'parallel builds (mixed v6/v7)' },
    { title: 'Benchmark', detail: 'serial 100-round scoring' },
  ],
}

// V=kernel version (6=single-buffer frags, 7=double-buffer). All 64x64 warp tile.
const CONFIGS = [
  { id: 1,  V:7, BM:128, BN:128, BK:32, S:3, WM:2, WN:2, MB:2 }, // current best 176
  { id: 2,  V:7, BM:128, BN:128, BK:32, S:4, WM:2, WN:2, MB:2 }, // S4
  { id: 3,  V:7, BM:128, BN:128, BK:32, S:5, WM:2, WN:2, MB:2 }, // S5
  { id: 4,  V:7, BM:256, BN:128, BK:32, S:3, WM:4, WN:2, MB:1 }, // big block, 8 warps
  { id: 5,  V:7, BM:256, BN:128, BK:32, S:4, WM:4, WN:2, MB:1 }, // big S4
  { id: 6,  V:6, BM:128, BN:128, BK:32, S:3, WM:2, WN:2, MB:2 }, // single-buffer vs #1
  { id: 7,  V:6, BM:128, BN:128, BK:32, S:4, WM:2, WN:2, MB:2 }, // single-buf S4
  { id: 8,  V:6, BM:128, BN:128, BK:32, S:3, WM:2, WN:2, MB:3 }, // single-buf, aim 3 blocks
  { id: 9,  V:6, BM:128, BN:128, BK:32, S:4, WM:2, WN:2, MB:3 }, // single-buf S4 3 blocks
  { id: 10, V:6, BM:256, BN:128, BK:32, S:3, WM:4, WN:2, MB:2 }, // single-buf big, 2 blocks
  { id: 11, V:6, BM:256, BN:128, BK:32, S:4, WM:4, WN:2, MB:1 }, // single-buf big S4
  { id: 12, V:7, BM:128, BN:256, BK:32, S:3, WM:2, WN:4, MB:1 }, // double-buf 64x64 other shape
]

const REPO = '/home/daixu/code/github_code/VibeKernel/worktrees/dynamic_workflow'

function buildCmd(c) {
  const dir = `${REPO}/build_t10/cfg${c.id}`
  const defs = `-DPG_BM=${c.BM} -DPG_BN=${c.BN} -DPG_BK=${c.BK} -DPG_STAGES=${c.S} -DPG_WARP_M=${c.WM} -DPG_WARP_N=${c.WN} -DPG_MIN_BLOCKS=${c.MB}`
  const log = `${REPO}/build_t10/cfg${c.id}.log`
  return `mkdir -p ${REPO}/build_t10 && cd ${REPO} && cmake -S ./task-1 -B ${dir} -G Ninja -DCMAKE_TOOLCHAIN_FILE="$VCPKG_HOME/scripts/buildsystems/vcpkg.cmake" -DMATMUL_VERSION=${c.V} -DTEST_DATA_TYPE=float16 -DSTDOUT_IS_TERMINAL=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=20 -DCMAKE_CUDA_STANDARD=20 -DCMAKE_CUDA_FLAGS="${defs}" > ${log} 2>&1 && cmake --build ${dir} --parallel 3 >> ${log} 2>&1`
}

const BUILD_SCHEMA = { type: 'object', properties: { id: { type: 'integer' }, built: { type: 'boolean' }, note: { type: 'string' } }, required: ['id', 'built', 'note'] }

phase('Build')
const built = await parallel(CONFIGS.map((c) => () =>
  agent(
    `Run ONE deterministic CUDA build. Do NOT run GPU code. Run exactly:\n\n${buildCmd(c)}\n\n` +
    `Then: ls ${REPO}/build_t10/cfg${c.id}/src/task1_float16_v${c.V}\n` +
    `built=true if binary exists, else false with first error (grep -m1 -iE "error" ${REPO}/build_t10/cfg${c.id}.log) in note (<200 chars). id=${c.id}.`,
    { label: `build:cfg${c.id}`, phase: 'Build', schema: BUILD_SCHEMA }
  )
))

const okIds = built.filter(Boolean).filter((b) => b.built).map((b) => b.id)
const failed = built.filter(Boolean).filter((b) => !b.built)
log(`Build: ${okIds.length}/${CONFIGS.length} ok. Failed: ${failed.map((f) => `cfg${f.id}(${f.note})`).join('; ') || 'none'}`)

const cfgById = Object.fromEntries(CONFIGS.map((c) => [c.id, c]))
const benchList = okIds.map((id) => `cfg${id}: ${REPO}/build_t10/cfg${id}/src/task1_float16_v${cfgById[id].V}`).join('\n')

const BENCH_SCHEMA = { type: 'object', properties: { results: { type: 'array', items: { type: 'object', properties: { id: { type: 'integer' }, tflops: { type: 'number' }, error: { type: 'number' }, ok: { type: 'boolean' } }, required: ['id', 'tflops', 'error', 'ok'] } } }, required: ['results'] }

phase('Benchmark')
const benched = await agent(
  `Benchmark pre-built CUDA GEMM binaries on a SINGLE GPU, STRICTLY ONE AT A TIME (never parallel).\n` +
  `For each: <binary> -m 4096 -n 4096 -k 4096 -t 100 ; parse "TFLOPS: X; Average Error: Y". ok=true if ran and error<0.05; on crash ok=false tflops=0 error=999.\n\n` +
  `Binaries (serially):\n${benchList}\n\nReturn results for all.`,
  { label: 'bench:serial', phase: 'Benchmark', schema: BENCH_SCHEMA }
)

const rows = (benched?.results || []).map((r) => ({ ...r, cfg: cfgById[r.id] }))
rows.sort((a, b) => (b.ok ? b.tflops : -1) - (a.ok ? a.tflops : -1))
log('64x64 refine results (sorted):')
for (const r of rows) {
  const c = r.cfg
  log(`cfg${r.id} v${c.V} [${c.BM}x${c.BN} S${c.S} W${c.WM}x${c.WN} MB${c.MB}]: ${r.ok ? r.tflops.toFixed(1) + ' TF, err ' + r.error.toExponential(2) : 'FAILED'}`)
}

return { sorted: rows.map((r) => ({ id: r.id, cfg: r.cfg, tflops: r.tflops, error: r.error, ok: r.ok })), failedBuilds: failed }
