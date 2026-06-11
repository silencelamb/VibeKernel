export const meta = {
  name: 'gemm-occ-ilp-tournament',
  description: 'Probe occupancy-vs-ILP tradeoff for v7: small warp tiles (more blocks/SM) vs big 64x64 warps (more ILP). Build parallel, bench serial.',
  phases: [
    { title: 'Build', detail: 'parallel builds per config' },
    { title: 'Benchmark', detail: 'serial 100-round scoring on single GPU' },
  ],
}

// (BM,BN,BK,STAGES,WARP_M,WARP_N,MIN_BLOCKS). warp tile = (BM/WM)x(BN/WN).
const CONFIGS = [
  { id: 1,  BM:128, BN:256, BK:32, S:3, WM:2, WN:8, MB:1 }, // current best (64x32, 16 warps)
  { id: 2,  BM:128, BN:128, BK:32, S:4, WM:4, WN:4, MB:2 }, // 32x32, aim 2 blocks=32 warps
  { id: 3,  BM:128, BN:128, BK:32, S:3, WM:4, WN:4, MB:2 }, // 32x32, 2 blocks, S3
  { id: 4,  BM:64,  BN:256, BK:32, S:3, WM:2, WN:8, MB:2 }, // 32x32, 2 blocks
  { id: 5,  BM:256, BN:64,  BK:32, S:3, WM:8, WN:2, MB:2 }, // 32x32, 2 blocks
  { id: 6,  BM:256, BN:128, BK:32, S:3, WM:4, WN:2, MB:1 }, // 64x64 warp, 8 warps, max ILP
  { id: 7,  BM:128, BN:128, BK:32, S:3, WM:2, WN:2, MB:2 }, // 64x64 warp, MB2=8 warps
  { id: 8,  BM:128, BN:128, BK:32, S:4, WM:8, WN:2, MB:2 }, // 16x64 warp (M_TILES=1), low reg
  { id: 9,  BM:64,  BN:128, BK:32, S:4, WM:2, WN:4, MB:3 }, // 32x32 small, aim 3 blocks=24 warps
  { id: 10, BM:128, BN:128, BK:32, S:6, WM:2, WN:8, MB:1 }, // 64x16 warp, deep S6, 16 warps
  { id: 11, BM:128, BN:128, BK:32, S:5, WM:4, WN:4, MB:2 }, // 32x32, deep S5, 2 blocks
  { id: 12, BM:256, BN:256, BK:32, S:3, WM:4, WN:8, MB:1 }, // 64x32 warp, 32 warps in 1 block(1024thr)
]

const REPO = '/home/daixu/code/github_code/VibeKernel/worktrees/dynamic_workflow'

function buildCmd(c) {
  const dir = `${REPO}/build_t9/cfg${c.id}`
  const defs = `-DPG_BM=${c.BM} -DPG_BN=${c.BN} -DPG_BK=${c.BK} -DPG_STAGES=${c.S} -DPG_WARP_M=${c.WM} -DPG_WARP_N=${c.WN} -DPG_MIN_BLOCKS=${c.MB}`
  const log = `${REPO}/build_t9/cfg${c.id}.log`
  return `mkdir -p ${REPO}/build_t9 && cd ${REPO} && cmake -S ./task-1 -B ${dir} -G Ninja -DCMAKE_TOOLCHAIN_FILE="$VCPKG_HOME/scripts/buildsystems/vcpkg.cmake" -DMATMUL_VERSION=7 -DTEST_DATA_TYPE=float16 -DSTDOUT_IS_TERMINAL=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=20 -DCMAKE_CUDA_STANDARD=20 -DCMAKE_CUDA_FLAGS="${defs}" > ${log} 2>&1 && cmake --build ${dir} --parallel 3 >> ${log} 2>&1`
}

const BUILD_SCHEMA = {
  type: 'object',
  properties: { id: { type: 'integer' }, built: { type: 'boolean' }, note: { type: 'string' } },
  required: ['id', 'built', 'note'],
}

phase('Build')
const built = await parallel(CONFIGS.map((c) => () =>
  agent(
    `Run ONE deterministic CUDA build. Do NOT run GPU code. Run exactly:\n\n${buildCmd(c)}\n\n` +
    `Then: ls ${REPO}/build_t9/cfg${c.id}/src/task1_float16_v7\n` +
    `built=true if binary exists, else built=false with first error (grep -m1 -iE "error" ${REPO}/build_t9/cfg${c.id}.log) in note (<200 chars). id=${c.id}.`,
    { label: `build:cfg${c.id}`, phase: 'Build', schema: BUILD_SCHEMA }
  )
))

const okIds = built.filter(Boolean).filter((b) => b.built).map((b) => b.id)
const failed = built.filter(Boolean).filter((b) => !b.built)
log(`Build: ${okIds.length}/${CONFIGS.length} ok. Failed: ${failed.map((f) => `cfg${f.id}(${f.note})`).join('; ') || 'none'}`)

const cfgById = Object.fromEntries(CONFIGS.map((c) => [c.id, c]))
const benchList = okIds.map((id) => `cfg${id}: ${REPO}/build_t9/cfg${id}/src/task1_float16_v7`).join('\n')

const BENCH_SCHEMA = {
  type: 'object',
  properties: {
    results: { type: 'array', items: { type: 'object', properties: {
      id: { type: 'integer' }, tflops: { type: 'number' }, error: { type: 'number' }, ok: { type: 'boolean' },
    }, required: ['id', 'tflops', 'error', 'ok'] } },
  },
  required: ['results'],
}

phase('Benchmark')
const benched = await agent(
  `Benchmark pre-built CUDA GEMM binaries on a SINGLE GPU, STRICTLY ONE AT A TIME (never parallel).\n` +
  `For each: <binary> -m 4096 -n 4096 -k 4096 -t 100 ; parse "TFLOPS: X; Average Error: Y". ok=true if ran and error<0.05; on crash ok=false tflops=0 error=999.\n\n` +
  `Binaries (serially):\n${benchList}\n\nReturn results for all.`,
  { label: 'bench:serial', phase: 'Benchmark', schema: BENCH_SCHEMA }
)

const rows = (benched?.results || []).map((r) => ({ ...r, cfg: cfgById[r.id] }))
rows.sort((a, b) => (b.ok ? b.tflops : -1) - (a.ok ? a.tflops : -1))
log('occ/ILP tournament results (sorted):')
for (const r of rows) {
  const c = r.cfg
  log(`cfg${r.id} [${c.BM}x${c.BN} BK${c.BK} S${c.S} W${c.WM}x${c.WN}(tile ${c.BM/c.WM}x${c.BN/c.WN}) MB${c.MB}]: ${r.ok ? r.tflops.toFixed(1) + ' TF, err ' + r.error.toExponential(2) : 'FAILED'}`)
}

return { sorted: rows.map((r) => ({ id: r.id, cfg: r.cfg, tflops: r.tflops, error: r.error, ok: r.ok })), failedBuilds: failed }
