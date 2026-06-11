export const meta = {
  name: 'gemm-v7-tournament',
  description: 'Build ~12 v7 (swizzle+regpipe) configs in parallel, benchmark serially on single GPU, return sorted TFLOPS',
  phases: [
    { title: 'Build', detail: 'parallel cmake+ninja per config into isolated build dirs' },
    { title: 'Benchmark', detail: 'one agent runs all built binaries serially (100-round scoring)' },
  ],
}

// (BM,BN,BK,STAGES,WARP_M,WARP_N,MIN_BLOCKS). Swizzle halves shared, so deeper
// stages and bigger blocks now fit. Warp tile kept small (<=64x32) to bound regs
// (register double-buffer doubles fragment regs).
const CONFIGS = [
  { id: 1,  BM:128, BN:128, BK:32, S:3, WM:2, WN:4, MB:2 }, // baseline v7 = 148
  { id: 2,  BM:128, BN:128, BK:32, S:4, WM:2, WN:4, MB:2 }, // STAGES=4
  { id: 3,  BM:128, BN:128, BK:32, S:5, WM:2, WN:4, MB:2 }, // STAGES=5
  { id: 4,  BM:128, BN:128, BK:64, S:2, WM:2, WN:4, MB:2 }, // BK64 S2 (K_TILES=4)
  { id: 5,  BM:128, BN:128, BK:16, S:4, WM:2, WN:4, MB:2 }, // BK16 S4
  { id: 6,  BM:128, BN:128, BK:16, S:6, WM:2, WN:4, MB:2 }, // BK16 deep
  { id: 7,  BM:256, BN:128, BK:32, S:3, WM:4, WN:4, MB:1 }, // big block, 16 warps, warp 64x32
  { id: 8,  BM:256, BN:128, BK:32, S:4, WM:4, WN:4, MB:1 }, // big block S4
  { id: 9,  BM:128, BN:256, BK:32, S:3, WM:2, WN:8, MB:1 }, // big block other orient
  { id: 10, BM:256, BN:128, BK:16, S:4, WM:4, WN:4, MB:1 }, // big block BK16
  { id: 11, BM:128, BN:128, BK:32, S:6, WM:2, WN:4, MB:1 }, // S6 single block
  { id: 12, BM:256, BN:128, BK:64, S:2, WM:4, WN:4, MB:1 }, // big BK64
]

const REPO = '/home/daixu/code/github_code/VibeKernel/worktrees/dynamic_workflow'

function buildCmd(c) {
  const dir = `${REPO}/build_t7/cfg${c.id}`
  const defs = `-DPG_BM=${c.BM} -DPG_BN=${c.BN} -DPG_BK=${c.BK} -DPG_STAGES=${c.S} -DPG_WARP_M=${c.WM} -DPG_WARP_N=${c.WN} -DPG_MIN_BLOCKS=${c.MB}`
  const log = `${REPO}/build_t7/cfg${c.id}.log`
  return `mkdir -p ${REPO}/build_t7 && cd ${REPO} && cmake -S ./task-1 -B ${dir} -G Ninja -DCMAKE_TOOLCHAIN_FILE="$VCPKG_HOME/scripts/buildsystems/vcpkg.cmake" -DMATMUL_VERSION=7 -DTEST_DATA_TYPE=float16 -DSTDOUT_IS_TERMINAL=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=20 -DCMAKE_CUDA_STANDARD=20 -DCMAKE_CUDA_FLAGS="${defs}" > ${log} 2>&1 && cmake --build ${dir} --parallel 3 >> ${log} 2>&1`
}

const BUILD_SCHEMA = {
  type: 'object',
  properties: {
    id: { type: 'integer' },
    built: { type: 'boolean' },
    note: { type: 'string', description: 'one-line compile error if failed, else empty' },
  },
  required: ['id', 'built', 'note'],
}

phase('Build')
const built = await parallel(CONFIGS.map((c) => () =>
  agent(
    `Run ONE deterministic CUDA build for a config tournament. Do NOT run GPU code.\n` +
    `Run exactly:\n\n${buildCmd(c)}\n\n` +
    `Then: ls -la ${REPO}/build_t7/cfg${c.id}/src/task1_float16_v7\n` +
    `If the binary exists set built=true. Else built=false and put the first compile error (grep -m1 -iE "error" ${REPO}/build_t7/cfg${c.id}.log) in note (<200 chars). id=${c.id}.`,
    { label: `build:cfg${c.id}`, phase: 'Build', schema: BUILD_SCHEMA }
  )
))

const okIds = built.filter(Boolean).filter((b) => b.built).map((b) => b.id)
const failed = built.filter(Boolean).filter((b) => !b.built)
log(`Build: ${okIds.length}/${CONFIGS.length} ok. Failed: ${failed.map((f) => `cfg${f.id}(${f.note})`).join('; ') || 'none'}`)

const cfgById = Object.fromEntries(CONFIGS.map((c) => [c.id, c]))
const benchList = okIds.map((id) => `cfg${id}: ${REPO}/build_t7/cfg${id}/src/task1_float16_v7`).join('\n')

const BENCH_SCHEMA = {
  type: 'object',
  properties: {
    results: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          id: { type: 'integer' },
          tflops: { type: 'number' },
          error: { type: 'number' },
          ok: { type: 'boolean' },
        },
        required: ['id', 'tflops', 'error', 'ok'],
      },
    },
  },
  required: ['results'],
}

phase('Benchmark')
const benched = await agent(
  `Benchmark pre-built CUDA GEMM binaries on a SINGLE shared GPU. Run them STRICTLY ONE AT A TIME (never parallel).\n` +
  `For each, run: <binary> -m 4096 -n 4096 -k 4096 -t 100\n` +
  `Parse "TFLOPS: <X>; Average Error: <Y>". ok=true if it ran and error<0.05. If crash, ok=false tflops=0 error=999.\n\n` +
  `Binaries (one at a time, wait for each):\n${benchList}\n\nReturn results for all listed.`,
  { label: 'bench:serial', phase: 'Benchmark', schema: BENCH_SCHEMA }
)

const rows = (benched?.results || []).map((r) => ({ ...r, cfg: cfgById[r.id] }))
rows.sort((a, b) => (b.ok ? b.tflops : -1) - (a.ok ? a.tflops : -1))
log('v7 tournament results (sorted):')
for (const r of rows) {
  const c = r.cfg
  log(`cfg${r.id} [${c.BM}x${c.BN} BK${c.BK} S${c.S} W${c.WM}x${c.WN} MB${c.MB}]: ${r.ok ? r.tflops.toFixed(1) + ' TFLOPS, err ' + r.error.toExponential(2) : 'FAILED'}`)
}

return { sorted: rows.map((r) => ({ id: r.id, cfg: r.cfg, tflops: r.tflops, error: r.error, ok: r.ok })), failedBuilds: failed }
