export const meta = {
  name: 'gemm-config-tournament',
  description: 'Build ~14 macro configs of matmul_f16 v5 in parallel, benchmark serially on the single GPU, return sorted TFLOPS',
  phases: [
    { title: 'Build', detail: 'parallel cmake+ninja per config into isolated build dirs' },
    { title: 'Benchmark', detail: 'one agent runs all built binaries serially (100-round task1 scoring)' },
  ],
}

// (BM,BN,BK,STAGES,WARP_M,WARP_N,MIN_BLOCKS)
const CONFIGS = [
  { id: 1,  BM:128, BN:128, BK:32, S:3, WM:2, WN:4, MB:2 }, // baseline v4
  { id: 2,  BM:128, BN:128, BK:32, S:4, WM:2, WN:4, MB:2 }, // STAGES=4
  { id: 3,  BM:128, BN:128, BK:32, S:5, WM:2, WN:4, MB:1 }, // STAGES=5
  { id: 4,  BM:128, BN:128, BK:32, S:4, WM:4, WN:2, MB:2 }, // warp 32x64
  { id: 5,  BM:128, BN:128, BK:32, S:4, WM:2, WN:8, MB:2 }, // 16 warps
  { id: 6,  BM:128, BN:128, BK:32, S:3, WM:2, WN:8, MB:2 }, // 16 warps S3
  { id: 7,  BM:128, BN:128, BK:64, S:2, WM:2, WN:4, MB:2 }, // BK64 S2
  { id: 8,  BM:128, BN:128, BK:64, S:3, WM:2, WN:4, MB:1 }, // BK64 S3
  { id: 9,  BM:256, BN:128, BK:32, S:3, WM:4, WN:2, MB:1 }, // big tile warp64x64
  { id: 10, BM:256, BN:128, BK:32, S:4, WM:4, WN:2, MB:1 }, // big tile S4
  { id: 11, BM:128, BN:256, BK:32, S:3, WM:2, WN:4, MB:1 }, // warp64x64 other
  { id: 12, BM:256, BN:128, BK:64, S:2, WM:4, WN:2, MB:1 }, // big+BK64
  { id: 13, BM:256, BN:256, BK:32, S:3, WM:4, WN:4, MB:1 }, // 256x256
  { id: 14, BM:64,  BN:128, BK:32, S:4, WM:1, WN:4, MB:3 }, // small tile hi-occ
]

const REPO = '/home/daixu/code/github_code/VibeKernel/worktrees/dynamic_workflow'

function buildCmd(c) {
  const dir = `${REPO}/build_tourney/cfg${c.id}`
  const defs = `-DPG_BM=${c.BM} -DPG_BN=${c.BN} -DPG_BK=${c.BK} -DPG_STAGES=${c.S} -DPG_WARP_M=${c.WM} -DPG_WARP_N=${c.WN} -DPG_MIN_BLOCKS=${c.MB}`
  const log = `${REPO}/build_tourney/cfg${c.id}.log`
  return `mkdir -p ${REPO}/build_tourney && cd ${REPO} && cmake -S ./task-1 -B ${dir} -G Ninja -DCMAKE_TOOLCHAIN_FILE="$VCPKG_HOME/scripts/buildsystems/vcpkg.cmake" -DMATMUL_VERSION=5 -DTEST_DATA_TYPE=float16 -DSTDOUT_IS_TERMINAL=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=20 -DCMAKE_CUDA_STANDARD=20 -DCMAKE_CUDA_FLAGS="${defs}" > ${log} 2>&1 && cmake --build ${dir} --parallel 3 >> ${log} 2>&1`
}

const BUILD_SCHEMA = {
  type: 'object',
  properties: {
    id: { type: 'integer' },
    built: { type: 'boolean', description: 'true if the binary task1_float16_v5 was produced' },
    note: { type: 'string', description: 'short note: compile error summary if failed, else empty' },
  },
  required: ['id', 'built', 'note'],
}

phase('Build')
const built = await parallel(CONFIGS.map((c) => () =>
  agent(
    `You are running ONE deterministic build for a CUDA GEMM config tournament. Do NOT run any GPU code or benchmark.\n` +
    `Run exactly this shell command (it configures + compiles config #${c.id} into an isolated build dir):\n\n${buildCmd(c)}\n\n` +
    `After it finishes, check whether the binary exists:\n  ls -la ${REPO}/build_tourney/cfg${c.id}/src/task1_float16_v5\n` +
    `If the binary exists, set built=true. If the build failed, set built=false and put a ONE-LINE summary of the first compile error (grep -m1 -iE "error" ${REPO}/build_tourney/cfg${c.id}.log) into note. Keep note under 200 chars. id=${c.id}.`,
    { label: `build:cfg${c.id}`, phase: 'Build', schema: BUILD_SCHEMA }
  )
))

const okIds = built.filter(Boolean).filter((b) => b.built).map((b) => b.id)
const failed = built.filter(Boolean).filter((b) => !b.built)
log(`Build phase done: ${okIds.length}/${CONFIGS.length} built OK. Failed: ${failed.map((f) => `cfg${f.id}(${f.note})`).join('; ') || 'none'}`)

const cfgById = Object.fromEntries(CONFIGS.map((c) => [c.id, c]))
const benchList = okIds.map((id) => {
  const c = cfgById[id]
  return `cfg${id}: BM=${c.BM} BN=${c.BN} BK=${c.BK} STAGES=${c.S} WM=${c.WM} WN=${c.WN} MB=${c.MB} -> ${REPO}/build_tourney/cfg${id}/src/task1_float16_v5`
}).join('\n')

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
          ok: { type: 'boolean', description: 'true if it ran and error < 0.05 (correct)' },
        },
        required: ['id', 'tflops', 'error', 'ok'],
      },
    },
  },
  required: ['results'],
}

phase('Benchmark')
const benched = await agent(
  `You are benchmarking pre-built CUDA GEMM binaries on a SINGLE shared GPU. You MUST run them strictly ONE AT A TIME (serially) — never in parallel — so timings are clean.\n\n` +
  `For EACH binary below, run exactly:\n  <binary_path> -m 4096 -n 4096 -k 4096 -t 100\n` +
  `It prints a line like: "[Playground] Result >>> TFLOPS: <X>; Average Error: <Y>". Parse X (tflops) and Y (error).\n` +
  `Run them one after another (wait for each to finish before starting the next). Set ok=true if it ran and error < 0.05.\n\n` +
  `Binaries:\n${benchList}\n\n` +
  `Return the results array for ALL listed configs (if one crashes, set ok=false, tflops=0, error=999).`,
  { label: 'bench:serial', phase: 'Benchmark', schema: BENCH_SCHEMA }
)

const rows = (benched?.results || []).map((r) => ({ ...r, cfg: cfgById[r.id] }))
rows.sort((a, b) => (b.ok ? b.tflops : -1) - (a.ok ? a.tflops : -1))
log('Tournament results (sorted):')
for (const r of rows) {
  const c = r.cfg
  log(`cfg${r.id} [${c.BM}x${c.BN} BK${c.BK} S${c.S} W${c.WM}x${c.WN} MB${c.MB}]: ${r.ok ? r.tflops.toFixed(1) + ' TFLOPS, err ' + r.error.toExponential(2) : 'FAILED'}`)
}

return { sorted: rows.map((r) => ({ id: r.id, cfg: r.cfg, tflops: r.tflops, error: r.error, ok: r.ok })), failedBuilds: failed }
