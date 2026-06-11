export const meta = {
  name: 'gemm-occ16-tournament',
  description: 'Post-fix: does 16-warp (v6 single-buffer, 32x64/64x32 tiles) beat 8-warp (v7 64x64)? Build parallel, bench serial.',
  phases: [{ title: 'Build' }, { title: 'Benchmark' }],
}

// V=6 single-buffer (low reg, high occ) / 7 double-buffer (high ILP). tile=(BM/WM)x(BN/WN)
const CONFIGS = [
  { id: 1,  V:7, BM:256, BN:128, BK:64, S:2, WM:4, WN:2, MB:1 }, // current best 192 (64x64, 8 warps)
  { id: 2,  V:6, BM:256, BN:128, BK:32, S:3, WM:8, WN:2, MB:1 }, // 32x64, 16 warps, single
  { id: 3,  V:6, BM:256, BN:128, BK:32, S:4, WM:8, WN:2, MB:1 }, // 32x64, 16 warps, S4
  { id: 4,  V:6, BM:256, BN:128, BK:64, S:2, WM:8, WN:2, MB:1 }, // 32x64, 16 warps, BK64
  { id: 5,  V:6, BM:256, BN:128, BK:32, S:3, WM:4, WN:4, MB:1 }, // 64x32, 16 warps
  { id: 6,  V:6, BM:256, BN:128, BK:64, S:2, WM:4, WN:4, MB:1 }, // 64x32, 16 warps, BK64
  { id: 7,  V:6, BM:128, BN:256, BK:32, S:3, WM:4, WN:4, MB:1 }, // 32x64, 16 warps
  { id: 8,  V:6, BM:256, BN:128, BK:64, S:3, WM:8, WN:2, MB:1 }, // 32x64, 16 warps, BK64 S3
  { id: 9,  V:6, BM:256, BN:128, BK:64, S:2, WM:4, WN:2, MB:1 }, // 64x64 single-buffer, 8 warps (vs #1)
  { id: 10, V:6, BM:256, BN:256, BK:32, S:2, WM:8, WN:4, MB:1 }, // 32x64, 32 warps(1024 thr) single
  { id: 11, V:6, BM:256, BN:128, BK:32, S:5, WM:8, WN:2, MB:1 }, // 32x64, 16 warps, deep S5
  { id: 12, V:7, BM:256, BN:128, BK:64, S:2, WM:8, WN:2, MB:1 }, // 32x64 DOUBLE buffer 16 warps (may spill)
]

const REPO = '/home/daixu/code/github_code/VibeKernel/worktrees/dynamic_workflow'
function buildCmd(c) {
  const dir = `${REPO}/build_t12/cfg${c.id}`
  const defs = `-DPG_BM=${c.BM} -DPG_BN=${c.BN} -DPG_BK=${c.BK} -DPG_STAGES=${c.S} -DPG_WARP_M=${c.WM} -DPG_WARP_N=${c.WN} -DPG_MIN_BLOCKS=${c.MB}`
  const log = `${REPO}/build_t12/cfg${c.id}.log`
  return `mkdir -p ${REPO}/build_t12 && cd ${REPO} && cmake -S ./task-1 -B ${dir} -G Ninja -DCMAKE_TOOLCHAIN_FILE="$VCPKG_HOME/scripts/buildsystems/vcpkg.cmake" -DMATMUL_VERSION=${c.V} -DTEST_DATA_TYPE=float16 -DSTDOUT_IS_TERMINAL=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=20 -DCMAKE_CUDA_STANDARD=20 -DCMAKE_CUDA_FLAGS="${defs}" > ${log} 2>&1 && cmake --build ${dir} --parallel 3 >> ${log} 2>&1`
}
const BUILD_SCHEMA = { type: 'object', properties: { id: { type: 'integer' }, built: { type: 'boolean' }, note: { type: 'string' } }, required: ['id', 'built', 'note'] }

phase('Build')
const built = await parallel(CONFIGS.map((c) => () =>
  agent(`Run ONE deterministic CUDA build. No GPU. Run exactly:\n\n${buildCmd(c)}\n\nThen: ls ${REPO}/build_t12/cfg${c.id}/src/task1_float16_v${c.V}\nbuilt=true if binary exists else false with first error (grep -m1 -iE error ${REPO}/build_t12/cfg${c.id}.log) in note(<200). id=${c.id}.`,
    { label: `build:cfg${c.id}`, phase: 'Build', schema: BUILD_SCHEMA })))

const okIds = built.filter(Boolean).filter((b) => b.built).map((b) => b.id)
const failed = built.filter(Boolean).filter((b) => !b.built)
log(`Build: ${okIds.length}/${CONFIGS.length} ok. Failed: ${failed.map((f) => `cfg${f.id}(${f.note})`).join('; ') || 'none'}`)
const cfgById = Object.fromEntries(CONFIGS.map((c) => [c.id, c]))
const benchList = okIds.map((id) => `cfg${id}: ${REPO}/build_t12/cfg${id}/src/task1_float16_v${cfgById[id].V}`).join('\n')
const BENCH_SCHEMA = { type: 'object', properties: { results: { type: 'array', items: { type: 'object', properties: { id: { type: 'integer' }, tflops: { type: 'number' }, error: { type: 'number' }, ok: { type: 'boolean' } }, required: ['id', 'tflops', 'error', 'ok'] } } }, required: ['results'] }

phase('Benchmark')
const benched = await agent(`Benchmark pre-built binaries on a SINGLE GPU, STRICTLY ONE AT A TIME.\nFor each: <binary> -m 4096 -n 4096 -k 4096 -t 100 ; parse "TFLOPS: X; Average Error: Y". ok=true if ran and error<0.05; crash->ok=false tflops=0 error=999.\n\nSerially:\n${benchList}\n\nReturn all.`,
  { label: 'bench:serial', phase: 'Benchmark', schema: BENCH_SCHEMA })

const rows = (benched?.results || []).map((r) => ({ ...r, cfg: cfgById[r.id] }))
rows.sort((a, b) => (b.ok ? b.tflops : -1) - (a.ok ? a.tflops : -1))
log('occ16 tournament results (sorted):')
for (const r of rows) { const c = r.cfg; log(`cfg${r.id} v${c.V} [${c.BM}x${c.BN} BK${c.BK} S${c.S} W${c.WM}x${c.WN}(tile ${c.BM/c.WM}x${c.BN/c.WN}) MB${c.MB}]: ${r.ok ? r.tflops.toFixed(1) + ' TF, err ' + r.error.toExponential(2) : 'FAILED'}`) }
return { sorted: rows.map((r) => ({ id: r.id, cfg: r.cfg, tflops: r.tflops, error: r.error, ok: r.ok })), failedBuilds: failed }
