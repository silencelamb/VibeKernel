---
name: playground-base-clean-fork-topology
description: playground-base=去cublas/cutlass干净基座(只当worktree源、永不被改);2026-06-07起【git worktree流】(每跑开worktrees/<run-name>→独立slug隔离transcript/memory→finish_run归档到results/<run-name>+删worktree;多cycle换run-name),弃单工作区原地reset与per-method-repo/submodule
metadata: 
  node_type: memory
  type: project
  originSessionId: bf1eac92-4b8c-4f65-be37-127078778259
---

**实验范式 2026-06-05 转向:去掉 cuBLAS 作为目标。** 不再"超越 cuBLAS"——`v0`=cBLAS(CPU)**仅作正确性 ground truth**(main.cpp `calculateAvgErr` 拿它当 golden,删 cublas 不影响校验),性能目标改为**逼近 A100 fp16 峰值 312**,全程无库基线对标。Agent 写的 kernel **从 v1 起**(原 v1=cublas 已删)。

**干净基座 = `silencelamb/playground-base`**(commit **0a8f197**,已 push;VibeKernel submodule 已 bump 到它;本地 `./playground-base/`)。从 PJLAB-CHIP/playground fork 而来,strip 掉 cublas/cutlass(删若干 .cu/.hpp、CMake/vcpkg 去 cublas+cutlass、task1.sh 默认 ver 0)。**task1.sh 在 base 根、kernel 在 `task-1/src/matmul_f16/`**。git 里给 task1.sh+scripts 打了 +x。已 smoke-build+跑通 v0(Error=0)。

**⭐ 2026-06-08 base build 优化(0a8f197,已测已 push):** ① **选择性编译**——`src/CMakeLists.txt` 不再 `file(GLOB *)` 全目录,只编 `main + v0(cBLAS 参照)+ 选中的 --ver N`(main.cpp 只 link `matmul<DataType,MATMUL_VERSION>`+v0)→ 建任一版本只编 2–3 个文件、**坏的别的版本不再挡构建**(旧 GLOB 全编=一个坏全挂)。② **快 ninja 通路**——`build-task1.sh` 当 build 目录已按同 version/dtype/buildtype 配过就跳过 cmake 重配置、直接 ninja(同版本改文件 0 重配置开销;换 --ver 才重配)。③ **修 /0**——`test_data.hpp calculateAvgErr` 旧式 `|GT-C|/|GT|` 在 GT 恰好=0(fp16 偶发,旧"~1/8 跑 inf 彩票")时 inf→PLAYGROUND_CHECK 中止整跑;改为**跳过 GT==0 项、除以有效计数**,C 出 inf/nan 仍照样判错(真坏 kernel 仍 fail);**无零参照时误差数值不变=对历史结果可比**。worker 手册 [[vibekernel-result-harness]] 对应段已更新(别的版本坏不挡你/同版本改文件直接 ninja)。**未来 worktree 从新 HEAD 开 → 自动吃到这三项。**

**⭐ 2026-06-07 工程转向:git worktree 流(取代 06-06 的单工作区原地 reset;更早是 per-method-repo+submodule)。** 每次跑 = `playground-base` 上一个独立 worktree `worktrees/<run-name>/`——独立文件夹名 → **独立 cwd-slug → transcript/worker-memory 隔离**(方法/cycle 间零串味);**基座 playground-base 永不被改**(只当源,共享 .git)——
- 跑:`scripts/launch_<方法>.sh [run-name]`(默认=方法名;多 cycle 传 naive_cycle2 等)。经 `scripts/_run_common.sh`:防覆盖闸(worktree 或 results/<run> 已存在则拒跑)→ `git worktree add --detach worktrees/<run> HEAD` → cp CLAUDE.md(写共享 `playground-base/.git/info/exclude` 免进 patch)→ 起 worker 落 results/<run>/run.jsonl。
- 收尾:`scripts/finish_run.sh <run-name>` → 归档 transcript(slug 由 worktree 路径推导、唯一,不用猜)+ check_handwritten + 快照 src/(include/ 仅 worker 动过 dispatcher 头时)/logs/worker.patch → parse_run → **`git worktree remove --force` 删 worktree**(用完即焚)。
- 复现 = base remote + `results/<run>/worker.patch`(git apply);`src/` 易读快照。
- **为啥比单工作区好**:每跑独立 slug → transcript 不再共用 playground-base slug 混在一个目录(归档不用猜哪个 session);worker 若自己写 memory 也各自一份、不跨方法串。
- ⚠️ `git -C base worktree add <path>` 的 `<path>` 必须**绝对路径**(相对路径会落到 base 内部)。脚本里 WT=$ROOT/worktrees/$RUN 是绝对的。
- ⚠️ **worktree 只拿基座 HEAD(已提交内容)→ 基座必须保持全提交**:若在 playground-base 留未提交改动(改框架头 / 加文件没 commit),新 worktree 不会有 → "上个 run 能编、这个缺文件编不过"假差异。`_run_common.sh` 已加**软警告**(基座 dirty 就提示)。实测我们的 task **自洽、此坑当前不咬**(基座 status=0、.gitignore 只忽略派生物 *.o/build/logs/.cache、无外部数据集/软链;naive_cycle2 纯 worktree 已 build+计分),但**改基座框架记得 commit**。每个 worktree 自带 build/(各自重装 vcpkg,~65M)= 唯一成本,非正确性问题。
- 🚨🚨 **大坑(2026-06-07 踩了):worker auto-memory 按【git 仓库】键(git-common-dir),不按 cwd → 所有 worktree 与 base 共享同一份 worker-memory**(落 base 的 cwd-slug `~/.claude/projects/-…-playground-base/memory`)。transcript 按 cwd 隔离(各自新 slug 文件夹)**但 memory 不是**。后果:上个 worker 把 kernel 设计写进 `fp16-gemm-best-kernel.md`,下个 worker 读到→直接重建最优 kernel = **抄答案污染**(naive_cycle2 因此作废,见 [[goal-cycle1-result]])。**编排 memory(VibeKernel slug,= .claude/memory)没漏**(worker transcript 里"看门狗/对比/reward-hack"0 次)——漏的是 worker 之间的设计知识。**修复(主):`_run_common.sh` 每 launch 前 `rm -rf <base-slug>/memory`(清 base-slug worker auto-memory → 每 run 白板;base-slug≠VibeKernel-slug,不碰编排 memory)。⭐ auto-memory【保持开启】——这样 worker 在【本 run 内】跨 context 压缩还能记/取关键发现(长跑需要),只是跨-run 被清掉、不串。两头都要(2026-06-07 用户提醒:别为防跨-run 把 run 内持久化也关了)。✅ 2026-06-07 已经 naive_cycle3 实测验证:泄漏标记(fp16-gemm-best-kernel/206.7/206.8/L2-persist/PAD=8)全 0、worker 自写自己的 `gemm-f16-best-config.md`(文件名都不同)=白板坐实;跑完它又写回 base-slug,下次 launch 再清,设计自洽。详见 [[goal-cycle1-result]] 的干净对照段。** 另:env `CLAUDE_CODE_DISABLE_AUTO_MEMORY=1` 能彻底关 auto-memory(读+写;**只关 memory 不关 CLAUDE.md**,比 --bare 干净,验于 bundle 2.1.167 m1() env-gate),但会连 run 内持久化一起丢 → 默认不用,`_run_common` 里留作注释 opt-out。 教训:**别假设"不同文件夹=隔离 memory";独立 repo 才天然隔离 memory,worktree/单工作区都共享 base 的 worker-memory。**
- META §3/§5/§6、`.gitignore`、`runbooks/{naive,goal}.md` 已同步此流。

**历史(为何当初想 per-method-repo,现已弃):** GitHub 一账号对同一上游只能一个 fork,silencelamb 对 PJLAB/playground 那个**已用掉=playground-base**(早先 playground-naive 改名来),故第二方法没法再 fork、fork 也不能转 template → 当时结论"方法仓必须独立 repo"。**已被单工作区流取代,不再建新方法 repo。** 连 `playground-naive-clean` submodule 也已退役(naive cycle2 的 kernel 抽进 `results/naive/{src,worker.patch}`);**per-method submodule 全无**。⭐ **2026-06-07(转 public 前)`playground-base` 本身重新以 git submodule 纳入 VibeKernel**(`.gitmodules` 指向 `silencelamb/playground-base`,gitlink 现 @0a8f197,公开后 `--recurse-submodules` clone 即得基座;之前是 gitignored 本地克隆)——基座内容仍永不被改、worktree 流不变。即:**唯一 submodule = 基座 playground-base,per-method kernel 仍只存 `results/` 快照**。

**归档**:含 cublas 那次 naive 跑已归档到 `results/naive_ncu_cublas/`;更早根部 `naive_no_ncu/`(无 ncu+曾偷 cutlass)。

相关:[[vibekernel-result-harness]] [[goal-method-harness]] [[orchestration-overview]]
