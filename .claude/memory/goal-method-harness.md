---
name: goal-method-harness
description: "/goal 方法 harness 机制与坑:evaluator=haiku槽(ANTHROPIC_DEFAULT_HAIKU_MODEL可改opus,我们默认留Haiku);condition=与naive同一份seed_gemm.txt;--max-turns是错的界(用--max-budget-usd/自然终止);NEVER STOP下看门狗只在worker自愿停时触发"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 6a0f753c-03d7-4ba8-bbe3-2d966510d17f
---

**/goal = naive + 一层最薄的"不让它自愿停"看门狗。** `claude -p "/goal <condition>"`,/goal 是 session 级 **Stop hook**:worker 每次吐出"无下一步动作"的消息(想交还控制权)时,一个独立 evaluator 读整段 transcript 判 condition;未达成就强制再续一 turn。

关键机制/坑(2026-06-06 查实于 Claude Code 2.1.167 + 官方 docs):
- **condition 一物两用**:既是 worker 第一 turn 的指令,又是 evaluator 判停依据。evaluator **只读 transcript、不跑工具**。为公平,我们让 condition = **与 naive 完全相同的 seed**(共享 `scripts/seed_gemm.txt`,byte-identical;两个 launcher 都 `cat` 它)。
- **NEVER STOP ⇒ 看门狗很少触发**:worker 把"下一步"挂在同条消息里、不交还 → Stop hook 稀疏触发,只在它真想停那刻。所以 /goal 相对 naive 唯一增量 = "在 naive 会死/停的点把它顶回去续";自愿停点之前两者完全相同。看门狗挡不了崩溃(ECONNRESET/usage/context 爆)。
- **evaluator 模型 = Claude Code 的 "small fast model"/haiku 槽** → `ANTHROPIC_DEFAULT_HAIKU_MODEL=claude-opus-4-8` 可改成 opus(`ANTHROPIC_SMALL_FAST_MODEL` 已废弃)。**但**无独立 effort 旋钮(一次性 yes/no);opus 上每 turn 重读全 transcript = **O(N²)** 贵一个数量级。**我们默认留 Haiku**(launch_goal.sh 里那行注释掉),判停一样准,且 evaluator 模型不影响被测的 worker。
- **界**:**别用 `--max-turns`**(单位"agentic turn"≈每次推理,粒度≠优化轮,且达上限 error 退出污染 run.jsonl)。要界用 `--max-budget-usd <N>`(优雅停)或不设界跑到自然终止(同 naive)。无终止态 condition ⇒ evaluator 永判未达成 ⇒ 可一直跑。
- **前置**:/goal 是 Stop hook → 需 workspace trust + hooks 未禁(disableAllHooks/allowManagedHooksOnly 任一会让它不可用)。启动后看 run.jsonl 确认 evaluator 真介入,否则退化成一次性跑(≠/goal 方法)。
- **token 口径**:evaluator 也花 token,跑完核对它是否以独立 message.id 进 transcript(会被 parse_run 去重累加进曲线),据此报"方法总成本(含evaluator)"还是只 worker 份额,写进 result.md。
- **★如何数/定位 evaluator 接入(跑完审计关键,2026-06-07 查实)**:权威信号在 **transcript** 的 `type:"attachment"` 行嵌的 **`goal_status`**(run.jsonl 里**没有**)——字段 `met`(true=判达成/false=顶回去)、`sentinel`(true=**启动占位、不算接入**)、`condition`。**接入次数 = 非-sentinel 的 goal_status 数**;`met:true` 全程可能一次没有(NEVER STOP 下 evaluator 一直 met:false,run 最后不是被判达成而停的)。jq:`jq -c 'select((.|tostring)|test("goal_status"))|{met:((.|tostring)|test("met.:true")),sentinel:((.|tostring)|test("sentinel.:true"))}' transcript.jsonl`。旁证(更糙):user 消息 `"Stop hook feedback:"`(每次顶回去,run.jsonl 也有)+ `"Stop hook is now active"`(启动);**优先 goal_status**(一个信号区分 启动/met/not-met)。`parse_run.sh` 已自动读它、在曲线画**红竖虚线**(met:false 红/met:true 绿)+ 控制台打接入时累计 token。**必查头一次接入时已到多少 TFLOPS**(见 [[goal-cycle1-result]]:worker 自己到 205.5 才头次接入,看门狗净加仅 +1.3——别把更高峰值想当然归 harness)。

详见 `runbooks/goal.md` + `scripts/launch_goal.sh`。相关:[[playground-base-clean-fork-topology]] [[naive-cycle1-econnreset-and-gotchas]] [[orchestration-overview]]
