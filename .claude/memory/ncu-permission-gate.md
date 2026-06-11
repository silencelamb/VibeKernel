---
name: ncu-permission-gate
description: ncu ERR_NVGPUCTRPERM 的两道闸门(CAP_SYS_ADMIN + 驱动 RmProfilingAdminOnly)、如何验证、容器内无法修复
metadata: 
  node_type: memory
  type: reference
  originSessionId: 7f4ad05a-b435-4e15-b2a5-ac7cab195e28
---

ncu(Nsight Compute,读硬件计数器)报 `ERR_NVGPUCTRPERM` 由两道闸门决定,开一道即可:
1. 进程有 `CAP_SYS_ADMIN`(容器靠 `docker run --cap-add=SYS_ADMIN` 或 `--privileged`)。
2. 驱动 `RmProfilingAdminOnly=0`(host 设 `/etc/modprobe.d` `options nvidia NVreg_RestrictProfilingToAdminUsers=0` 后重载模块/重启)。

**验证 CAP_SYS_ADMIN(容器内)**:`CE=$(grep CapEff /proc/self/status|awk '{print $2}'); (( 0x$CE & 0x200000 ))` → 真即有。或 `capsh --print | grep sys_admin`(带 `!` 前缀=没有)。驱动闸门:`grep -i RmProfilingAdminOnly /proc/driver/nvidia/params`。一键自检:`scripts/ncu-doctor.sh [二进制]`。

**关键**:容器内修不了(没 cap 且不在 bounding set;sysfs 参数不可运行时改;host 模块 reload 不到)——必须在 host/启动层改。`!<cmd>` 也没用(在容器内跑)。

**nsys / torch.profiler 不受此限**:它们走 CUPTI 时间线 tracing,不读硬件计数器,所以不需要这个 cap(这就是为啥之前 vLLM profiling 没报错——用的是时间线工具或另一个带 cap 的容器)。

VibeKernel 第一个跑 naive 的容器**没有** CAP_SYS_ADMIN(`a80425fb`=docker 默认)且 `RmProfilingAdminOnly=1`,故 ncu 全程被挡(那次叫 naive_no_ncu)。`CLAUDE_For_KernelAgent.md`(原根 CLAUDE.md)旧称"SYS_ADMIN 已具备"是错的,已改。换有 cap 的容器后先 ncu-doctor.sh 证实。相关 [[vibekernel-result-harness]]。
