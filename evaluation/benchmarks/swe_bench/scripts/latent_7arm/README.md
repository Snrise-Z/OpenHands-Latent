# SWE-Bench Verified 七臂评测驱动

`eval7.sh` 在一台 8 卡机上跑 500 题 x 7 个观察策略 = 3,500 组 rollout + 判分。
分两阶段:阶段 1 用裸 vLLM 跑明文对照臂(allhard)500 题;阶段 2 起 shim 跑 6 个
latent 臂共 3,000 组。每题 rollout 结束立即判分,不等同批其他题。

## 运行

```bash
source evaluation/benchmarks/swe_bench/scripts/latent_7arm/eval_lib.sh
bash  evaluation/benchmarks/swe_bench/scripts/latent_7arm/eval7.sh
```

`eval_lib.sh` 必须先 source:它设置 VLLM_CACHE_ROOT / TRITON_CACHE_DIR / TMPDIR /
LD_LIBRARY_PATH。AutoDL 系统盘只剩几 GB,不改这些变量时 vLLM 会在权重加载完之后
才因 `os.makedirs` 抛 FileNotFoundError 崩掉,现象是八张卡功率停在 85W 而无报错。

脚本可重复执行:已有 done 标记的组合自动跳过,所以中断后重跑即续跑。
`$ROOT/priority.txt` 里列出的 `<arm> <instance_id>` 会被排到队首。

## 两条正确性约束

**done 标记必须等判分产出结论才写。** rollout 完成但判分被中断时若已写 done,
重启后该题被永久跳过,产生静默缺题。这里只在判分返回 True/False 时写标记;
无结论则累加 `attempts/<tag>`,连续 3 次仍无结论才封盘并记入 `stuck.txt`,
避免坏题无限重试。

**阶段之间要核对完成度,不能只看退出码。** `xargs` 退出不等于该阶段跑完
(被 kill、被信号中断都会正常退出)。阶段 1 结束后逐题核对 done 标记,缺一道就
拒绝进入阶段 2 并保留 vLLM 存活,提示排查后重跑续跑。同类错误:接管脚本
`takeover.sh` 判断阶段切换时必须记录启动时的日志行数,只检查其后新增的行,
否则会匹配到历史日志而立即误触发。

## 口径

模型渲染 `OH_FNCALL_STYLE=swemaster`(与训练渲染逐字节一致),100 步上限,
131,072 上下文窗口,`enable_history_truncation=false`(超窗即停而非截断重试),
温度 0.7 / top_p 0.8 / top_k 20。(臂, 题) 绑定 vLLM 副本以吃满前缀缓存,
实测命中 92-98%。

启动时把 harness 的三个指纹写进日志作为闸门,换 harness 后指纹变化即可发现:
编辑器模糊层、运行时、系统提示各一个 md5。

## 超窗必须停下,而不是压缩后重试

`config.lclm-eval.toml` 的 `[agent]` 段必须显式设 `enable_history_truncation = false`。
它的默认值是 `true`,走的是「抛 ContextWindowExceededError 后发一个
CondensationRequestAction」这条路;而本项目 `[condenser]` 是 `noop`,压缩器什么都不做,
上下文仍然超窗,于是一步一个 condensation_request 空转到步数耗尽。实测 21 道题这样
烧掉步数预算且全部未解决,首次触发中位在第 90 步,最多吃掉 24% 的预算。

设成 `false` 后改走 `raise LLMContextWindowExceedError`:智能体立刻停止,而该异常不在
`evaluation/utils/shared.py` 的 `FATAL_EXCEPTIONS` 里,所以不会触发 run_infer 的重跑,
run_infer 会照常抽取 git 补丁并把 error 写进产物 —— 停下,且留下结果。

对应地,`one_job` 的超窗守卫只记录不立刻杀:发现关键字后记一次账,给最多 `OVF_GRACE`
秒(缺省 420)让 run_infer 完成抽补丁与写产物,仍不退出才强杀兜底。早先的版本一发现
关键字就 SIGTERM 再 SIGKILL,会把这段收尾打断,产物丢失、判分拿不到结论,该题反而进
重试计数 —— 与设 false 的目的正好相反。
