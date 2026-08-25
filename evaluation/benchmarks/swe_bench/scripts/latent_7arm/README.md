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

## 超窗必须停下, 而不是压缩后重试

**设置点在 `run_infer.py` 的 `get_config`, 不在配置文件。** 该函数用
`AgentConfig(...)` + `set_agent_config` 整体覆盖 `config.toml` 的 `[agent]` 段,
所以配置文件里写 `enable_history_truncation = false` 到不了智能体, 用的是字段默认值
`True`。默认值走的是「抛 ContextWindowExceededError -> 发 CondensationRequestAction」
这条路, 而本项目 condenser 是 `noop`, 压缩器什么都不做、上下文仍然超窗, 于是一步一个
`condensation_request` 空转到步数耗尽。实测 25 道题这样烧掉预算且全部未解决,
首次触发中位在第 90 步, 最多吃掉 24% 的预算。

因此 `AgentConfig(...)` 里必须显式传 `enable_history_truncation=False`。传了之后改走
`raise LLMContextWindowExceedError`: 智能体立刻停止, 且该异常不在
`evaluation/utils/shared.py` 的 `FATAL_EXCEPTIONS` 里, 不会触发 run_infer 重跑,
run_infer 会照常抽 git 补丁并把 error 写进产物 —— 停下, 且留下结果。

**闸门要查生效值, 不要 grep 配置文件。** 早先 driver 的闸门 grep `config.toml` 里那一行,
文本在、闸门放行, 但值被 `set_agent_config` 覆盖掉了, 于是静默失效了整整一轮。
现在 `get_config` 会打一行 `[harness-effective] enable_history_truncation=... `,
闸门应当查这一行。

对应地, `one_job` 的超窗守卫只记录不立刻杀: 发现关键字后记一次账, 给最多 `OVF_GRACE`
秒(缺省 420)让 run_infer 完成抽补丁与写产物, 仍不退出才强杀兜底。原版一发现关键字
就 SIGTERM 再 SIGKILL, 会把收尾打断, 产物丢失、判分无结论, 该题反而进重试计数。

## 单题超时必须由进程内的闹钟先响

驱动脚本给每题套了外部 `timeout -s TERM 7200`。如果进程内的单题超时比它更晚(或者
根本没装上), 外部信号先到、进程被直接杀掉, 这道题**一行产物都不留** —— 判分拿不到
结论, 该题进重试计数, 最终分母被削。实测本轮 500 题里 50 个 `output.jsonl` 为空、
71 题进过 `timeout.txt`。

这里原本有两层问题, 都已修:

1. `evaluation/utils/shared.py` 的 `run_evaluation`: 多 worker 分支把 `timeout_seconds`
   传给了 `_process_instance_wrapper`, **单 worker 分支漏传**, 于是取缺省 `None`,
   `with timeout(...)` 整段被跳过。评测用 `--eval-num-workers 1`, 正好落在这条路上,
   所以那个闹钟从来没装上过, `EvalTimeoutException` 那段一直是死代码。
2. `run_infer.py` 把 `timeout_seconds` 写死成 8 小时, 比外部的 2 小时还长。现改为读
   环境变量 `EVAL_INSTANCE_TIMEOUT`, 缺省仍是 8 小时。

**两处必须配套**: 只补第 1 处, 超时仍是 8 小时、外部还是先到; 只改第 2 处, 参数传不下去。

用法: 驱动脚本设 `EVAL_INSTANCE_TIMEOUT` 为略小于外部超时的值(例如外部 7200 时设
6600), 让内部闹钟先响。

**已知局限**: 走内部超时返回的 `EvalOutput` 里 `test_result={}`、`history` 为空,
拿到的是「一条确定的未解决结果」, 不是抢救出补丁。它解决的是分母诚实与停止重试,
不是提分。要连补丁一起保住, 需要在超时路径里补一次 `complete_runtime`, 那是更大的改动。

验证: `EVAL_INSTANCE_TIMEOUT=150`、外部 900 的真实单题冒烟, 150 秒时抛出
`EvalTimeoutException: Function timed out after 150 seconds`, 写出 1 行产物
(`error="Timeout after 150 seconds"`), 容器正常回收, 进程干净退出。

## 系统提示与 sed:回到训练分布

`sed -i` 禁令与配套的系统提示改写已全部撤回。依据是三组同题对照:A(完全未改的
harness)与 C(全套改动)在 262 题交集上 78 vs 80、差 2 题;B 与 C 在 156 题交集上
50 vs 47、McNemar p = 0.701;而 C 相对 B 多出的 path/view_range 提示本轮触发了 995 次、
涉及 239 题 —— 触发量很大却测不出任何分数差。翻转噪声底是 17%,配对差值标准差约
3.4 个百分点,这个量级的改动本来也测不出来。

既然拿不到收益,就不该为它付口径的代价:系统提示进每一次请求,偏离训练分布的成本
是确定的,而 SWE-Master-4B-RL 正是在那份提示下做的 RL。

因此:
- `openhands/llm/swemaster_format.py` 的 `SYSTEM_PROMPT` 逐字节恢复成训练时的样子
  (长度 15496,md5 前 12 位 701e9bd962e8)。驱动脚本的闸门断言这两个值,
  查生效值而不是 grep 文件文本。
- `cli_runtime.py` 里的 `sed -i` 拦截整段移除。自检改为反向断言:用
  `inspect.getsource(CLIRuntime.run)` 确认运行时里不再残留 sed 相关逻辑。
- 保留的是模糊层与三个无训练闸门 —— 它们只在模型已经犯错的那一步出现,不进入
  正常步骤看到的输入。

## 不要复活按时长杀进程的看门狗

`mem_reaper2.sh` / `mem_reaper3.sh` 按 6000 秒杀掉匹配到的进程,比驱动自己给每题的
`timeout -s TERM 7200` 还早。它们会把跑长的题杀成无产出 —— 实测被驱动的闸门段复活后,
19 点杀了 36 个 rollout、21 点又杀了 8 个,直接制造「撞上限却一行产物都没有」。

时长上限已经由外部 `timeout` 与 `EVAL_INSTANCE_TIMEOUT` 两层负责,不要再叠一层。
驱动现在只拉起 `mem_reaper.sh`(只看内存,阈值 60G,排除 vLLM 与训练进程)——
那个是有用的:实测抓到过三个各占 164 GB、已跑 2.7 天的孤儿测试进程。
