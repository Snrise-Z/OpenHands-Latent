#!/bin/bash
# 3500 组(7 臂 x 500 题)全量评测, 新 harness(提交 5852af5)。
# 七臂 x 500 题 = 3500 组, 单一全局队列, 七臂交错推进。
# 全部跑在同一套 serve_shim 上: 裸 vLLM 与 shim 对同样的贪心请求会给出不同输出
# (冷热缓存下分歧点相同, 各自内部确定), 根因是分块预填充粒度 8192 与 2048 之别,
# 而 serve_shim 不暴露该参数。同栈才能让臂间对比不混入服务栈的影响。
# 每题一条流水线: rollout 完立即判分, 不等其他题。
# 每题一条流水线: rollout 完立即判分, 不等其他题。
set -u
T=/root/autodl-tmp
OH=$T/OpenHands-Latent
VENV=$T/envs/openhands-0620
VLLM_ENV=$T/vllm-env
LCLM=$T/LCLM
BAKED=$T/swemd_final_eval
ROOT=/root/autodl-fs/eval_swemd_7arm_v2
OUTB=$OH/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent
L=$ROOT/driver.log
IDS=/root/autodl-fs/eval_sets/verified_all500_ids.txt
mkdir -p $ROOT/{logs,done,fuzzy_logs,shards,attempts,logs/serve}
say() { echo "[$(date '+%F %T')] $*" | tee -a $L; }

# 必须先 source: 它把 vLLM/Triton 缓存、TMPDIR、LD_LIBRARY_PATH 挪出系统盘(仅 3.7G 空闲),
# 否则引擎在 os.makedirs 缓存目录时直接失败(实测: 权重加载成功后仍崩)。
source $T/eval_lib.sh 2>/dev/null || { echo "!! eval_lib.sh 缺失"; exit 1; }
export EVAL_ROOT=$ROOT LOGDIR=$ROOT/logs
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TMPDIR" 2>/dev/null

NG=$(nvidia-smi -L | wc -l)
CONC=${CONC:-64}   # 全局并发
say "===== 3500 组评测开始(单一全局队列, 七臂同栈) NG=$NG CONC=$CONC ====="
say "缓存 VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT TMPDIR=$TMPDIR 系统盘余 $(df -BG / | awk 'NR==2{print $4}')"

# ---------- 闸门: 三处改动必须在位 ----------
FZ=$OH/openhands/runtime/latent_fuzzy_editor.py
CR=$OH/openhands/runtime/impl/cli/cli_runtime.py
grep -q "forgot to put the modification into" $FZ || { say "!! identical 提示未更新"; exit 1; }
grep -q "_MISSING_HINTS" $FZ || { say "!! 参数缺失提示缺失"; exit 1; }
grep -q "enable_history_truncation = false" $OH/config.toml || { say "!! 未设超窗即停"; exit 1; }
$VENV/bin/python -c "
import sys; sys.path.insert(0,'$OH')
import inspect
from openhands.runtime.impl.cli.cli_runtime import CLIRuntime as _CR
from openhands.runtime.latent_fuzzy_editor import _IDENTICAL_HINT, _MISSING_HINTS
assert 'sed' not in inspect.getsource(_CR.run).lower(), 'sed 拦截未彻底移除'
assert 'forgot to put' in _IDENTICAL_HINT and len(_MISSING_HINTS) >= 4
print('harness 闸门 OK')
" 2>&1 | tee -a $L | grep -q OK || { say "!! harness 自检失败"; exit 1; }
SP_MD5=$($VENV/bin/python -c "
import sys, hashlib; sys.path.insert(0,'$OH')
from openhands.llm.swemaster_format import SYSTEM_PROMPT as S
print(hashlib.md5(S.encode()).hexdigest()[:12], len(S))
" 2>/dev/null)
say "闸门通过 editor_md5=$(md5sum $FZ | cut -c1-12) runtime_md5=$(md5sum $CR | cut -c1-12) sysprompt=$SP_MD5"
# 系统提示必须与模型训练时看到的逐字节一致 —— 它进每一次请求, 是「在训练分布下评测」
# 这一说法的依据。查生效值而不是 grep 文件: 长度与 md5 都要对上。
[ "$SP_MD5" = "701e9bd962e8 15496" ] || { say "!! 系统提示与训练不一致: 期望 [701e9bd962e8 15496], 实得 [$SP_MD5]"; exit 1; }
grep -q "_PATH_HINTS" $FZ || { say "!! path 提示补丁缺失"; exit 1; }

# 只拉起「只看内存」的看门狗。mem_reaper2/3 按 6000 秒杀进程, 比驱动自己的
# timeout 7200 还早, 会把跑长的题杀成无产出 —— 实测它被复活后在 19 点杀了 36 个、
# 21 点又杀了 8 个。时长上限已由 timeout 与 EVAL_INSTANCE_TIMEOUT 负责, 不要重复。
pgrep -f "mem_reaper\.s[h]" >/dev/null || { setsid nohup bash $T/mem_reaper.sh >/dev/null 2>&1 </dev/null & }
pgrep -f "reaper_relaxed\.s[h]" >/dev/null || { setsid nohup bash $T/reaper_relaxed.sh >/dev/null 2>&1 </dev/null & }

cd $OH
export PYTHONPATH=$OH RUNTIME=udocker_runtime.UDockerRuntime ITERATIVE_EVAL_MODE=false
export EVAL_DOCKER_IMAGE_PREFIX=docker.io/swebench
export OH_FNCALL_STYLE=swemaster OH_SWEMASTER_MAX_STEPS=100
export OH_FUZZY_STR_REPLACE=1 OH_FUZZY_SYNTAX_GUARD=1
export UDOCKER_EXE=/root/miniconda3/bin/udocker UDOCKER_DIR=$T/.udocker
export OH VENV OUTB T L ROOT NG VLLM_CACHE_ROOT TRITON_CACHE_DIR TMPDIR LD_LIBRARY_PATH

one_job() {
  arm=$1; iid=$2; rep=$3; cfg=$4
  tag="${arm}-${iid}"
  [ -f "$ROOT/done/$tag" ] && return 0
  case "$arm" in
    allhard)   export LATENT_OBS_POLICY=none; unset LATENT_HARD_WINDOW ;;
    alllatent) export LATENT_OBS_POLICY=all-latent; unset LATENT_HARD_WINDOW; export LATENT_MIN_OBS_CHARS=128 ;;
    hardlast*) export LATENT_OBS_POLICY=lastk-hard; export LATENT_HARD_WINDOW=${arm#hardlast}; export LATENT_MIN_OBS_CHARS=128 ;;
  esac
  for w in $(seq 1 60); do
    fg=$(df -BG /root/autodl-tmp | awk 'NR==2{gsub("G","",$4); print $4}')
    [ "$fg" -ge 250 ] && break
    sleep 60
  done
  SF=$ROOT/shards/$tag.txt; echo "$iid" > $SF
  RLOG=$ROOT/logs/rollout_$tag.log
  EVAL_IDS_FILE=$SF OH_FUZZY_LOG=$ROOT/fuzzy_logs/$tag.jsonl \
    timeout -s TERM 7200 $VENV/bin/python evaluation/benchmarks/swe_bench/run_infer.py \
    --agent-cls CodeActAgent --llm-config ${cfg}$rep --max-iterations 100 \
    --eval-num-workers 1 --eval-note "a7-$tag" \
    --dataset princeton-nlp/SWE-bench_Verified --split test > $RLOG 2>&1 &
  RPID=$!
  # 超窗守卫: 只记录并留出收尾时间, 不再一发现就杀。
  # 配置已设 enable_history_truncation=false, 超窗会抛 LLMContextWindowExceedError,
  # 智能体自行停止, run_infer 随后照常抽 git 补丁并把 error 写进产物。若在这段收尾
  # 期间把进程杀掉, 产物会丢失、判分拿不到结论、该题反而进重试计数。
  # 因此: 记一次账 -> 等它自己退出(最多 OVF_GRACE 秒)-> 仍不退才强杀兜底。
  OVF_GRACE=${OVF_GRACE:-420}
  ( while kill -0 $RPID 2>/dev/null; do
      grep -qiE "maximum context length|context_length_exceeded|ContextWindowExceed|LLMContextWindowExceed" $RLOG 2>/dev/null && {
        echo "$arm $iid" >> $ROOT/context_overflow.txt
        w=0
        while kill -0 $RPID 2>/dev/null && [ $w -lt $OVF_GRACE ]; do sleep 10; w=$((w+10)); done
        kill -0 $RPID 2>/dev/null && {
          echo "$arm $iid grace-expired" >> $ROOT/context_overflow.txt
          kill -TERM $RPID 2>/dev/null; sleep 10; kill -KILL $RPID 2>/dev/null; }
        break; }
      sleep 20
    done ) & WPID=$!
  wait $RPID 2>/dev/null; RC=$?
  kill $WPID 2>/dev/null
  [ $RC -eq 124 ] && echo "$arm $iid" >> $ROOT/timeout.txt
  D=$(ls -d $OUTB/*a7-$tag 2>/dev/null | head -1)
  N=$(wc -l < "$D/output.jsonl" 2>/dev/null || echo 0)
  R=NA
  if [ "$N" -ge 1 ]; then
    # 判分必须有时限。没有时限时实测会卡在加载完预测之后的一次 socket 读上永不返回
    # (最长 37 小时), 把 xargs 槽位永久占住, 整轮评测因此停滞。单题判分正常几分钟,
    # 最重的仓库也在十几分钟内, 一小时是很宽的上界。-k 60 保证 TERM 无效时补 KILL。
    timeout -s TERM -k 60 3600 $VENV/bin/python evaluation/benchmarks/swe_bench/eval_infer.py \
      --input-file "$D/output.jsonl" --dataset princeton-nlp/SWE-bench_Verified --split test \
      --eval-num-workers 1 > $ROOT/logs/grade_$tag.log 2>&1
    [ $? -eq 124 ] && echo "$arm $iid" >> $ROOT/grade_timeout.txt
    R=$(python3 -c "
import json
try:
    d=json.loads(open('$D/output.swebench_eval.jsonl').readline())
    v=(d.get('test_result',{}).get('report',{}) or {}).get('resolved')
    print(v if v in (True, False) else 'NA')
except Exception: print('NA')
" 2>/dev/null)
    echo "[$(date '+%F %T')]   [$arm] $iid resolved=$R" >> $L
  fi
  # done 只在判分产出有效结论后才写。被中途杀掉时不会留下假标记, 重启即可续跑。
  # 连续失败 3 次仍无结论则封盘并记入 stuck.txt, 防止无限重试。
  if [ "$R" = "True" ] || [ "$R" = "False" ]; then
    touch $ROOT/done/$tag
  else
    A=$ROOT/attempts/$tag
    mkdir -p $ROOT/attempts
    n=$(( $(cat $A 2>/dev/null || echo 0) + 1 ))
    echo $n > $A
    echo "[$(date '+%F %T')]   [$arm] $iid 无有效判分(第 $n 次), 不打 done" >> $L
    if [ "$n" -ge 3 ]; then
      echo "$arm $iid" >> $ROOT/stuck.txt
      touch $ROOT/done/$tag
      echo "[$(date '+%F %T')]   [$arm] $iid 连续 3 次无结论, 封盘记入 stuck.txt" >> $L
    fi
  fi
}
export -f one_job

# 缓存命中率采样。注意: 不能写成 SP=$(start_stats) —— 命令替换会等管道 EOF,
# 而后台 while 循环持有写端永不关闭, 主脚本会永久卡在 pipe_read(实测踩过)。
start_stats() {   # start_stats <tag>; PID 写入全局 STATS_PID
  local tag=$1
  ( while :; do
      line="[$(date '+%F %T')] $tag 缓存"
      for i in $(seq 0 $((NG-1))); do
        cr=$(curl -s -m 3 --noproxy '*' http://127.0.0.1:$((8600+i))/stats 2>/dev/null \
             | python3 -c "import json,sys; print(json.load(sys.stdin).get('cached_ratio','-'))" 2>/dev/null)
        line="$line r$i=${cr:--}"
      done
      echo "$line 完成 $(ls $ROOT/done 2>/dev/null | wc -l)/3500" >> $ROOT/cache_stats.log
      sleep 600
    done ) >/dev/null 2>&1 &
  STATS_PID=$!
}

# ================= 阶段 1: allhard(裸 vLLM) =================
SDIR=$T/serving/swemd_final_7arm
say "构建 shim 服务目录(七臂共用同一套栈)"
if [ -f "$SDIR/config.json" ]; then
  say "  服务目录已在, 跳过构建"
else
  cd $LCLM && PYTHONPATH=$LCLM/latent_context/vllm_plugin $VLLM_ENV/bin/python -m vllm_lclm.build_serving_dir \
    --eval-dir $BAKED --out $SDIR --repo-dir $LCLM > $ROOT/logs/serve/build.log 2>&1 \
    || { say "!! build_serving_dir 失败"; exit 1; }
fi
say "起 $NG 个 shim 副本"
for i in $(seq 0 $((NG-1))); do
  # 判据不能写死模型名 —— 服务目录里的 id 是 qwen3-4b-2507, 匹配 swe-master 永远为假。
  curl -s -m 3 --noproxy '*' http://127.0.0.1:$((8600+i))/v1/models 2>/dev/null | grep -q '"object":"model"' && {
    say "  副本 $i 已在, 跳过"; continue; }
  CUDA_VISIBLE_DEVICES=$i PYTHONPATH=$LCLM/latent_context/vllm_plugin \
    setsid nohup $VLLM_ENV/bin/python -m vllm_lclm.serve_shim \
    --serving-dir $SDIR --port $((8600+i)) --max-model-len 131072 \
    --gpu-memory-utilization 0.88 --max-num-seqs 24 --kv-cache-dtype auto \
    --max-num-batched-tokens 8192 --enable-prefix-caching 1 \
    --default-top-p 0.8 --default-top-k 20 --default-min-p 0.0 \
    --metrics-log $ROOT/logs/serve/metrics_$((8600+i)).jsonl \
    > $ROOT/logs/serve/shim_$((8600+i)).log 2>&1 < /dev/null &
done
for w in $(seq 1 90); do
  ok=0; for i in $(seq 0 $((NG-1))); do curl -s -m 3 --noproxy '*' -o /dev/null http://127.0.0.1:$((8600+i))/v1/models && ok=$((ok+1)); done
  [ "$ok" -eq "$NG" ] && break; sleep 20
done
[ "${ok:-0}" -eq "$NG" ] || { say "!! 仅 ${ok:-0}/$NG shim 就绪"; exit 1; }
# 闸门: 查引擎实际生效的分块预填充上限, 不是查命令行写了什么。
# 这个值不显式设会被多模态路径压到编码器预算(实测 2048), 与裸 vLLM 的 8192 不同,
# 会改变浮点归约顺序进而改变贪心输出 —— 必须确认真的是 8192。
VREC=$ROOT/logs/serve/verified_effective.txt
MNBT=$(grep -o "max_num_batched_tokens=[0-9]*" $ROOT/logs/serve/shim_8600.log 2>/dev/null | head -1 | cut -d= -f2)
PFX=$(grep -o "enable_prefix_caching=[A-Za-z]*" $ROOT/logs/serve/shim_8600.log 2>/dev/null | head -1 | cut -d= -f2)
if [ -n "$MNBT" ]; then
  # 日志里有引擎配置行: 读生效值, 并把结论连同进程身份存档, 供下次复用 shim 时取证。
  SPID=$(pgrep -f "serve_shim.*--port 8600" | head -1)
  echo "$SPID $(stat -c %Y /proc/${SPID:-0} 2>/dev/null) $MNBT $PFX" > $VREC
else
  # 日志被截断。只要存档里那个进程还活着且启动时刻没变, 当前服务的就仍是当初验过的同一批引擎。
  read -r SPID SST MNBT PFX < $VREC 2>/dev/null
  NOW=$(stat -c %Y /proc/${SPID:-0} 2>/dev/null)
  { [ -n "${SPID:-}" ] && [ -n "$NOW" ] && [ "$NOW" = "$SST" ]; } \
    || { say "!! 无法确认生效的 max_num_batched_tokens(shim 日志已截断, 且存档与在跑进程对不上)"; exit 1; }
  say "  shim 日志已被截断, 改用存档核验: pid=$SPID 启动时刻未变, 仍是验过的那批引擎"
fi
[ "${MNBT:-0}" = "8192" ] || { say "!! 生效的 max_num_batched_tokens=${MNBT:-未知}, 期望 8192"; exit 1; }
say "引擎生效值 max_num_batched_tokens=$MNBT enable_prefix_caching=$PFX"
say "$NG 个 shim 就绪, 3500 组单一全局队列, 并发 $CONC"
cd $OH
# 单一全局队列: 按题为主序, 每道题连出七个臂 -> 七臂的完成数同步增长,
# 不会出现一个臂全跑完才轮到下一个。副本按题号绑定, 同一道题的七个臂落在同一副本,
# 共享该题问题陈述的前缀, 前缀缓存命中更好。全部用 shim 配置 sml。
# 队列先写本地盘再整体拷到共享盘: 共享盘是网络文件系统, 逐行 >> 实测只有约
# 2.6 行/秒, 3500 行要 22 分钟。整段一次重定向再拷贝, 一秒内完成。
QTMP=$TMPDIR/q_$$.txt
k=0
{
  while read -r iid; do
    [ -z "$iid" ] && continue
    for arm in allhard alllatent hardlast8 hardlast4 hardlast3 hardlast2 hardlast1; do
      echo "$arm $iid $((k % NG)) sml"
    done
    k=$((k+1))
  done < $IDS
} > $QTMP
cp $QTMP $ROOT/q.txt
say "队列 $(wc -l < $ROOT/q.txt) 组(七臂 x $(wc -l < $IDS) 题)"

start_stats 全局; SP=$STATS_PID
# 队列最多自动跑 3 遍: 无结论的组每遍重试一次(done 标记的组直接跳过),
# 判分偶发卡壳不再需要人工重新拉驱动 —— v2 与教师两轮各被人工重启了 2-3 次, 全是这一步。
MISS=0
for PASS in 1 2 3; do
  say "队列第 $PASS 遍"
  cat $ROOT/q.txt | xargs -P $CONC -L1 bash -c 'one_job "$@"' _
  MISS=$(awk '{print $1"-"$2}' $ROOT/q.txt | while read -r t; do [ -f "$ROOT/done/$t" ] || echo x; done | wc -l)
  [ "$MISS" -eq 0 ] && break
  say "第 $PASS 遍后仍缺 $MISS 组, 自动续跑"
done
kill $SP 2>/dev/null

# 完成度闸门: 逐组核对 done 标记。xargs 退出不等于跑完 —— 被 kill、被信号中断都会
# 正常退出, 这里拦住才不会把「没跑完」当成「跑完了」。
if [ "$MISS" -gt 0 ]; then
  say "!! 尚有 $MISS 组无 done 标记, 判定为未完成。"
  say "   shim 保持运行; 排查后重跑本脚本即可续跑(已完成的组会自动跳过)。"
  exit 1
fi
say "全部 $(ls $ROOT/done | wc -l) 组完成"

# ================= 汇总 =================
$VENV/bin/python - <<'PYEOF' 2>&1 | tee -a $L
import json, glob, os, re
from collections import defaultdict
agg = defaultdict(lambda: [0, 0])
for g in glob.glob("/root/autodl-tmp/OpenHands-Latent/evaluation/evaluation_outputs/outputs/*/CodeActAgent/*a7-*/output.swebench_eval.jsonl"):
    m = re.search(r"a7-(allhard|alllatent|hardlast\d+)-", g)
    if not m: continue
    for line in open(g, errors="ignore"):
        line = line.strip()
        if not line: continue
        try: d = json.loads(line)
        except Exception: continue
        rep = (d.get("test_result") or {}).get("report") or {}
        agg[m.group(1)][1] += 1
        agg[m.group(1)][0] += bool(rep.get("resolved"))
print("=== 七臂成绩(新 harness) ===")
for arm in ("allhard", "hardlast8", "hardlast4", "hardlast3", "hardlast2", "hardlast1", "alllatent"):
    ok, n = agg.get(arm, [0, 0])
    print("  %-12s %3d / %3d  %.1f%%" % (arm, ok, n, 100.0*ok/max(n,1)))
json.dump({k: {"resolved": v[0], "n": v[1]} for k, v in agg.items()},
          open(os.environ["EVAL_ROOT"] + "/summary.json", "w"), indent=1)
PYEOF
say "EVAL7_ALL_DONE"
