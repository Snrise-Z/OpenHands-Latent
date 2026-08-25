#!/bin/bash
# 3500 组(7 臂 x 500 题)全量评测, 新 harness(提交 5852af5)。
# 阶段1: allhard(裸 vLLM);阶段2: 6 个 latent 臂共用一套 shim, 3000 组全局队列。
# 每题一条流水线: rollout 完立即判分, 不等其他题。
set -u
T=/root/autodl-tmp
OH=$T/OpenHands-Latent
VENV=$T/envs/openhands-0620
VLLM_ENV=$T/vllm-env
LCLM=$T/LCLM
BAKED=$T/swemd_final_eval
ROOT=/root/autodl-fs/eval_swemd_7arm
OUTB=$OH/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent
L=$ROOT/driver.log
IDS=/root/autodl-fs/eval_sets/verified_all500_ids.txt
mkdir -p $ROOT/{logs,done,fuzzy_logs,shards,logs/serve}
say() { echo "[$(date '+%F %T')] $*" | tee -a $L; }

# 必须先 source: 它把 vLLM/Triton 缓存、TMPDIR、LD_LIBRARY_PATH 挪出系统盘(仅 3.7G 空闲),
# 否则引擎在 os.makedirs 缓存目录时直接失败(实测: 权重加载成功后仍崩)。
source $T/eval_lib.sh 2>/dev/null || { echo "!! eval_lib.sh 缺失"; exit 1; }
export EVAL_ROOT=$ROOT LOGDIR=$ROOT/logs
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TMPDIR" 2>/dev/null

NG=$(nvidia-smi -L | wc -l)
say "===== 3500 组评测开始(新 harness) NG=$NG ====="
say "缓存 VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT TMPDIR=$TMPDIR 系统盘余 $(df -BG / | awk 'NR==2{print $4}')"

# ---------- 闸门: 三处改动必须在位 ----------
FZ=$OH/openhands/runtime/latent_fuzzy_editor.py
CR=$OH/openhands/runtime/impl/cli/cli_runtime.py
grep -q "forgot to put the modification into" $FZ || { say "!! identical 提示未更新"; exit 1; }
grep -q "_MISSING_HINTS" $FZ || { say "!! 参数缺失提示缺失"; exit 1; }
grep -q "_uses_sed_inplace" $CR || { say "!! sed -i 禁用未生效"; exit 1; }
grep -q "enable_history_truncation = false" $OH/config.toml || { say "!! 未设超窗即停"; exit 1; }
$VENV/bin/python -c "
import sys; sys.path.insert(0,'$OH')
from openhands.runtime.impl.cli.cli_runtime import _uses_sed_inplace as B
from openhands.runtime.latent_fuzzy_editor import _IDENTICAL_HINT, _MISSING_HINTS
assert B(\"sed -i 's/a/b/' x.py\") and not B(\"sed -n '1,2p' x.py\")
assert 'forgot to put' in _IDENTICAL_HINT and len(_MISSING_HINTS) >= 4
print('harness 闸门 OK')
" 2>&1 | tee -a $L | grep -q OK || { say "!! harness 自检失败"; exit 1; }
SP_MD5=$($VENV/bin/python -c "
import sys, hashlib; sys.path.insert(0,'$OH')
from openhands.llm.swemaster_format import SYSTEM_PROMPT as S
print(hashlib.md5(S.encode()).hexdigest()[:12], len(S))
" 2>/dev/null)
say "闸门通过 editor_md5=$(md5sum $FZ | cut -c1-12) runtime_md5=$(md5sum $CR | cut -c1-12) sysprompt=$SP_MD5"
grep -q "In-place shell editing is disabled" $OH/openhands/llm/swemaster_format.py \
  || { say "!! 系统提示未含 sed 禁令"; exit 1; }
grep -q "_PATH_HINTS" $FZ || { say "!! path 提示补丁缺失"; exit 1; }

pgrep -f "mem_reaper2\.s[h]" >/dev/null || { setsid nohup bash $T/mem_reaper2.sh >/dev/null 2>&1 </dev/null & }
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
    $VENV/bin/python evaluation/benchmarks/swe_bench/eval_infer.py \
      --input-file "$D/output.jsonl" --dataset princeton-nlp/SWE-bench_Verified --split test \
      --eval-num-workers 1 > $ROOT/logs/grade_$tag.log 2>&1
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
say "阶段1 起 $NG 个裸 vLLM(allhard)"
for i in $(seq 0 $((NG-1))); do
  curl -s -m 3 --noproxy '*' http://127.0.0.1:$((8600+i))/v1/models 2>/dev/null | grep -q swe-master && {
    say "  副本 $i 已在, 跳过"; continue; }
  CUDA_VISIBLE_DEVICES=$i setsid nohup $VLLM_ENV/bin/python -m vllm.entrypoints.openai.api_server \
    --model $BAKED/decoder --served-model-name swe-master \
    --port $((8600+i)) --max-model-len 131072 --gpu-memory-utilization 0.90 \
    --enable-prefix-caching --max-num-seqs 16 \
    --no-enable-log-requests --enable-prompt-tokens-details --enable-force-include-usage \
    --override-generation-config '{"temperature":0.7,"top_p":0.8,"top_k":20,"min_p":0.0}' \
    > $ROOT/logs/serve/vllm_$((8600+i)).log 2>&1 < /dev/null &
done
for w in $(seq 1 90); do
  ok=0; for i in $(seq 0 $((NG-1))); do curl -s -m 3 --noproxy '*' -o /dev/null http://127.0.0.1:$((8600+i))/v1/models && ok=$((ok+1)); done
  [ "$ok" -eq "$NG" ] && break; sleep 20
done
[ "${ok:-0}" -eq "$NG" ] || { say "!! 阶段1 仅 ${ok:-0}/$NG 就绪"; exit 1; }
say "阶段1 $NG 副本就绪, 500 题并发 64"
: > $ROOT/q1.txt
# 优先队列: $ROOT/priority.txt 里的题排到最前(用于补齐跨轮对照所需的题)
k=0
if [ -s "$ROOT/priority.txt" ]; then
  while read -r iid; do
    [ -n "$iid" ] && echo "allhard $iid $((k % NG)) sm" >> $ROOT/q1.txt && k=$((k+1))
  done < $ROOT/priority.txt
  say "优先队列 $k 题排在最前"
fi
while read -r iid; do
  [ -z "$iid" ] && continue
  grep -qxF "$iid" "$ROOT/priority.txt" 2>/dev/null && continue
  echo "allhard $iid $((k % NG)) sm" >> $ROOT/q1.txt; k=$((k+1))
done < $IDS
start_stats 阶段1; SP=$STATS_PID
cat $ROOT/q1.txt | xargs -P 64 -L1 bash -c 'one_job "$@"' _
kill $SP 2>/dev/null
# 完成度闸门: 队列里每一题都要有 done 标记才算这一阶段真的结束。
# xargs 若被人为杀掉或异常退出, 这里会拦住, 不会误入阶段 2(踩过一次: 杀 xargs
# 导致主脚本顺势杀光 vLLM 并开始构建 shim)。
MISS=$(awk '{print $1"-"$2}' $ROOT/q1.txt | while read -r t; do [ -f "$ROOT/done/$t" ] || echo x; done | wc -l)
if [ "$MISS" -gt 0 ]; then
  say "!! 阶段1 尚有 $MISS 题无 done 标记, 判定为未完成, 不进入阶段2。"
  say "   vLLM 保持运行; 排查后重跑本脚本即可续跑(已完成的题会自动跳过)。"
  exit 1
fi
say "阶段1 完成: $(ls $ROOT/done | wc -l)/500"
pkill -f "vllm.entrypoints.openai.api_serve[r]"; sleep 30

# ================= 阶段 2: 6 个 latent 臂(shim) =================
SDIR=$T/serving/swemd_final_7arm
say "阶段2 构建 shim 服务目录"
cd $LCLM && PYTHONPATH=$LCLM/latent_context/vllm_plugin $VLLM_ENV/bin/python -m vllm_lclm.build_serving_dir \
  --eval-dir $BAKED --out $SDIR --repo-dir $LCLM > $ROOT/logs/serve/build.log 2>&1 \
  || { say "!! build_serving_dir 失败"; exit 1; }
for i in $(seq 0 $((NG-1))); do
  CUDA_VISIBLE_DEVICES=$i PYTHONPATH=$LCLM/latent_context/vllm_plugin \
    setsid nohup $VLLM_ENV/bin/python -m vllm_lclm.serve_shim \
    --serving-dir $SDIR --port $((8600+i)) --max-model-len 131072 \
    --gpu-memory-utilization 0.88 --max-num-seqs 24 \
    --metrics-log $ROOT/logs/serve/metrics_$((8600+i)).jsonl \
    > $ROOT/logs/serve/shim_$((8600+i)).log 2>&1 < /dev/null &
done
for w in $(seq 1 90); do
  ok=0; for i in $(seq 0 $((NG-1))); do curl -s -m 3 --noproxy '*' -o /dev/null http://127.0.0.1:$((8600+i))/v1/models && ok=$((ok+1)); done
  [ "$ok" -eq "$NG" ] && break; sleep 20
done
[ "${ok:-0}" -eq "$NG" ] || { say "!! 阶段2 仅 ${ok:-0}/$NG shim 就绪"; exit 1; }
say "阶段2 $NG shim 副本就绪, 3000 组全局队列并发 48"
cd $OH
: > $ROOT/q2.txt
k=0
while read -r iid; do
  [ -z "$iid" ] && continue
  for arm in alllatent hardlast8 hardlast4 hardlast3 hardlast2 hardlast1; do
    echo "$arm $iid $((k % NG)) sml" >> $ROOT/q2.txt; k=$((k+1))
  done
done < $IDS
start_stats 阶段2; SP=$STATS_PID
cat $ROOT/q2.txt | xargs -P 48 -L1 bash -c 'one_job "$@"' _
kill $SP 2>/dev/null
MISS2=$(awk '{print $1"-"$2}' $ROOT/q2.txt | while read -r t; do [ -f "$ROOT/done/$t" ] || echo x; done | wc -l)
[ "$MISS2" -gt 0 ] && say "!! 阶段2 尚有 $MISS2 组无 done 标记(汇总仍会输出, 但不完整)"

# ================= 汇总 =================
$VENV/bin/python - <<'PYEOF' 2>&1 | tee -a $L
import json, glob, re
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
          open("/root/autodl-fs/eval_swemd_7arm/summary.json", "w"), indent=1)
PYEOF
say "EVAL7_ALL_DONE"
