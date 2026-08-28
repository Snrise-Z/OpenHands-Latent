#!/bin/bash
# SWE-Master-4B-RL 原始权重 x 现行 harness x Verified 500 题(单臂, 明文观察)。
# 目的: 给七臂里的 allhard 一个"未蒸馏教师模型"参照, 把蒸馏损失与模型/harness 上限分开。
# 口径与七臂 v2 完全一致: 同 harness 闸门、同采样(0.7/0.8/20)、同引擎参数
# (max_model_len 131072 / max_num_seqs 24 / max_num_batched_tokens 8192 / 前缀缓存开)、
# 同 100 步上限、同判分链(含 3600 秒判分时限)。区别仅两点: 权重是原始 RL 模型;
# 服务用裸 vLLM(原模型不需要 latent 插件), 聊天模板显式指定逐字节验证过的
# chat_template_swemaster_rl.jinja(tokenizer_config 内嵌的是通用模板, 不能用)。
set -u
T=/root/autodl-tmp
OH=$T/OpenHands-Latent
VENV=$T/envs/openhands-0620
VLLM_ENV=$T/vllm-env
MODEL=$T/models/SWE-Master-4B-RL
ROOT=/root/autodl-fs/eval_swemorig_500
OUTB=$OH/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent
L=$ROOT/driver.log
IDS=/root/autodl-fs/eval_sets/verified_all500_ids.txt
mkdir -p $ROOT/logs $ROOT/done $ROOT/fuzzy_logs $ROOT/shards $ROOT/attempts $ROOT/logs/serve
say() { echo "[$(date '+%F %T')] $*" | tee -a $L; }

source $T/eval_lib.sh 2>/dev/null || { echo "!! eval_lib.sh 缺失"; exit 1; }
export EVAL_ROOT=$ROOT LOGDIR=$ROOT/logs
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TMPDIR" 2>/dev/null

NG=$(nvidia-smi -L | wc -l)
CONC=${CONC:-64}
say "===== SWE-Master-4B-RL 原始模型 500 题评测开始 NG=$NG CONC=$CONC ====="

# ---------- 模型完备性 ----------
for f in config.json model.safetensors.index.json tokenizer.json chat_template_swemaster_rl.jinja; do
  [ -f "$MODEL/$f" ] || { say "!! 模型文件缺失: $f"; exit 1; }
done
# 与共享盘源逐字节核对模板(换机/拷贝分叉的老坑)
cmp -s $MODEL/chat_template_swemaster_rl.jinja /root/autodl-fs/model_zoo/SWE-Master-4B-RL/chat_template_swemaster_rl.jinja \
  || { say "!! 本地聊天模板与共享盘不一致"; exit 1; }

# ---------- 闸门: harness 三处改动必须在位(与七臂 v2 同一套) ----------
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
[ "$SP_MD5" = "701e9bd962e8 15496" ] || { say "!! 系统提示与训练不一致: 期望 [701e9bd962e8 15496], 实得 [$SP_MD5]"; exit 1; }
grep -q "_PATH_HINTS" $FZ || { say "!! path 提示补丁缺失"; exit 1; }
grep -q "\[llm.smo0\]" $OH/config.toml || { say "!! config.toml 缺 smo 配置"; exit 1; }

pgrep -f "mem_reaper\.s[h]" >/dev/null || { setsid nohup bash $T/mem_reaper.sh >/dev/null 2>&1 </dev/null & }
pgrep -f "reaper_relaxed\.s[h]" >/dev/null || { setsid nohup bash $T/reaper_relaxed.sh >/dev/null 2>&1 </dev/null & }

cd $OH
export PYTHONPATH=$OH RUNTIME=udocker_runtime.UDockerRuntime ITERATIVE_EVAL_MODE=false
export EVAL_DOCKER_IMAGE_PREFIX=docker.io/swebench
export OH_FNCALL_STYLE=swemaster OH_SWEMASTER_MAX_STEPS=100
export OH_FUZZY_STR_REPLACE=1 OH_FUZZY_SYNTAX_GUARD=1
export UDOCKER_EXE=/root/miniconda3/bin/udocker UDOCKER_DIR=$T/.udocker
export LATENT_OBS_POLICY=none
export OH VENV OUTB T L ROOT NG VLLM_CACHE_ROOT TRITON_CACHE_DIR TMPDIR LD_LIBRARY_PATH

one_job() {
  arm=$1; iid=$2; rep=$3; cfg=$4
  tag="${arm}-${iid}"
  [ -f "$ROOT/done/$tag" ] && return 0
  export LATENT_OBS_POLICY=none; unset LATENT_HARD_WINDOW
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
    --eval-num-workers 1 --eval-note "$tag" \
    --dataset princeton-nlp/SWE-bench_Verified --split test > $RLOG 2>&1 &
  RPID=$!
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
  D=$(ls -d $OUTB/*N_$tag 2>/dev/null | head -1)
  N=$(wc -l < "$D/output.jsonl" 2>/dev/null || echo 0)
  R=NA
  if [ "$N" -ge 1 ]; then
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

# ---------- 起 8 个裸 vLLM 副本 ----------
say "起 $NG 个裸 vLLM 副本(端口 8700 起)"
for i in $(seq 0 $((NG-1))); do
  curl -s -m 3 --noproxy '*' http://127.0.0.1:$((8700+i))/v1/models 2>/dev/null | grep -q '"object":"model"' && {
    say "  副本 $i 已在, 跳过"; continue; }
  CUDA_VISIBLE_DEVICES=$i setsid nohup $VLLM_ENV/bin/python -m vllm.entrypoints.openai.api_server \
    --model $MODEL --served-model-name swe-master-rl \
    --chat-template $MODEL/chat_template_swemaster_rl.jinja \
    --port $((8700+i)) --max-model-len 131072 --gpu-memory-utilization 0.88 \
    --max-num-seqs 24 --max-num-batched-tokens 8192 --enable-prefix-caching \
    --no-enable-log-requests --enable-prompt-tokens-details --enable-force-include-usage \
    --override-generation-config '{"temperature":0.7,"top_p":0.8,"top_k":20,"min_p":0.0}' \
    > $ROOT/logs/serve/vllm_$((8700+i)).log 2>&1 < /dev/null &
done
for w in $(seq 1 90); do
  ok=0; for i in $(seq 0 $((NG-1))); do curl -s -m 3 --noproxy '*' -o /dev/null http://127.0.0.1:$((8700+i))/v1/models && ok=$((ok+1)); done
  [ "$ok" -eq "$NG" ] && break; sleep 20
done
[ "${ok:-0}" -eq "$NG" ] || { say "!! 仅 ${ok:-0}/$NG 副本就绪"; exit 1; }

# 生效值闸门: 查引擎日志里的真实值, 日志被截断时用存档核验(与七臂 v2 同款)
VREC=$ROOT/logs/serve/verified_effective.txt
MNBT=$(grep -o "max_num_batched_tokens=[0-9]*" $ROOT/logs/serve/vllm_8700.log 2>/dev/null | head -1 | cut -d= -f2)
PFX=$(grep -o "enable_prefix_caching=[A-Za-z]*" $ROOT/logs/serve/vllm_8700.log 2>/dev/null | head -1 | cut -d= -f2)
if [ -n "$MNBT" ]; then
  SPID=$(pgrep -f "api_server.*--port 8700" | head -1)
  echo "$SPID $(stat -c %Y /proc/${SPID:-0} 2>/dev/null) $MNBT $PFX" > $VREC
else
  read -r SPID SST MNBT PFX < $VREC 2>/dev/null
  NOW=$(stat -c %Y /proc/${SPID:-0} 2>/dev/null)
  { [ -n "${SPID:-}" ] && [ -n "$NOW" ] && [ "$NOW" = "$SST" ]; } \
    || { say "!! 无法确认生效的 max_num_batched_tokens"; exit 1; }
  say "  引擎日志已截断, 存档核验通过: pid=$SPID"
fi
[ "${MNBT:-0}" = "8192" ] || { say "!! 生效的 max_num_batched_tokens=${MNBT:-未知}, 期望 8192"; exit 1; }
say "引擎生效值 max_num_batched_tokens=$MNBT enable_prefix_caching=$PFX"
say "$NG 个副本就绪, 500 题单队列, 并发 $CONC"
cd $OH

# ---------- 队列(先写本地盘再拷共享盘) ----------
QTMP=$TMPDIR/qo_$$.txt
k=0
{
  while read -r iid; do
    [ -z "$iid" ] && continue
    echo "swemorig $iid $((k % NG)) smo"
    k=$((k+1))
  done < $IDS
} > $QTMP
cp $QTMP $ROOT/q.txt
say "队列 $(wc -l < $ROOT/q.txt) 组"

# ---------- 冒烟: 前 2 题串行, 至少 1 题拿到有效判分才放行 ----------
if [ ! -f $ROOT/.smoke_ok ]; then
  say "冒烟: 串行跑前 2 题"
  head -2 $ROOT/q.txt | while read -r a b c d; do one_job "$a" "$b" "$c" "$d"; done
  okc=0
  for t in $(head -2 $ROOT/q.txt | awk '{print $1"-"$2}'); do [ -f $ROOT/done/$t ] && okc=$((okc+1)); done
  [ "${okc:-0}" -ge 1 ] || { say "!! 冒烟失败: 前 2 题无一有效判分, 停"; exit 1; }
  touch $ROOT/.smoke_ok
  say "冒烟通过($okc/2), 放行全量"
fi

# 队列最多自动跑 3 遍: 无结论的组每遍重试一次(done 标记的组直接跳过)。
MISS=0
for PASS in 1 2 3; do
  say "队列第 $PASS 遍"
  cat $ROOT/q.txt | xargs -P $CONC -L1 bash -c 'one_job "$@"' _
  MISS=$(awk '{print $1"-"$2}' $ROOT/q.txt | while read -r t; do [ -f "$ROOT/done/$t" ] || echo x; done | wc -l)
  [ "$MISS" -eq 0 ] && break
  say "第 $PASS 遍后仍缺 $MISS 组, 自动续跑"
done

# ---------- 完成度闸门与汇总 ----------
if [ "$MISS" -gt 0 ]; then
  say "!! 尚有 $MISS 组无 done 标记, 判定为未完成; 重跑本脚本即可续跑"
  exit 1
fi
say "全部 $(ls $ROOT/done | wc -l) 组完成"
$VENV/bin/python - <<'PYEOF' 2>&1 | tee -a $L
import json, glob, os
ok = n = 0
for g in glob.glob("/root/autodl-tmp/OpenHands-Latent/evaluation/evaluation_outputs/outputs/*/CodeActAgent/*N_swemorig-*/output.swebench_eval.jsonl"):
    for line in open(g, errors="ignore"):
        line = line.strip()
        if not line: continue
        try: d = json.loads(line)
        except Exception: continue
        rep = (d.get("test_result") or {}).get("report") or {}
        n += 1; ok += bool(rep.get("resolved"))
print("=== SWE-Master-4B-RL 原始模型 x 现行 harness ===")
print("  resolved %d / %d = %.1f%%" % (ok, n, 100.0*ok/max(n,1)))
json.dump({"resolved": ok, "n": n}, open(os.environ["EVAL_ROOT"] + "/summary.json", "w"))
PYEOF
say "SWEMORIG_ALL_DONE"
