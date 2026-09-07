#!/bin/bash
# 四种上下文压缩对比方法 x 两个训练后检查点 x SWE-bench Verified 500 题。
# 两家族各占 4 卡、各起一个驱动进程, 互不等待。用法: FAM=qwen|swem [CONC=32] bash eval_cmp2.sh
#   qwen  v3full 检查点(Qwen3-4B-Instruct-2507 解码器 + r64 读取补丁, 与七臂轮 R2/R4 同检查点):
#         GPU 0-3, 端口 8600-8603, 配置 lclm0-3(原生工具调用), Qwen 原生聊天模板,
#         200 步 / 262,144 窗口 —— 与 Qwen 家族七臂轮同口径。
#   swem  swemdistill41k 检查点(SWE-Master-4B-RL 解码器 + 自锚蒸馏, 与七臂轮 R5 同检查点):
#         GPU 4-7, 端口 8604-8607, 配置 sml4-7(swemaster 文本工具协议), SWE-Master 聊天模板
#         (chat_template_swemaster_rl.jinja 由服务目录副本携带, 以 tokenizer 生效值做闸门),
#         100 步 / 131,072 窗口 —— 与 SWE-Master 七臂轮同口径。
# 对比方法(匹配口径, 定义在 adapter_realign/baselines/arms.json): trunclast3 / swepruner / selfprune / longcodezip。
# 剪枝服务 8700(SWE-Pruner)/8701(LongCodeZip)两家族共用, 跑在 GPU 0; Self-Prune 指向本题绑定的解码器副本。
# 成本三路记录, 缺一不可:
#   服务端 logs/serve/metrics_<端口>.jsonl   逐请求 prompt/completion/num_cached_tokens(缓存命中的准确来源)
#   轨迹   output.jsonl metrics.token_usages  逐调用 prompt/completion/cache_read
#   通道   obs_metrics/<臂>/<标签>.jsonl      逐观察 origin/kept/aux 词元与 cached 标记(只有 cached=false 是真实计算)
set -u
FAM=${FAM:?用法: FAM=qwen|swem bash eval_cmp2.sh}
T=/root/autodl-tmp
OH=$T/OpenHands-Latent
VENV=$T/envs/openhands-0620
VLLM_ENV=$T/vllm-env
LCLM=$T/LCLM
BL=$LCLM/adapter_realign/baselines
IDS=/root/autodl-fs/eval_sets/verified_all500_ids.txt
OUTB=$OH/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent
ARMS="trunclast3 swepruner selfprune longcodezip"
case "$FAM" in
  qwen) GPUS="0 1 2 3"; CFG=lclm; SDIR=$T/serving/r64-4b64-full_b2;      MAXLEN=262144; MAXIT=200; TPL_MD5=5795f12e; PFX=cmpq ;;
  swem) GPUS="4 5 6 7"; CFG=sml;  SDIR=$T/serving/swemd_final_7arm_smtpl; MAXLEN=131072; MAXIT=100; TPL_MD5=11c6a8d4; PFX=cmps ;;
  *) echo "未知家族 $FAM"; exit 1 ;;
esac
ROOT=$T/eval_cmp2_$FAM
L=$ROOT/driver.log
CONC=${CONC:-32}
SMOKE_ID=${SMOKE_ID:-django__django-11099}
mkdir -p $ROOT/logs $ROOT/done $ROOT/fuzzy_logs $ROOT/shards $ROOT/attempts $ROOT/obs_metrics $ROOT/logs/serve
say() { echo "[$(date '+%F %T')] $*" | tee -a $L; }

# 先设 EVAL_ROOT 再 source: eval_lib 把 vLLM/Triton 缓存、TMPDIR、udocker 分层、离线开关、
# 进程内超时 EVAL_INSTANCE_TIMEOUT=6600 一并设好; 全部状态目录放本地盘(共享盘 inode 只剩 8 千)。
export EVAL_ROOT=$ROOT LOGDIR=$ROOT/logs
source $T/eval_lib.sh 2>/dev/null || { echo "!! eval_lib.sh 缺失"; exit 1; }
mkdir -p "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TMPDIR" 2>/dev/null
NQ=$(( $(wc -l < $IDS) * 4 ))
say "===== 对比方法评测 家族=$FAM 卡=[$GPUS] 配置=$CFG 步数=$MAXIT 窗口=$MAXLEN 并发=$CONC 组数=$NQ ====="

# ---------- 闸门 1: harness(与七臂轮同一套三处改动 + 观察压缩通道在位) ----------
FZ=$OH/openhands/runtime/latent_fuzzy_editor.py
CR=$OH/openhands/runtime/impl/cli/cli_runtime.py
grep -q "forgot to put the modification into" $FZ || { say "!! identical 提示未更新"; exit 1; }
grep -q "_MISSING_HINTS" $FZ || { say "!! 参数缺失提示缺失"; exit 1; }
grep -q "_PATH_HINTS" $FZ || { say "!! path 提示补丁缺失"; exit 1; }
grep -q "enable_history_truncation = false" $OH/config.toml || { say "!! 未设超窗即停"; exit 1; }
for i in $GPUS; do grep -q "^\[llm.${CFG}${i}\]" $OH/config.toml || { say "!! config.toml 缺 [llm.${CFG}${i}]"; exit 1; }; done
$VENV/bin/python -c "
import sys; sys.path.insert(0,'$OH')
import inspect
from openhands.runtime.impl.cli.cli_runtime import CLIRuntime as _CR
from openhands.runtime.latent_fuzzy_editor import _IDENTICAL_HINT, _MISSING_HINTS
from openhands.memory.obs_compress import apply_observation_compression
assert 'sed' not in inspect.getsource(_CR.run).lower(), 'sed 拦截未彻底移除'
assert 'forgot to put' in _IDENTICAL_HINT and len(_MISSING_HINTS) >= 4
print('harness 闸门 OK')
" 2>&1 | tee -a $L | grep -q OK || { say "!! harness 自检失败"; exit 1; }
HCOMMIT=$(cd $OH && git rev-parse --short HEAD 2>/dev/null)
say "harness 提交 $HCOMMIT editor_md5=$(md5sum $FZ | cut -c1-12) runtime_md5=$(md5sum $CR | cut -c1-12)"
if [ "$FAM" = swem ]; then
  # 系统提示必须与 SWE-Master 训练时看到的逐字节一致(查生效值, 不是 grep 文件)
  SP_MD5=$($VENV/bin/python -c "
import sys, hashlib; sys.path.insert(0,'$OH')
from openhands.llm.swemaster_format import SYSTEM_PROMPT as S
print(hashlib.md5(S.encode()).hexdigest()[:12], len(S))" 2>/dev/null)
  [ "$SP_MD5" = "701e9bd962e8 15496" ] || { say "!! swemaster 系统提示与训练不一致: 期望 [701e9bd962e8 15496] 实得 [$SP_MD5]"; exit 1; }
  say "swemaster 系统提示核验通过 [$SP_MD5]"
fi

# ---------- 闸门 2: 聊天模板身份 —— 看 tokenizer 实际加载出来的模板, 不看文件名 ----------
[ -f $SDIR/config.json ] || { say "!! 服务目录缺失 $SDIR"; exit 1; }
EFF=$($VLLM_ENV/bin/python -c "
import hashlib
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('$SDIR')
print('TPL', hashlib.md5((t.chat_template or '').encode()).hexdigest()[:8], len(t.chat_template or ''))" 2>/dev/null | awk '/^TPL/{print $2" "$3}')
[ "${EFF%% *}" = "$TPL_MD5" ] || { say "!! 服务目录生效模板 md5=[$EFF], 期望 $TPL_MD5"; exit 1; }
if [ "$FAM" = swem ]; then
  cmp -s $SDIR/chat_template.jinja /root/autodl-fs/model_zoo/SWE-Master-4B-RL/chat_template_swemaster_rl.jinja \
    || { say "!! swemaster 模板与共享盘源不一致"; exit 1; }
fi
say "模板闸门通过: $SDIR 生效模板 md5/长度 = $EFF"

# ---------- 闸门 3: 剪枝服务与臂定义 ----------
for p in 8700 8701; do
  curl -s -m 5 --noproxy '*' http://127.0.0.1:$p/health 2>/dev/null | grep -q '"model_loaded":true' \
    || { say "!! 剪枝服务 $p 未就绪(先 start_servers.sh all 0)"; exit 1; }
done
for a in $ARMS; do
  python3 -c "
import json, shlex
d = json.load(open('$BL/arms.json'))['$a']
print('\n'.join('export %s=%s' % (k, shlex.quote(str(v))) for k, v in d.items()))" > $ROOT/armenv_$a.sh \
    || { say "!! arms.json 缺臂 $a"; exit 1; }
done
say "剪枝服务 8700/8701 就绪; 臂环境已渲染到 $ROOT/armenv_*.sh"

pgrep -f "mem_reaper\.s[h]" >/dev/null || { setsid nohup bash $T/mem_reaper.sh >/dev/null 2>&1 </dev/null & }
pgrep -f "reaper_relaxed\.s[h]" >/dev/null || { setsid nohup bash $T/reaper_relaxed.sh >/dev/null 2>&1 </dev/null & }

# ---------- 起本家族 4 个 shim 副本(端口 8600+卡号) ----------
for i in $GPUS; do
  port=$((8600+i))
  curl -s -m 3 --noproxy '*' http://127.0.0.1:$port/v1/models 2>/dev/null | grep -q '"object":"model"' && { say "  副本 $i(端口 $port)已在, 复用"; continue; }
  util=0.88; [ "$i" = 0 ] && util=0.80   # GPU 0 与两个剪枝服务共卡(它们占约 6.3G)
  CUDA_VISIBLE_DEVICES=$i PYTHONPATH=$LCLM/latent_context/vllm_plugin \
    setsid nohup $VLLM_ENV/bin/python -m vllm_lclm.serve_shim \
    --serving-dir $SDIR --port $port --max-model-len $MAXLEN \
    --gpu-memory-utilization $util --max-num-seqs 24 --kv-cache-dtype auto \
    --max-num-batched-tokens 8192 --enable-prefix-caching 1 \
    --default-top-p 0.8 --default-top-k 20 --default-min-p 0.0 \
    --metrics-log $ROOT/logs/serve/metrics_$port.jsonl \
    > $ROOT/logs/serve/shim_$port.log 2>&1 < /dev/null &
  say "  副本 $i(端口 $port)启动, gpu_util=$util"
done
for w in $(seq 1 90); do
  ok=0; for i in $GPUS; do curl -s -m 3 --noproxy '*' -o /dev/null http://127.0.0.1:$((8600+i))/v1/models && ok=$((ok+1)); done
  [ "$ok" -eq 4 ] && break; sleep 20
done
[ "${ok:-0}" -eq 4 ] || { say "!! 仅 ${ok:-0}/4 shim 就绪"; exit 1; }

# 生效值闸门: 查引擎日志里真实生效的分块预填充上限(与七臂轮同款, 日志截断时用存档核验)
P0=$((8600 + ${GPUS%% *}))
VREC=$ROOT/logs/serve/verified_effective.txt
MNBT=$(grep -o "max_num_batched_tokens=[0-9]*" $ROOT/logs/serve/shim_$P0.log 2>/dev/null | head -1 | cut -d= -f2)
PFC=$(grep -o "enable_prefix_caching=[A-Za-z]*" $ROOT/logs/serve/shim_$P0.log 2>/dev/null | head -1 | cut -d= -f2)
if [ -n "$MNBT" ]; then
  SPID=$(pgrep -f "serve_shim.*--port $P0" | head -1)
  echo "$SPID $(stat -c %Y /proc/${SPID:-0} 2>/dev/null) $MNBT $PFC" > $VREC
else
  read -r SPID SST MNBT PFC < $VREC 2>/dev/null
  NOW=$(stat -c %Y /proc/${SPID:-0} 2>/dev/null)
  { [ -n "${SPID:-}" ] && [ -n "$NOW" ] && [ "$NOW" = "$SST" ]; } \
    || { say "!! 无法确认生效的 max_num_batched_tokens(日志截断且存档与在跑进程对不上)"; exit 1; }
  say "  shim 日志已截断, 存档核验通过: pid=$SPID 启动时刻未变"
fi
[ "${MNBT:-0}" = "8192" ] || { say "!! 生效的 max_num_batched_tokens=${MNBT:-未知}, 期望 8192"; exit 1; }
SERVED_ID=$(curl -s -m 5 --noproxy '*' http://127.0.0.1:$P0/v1/models | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
[ -n "$SERVED_ID" ] || { say "!! 取不到服务模型 id"; exit 1; }
say "引擎生效值 max_num_batched_tokens=$MNBT enable_prefix_caching=$PFC 服务模型 id=$SERVED_ID"

cd $OH
export PYTHONPATH=$OH RUNTIME=udocker_runtime.UDockerRuntime ITERATIVE_EVAL_MODE=false
export EVAL_DOCKER_IMAGE_PREFIX=docker.io/swebench
export OH_FUZZY_STR_REPLACE=1 OH_FUZZY_SYNTAX_GUARD=1
export UDOCKER_EXE=/root/miniconda3/bin/udocker UDOCKER_DIR=$T/.udocker
if [ "$FAM" = swem ]; then export OH_FNCALL_STYLE=swemaster OH_SWEMASTER_MAX_STEPS=100; else unset OH_FNCALL_STYLE OH_SWEMASTER_MAX_STEPS; fi
# 观察压缩通道与 latent 通道互斥: 本轮解码器全明文, 压缩在客户端由各方法完成
export LATENT_OBS_POLICY=none
unset LATENT_HARD_WINDOW LATENT_MIN_OBS_CHARS
export OBS_TOKENIZER=$SDIR          # 计量口径 = 本家族解码器分词器
export OH VENV OUTB T L ROOT CFG MAXIT PFX SERVED_ID VLLM_CACHE_ROOT TRITON_CACHE_DIR TMPDIR LD_LIBRARY_PATH

# 判分要访问 GitHub raw(SWE-bench harness 按提交拉 requirements / environment.yml)。这台机器直连时通时断
# (冒烟时直连全部超时, 上一轮 500 题里也有 5 题因此判分失败)。按用户要求, 网络问题用 `clash on` 起代理
# (mihomo, 127.0.0.1:7890, 不做 TLS 拦截, GitHub raw 约 1 秒); 它会悄悄死掉, 所以每次判分前探测, 掉了就重新拉起。
# 路线顺序: clash -> 直连 -> AutoDL 学术代理(做 TLS 拦截, 需系统证书链) -> 等 60 秒再探, 最多 10 分钟;
# 判分日志出现网络错误则换路再判一次。rollout 本身只访问本机服务, 不走代理。
GH_PROBE=https://raw.githubusercontent.com/django/django/419a78300f7cd27611196e1e464d50fd0385ff27/setup.py
CLASH_PROXY=http://127.0.0.1:7890
TURBO_PROXY=$(bash -c 'source /etc/network_turbo >/dev/null 2>&1; echo ${https_proxy:-}')
clash_ensure() {   # 7890 通则返回 0; 否则用 clashctl on 拉起(幂等)再探一次; 加锁防并发重复拉起
  curl -s -m 8 -x "$CLASH_PROXY" -o /dev/null -f "$GH_PROBE" && return 0
  ( flock -w 120 9 || exit 1
    curl -s -m 8 -x "$CLASH_PROXY" -o /dev/null -f "$GH_PROBE" && exit 0
    echo "[$(date '+%F %T')] clash 不通, 重新拉起" >> $ROOT/clash_restart.log
    bash -c 'source /root/tools/mihomo/script/common.sh && source /root/tools/mihomo/script/clashctl.sh && clashctl on' >> $ROOT/clash_restart.log 2>&1
    sleep 5
    curl -s -m 8 -x "$CLASH_PROXY" -o /dev/null -f "$GH_PROBE"
  ) 9>$ROOT/.clash.lock
}
pick_route() {   # 输出可用路线(空串 = 直连, 否则为代理 URL); 十分钟内都不通返回 1
  local i
  for i in $(seq 1 10); do
    clash_ensure && { echo "$CLASH_PROXY"; return 0; }
    env -u http_proxy -u https_proxy curl -s -m 10 -o /dev/null -f "$GH_PROBE" && { echo ""; return 0; }
    [ -n "$TURBO_PROXY" ] && curl -s -m 15 -x "$TURBO_PROXY" -o /dev/null -f "$GH_PROBE" && { echo "$TURBO_PROXY"; return 0; }
    sleep 60
  done
  return 1
}
export GH_PROBE CLASH_PROXY TURBO_PROXY; export -f clash_ensure pick_route
clash_ensure >/dev/null 2>&1
say "判分路线探测: clash=$(curl -s -m 8 -x $CLASH_PROXY -o /dev/null -w '%{http_code}' $GH_PROBE) 直连=$(env -u http_proxy -u https_proxy curl -s -m 10 -o /dev/null -w '%{http_code}' $GH_PROBE) 学术代理($TURBO_PROXY)=$(curl -s -m 15 -x "$TURBO_PROXY" -o /dev/null -w '%{http_code}' $GH_PROBE)"

one_job() {   # one_job <臂> <题号> <卡号>; 配置 ${CFG}<卡号>, 端口 8600+<卡号>
  arm=$1; iid=$2; rep=$3
  tag="${PFX}-${arm}-${iid}"
  [ -f "$ROOT/done/$tag" ] && return 0
  source $ROOT/armenv_$arm.sh
  mkdir -p $ROOT/obs_metrics/$arm
  export OBS_METRICS_PATH=$ROOT/obs_metrics/$arm/$tag.jsonl OBS_ARM=$arm OBS_INSTANCE_ID=$iid
  # Self-Prune 走与智能体同一个副本: 负载分布与缓存行为一致, 成本也落在同一份服务端计量里
  export SELF_PRUNE_BASE_URL=http://127.0.0.1:$((8600+rep))/v1 SELF_PRUNE_MODEL=$SERVED_ID
  for w in $(seq 1 60); do
    fg=$(df -BG /root/autodl-tmp | awk 'NR==2{gsub("G","",$4); print $4}')
    [ "$fg" -ge 250 ] && break
    sleep 60
  done
  SF=$ROOT/shards/$tag.txt; echo "$iid" > $SF
  RLOG=$ROOT/logs/rollout_$tag.log
  EVAL_IDS_FILE=$SF OH_FUZZY_LOG=$ROOT/fuzzy_logs/$tag.jsonl \
    timeout -s TERM 7200 $VENV/bin/python evaluation/benchmarks/swe_bench/run_infer.py \
    --agent-cls CodeActAgent --llm-config ${CFG}$rep --max-iterations $MAXIT \
    --eval-num-workers 1 --eval-note "$tag" \
    --dataset princeton-nlp/SWE-bench_Verified --split test > $RLOG 2>&1 &
  RPID=$!
  # 超窗守卫: 只记账并留收尾时间(配置已设超窗即停, 智能体自行结束并抽补丁), 宽限期过了才强杀
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
    for gtry in 1 2; do
      route=$(pick_route) || { echo "$arm $iid 判分前十分钟内 GitHub 不可达, 仍按直连尝试" >> $ROOT/grade_netfail.txt; route=""; }
      # 学术代理做 TLS 拦截, requests 默认的 certifi 证书链验不过(实测 SSLCertVerificationError), 走代理时改用系统证书链
      # (含 /usr/local/share/ca-certificates/autodl-signed.crt); venv 的 certifi 也已追加该 CA, 这里是双保险。
      cab=""; [ -n "$route" ] && [ "$route" = "$TURBO_PROXY" ] && cab=/etc/ssl/certs/ca-certificates.crt
      # 判分必须有时限(无时限时实测卡在 socket 读上 37 小时占死槽位); -k 60 保证 TERM 无效时补 KILL
      # 用 env 传变量: 由参数展开得到的 "NAME=值" 词不会被 bash 当作前缀赋值, 直接写会被当成命令名
      env http_proxy=$route https_proxy=$route HTTP_PROXY=$route HTTPS_PROXY=$route \
        ${cab:+REQUESTS_CA_BUNDLE=$cab} ${cab:+SSL_CERT_FILE=$cab} \
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
      { [ "$R" = "True" ] || [ "$R" = "False" ]; } && break
      grep -q "requests.exceptions.ConnectionError\|requests.exceptions.SSLError\|SSLCertVerificationError\|Max retries exceeded\|ConnectTimeout\|ReadTimeout\|Connection reset" $ROOT/logs/grade_$tag.log || break
      echo "$arm $iid 第 $gtry 次判分遇网络错误(路线 [${route:-直连}]), 换路重判" >> $ROOT/grade_netfail.txt
      sleep 30
    done
    echo "[$(date '+%F %T')]   [$arm] $iid resolved=$R 路线=[${route:-直连}]" >> $L
  fi
  # done 只在判分产出有效结论后才写; 连续 3 次无结论封盘记入 stuck.txt
  if [ "$R" = "True" ] || [ "$R" = "False" ]; then
    touch $ROOT/done/$tag
  else
    A=$ROOT/attempts/$tag
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

# 缓存命中率采样(后台循环, 不能用命令替换接它)
start_stats() {
  ( while :; do
      line="[$(date '+%F %T')] 缓存"
      for i in $GPUS; do
        cr=$(curl -s -m 3 --noproxy '*' http://127.0.0.1:$((8600+i))/stats 2>/dev/null \
             | python3 -c "import json,sys; print(json.load(sys.stdin).get('cached_ratio','-'))" 2>/dev/null)
        line="$line r$i=${cr:--}"
      done
      echo "$line 完成 $(ls $ROOT/done 2>/dev/null | wc -l)/$NQ" >> $ROOT/cache_stats.log
      sleep 600
    done ) >/dev/null 2>&1 &
  STATS_PID=$!
}

# ---------- 队列: 按题为主序, 每题连出四臂; 副本按题号绑定(同题四臂同副本, 共享题面前缀) ----------
QTMP=$TMPDIR/qc_${FAM}_$$.txt
k=0
{
  while read -r iid; do
    [ -z "$iid" ] && continue
    g=$(echo $GPUS | awk -v k=$((k % 4)) '{print $(k+1)}')
    for arm in $ARMS; do echo "$arm $iid $g"; done
    k=$((k+1))
  done < $IDS
} > $QTMP
cp $QTMP $ROOT/q.txt
say "队列 $(wc -l < $ROOT/q.txt) 组(四臂 x $(wc -l < $IDS) 题)"

# ---------- 冒烟: 一题四臂并行, 逐项核验 rollout/判分/三路成本记录后才放行 ----------
if [ ! -f $ROOT/.smoke_ok ]; then
  say "冒烟: $SMOKE_ID 四臂并行"
  grep " $SMOKE_ID " $ROOT/q.txt | xargs -P 4 -L1 bash -c 'one_job "$@"' _
  $VENV/bin/python $T/cmp2_smoke_check.py $ROOT $PFX $SMOKE_ID $ARMS 2>&1 | tee -a $L | grep -q "^SMOKE_OK" \
    || { say "!! 冒烟未通过, 停(详情见 driver.log)"; exit 1; }
  touch $ROOT/.smoke_ok
  say "冒烟通过, 放行全量"
fi

start_stats; SP=$STATS_PID
cat $ROOT/q.txt | xargs -P $CONC -L1 bash -c 'one_job "$@"' _
kill $SP 2>/dev/null

# ---------- 完成度闸门(逐组核对 done 标记, xargs 退出不等于跑完)与汇总 ----------
MISS=$(awk '{print $1"-"$2}' $ROOT/q.txt | while read -r t; do [ -f "$ROOT/done/$PFX-$t" ] || echo x; done | wc -l)
if [ "$MISS" -gt 0 ]; then
  say "!! 尚有 $MISS 组无 done 标记, 判定为未完成; shim 保持运行, 重跑本脚本即可续跑"
  exit 1
fi
say "全部 $(ls $ROOT/done | wc -l) 组完成"
$VENV/bin/python - "$OUTB" "$PFX" "$ROOT" <<'PYEOF' 2>&1 | tee -a $L
import glob, json, re, sys
outb, pfx, root = sys.argv[1:4]
agg = {}
for g in glob.glob(f"{outb}/*N_{pfx}-*/output.swebench_eval.jsonl"):
    m = re.search(rf"N_{pfx}-([a-z0-9]+)-", g)
    if not m: continue
    arm = m.group(1); a = agg.setdefault(arm, [0, 0])
    for line in open(g, errors="ignore"):
        line = line.strip()
        if not line: continue
        try: d = json.loads(line)
        except Exception: continue
        rep = (d.get("test_result") or {}).get("report") or {}
        a[1] += 1; a[0] += bool(rep.get("resolved"))
print(f"=== 对比方法成绩 家族 {pfx} ===")
for arm in ("trunclast3", "swepruner", "selfprune", "longcodezip"):
    ok, n = agg.get(arm, [0, 0])
    print("  %-12s %3d / %3d  %.1f%%" % (arm, ok, n, 100.0 * ok / max(n, 1)))
json.dump({k: {"resolved": v[0], "n": v[1]} for k, v in agg.items()}, open(f"{root}/summary.json", "w"), indent=1)
PYEOF
say "CMP2_${FAM}_ALL_DONE"
