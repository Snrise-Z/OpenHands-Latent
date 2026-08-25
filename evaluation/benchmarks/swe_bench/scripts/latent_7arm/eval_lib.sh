#!/bin/bash
# 评测流水线公共函数(gpu8)。被 eval_arm.sh / eval_all.sh source。
export EVAL_ROOT=${EVAL_ROOT:-/root/autodl-fs/eval_newharness_verified500}          # 全部产物落共享盘(可用环境变量覆盖)
export EVAL_MODEL_ROOT=${EVAL_MODEL_ROOT:-/root/autodl-fs/model_zoo/eval_v3s3514_adamw}   # 烘焙好的评测目录根
export OH=/root/autodl-tmp/OpenHands-Latent
export VENV=/root/autodl-tmp/envs/openhands-0620
export VLLM_ENV=/root/autodl-tmp/vllm-env
export LCLM=/root/autodl-tmp/LCLM
export IDS20=/root/autodl-fs/eval_sets/qwen3_4b_resolved_subset20_ids.txt
export LOGDIR=$EVAL_ROOT/logs
mkdir -p "$LOGDIR" /root/autodl-tmp/eval_v3s

# ---- udocker(与基线同一套分层与陷阱规避) ----
export UDOCKER_EXE=/root/miniconda3/bin/udocker
unset UDOCKER_BIN
export UDOCKER_DIR=/root/autodl-tmp/.udocker
export UDOCKER_STORE=/root/autodl-fs/udocker_store
export UDOCKER_REPOS=$UDOCKER_STORE/repos
export UDOCKER_LAYERS=$UDOCKER_STORE/layers
export UDOCKER_CONTAINERS=$UDOCKER_DIR/containers
export UDOCKER_TMP=/root/autodl-tmp/.proot-tmp
export UDOCKER_EXECMODE=F3
export UDOCKER_TEMPLATE_DIR=/root/autodl-tmp/.udocker/templates   # 每镜像解压一次做模板,任务容器用 XFS reflink 克隆
export CONTAINER_PROXY=http://127.0.0.1:7890
mkdir -p "$UDOCKER_CONTAINERS" "$UDOCKER_TMP"

# ---- 通用环境 ----
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TMPDIR=/root/autodl-tmp/.tmp; mkdir -p "$TMPDIR"
export VLLM_CACHE_ROOT=/root/autodl-tmp/.vllm_cache TRITON_CACHE_DIR=/root/autodl-tmp/.triton_cache
export LD_LIBRARY_PATH=$VLLM_ENV/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}
export no_proxy=localhost,127.0.0.1,::1; export NO_PROXY=$no_proxy

# 单题的进程内超时, 必须小于驱动给每题的外部 timeout -s TERM(7200), 留出收尾余量。
# 外部信号先到时进程被直接杀掉、一行产物都不留, 判分拿不到结论, 该题进重试计数,
# 最终分母被削(实测 500 题里 50 个 output.jsonl 为空)。内部闹钟先响则会写出一行
# 带 error 的确定结果。注意 evaluation/utils/shared.py 单 worker 分支原本漏传这个
# 参数(已修), 两处配套才生效。
export EVAL_INSTANCE_TIMEOUT=6600
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGDIR/eval_driver.log"; }

GPU_ALL="0 1 2 3 4 5 6 7"
# 指定 GPU 编号上的计算进程 PID(按 PCI 总线号匹配;含 vLLM 引擎子进程)
gpu_pids() {
  local want=" ${*:-$GPU_ALL} "
  nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader | while IFS=, read -r idx bus; do
    idx=$(echo $idx); bus=$(echo $bus)
    case "$want" in *" $idx "*) nvidia-smi --query-compute-apps=pid,gpu_bus_id --format=csv,noheader | awk -F', *' -v b="$bus" '$2==b{print $1}';; esac
  done
}
# 停掉指定副本(GPU 编号列表,默认全部)上的推理服务:按显卡杀计算进程(引擎子进程不随父进程退),再按端口杀前端
stop_serving() {
  local reps=${*:-$GPU_ALL}
  for p in $(gpu_pids $reps); do kill -9 "$p" 2>/dev/null; done
  for i in $reps; do
    for p in $(ps -eo pid,args | grep -E "[s]erve_shim|[v]llm.entrypoints" | grep -E -- "--port $((8600 + i))( |$)" | awk '{print $1}'); do kill -9 "$p" 2>/dev/null; done
  done
  sleep 10
}

# 在指定副本上起 latent 服务(训练后的检查点),副本 i 用 GPU i、端口 8600+i;默认 8 个全起
serve_arm() {   # serve_arm <arm> [replicas...]
  local arm=$1; shift; local reps=${*:-$GPU_ALL}
  local evald=$EVAL_MODEL_ROOT/$arm
  local sdir=/root/autodl-tmp/serving/$arm
  mkdir -p "$sdir" "$LOGDIR/serve"
  cd "$LCLM" && "$VLLM_ENV/bin/python" -m vllm_lclm.build_serving_dir --eval-dir "$evald" --out "$sdir" --repo-dir "$LCLM" > "$LOGDIR/serve/build_$arm.log" 2>&1 || return 1
  for i in $reps; do
    CUDA_VISIBLE_DEVICES=$i "$VLLM_ENV/bin/python" -m vllm_lclm.serve_shim \
      --serving-dir "$sdir" --port $((8600 + i)) --max-model-len 262144 \
      --gpu-memory-utilization 0.85 --max-num-seqs 32 \
      --metrics-log "$LOGDIR/serve/metrics_${arm}_$((8600 + i)).jsonl" \
      > "$LOGDIR/serve/shim_${arm}_$((8600 + i)).log" 2>&1 &
  done
}

# 在指定副本上起原生 vLLM(原版 Qwen3-4B 参照臂),端口同为 8600+i,口径与基线一致
serve_base() {   # serve_base [replicas...]
  local reps=${*:-$GPU_ALL}
  mkdir -p "$LOGDIR/serve"
  for i in $reps; do
    CUDA_VISIBLE_DEVICES=$i "$VLLM_ENV/bin/python" -m vllm.entrypoints.openai.api_server \
      --model /root/autodl-fs/model_zoo/Qwen3-4B-Instruct-2507 --served-model-name qwen3-4b-2507 \
      --port $((8600 + i)) --max-model-len 262144 --gpu-memory-utilization 0.85 \
      --enable-prefix-caching --max-num-seqs 32 --enable-auto-tool-choice --tool-call-parser hermes \
      --no-enable-log-requests --enable-prompt-tokens-details --enable-force-include-usage \
      --override-generation-config '{"temperature":0.7,"top_p":0.8,"top_k":20,"min_p":0.0}' \
      > "$LOGDIR/serve/vllm_base_$((8600 + i)).log" 2>&1 &
  done
}

wait_serving() {   # wait_serving [replicas...]  等端点就绪,最多 15 分钟
  local reps=${*:-$GPU_ALL} n=0 t=0
  for i in $reps; do n=$((n + 1)); done
  while [ $t -lt 900 ]; do
    local ok=0
    for i in $reps; do
      curl -s --noproxy '*' --max-time 5 "http://127.0.0.1:$((8600 + i))/v1/models" >/dev/null 2>&1 && ok=$((ok + 1))
    done
    [ $ok -eq $n ] && return 0
    sleep 15; t=$((t + 15))
  done
  return 1
}

# 一个 rollout 分片:<arm> <policy> <replica> <ids文件> [<标签>]
# 策略只差环境变量(fork 的 latent_observation_policy.py 读取)
# 标签默认 r<replica>;分片数超过副本数时两个分片会落到同一副本,必须传各自的标签(s<k>),
# 否则 eval-note(输出目录)和日志同名,两个进程互相覆盖、往同一个 output.jsonl 并发追加。
run_shard() {
  local arm=$1 policy=$2 rep=$3 ids=$4 tag=${5:-r$3}
  case "$policy" in
    allhard)   export LATENT_OBS_POLICY=none ;;
    alllatent) export LATENT_OBS_POLICY=all-latent; export LATENT_MIN_OBS_CHARS=128 ;;
    hardlast3) export LATENT_OBS_POLICY=lastk-hard; export LATENT_HARD_WINDOW=3; export LATENT_MIN_OBS_CHARS=128 ;;
    *) echo "未知策略 $policy"; return 1 ;;
  esac
  cd "$OH"
  PYTHONPATH="$OH" RUNTIME=udocker_runtime.UDockerRuntime ITERATIVE_EVAL_MODE=false \
  EVAL_IDS_FILE="$ids" EVAL_DOCKER_IMAGE_PREFIX=docker.io/swebench \
  "$VENV/bin/python" evaluation/benchmarks/swe_bench/run_infer.py \
    --agent-cls CodeActAgent --llm-config "lclm$rep" --max-iterations 200 \
    --eval-num-workers 1 --eval-note "v3s-${arm}-${policy}-${tag}" \
    --dataset princeton-nlp/SWE-bench_Verified --split test \
    > "$LOGDIR/rollout_${arm}_${policy}_${tag}.log" 2>&1
}

# 判分:<输入 output.jsonl> <并发>
grade() {
  local inp=$1 nw=${2:-4}
  cd "$OH"
  PYTHONPATH="$OH" RUNTIME=udocker_runtime.UDockerRuntime EVAL_DOCKER_IMAGE_PREFIX=docker.io/swebench \
  "$VENV/bin/python" evaluation/benchmarks/swe_bench/eval_infer.py \
    --input-file "$inp" --dataset princeton-nlp/SWE-bench_Verified --split test \
    --eval-num-workers "$nw" > "${inp%.jsonl}.grade.log" 2>&1
}

# 回收闲置容器(评测进程被杀时会留孤儿)
reap_containers() {   # reap_containers <闲置分钟>
  local idle=${1:-30} now; now=$(date +%s)
  for d in "$UDOCKER_CONTAINERS"/*/; do
    [ -d "$d" ] || continue
    local f="$d/ROOT/tmp/.oh_cwd" age
    if [ -f "$f" ]; then age=$(( (now - $(stat -c %Y "$f")) / 60 )); else age=$(( (now - $(stat -c %Y "$d")) / 60 )); fi
    [ "$age" -gt "$idle" ] && { "$UDOCKER_EXE" --allow-root rm "$(basename "$d")" >/dev/null 2>&1 || rm -rf "$d"; }
  done
}
