#!/bin/bash
# 阶段1 结束时接管: 停旧 driver, 用修复版 eval7_v2.sh 跑阶段2。
# 旧版会在阶段1 后直接杀 vLLM 进阶段2, 这里抢在它构建 shim 之前接手。
set -u
T=/root/autodl-tmp
R=/root/autodl-fs/eval_swemd_7arm
L=$R/takeover.log
say() { echo "[$(date +%F %T)] $*" | tee -a $L; }
START_LINES=$(wc -l < $R/driver.log 2>/dev/null || echo 0)
say "接管器就位(基线 $START_LINES 行): 等阶段1 的 500 题全部 done"
while :; do
  n=$(ls $R/done 2>/dev/null | grep -c "^allhard-")
  # 只认接管器启动之后新写的日志。踩过一次: 直接 grep 全文会匹配到历史里的
  # "阶段2 构建 shim", 让接管器一启动就误触发。
  if [ $(wc -l < $R/driver.log) -gt $START_LINES ]; then
    tail -n +$((START_LINES+1)) $R/driver.log | grep -q "阶段2 构建 shim" && \
      { say "旧 driver 已进阶段2, 立即接管"; break; }
  fi
  [ "$n" -ge 500 ] && { say "阶段1 完成 500/500, 接管"; break; }
  ps -ef | awk "/[e]val7.sh/" | grep -q . || { say "旧 driver 已退出(阶段1 完成 $n/500), 接管"; break; }
  sleep 60
done
ps -ef | awk "/[e]val7.sh/ {print \$2}" | xargs -r kill -9 2>/dev/null
sleep 3
ps -ef | awk "/[b]uild_serving_dir|[s]erve_shim/ {print \$2}" | xargs -r kill -9 2>/dev/null
say "旧 driver 已停, 启动修复版 eval7_v2.sh(done 时机 + 阶段闸门)"
setsid nohup bash $T/eval7_v2.sh > $T/eval7_v2_nohup.log 2>&1 < /dev/null &
say "TAKEOVER_DONE"
