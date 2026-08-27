#!/bin/bash
# 等七臂 v2 驱动收官 -> 出终表 -> 停 shim 释放显存 -> 等模型拷贝完成 -> 起 500 题评测
LOG=/root/autodl-tmp/chain_swemorig.log
say() { echo "$(date '+%F %T') $*" >> $LOG; }
say "等 eval7_v6 退出..."
while pgrep -f "eval7_v6\.s[h]" > /dev/null; do sleep 60; done
say "v6 已退出, done=$(ls /root/autodl-fs/eval_swemd_7arm_v2/done | wc -l)"
python3 /root/autodl-tmp/arm_stats.py > /root/autodl-fs/eval_swemd_7arm_v2/final_arm_table.txt 2>&1
say "终表已写 final_arm_table.txt"
for p in $(pgrep -f "[s]erve_shim"); do kill $p 2>/dev/null; done
sleep 20
for p in $(pgrep -f "[s]erve_shim"); do kill -9 $p 2>/dev/null; done
say "shim 已停, 显存: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"
n=0
while [ ! -f /root/autodl-tmp/models/.copy_done ]; do
  sleep 30; n=$((n+1))
  [ $n -gt 60 ] && { say "!! 模型拷贝 30 分钟未完成, 停"; exit 1; }
done
say "模型拷贝完成, 起评测驱动"
cd /root/autodl-tmp
setsid nohup bash ./eval_swemorig_v1.sh > ./eval_swemorig_v1_nohup.log 2>&1 < /dev/null &
say "驱动 pid=$!"
