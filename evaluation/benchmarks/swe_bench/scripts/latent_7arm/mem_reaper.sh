#!/bin/bash
# 内存看门狗。两条判据:
#   1) 无主进程: 父链走到 init 且祖先里没有 run_infer -> 它的产出没人要, RSS > 8G 就杀。
#      评测里智能体自写的复现脚本会在任务被超时杀掉后继续跑, 实测 15 个孤儿合计 587G,
#      单个只有 35-51G, 全都低于原来的 60G 阈值, 所以旧规则一个都抓不到。
#   2) 单进程 RSS > 60G 且不在长驻名单里 -> 直接杀(保留原规则做第二道网)。
LOG=/root/autodl-tmp/mem_reaper.log
while :; do
  python3 - <<'PY' >> $LOG 2>&1
import os, time

def rd(p):
    try:
        return open(p, 'rb').read()
    except Exception:
        return b''

KEEP = ('vllm', 'serve_shim', 'run_infer', 'eval7', 'reaper', 'reap_containers',
        'jupyter', 'tensorboard', 'sshd', 'tmux', 'wandb', 'nvidia', 'dockerd',
        'systemd', 'containerd', 'torchrun', 'eval_infer')

procs = {}
for d in os.listdir('/proc'):
    if not d.isdigit():
        continue
    pid = int(d); st = rd('/proc/%d/stat' % pid)
    if not st:
        continue
    try:
        ppid = int(st[st.rindex(b')') + 2:].split()[1])
    except Exception:
        continue
    rss = 0
    for line in rd('/proc/%d/status' % pid).splitlines():
        if line.startswith(b'VmRSS:'):
            rss = int(line.split()[1]); break
    procs[pid] = dict(ppid=ppid, rss=rss,
                      args=rd('/proc/%d/cmdline' % pid).replace(b'\0', b' ').decode('utf-8', 'replace').strip())

ts = time.strftime('%F %T')
for pid, p in procs.items():
    if not p['args'] or any(k in p['args'] for k in KEEP):
        continue
    big, huge = p['rss'] > 8 * 1024 * 1024, p['rss'] > 60 * 1024 * 1024
    if not big:
        continue
    cur, has_infer = p['ppid'], False
    for _ in range(12):
        if cur in (0, 1) or cur not in procs:
            break
        if 'run_infer' in procs[cur]['args']:
            has_infer = True; break
        cur = procs[cur]['ppid']
    if has_infer and not huge:
        continue
    why = '超 60G' if huge else '无主'
    print('%s 杀 %d rss=%dG (%s) %s' % (ts, pid, p['rss'] // 1048576, why, p['args'][:70]))
    try:
        os.kill(pid, 9)
    except Exception:
        pass
PY
  sleep 60
done
