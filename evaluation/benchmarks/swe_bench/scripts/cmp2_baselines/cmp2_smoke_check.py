#!/usr/bin/env python
"""对比方法评测的冒烟核验: 每个臂一题, 逐项检查 rollout、判分与三路成本记录是否齐全可信。

用法: cmp2_smoke_check.py <评测根> <标签前缀> <题号> <臂...>
通过判据(任一不满足即 SMOKE_FAIL):
  每臂有输出目录与轨迹、判分给出 True/False、轨迹不带 error;
  轨迹 token_usages 有缓存读数(调用数 >= 2 时 cache_read 必须 > 0, 这正是上一轮漏掉的项);
  观察压缩通道有计量记录、无报错、剪枝类臂有辅助模型词元、真实记录里保留词元少于原始词元;
  服务端逐请求计量非空且 num_cached_tokens 不恒为 0。
"""
import glob
import json
import os
import sys

root, pfx, iid, *arms = sys.argv[1:]
OUTB = ('/root/autodl-tmp/OpenHands-Latent/evaluation/evaluation_outputs/outputs/'
        'princeton-nlp__SWE-bench_Verified-test/CodeActAgent')
bad = []


def note(msg):
    bad.append(msg)
    print('   !! ' + msg)


for arm in arms:
    tag = f'{pfx}-{arm}-{iid}'
    ds = glob.glob(f'{OUTB}/*N_{tag}')
    if not ds:
        print(f'[{arm}] 无输出目录')
        note(f'{arm} 无输出目录')
        continue
    d = ds[0]
    rows = []
    try:
        rows = [json.loads(l) for l in open(f'{d}/output.jsonl') if l.strip()]
    except Exception:
        pass
    if not rows:
        print(f'[{arm}] output.jsonl 为空')
        note(f'{arm} 无轨迹')
        continue
    r = rows[0]
    err = r.get('error')
    patch = (r.get('test_result') or {}).get('git_patch') or ''
    tu = (r.get('metrics') or {}).get('token_usages') or []
    P = sum(int(u.get('prompt_tokens') or 0) for u in tu)
    C = sum(int(u.get('completion_tokens') or 0) for u in tu)
    CR = sum(int(u.get('cache_read_tokens') or 0) for u in tu)
    res = None
    g = f'{d}/output.swebench_eval.jsonl'
    if os.path.exists(g):
        try:
            res = ((json.loads(open(g).readline()).get('test_result') or {}).get('report') or {}).get('resolved')
        except Exception:
            pass
    mf = f'{root}/obs_metrics/{arm}/{tag}.jsonl'
    recs = []
    if os.path.exists(mf):
        for l in open(mf):
            try:
                recs.append(json.loads(l))
            except Exception:
                pass
    real = [m for m in recs if not m.get('cached')]
    O = sum(int(m.get('origin_tokens') or 0) for m in real)
    K = sum(int(m.get('kept_tokens') or 0) for m in real)
    AUX = sum(int(m.get('aux_prompt_tokens') or 0) + int(m.get('aux_completion_tokens') or 0) for m in real)
    errs = [m for m in recs if m.get('error')]
    backends = sorted({str(m.get('backend')) for m in recs})
    print(f'[{arm}] resolved={res} error={"是" if err else "否"} 调用={len(tu)} 提示={P} 生成={C} 缓存读={CR} '
          f'补丁={len(patch)}B | 通道记录={len(recs)} 真实={len(real)} 原始={O} 保留={K} 辅助={AUX} '
          f'报错={len(errs)} backend={backends}')
    if res not in (True, False):
        note(f'{arm} 无有效判分')
    # 智能体自身的终止方式(卡循环、步数用尽)是合法结果, 不是 harness 故障; 其余 error 才算冒烟失败
    if err and not any(k in str(err) for k in ('AgentStuckInLoopError', 'maximum iteration', 'max_iterations', 'reached maximum')):
        note(f'{arm} 轨迹带 harness 类 error: {str(err)[:160]}')
    if len(tu) >= 2 and CR == 0:
        note(f'{arm} 缓存读数为 0(cache_read 未记录)')
    if not recs:
        note(f'{arm} 观察压缩通道无计量记录')
    if errs:
        note(f'{arm} 压缩通道报错: {str(errs[0].get("error"))[:160]}')
    if arm in ('swepruner', 'selfprune', 'longcodezip') and real and AUX == 0:
        note(f'{arm} 辅助模型词元为 0')
    if real and K >= O:
        # LongCodeZip 粗粒度阶段按函数切块再排序选块: 观察切不出第二块(selected_chunks=1)时整块保留, 是方法固有行为
        single = all(int((m.get('extra') or {}).get('selected_chunks') or 0) <= 1 for m in real)
        if arm == 'longcodezip' and single:
            print(f'   注: longcodezip 本题所有真实记录都只切出单块(selected_chunks=1), 粗粒度排序无从压缩, 不计失败')
        else:
            note(f'{arm} 未见压缩(保留 {K} >= 原始 {O})')

n = cached = 0
for f in glob.glob(f'{root}/logs/serve/metrics_*.jsonl'):
    for l in open(f):
        try:
            m = json.loads(l)
        except Exception:
            continue
        n += 1
        cached += int(m.get('num_cached_tokens') or 0)
print(f'[服务端] 逐请求计量 {n} 条, num_cached_tokens 合计 {cached}')
if n == 0:
    note('服务端逐请求计量为空')
if n >= 4 and cached == 0:
    note('服务端 num_cached_tokens 恒为 0')
print('SMOKE_OK' if not bad else 'SMOKE_FAIL')
sys.exit(0 if not bad else 1)
