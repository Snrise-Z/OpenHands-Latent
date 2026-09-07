#!/usr/bin/env python
"""对比方法双家族轮的成绩与成本表(三路记录合一)。

用法: cmp2_cost_report.py <家族根目录> <标签前缀> [臂1,臂2,...]
  家族根目录 = /root/autodl-tmp/eval_cmp2_qwen 或 eval_cmp2_swem; 前缀 cmpq / cmps。
输出: <根>/cost_report.json、cost_report.txt(同时打印)、per_task.csv(逐题, 供与七臂轮配对)。

列(与论文 500 题表同口径, 见 results_tables/成本表加列建议.txt):
  resolved | calls | dec in | dec out | cached | aux tok | ratio | dec PF | aux PF
  dec in/out  轨迹 metrics.token_usages 逐调用 prompt / completion 之和(每题平均)
  cached      解码器缓存命中率 = Σcache_read / Σprompt(轨迹侧); 并列服务端 num_cached_tokens 口径交叉核验
  aux tok     压缩通道逐观察计量里 cached=false 的 aux_prompt + aux_completion(每题平均); 截断臂恒 0
  ratio       观察级压缩比 = Σorigin / Σkept, 只统计真正被压过(kept != origin)的真实记录
  dec PF      稠密 2 x 3.63B x (未命中前填 + 全部解码) + 注意力二次项 4 x 36 x 2560 x ((T^2-c^2)/2 + mT + m^2/2), 逐调用累加
  aux PF      辅助模型只计稠密: SWE-Pruner 0.6B(Qwen3-Reranker-0.6B)、LongCodeZip 1.31B(Qwen2.5-Coder-1.5B-Instruct 非嵌入)、
              Self-Prune 3.63B(解码器本尊; 其前填缓存按服务端未归属请求实测命中率扣除)
服务端交叉核验: 轨迹逐调用 response_id 与服务端逐请求 id 精确连接; 连不上的退回按端口时间序贪心匹配 (prompt, completion)。
  未匹配的服务端请求归为辅助调用(Self-Prune 走同一副本), 其 num_cached_tokens 用于 Self-Prune 的 aux 缓存扣除。
"""
from __future__ import annotations

import collections
import csv
import glob
import json
import os
import sys

OUTB = ('/root/autodl-tmp/OpenHands-Latent/evaluation/evaluation_outputs/outputs/'
        'princeton-nlp__SWE-bench_Verified-test/CodeActAgent')
P_DEC = 3.63e9
L_DEC, D_DEC = 36, 2560
AUX_PARAMS = {'swepruner': 0.6e9, 'longcodezip': 1.31e9, 'selfprune': P_DEC, 'trunclast3': 0.0}
PF = 1e15


def dec_flops(prompt: int, cached: int, comp: int) -> float:
    dense = 2.0 * P_DEC * (max(prompt - cached, 0) + comp)
    attn = 4.0 * L_DEC * D_DEC * ((prompt ** 2 - cached ** 2) / 2.0 + comp * prompt + comp ** 2 / 2.0)
    return dense + attn


def load_jsonl(path):
    out = []
    try:
        for line in open(path, errors='ignore'):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    except FileNotFoundError:
        pass
    return out


def main() -> None:
    root, pfx = sys.argv[1], sys.argv[2]
    arms = sys.argv[3].split(',') if len(sys.argv) > 3 else ['trunclast3', 'swepruner', 'selfprune', 'longcodezip']

    # 题 -> 端口(队列里的副本绑定)
    port_of = {}
    for line in open(os.path.join(root, 'q.txt')):
        p = line.split()
        if len(p) == 3:
            port_of[(p[0], p[1])] = 8600 + int(p[2])

    # 服务端逐请求记录: 按 id 索引 + 按端口时间序
    srv_by_id = {}
    srv_by_port = collections.defaultdict(list)
    for f in glob.glob(os.path.join(root, 'logs', 'serve', 'metrics_*.jsonl')):
        port = int(os.path.basename(f).split('_')[1].split('.')[0])
        for r in load_jsonl(f):
            r['_port'] = port
            srv_by_id[r.get('id')] = r
            srv_by_port[port].append(r)
    for port in srv_by_port:
        srv_by_port[port].sort(key=lambda r: r.get('ts', 0))
    matched_ids = set()

    rows, per_task = [], []
    lines = ['%-12s %5s %9s %6s %9s %8s %7s %8s %7s %6s %7s %7s | %8s %8s' % (
        '臂', '题数', 'resolved', '补丁', 'stuck/cap', 'calls', 'iters', 'dec in', 'dec out', 'cached',
        'aux tok', 'ratio', 'dec PF', 'aux PF')]
    for arm in arms:
        agg = collections.Counter()
        seen = set()
        srv_prompt = srv_cached = 0
        n_match = n_calls = 0
        dec_f = 0.0
        for d in sorted(glob.glob(f'{OUTB}/*N_{pfx}-{arm}-*')):
            traj = load_jsonl(os.path.join(d, 'output.jsonl'))
            if not traj:
                continue
            r = traj[0]
            iid = r.get('instance_id')
            if not iid or iid in seen:
                continue
            seen.add(iid)
            graded = None
            g = load_jsonl(os.path.join(d, 'output.swebench_eval.jsonl'))
            if g:
                graded = bool(((g[0].get('test_result') or {}).get('report') or {}).get('resolved'))
            err = r.get('error') or ''
            patch = ((r.get('test_result') or {}).get('git_patch') or '').strip()
            tus = (r.get('metrics') or {}).get('token_usages') or []
            P = sum(int(t.get('prompt_tokens') or 0) for t in tus)
            C = sum(int(t.get('completion_tokens') or 0) for t in tus)
            CR = sum(int(t.get('cache_read_tokens') or 0) for t in tus)
            iters = sum(1 for x in (r.get('history') or []) if x.get('action'))
            # 逐调用计算量: 缓存用轨迹侧 cache_read(与服务端 num_cached_tokens 同源, 交叉核验见下)
            f_task = 0.0
            for t in tus:
                f_task += dec_flops(int(t.get('prompt_tokens') or 0), int(t.get('cache_read_tokens') or 0),
                                    int(t.get('completion_tokens') or 0))
            # 服务端交叉核验: 先按 response_id 精确连接, 连不上再按端口时间序贪心匹配
            port = port_of.get((arm, iid))
            pool = srv_by_port.get(port, [])
            for t in tus:
                n_calls += 1
                s = srv_by_id.get(t.get('response_id'))
                if s is None:
                    want = (int(t.get('prompt_tokens') or 0), int(t.get('completion_tokens') or 0))
                    for cand in pool:
                        if cand.get('id') in matched_ids:
                            continue
                        if (int(cand.get('prompt_tokens') or 0), int(cand.get('completion_tokens') or 0)) == want:
                            s = cand
                            break
                if s is not None and s.get('id') not in matched_ids:
                    matched_ids.add(s.get('id'))
                    n_match += 1
                    srv_prompt += int(s.get('prompt_tokens') or 0)
                    srv_cached += int(s.get('num_cached_tokens') or 0)
            # 压缩通道
            recs = load_jsonl(os.path.join(root, 'obs_metrics', arm, f'{pfx}-{arm}-{iid}.jsonl'))
            real = [m for m in recs if not m.get('cached')]
            aux_p = sum(int(m.get('aux_prompt_tokens') or 0) for m in real)
            aux_c = sum(int(m.get('aux_completion_tokens') or 0) for m in real)
            comp = [m for m in real if int(m.get('kept_tokens') or 0) != int(m.get('origin_tokens') or 0)]
            o_tok = sum(int(m.get('origin_tokens') or 0) for m in comp)
            k_tok = sum(int(m.get('kept_tokens') or 0) for m in comp)
            n_err = sum(1 for m in recs if m.get('error'))

            agg['n'] += 1
            agg['graded'] += graded is not None
            agg['resolved'] += bool(graded)
            agg['patch'] += bool(patch)
            agg['stuck'] += 'AgentStuckInLoopError' in err
            agg['cap'] += ('AgentStuckInLoopError' not in err) and ('iteration' in err.lower())
            agg['timeout'] += ('timeout' in err.lower()) or ('timed out' in err.lower())
            agg['calls'] += len(tus)
            agg['iters'] += iters
            agg['P'] += P
            agg['C'] += C
            agg['CR'] += CR
            agg['aux_p'] += aux_p
            agg['aux_c'] += aux_c
            agg['o_tok'] += o_tok
            agg['k_tok'] += k_tok
            agg['n_comp'] += len(comp)
            agg['n_real'] += len(real)
            agg['n_rec'] += len(recs)
            agg['obs_err'] += n_err
            dec_f += f_task
            per_task.append({
                'family': pfx, 'arm': arm, 'instance_id': iid, 'resolved': graded, 'non_empty_patch': bool(patch),
                'error': err[:80], 'calls': len(tus), 'iterations': iters, 'dec_in': P, 'dec_out': C,
                'cache_read': CR, 'aux_prompt': aux_p, 'aux_completion': aux_c, 'obs_origin': o_tok,
                'obs_kept': k_tok, 'dec_pf': round(f_task / PF, 4)})

        n = agg['n']
        if n == 0:
            lines.append('%-12s %5d' % (arm, 0))
            continue
        # Self-Prune 的辅助调用走同一副本: 服务端未归属请求 = 辅助调用, 其缓存命中率用于扣除
        aux_cached_rate = 0.0
        unmatched_note = ''
        if arm == 'selfprune':
            up = uc = un = 0
            for port, pool in srv_by_port.items():
                for s in pool:
                    if s.get('id') not in matched_ids:
                        un += 1
                        up += int(s.get('prompt_tokens') or 0)
                        uc += int(s.get('num_cached_tokens') or 0)
            aux_cached_rate = (uc / up) if up else 0.0
            unmatched_note = f'服务端未归属请求 {un} 条, 提示 {up} 词元(通道记 aux 提示 {agg["aux_p"]}), 命中 {aux_cached_rate:.3f}'
        aux_params = AUX_PARAMS.get(arm, 0.0)
        aux_f = 2.0 * aux_params * (agg['aux_p'] * (1.0 - aux_cached_rate) + agg['aux_c'])
        cached_rate = agg['CR'] / agg['P'] if agg['P'] else 0.0
        srv_rate = srv_cached / srv_prompt if srv_prompt else 0.0
        ratio = agg['o_tok'] / agg['k_tok'] if agg['k_tok'] else float('inf') if agg['o_tok'] else 0.0
        row = {
            'arm': arm, 'tasks': n, 'graded': agg['graded'], 'resolved': agg['resolved'],
            'resolve_rate': round(100.0 * agg['resolved'] / n, 2),
            'non_empty_patch': agg['patch'], 'stuck_loop': agg['stuck'], 'iter_cap': agg['cap'], 'timeout_err': agg['timeout'],
            'calls_per_task': round(agg['calls'] / n, 1), 'iters_per_task': round(agg['iters'] / n, 1),
            'dec_in_per_task': round(agg['P'] / n), 'dec_out_per_task': round(agg['C'] / n),
            'cache_read_per_task': round(agg['CR'] / n), 'cached_rate_traj': round(cached_rate, 4),
            'server_matched_calls': n_match, 'server_match_rate': round(n_match / agg['calls'], 4) if agg['calls'] else 0,
            'cached_rate_server': round(srv_rate, 4),
            'aux_tok_per_task': round((agg['aux_p'] + agg['aux_c']) / n), 'aux_prompt_total': agg['aux_p'],
            'aux_completion_total': agg['aux_c'], 'aux_cached_rate': round(aux_cached_rate, 4),
            'obs_records': agg['n_rec'], 'obs_real': agg['n_real'], 'obs_compressed': agg['n_comp'],
            'obs_origin_tokens': agg['o_tok'], 'obs_kept_tokens': agg['k_tok'],
            'obs_ratio': round(ratio, 3) if ratio != float('inf') else 'inf', 'obs_errors': agg['obs_err'],
            'dec_pf_per_task': round(dec_f / n / PF, 4), 'aux_pf_per_task': round(aux_f / n / PF, 4),
            'aux_params': aux_params, 'note': unmatched_note,
        }
        rows.append(row)
        lines.append('%-12s %5d %9s %6s %9s %8.1f %7.1f %8.0f %7.0f %6.3f %7.0f %7s | %8.3f %8.3f' % (
            arm, n, f"{agg['resolved']}/{agg['graded']}", f"{100.0 * agg['patch'] / n:.0f}%",
            f"{agg['stuck']}/{agg['cap']}", agg['calls'] / n, agg['iters'] / n, agg['P'] / n, agg['C'] / n,
            cached_rate, (agg['aux_p'] + agg['aux_c']) / n, row['obs_ratio'], row['dec_pf_per_task'], row['aux_pf_per_task']))
        lines.append('             服务端核验: 匹配 %d/%d 调用 (%.1f%%), 服务端口径缓存 %.3f%s' % (
            n_match, agg['calls'], 100.0 * n_match / max(agg['calls'], 1), srv_rate,
            ('; ' + unmatched_note) if unmatched_note else ''))

    text = '\n'.join(lines)
    print(text)
    json.dump(rows, open(os.path.join(root, 'cost_report.json'), 'w'), indent=1, ensure_ascii=False)
    open(os.path.join(root, 'cost_report.txt'), 'w').write(text + '\n')
    if per_task:
        with open(os.path.join(root, 'per_task.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(per_task[0].keys()))
            w.writeheader()
            w.writerows(per_task)
    print(f'\n-> {root}/cost_report.json, cost_report.txt, per_task.csv ({len(per_task)} 行)')


if __name__ == '__main__':
    main()
