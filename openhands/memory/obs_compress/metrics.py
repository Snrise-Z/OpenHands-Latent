"""对比基线的成本计量:每次压缩追加一条 JSONL 记录。

字段与 serve_shim 的计量对齐,便于最终成本表把"辅助模型开销"与我们的编码器开销放同一列:
  origin_tokens        压缩前观察词元(解码器分词器口径)
  kept_tokens          压缩后观察词元
  aux_prompt_tokens    辅助模型输入词元
  aux_completion_tokens 辅助模型输出词元
  cached               是否命中缓存(命中不重复计成本)
路径由 OBS_METRICS_PATH 指定;未设则不落盘(冒烟测试可用 OBS_METRICS_STDERR=1 打到日志)。
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time

_lock = threading.Lock()
_path = None
_resolved = False


def _target() -> str | None:
    global _path, _resolved
    if _resolved:
        return _path
    with _lock:
        if not _resolved:
            p = os.environ.get('OBS_METRICS_PATH', '').strip()
            if p:
                try:
                    os.makedirs(os.path.dirname(p) or '.', exist_ok=True)
                except OSError:
                    p = ''
            _path = p or None
            _resolved = True
    return _path


def record(result, ctx) -> None:
    """写一条计量记录。任何异常都不能影响 rollout,因此整体吞掉。"""
    try:
        rec = {
            'ts': time.time(),
            'instance_id': os.environ.get('OBS_INSTANCE_ID', ''),
            'arm': os.environ.get('OBS_ARM', ''),
            'backend': result.backend,
            'origin_tokens': result.origin_tokens,
            'kept_tokens': result.kept_tokens,
            'aux_prompt_tokens': result.aux_prompt_tokens,
            'aux_completion_tokens': result.aux_completion_tokens,
            'latency_ms': round(result.latency_ms, 2),
            'cached': result.cached,
            'obs_index': ctx.index,
            'obs_total': ctx.total,
            'tool': ctx.tool_name,
            'has_focus_question': bool(ctx.focus_question),
            'error': result.error,
        }
        if result.extra:
            rec['extra'] = result.extra
        line = json.dumps(rec, ensure_ascii=False)
        p = _target()
        if p:
            with _lock:
                with open(p, 'a') as f:
                    f.write(line + '\n')
        if os.environ.get('OBS_METRICS_STDERR', '').strip() in ('1', 'true', 'yes'):
            print('[obs_compress] ' + line, file=sys.stderr, flush=True)
    except Exception:
        pass
