"""对比基线的观察压缩通道:截断 / SWE-Pruner / Self-Prune / LongCodeZip。

挂在与 latent 策略完全相同的调用点(ConversationMemory 把事件流渲染成消息之后),
所以四条基线臂与我们的 latent 臂共享脚手架、工具集、系统提示、判分链路,
臂间唯一的差别就是"滑窗之外的工具观察被替换成什么"。

环境变量:
  OBS_COMPRESS      none(缺省) | truncate | swe-pruner | self-prune | longcodezip
  OBS_HARD_WINDOW   最后 K 条工具观察保持原样(缺省 3,与 Hard Last 3 对齐)
  OBS_SCOPE         older-than-k | all
                    缺省随后端而定:截断只对窗口外生效;三个剪枝方法按原论文"到达即剪枝",
                    默认对所有观察生效。要做等预算对照时把它们也设成 older-than-k。
  OBS_MIN_CHARS     低于此长度的观察不动(缺省 128,与 latent 策略同口径)
  OBS_TOKENIZER     统计词元用的分词器路径(缺省不加载,退化成字符数估计)
  OBS_METRICS_PATH  逐次压缩的计量落盘路径
  OBS_INSTANCE_ID / OBS_ARM  计量记录里的标识

其余每个后端自己的开关见各模块文档。
"""

from __future__ import annotations

import json
import os
import re
import sys

from openhands.core.message import Message, TextContent

from .base import ObsContext, ResultCache
from . import metrics as _metrics

MEMORY_OPEN = '<|memory_start|>'
MEMORY_CLOSE = '<|memory_end|>'
_MARKER_RE = re.compile(re.escape(MEMORY_OPEN) + '|' + re.escape(MEMORY_CLOSE))

_ALL_SCOPE_BACKENDS = {'swe-pruner', 'self-prune', 'longcodezip'}

_cache = ResultCache()
_backend = None
_backend_name = None
_warned = False


def _make_backend(name: str):
    if name == 'truncate':
        from .truncate import TruncateCompressor

        return TruncateCompressor()
    if name == 'swe-pruner':
        from .swe_pruner import SwePrunerCompressor

        return SwePrunerCompressor()
    if name == 'self-prune':
        from .self_prune import SelfPruneCompressor

        return SelfPruneCompressor()
    if name == 'longcodezip':
        from .longcodezip import LongCodeZipCompressor

        return LongCodeZipCompressor()
    raise ValueError(f'未知的 OBS_COMPRESS 后端: {name}')


def _get_backend(name: str):
    global _backend, _backend_name
    if _backend is None or _backend_name != name:
        _backend = _make_backend(name)
        _backend_name = name
    return _backend


def _task_statement(messages: list[Message]) -> str:
    for m in messages:
        if m.role == 'user':
            for item in m.content:
                if isinstance(item, TextContent) and (item.text or '').strip():
                    return item.text
    return ''


def _tool_call_map(messages: list[Message]) -> dict[str, tuple[str, str]]:
    """tool_call_id -> (工具名, 参数 JSON 串)。"""
    out: dict[str, tuple[str, str]] = {}
    for m in messages:
        for tc in m.tool_calls or []:
            try:
                fn = tc.function
                out[tc.id] = (fn.name or '', fn.arguments or '')
            except Exception:
                continue
    return out


def _focus_question(arguments: str) -> str:
    if not arguments:
        return ''
    try:
        d = json.loads(arguments)
    except Exception:
        return ''
    v = d.get('context_focus_question') if isinstance(d, dict) else None
    return v.strip() if isinstance(v, str) else ''


def apply_observation_compression(messages: list[Message]) -> list[Message]:
    """按 OBS_COMPRESS 替换工具观察。缺省不做任何事,保持上游行为。"""
    name = os.environ.get('OBS_COMPRESS', 'none').strip().lower()
    if name in ('', 'none'):
        return messages

    latent = os.environ.get('LATENT_OBS_POLICY', 'none').strip().lower()
    if latent not in ('', 'none'):
        # 两套通道同时开着会得到一个既非基线也非本方法的臂,宁可当场失败也不要静默污染
        raise RuntimeError(
            f'OBS_COMPRESS={name} 与 LATENT_OBS_POLICY={latent} 不能同时启用'
        )

    hard_window = int(os.environ.get('OBS_HARD_WINDOW', '3') or 3)
    min_chars = int(os.environ.get('OBS_MIN_CHARS', '128') or 128)
    default_scope = 'all' if name in _ALL_SCOPE_BACKENDS else 'older-than-k'
    scope = os.environ.get('OBS_SCOPE', default_scope).strip().lower()

    tool_indices = [i for i, m in enumerate(messages) if m.role == 'tool']
    if not tool_indices:
        return messages
    if scope == 'older-than-k' and hard_window > 0:
        targets = tool_indices[:-hard_window]
    else:
        targets = list(tool_indices)
    if not targets:
        return messages

    try:
        backend = _get_backend(name)
    except Exception as exc:  # 配置错误应当立刻暴露
        raise RuntimeError(f'初始化观察压缩后端失败: {exc}') from exc

    call_map = _tool_call_map(messages)
    task = _task_statement(messages)
    total = len(tool_indices)
    pos = {idx: n for n, idx in enumerate(tool_indices)}

    for i in targets:
        msg = messages[i]
        tool_name, args = call_map.get(msg.tool_call_id or '', (msg.name or '', ''))
        ctx = ObsContext(
            task_statement=task,
            tool_name=tool_name,
            tool_arguments=args,
            focus_question=_focus_question(args),
            index=pos.get(i, -1),
            total=total,
        )
        new_content = []
        changed = False
        for item in msg.content:
            if not isinstance(item, TextContent):
                new_content.append(item)
                continue
            text = _MARKER_RE.sub('', item.text or '')
            if len(text) < min_chars:
                new_content.append(TextContent(text=text))
                continue
            key = ResultCache.key(name, ctx.focus_question or task, text)
            res = _cache.get(key)
            if res is None:
                try:
                    res = backend.compress(text, ctx)
                except Exception as exc:  # 后端异常不能打断 rollout
                    _warn(f'{name} 压缩异常,原样透传: {type(exc).__name__}: {exc}')
                    from .base import CompressResult, count_tokens

                    n = count_tokens(text)
                    res = CompressResult(
                        text=text, origin_tokens=n, kept_tokens=n, backend=name,
                        error=f'{type(exc).__name__}: {exc}',
                    )
                _cache.put(key, res)
            _metrics.record(res, ctx)
            new_content.append(TextContent(text=res.text))
            changed = changed or res.text != text
        if changed or new_content:
            messages[i] = msg.model_copy(update={'content': new_content})
    return messages


def _warn(msg: str) -> None:
    global _warned
    if not _warned:
        _warned = True
        print(f'[obs_compress] {msg}', file=sys.stderr, flush=True)
