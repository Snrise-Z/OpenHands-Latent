"""Self-Prune 基线:让智能体自己的骨干模型挑要保留的行,不引入任何额外模型。

这是成本最低的一档对照:推理用的就是已经在跑的那个服务,不占额外显存,
但会多出一次生成请求,提示里还要带上带行号的观察,所以"省词元"的账要认真算。

实现要点:
  - 观察按行编号后交给模型,要求只返回保留行号的 JSON。vLLM 支持 guided_json,
    默认开启;关掉时靠正则兜底解析。
  - 观察很长时按 SELF_PRUNE_MAX_LINES 分段处理,避免一次请求把上下文撑爆,
    也避免让模型输出上千个行号。
  - 失败一律原样透传并记错误,不退化成别的方法。
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

from .base import CompressResult, Compressor, ObsContext, count_tokens, derived_query

_SYSTEM = (
    'You are a precision filter. Given a numbered tool output and a focus question, '
    'return ONLY the line numbers that are needed to answer it. '
    'Keep lines that carry file paths, symbol definitions, error messages, diffs or values. '
    'Drop boilerplate, repeated separators and unrelated noise. '
    'Answer with JSON of the form {"keep": [1, 2, 5]} and nothing else.'
)

_SCHEMA = {
    'type': 'object',
    'properties': {'keep': {'type': 'array', 'items': {'type': 'integer'}}},
    'required': ['keep'],
}

_NUM_RE = re.compile(r'-?\d+')


class SelfPruneCompressor(Compressor):
    name = 'self-prune'

    def __init__(self) -> None:
        base = os.environ.get('SELF_PRUNE_BASE_URL') or os.environ.get(
            'OBS_LLM_BASE_URL', 'http://127.0.0.1:8600/v1'
        )
        self.url = base.rstrip('/') + '/chat/completions'
        self.model = os.environ.get('SELF_PRUNE_MODEL', 'default')
        self.max_lines = int(os.environ.get('SELF_PRUNE_MAX_LINES', '400') or 400)
        self.max_tokens = int(os.environ.get('SELF_PRUNE_MAX_TOKENS', '1024') or 1024)
        self.guided = os.environ.get('SELF_PRUNE_GUIDED', '1').strip() not in ('0', 'false', 'no')
        self.timeout = float(os.environ.get('OBS_HTTP_TIMEOUT', '180') or 180)
        self.retries = int(os.environ.get('OBS_HTTP_RETRIES', '2') or 2)

    def _ask(self, numbered: str, query: str, lo: int, hi: int) -> tuple[set[int], int, int, str | None]:
        user = (
            f'Focus question:\n{query}\n\n'
            f'Tool output (lines {lo}-{hi}):\n{numbered}\n\n'
            f'Return the line numbers to keep, between {lo} and {hi}.'
        )
        body = {
            'model': self.model,
            'messages': [{'role': 'system', 'content': _SYSTEM}, {'role': 'user', 'content': user}],
            'temperature': 0.0,
            'max_tokens': self.max_tokens,
        }
        if self.guided:
            body['guided_json'] = _SCHEMA
        payload = json.dumps(body).encode()
        last_err = None
        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(
                    self.url, data=payload, headers={'Content-Type': 'application/json'}
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    d = json.loads(resp.read())
                msg = ((d.get('choices') or [{}])[0].get('message') or {})
                content = msg.get('content') or ''
                usage = d.get('usage') or {}
                keep: set[int] = set()
                try:
                    obj = json.loads(content)
                    keep = {int(x) for x in (obj.get('keep') or [])}
                except Exception:
                    keep = {int(x) for x in _NUM_RE.findall(content)}
                keep = {n for n in keep if lo <= n <= hi}
                return (
                    keep,
                    int(usage.get('prompt_tokens') or 0),
                    int(usage.get('completion_tokens') or 0),
                    None,
                )
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                last_err = f'{type(exc).__name__}: {exc}'
                if attempt < self.retries:
                    time.sleep(1.0 + attempt)
        return set(), 0, 0, last_err

    def compress(self, text: str, ctx: ObsContext) -> CompressResult:
        t0 = time.time()
        origin = count_tokens(text)
        query = (ctx.focus_question or '').strip() or derived_query(ctx)
        lines = text.splitlines()
        keep: set[int] = set()
        p_tok = c_tok = 0
        err = None
        for start in range(0, len(lines), self.max_lines):
            seg = lines[start : start + self.max_lines]
            lo, hi = start + 1, start + len(seg)
            numbered = '\n'.join(f'{i}: {s}' for i, s in enumerate(seg, start=lo))
            k, p, c, e = self._ask(numbered, query, lo, hi)
            keep |= k
            p_tok += p
            c_tok += c
            if e:
                err = e
                break
        if err or not keep:
            return CompressResult(
                text=text, origin_tokens=origin, kept_tokens=origin,
                aux_prompt_tokens=p_tok, aux_completion_tokens=c_tok,
                latency_ms=(time.time() - t0) * 1000, backend=self.name,
                error=(f'self-prune 失败: {err}' if err else None),
                extra={'kept_nothing': not keep and not err},
            )
        out = _render(lines, keep)
        return CompressResult(
            text=out,
            origin_tokens=origin,
            kept_tokens=count_tokens(out),
            aux_prompt_tokens=p_tok,
            aux_completion_tokens=c_tok,
            latency_ms=(time.time() - t0) * 1000,
            backend=self.name,
            extra={'kept_lines': len(keep), 'total_lines': len(lines)},
        )


def _render(lines: list[str], keep: set[int]) -> str:
    """保留行原样输出,被丢掉的连续段折叠成一行标注。"""
    out: list[str] = []
    gap = 0
    for i, s in enumerate(lines, start=1):
        if i in keep:
            if gap:
                out.append(f'... ({gap} lines omitted) ...')
                gap = 0
            out.append(s)
        else:
            gap += 1
    if gap:
        out.append(f'... ({gap} lines omitted) ...')
    return '\n'.join(out)
