"""LongCodeZip 基线客户端(ASE 2025)。

对端是 servers/longcodezip_server.py:把官方 CodeCompressor 包成 HTTP 服务,
只跑粗粒度阶段(按函数切块、用相对于查询的条件困惑度排序、按预算选块,rank_only=True),
与 SWE-Pruner Pro 复现该基线时的用法一致。

之所以要做成服务:rollout 是几十上百个进程并行,不可能每个进程各加载一份排序模型。
契约与 SWE-Pruner 服务保持一致,便于两条臂共用同一套计量与报表。
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from .base import CompressResult, Compressor, ObsContext, count_tokens, derived_query


class LongCodeZipCompressor(Compressor):
    name = 'longcodezip'

    def __init__(self) -> None:
        self.url = os.environ.get('LONGCODEZIP_URL', 'http://127.0.0.1:8701/prune')
        self.rate = float(os.environ.get('LONGCODEZIP_RATE', '0.5') or 0.5)
        # 只跑粗粒度阶段(原论文两阶段里的第一阶段)。在智能体观察上,
        # 两阶段的排序模型开销高达观察本身的 160 倍,粗粒度则几乎不压缩,两者都要能选。
        self.rank_only = os.environ.get('LONGCODEZIP_RANK_ONLY', '1').strip().lower() in ('1', 'true', 'yes')
        self.timeout = float(os.environ.get('OBS_HTTP_TIMEOUT', '300') or 300)
        self.retries = int(os.environ.get('OBS_HTTP_RETRIES', '2') or 2)

    def compress(self, text: str, ctx: ObsContext) -> CompressResult:
        t0 = time.time()
        origin = count_tokens(text)
        query = (ctx.focus_question or '').strip() or derived_query(ctx)
        payload = json.dumps({'query': query, 'code': text, 'rate': self.rate,
                              'rank_only': self.rank_only}).encode()
        last_err = None
        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(
                    self.url, data=payload, headers={'Content-Type': 'application/json'}
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    d = json.loads(resp.read())
                pruned = d.get('pruned_code')
                if not isinstance(pruned, str) or not pruned.strip():
                    pruned = text
                return CompressResult(
                    text=pruned,
                    origin_tokens=origin,
                    kept_tokens=count_tokens(pruned),
                    aux_prompt_tokens=int(d.get('model_input_token_cnt') or 0),
                    aux_completion_tokens=0,
                    latency_ms=(time.time() - t0) * 1000,
                    backend=self.name,
                    extra={
                        'ranker_origin_tokens': d.get('origin_token_cnt'),
                        'ranker_left_tokens': d.get('left_token_cnt'),
                        'selected_chunks': d.get('selected_chunks'),
                        'rank_only': self.rank_only,
                        'attempt': attempt,
                    },
                )
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                last_err = f'{type(exc).__name__}: {exc}'
                if attempt < self.retries:
                    time.sleep(1.0 + attempt)
        return CompressResult(
            text=text, origin_tokens=origin, kept_tokens=origin,
            latency_ms=(time.time() - t0) * 1000, backend=self.name,
            error=f'longcodezip 服务不可用: {last_err}',
        )
