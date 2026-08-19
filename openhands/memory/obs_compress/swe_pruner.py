"""SWE-Pruner 基线客户端(arXiv 2601.16746)。

对端是官方 0.6B 剪枝模型的 HTTP 服务(servers/swe_pruner_server.py 拉起),
契约与官方 online_serving 一致:POST /prune {query, code, threshold}
返回 {score, pruned_code, origin_token_cnt, left_token_cnt, model_input_token_cnt, ...}。

查询模式(SWE_PRUNER_QUERY_MODE):
  derived (缺省) 由任务描述 + 当前工具调用拼出关注问题,不改工具签名。
                 这样基线臂与其它臂的动作空间完全相同,是更受控的对照。
  schema         忠实于原论文:关注问题由智能体自己写在工具参数 context_focus_question 里,
                 参数缺失就不剪枝(计入 skipped_no_question,用来报"触发率")。

失败时按原论文的做法原样透传并记下错误,绝不静默退化成别的方法。
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from .base import CompressResult, Compressor, ObsContext, count_tokens, derived_query


class SwePrunerCompressor(Compressor):
    name = 'swe-pruner'

    def __init__(self) -> None:
        self.url = os.environ.get('SWE_PRUNER_URL', 'http://127.0.0.1:8700/prune')
        self.threshold = float(os.environ.get('SWE_PRUNER_THRESHOLD', '0.5') or 0.5)
        self.query_mode = os.environ.get('SWE_PRUNER_QUERY_MODE', 'derived').strip().lower()
        self.timeout = float(os.environ.get('OBS_HTTP_TIMEOUT', '180') or 180)
        self.retries = int(os.environ.get('OBS_HTTP_RETRIES', '2') or 2)

    def build_query(self, ctx: ObsContext) -> str:
        if self.query_mode == 'schema':
            return (ctx.focus_question or '').strip()
        return derived_query(ctx)

    def compress(self, text: str, ctx: ObsContext) -> CompressResult:
        t0 = time.time()
        origin = count_tokens(text)
        query = self.build_query(ctx)
        if not query:
            # schema 模式下智能体没写关注问题:原样保留,并记一次未触发
            return CompressResult(
                text=text, origin_tokens=origin, kept_tokens=origin,
                latency_ms=(time.time() - t0) * 1000, backend=self.name,
                extra={'skipped_no_question': True},
            )

        payload = json.dumps({'query': query, 'code': text, 'threshold': self.threshold}).encode()
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
                        'score': d.get('score'),
                        'pruner_origin_tokens': d.get('origin_token_cnt'),
                        'pruner_left_tokens': d.get('left_token_cnt'),
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
            error=f'swe-pruner 服务不可用: {last_err}',
        )
