"""截断基线:不额外用任何模型,直接丢掉滑窗之外的观察内容。

这是"潜在记忆到底带没带信息"的最干净对照 —— 它与 Hard Last K 只差一点:
窗口之外的观察是被压成潜在记忆,还是直接消失。成本为零,不需要任何服务。

两种模式(OBS_TRUNCATE_MODE):
  drop     (缺省)整段替换成一行占位,只留行数与词元数。产生该观察的工具调用仍在
           助手消息里明文可见,所以占位里不重复命令,避免给这一臂额外信息。
  headtail 保留首 OBS_TRUNCATE_HEAD 行与末 OBS_TRUNCATE_TAIL 行,中间折叠。
           更贴近工程上常见的做法,作为更强的截断对照。
"""

from __future__ import annotations

import os
import time

from .base import CompressResult, Compressor, ObsContext, count_tokens


class TruncateCompressor(Compressor):
    name = 'truncate'

    def __init__(self) -> None:
        self.mode = os.environ.get('OBS_TRUNCATE_MODE', 'drop').strip().lower()
        self.head = int(os.environ.get('OBS_TRUNCATE_HEAD', '20') or 20)
        self.tail = int(os.environ.get('OBS_TRUNCATE_TAIL', '20') or 20)

    def compress(self, text: str, ctx: ObsContext) -> CompressResult:
        t0 = time.time()
        lines = text.splitlines()
        origin = count_tokens(text)
        if self.mode == 'headtail' and len(lines) > self.head + self.tail + 1:
            omitted = len(lines) - self.head - self.tail
            kept = (
                lines[: self.head]
                + [f'... [{omitted} lines omitted] ...']
                + (lines[-self.tail:] if self.tail else [])
            )
            out = '\n'.join(kept)
        else:
            out = f'[Earlier tool output omitted: {len(lines)} lines, {origin} tokens]'
        return CompressResult(
            text=out,
            origin_tokens=origin,
            kept_tokens=count_tokens(out),
            latency_ms=(time.time() - t0) * 1000,
            backend=self.name,
            extra={'mode': self.mode},
        )
