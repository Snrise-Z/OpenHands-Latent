"""对比基线的公共骨架:压缩结果、观察上下文、词元计数、按内容缓存。

设计要点:
  - 所有基线都挂在与 latent 策略相同的调用点(ConversationMemory 序列化消息之后),
    这样脚手架、工具集、提示词、判分链路完全一致,臂与臂之间只差"观察被替换成什么"。
  - OpenHands 每一轮都会把整段历史重新渲染一遍。若不缓存,一条观察会被反复剪枝
    (200 轮 x 数十条观察 = 上万次调用),既不忠实于"到达即剪枝"的原始设计,成本也不可接受。
    因此按 (后端, 查询, 观察内容) 哈希缓存,一条观察在一次 rollout 内只压缩一次。
  - 成本口径:辅助模型的输入/输出词元必须逐次记账,否则"省了多少词元"会被高估。
"""

from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass, field


@dataclass
class ObsContext:
    """一条工具观察的上下文,供各后端构造查询。"""

    task_statement: str = ''      # 任务描述(首条用户消息)
    tool_name: str = ''           # 产生该观察的工具名
    tool_arguments: str = ''      # 该工具调用的参数(原始 JSON 串)
    focus_question: str = ''      # 智能体自己写的关注问题(仅 schema 查询模式下非空)
    index: int = -1               # 该观察在全部工具观察中的序号
    total: int = 0                # 工具观察总数


@dataclass
class CompressResult:
    """一次压缩的结果与成本。"""

    text: str
    origin_tokens: int = 0
    kept_tokens: int = 0
    aux_prompt_tokens: int = 0        # 辅助模型的输入词元
    aux_completion_tokens: int = 0    # 辅助模型的输出词元
    latency_ms: float = 0.0
    backend: str = ''
    cached: bool = False
    error: str | None = None
    extra: dict = field(default_factory=dict)


class _Tokenizer:
    """解码器分词器。口径与成本表一致,拿不到就退化成字符数估计。"""

    _lock = threading.Lock()
    _tok = None
    _tried = False

    @classmethod
    def count(cls, text: str) -> int:
        if not text:
            return 0
        tok = cls._get()
        if tok is None:
            return max(1, len(text) // 4)
        try:
            if hasattr(tok, 'encode') and not hasattr(tok, 'convert_ids_to_tokens'):
                return len(tok.encode(text).ids)          # tokenizers.Tokenizer
            return len(tok(text, add_special_tokens=False)['input_ids'])   # transformers
        except Exception:
            return max(1, len(text) // 4)

    @classmethod
    def _get(cls):
        if cls._tried:
            return cls._tok
        with cls._lock:
            if cls._tried:
                return cls._tok
            cls._tried = True
            path = os.environ.get('OBS_TOKENIZER', '').strip()
            if path:
                cls._tok = _load_tokenizer(path)
        return cls._tok


def _load_tokenizer(path: str):
    """优先 transformers,退而用 tokenizers 直接读 tokenizer.json。

    rollout 进程所在的 OpenHands 环境通常没装 transformers(它不需要 torch),
    但一定有 tokenizers,所以两条路都要留,拿不到就退化成字符数估计。
    """
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    except Exception:
        pass
    try:
        from tokenizers import Tokenizer

        f = path if path.endswith('.json') else os.path.join(path, 'tokenizer.json')
        return Tokenizer.from_file(f)
    except Exception:
        return None


count_tokens = _Tokenizer.count


class ResultCache:
    """按内容哈希缓存压缩结果,保证一条观察只压缩一次。"""

    def __init__(self) -> None:
        self._d: dict[str, CompressResult] = {}
        self._lock = threading.Lock()

    @staticmethod
    def key(backend: str, query: str, text: str) -> str:
        h = hashlib.sha1()
        h.update(backend.encode('utf-8', 'ignore'))
        h.update(b'\x00')
        h.update(query.encode('utf-8', 'ignore'))
        h.update(b'\x00')
        h.update(text.encode('utf-8', 'ignore'))
        return h.hexdigest()

    def get(self, k: str) -> CompressResult | None:
        with self._lock:
            r = self._d.get(k)
        if r is None:
            return None
        # 命中时复制一份并标记,避免调用方改坏缓存里的对象
        return CompressResult(
            text=r.text, origin_tokens=r.origin_tokens, kept_tokens=r.kept_tokens,
            aux_prompt_tokens=r.aux_prompt_tokens, aux_completion_tokens=r.aux_completion_tokens,
            latency_ms=r.latency_ms, backend=r.backend, cached=True, error=r.error,
            extra=dict(r.extra),
        )

    def put(self, k: str, r: CompressResult) -> None:
        with self._lock:
            self._d[k] = r


class Compressor:
    """后端接口。实现只需覆写 compress。"""

    name = 'base'

    def compress(self, text: str, ctx: ObsContext) -> CompressResult:  # pragma: no cover
        raise NotImplementedError


def derived_query(ctx: ObsContext, max_task_chars: int = 400) -> str:
    """在不改工具签名的前提下构造关注问题。

    原始设计是让智能体自己在工具参数里写 context_focus_question。那样会改动作空间,
    使基线臂与其它臂在"智能体接口"上也不同;这里提供一个不改接口的替代:
    用任务描述 + 当前这次工具调用拼出查询。两种模式都保留,由 SWE_PRUNER_QUERY_MODE 选择。
    """
    task = (ctx.task_statement or '').strip().replace('\n', ' ')
    if len(task) > max_task_chars:
        task = task[:max_task_chars] + '...'
    args = (ctx.tool_arguments or '').strip().replace('\n', ' ')
    if len(args) > 300:
        args = args[:300] + '...'
    parts = []
    if task:
        parts.append(f'Task: {task}')
    if ctx.tool_name:
        parts.append(f'Command: {ctx.tool_name} {args}'.strip())
    parts.append('Which parts of the following output are needed to make progress on the task?')
    return '\n'.join(parts)
