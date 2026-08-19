"""给工具签名加上 SWE-Pruner 原论文要求的 context_focus_question 字段。

原论文的做法是让智能体自己在工具参数里写出"这次要关注什么",剪枝器再按这个问题筛行。
这会改动作空间,所以默认关闭:只有 OBS_TOOL_FOCUS_QUESTION=1 的臂(sweprunerpaper)才加,
其余臂的工具签名与我们自己的臂逐字节相同。

字段是可选的:智能体不写就不剪枝,计量里记 skipped_no_question,用来报触发率。
"""

from __future__ import annotations

import os

_DESC = (
    'Optional. A complete, self-contained question describing what you need from this '
    'command output (for example "Where is autoescape read in Engine.render_to_string?"). '
    'Long outputs are filtered against this question before you see them. '
    'Do not use keywords or phrases, write a full question.'
)


def enabled() -> bool:
    return os.environ.get('OBS_TOOL_FOCUS_QUESTION', '0').strip().lower() in ('1', 'true', 'yes')


def maybe_add_focus_question(tool):
    """就地给工具参数加上可选字段。拿不到预期结构就原样返回,绝不因此让智能体起不来。"""
    if not enabled():
        return tool
    try:
        props = tool['function']['parameters']['properties']
        if 'context_focus_question' not in props:
            props['context_focus_question'] = {'type': 'string', 'description': _DESC}
    except Exception:
        pass
    return tool
