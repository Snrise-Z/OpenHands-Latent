"""观察通道策略: 把工具观察按策略包裹为 latent 记忆区域。

服务端(vLLM 多模态插件)在聊天模板渲染后提取 <|memory_start|>…<|memory_end|>
包裹的区域, 压缩为 latent 表示喂给解码器; 客户端(本模块)只负责决定"哪些观察
走压缩通道", 对 LLM 端点而言一切仍是标准 OpenAI 聊天补全。

策略(环境变量配置, 缺省完全保持上游行为):
  LATENT_OBS_POLICY   none(缺省) | all-latent | lastk-hard
    - none       : 不做任何包裹(上游原样);
    - all-latent : 所有长度 >= LATENT_MIN_OBS_CHARS 的工具观察包裹为压缩区域;
    - lastk-hard : 最后 LATENT_HARD_WINDOW 条工具观察(按消息位置)保持明文,
                   其余同 all-latent。包裹在每次序列化时按当前位置重新计算,
                   因此一条观察会在滑出窗口时自然从明文降级为压缩形态。
  LATENT_HARD_WINDOW  滑窗大小(缺省 2)
  LATENT_MIN_OBS_CHARS 压缩下限(缺省 128; 短观察走明文, 压缩收益不抵占位开销)

设计依据(LCLM 仓库 adapter_realign/v1_lastk2_failure_modes.md):最后 k 条观察
保持明文可消除压缩通道下 str_replace 逐字引用的死锁(成功编辑 0 -> 10, 超过全
明文基线), 剩余失败与通道无关或属"远期压缩伤聚合"的已知残余。
"""

import os
import re

from openhands.core.message import Message, TextContent

MEMORY_OPEN = '<|memory_start|>'
MEMORY_CLOSE = '<|memory_end|>'
_MARKER_RE = re.compile(re.escape(MEMORY_OPEN) + '|' + re.escape(MEMORY_CLOSE))


def _policy() -> tuple[str, int, int]:
    policy = os.environ.get('LATENT_OBS_POLICY', 'none').strip().lower()
    hard_window = int(os.environ.get('LATENT_HARD_WINDOW', '2') or 2)
    min_chars = int(os.environ.get('LATENT_MIN_OBS_CHARS', '128') or 128)
    return policy, hard_window, min_chars


def apply_latent_observation_policy(messages: list[Message]) -> list[Message]:
    """按策略把工具观察的文本内容包裹为压缩区域。

    仅处理 role == 'tool' 的消息(函数调用模式下的工具结果)。观察文本中的
    杂散记忆标记一律剥除后再决定是否包裹, 保证标记只由本模块引入且不嵌套。
    """
    policy, hard_window, min_chars = _policy()
    if policy == 'none':
        return messages

    tool_indices = [i for i, m in enumerate(messages) if m.role == 'tool']
    hard = set(tool_indices[-hard_window:]) if policy == 'lastk-hard' and hard_window > 0 else set()

    for i in tool_indices:
        msg = messages[i]
        new_content = []
        for item in msg.content:
            if isinstance(item, TextContent):
                text = _MARKER_RE.sub('', item.text or '')
                if i not in hard and len(text) >= min_chars:
                    text = MEMORY_OPEN + text + MEMORY_CLOSE
                new_content.append(TextContent(text=text))
            else:
                new_content.append(item)
        messages[i] = msg.model_copy(update={'content': new_content})
    return messages
