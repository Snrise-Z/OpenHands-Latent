"""无训练的 harness 闸门:工具名映射提示、finish 空改动拦截、掉出工具通道催动。

三个闸门共同的设计约束(来自 500x7 主实验 2543 条轨迹的普查):

  只提示不代做   一律不替模型执行任何动作 —— 工具名映射只改写错误消息(不转译调用),
                 finish 拦截只退回一条说明(一次重试语义,第二次 finish 放行),
                 通道催动只把"请继续"升级成"必须调用工具"。
                 提示错的代价是线性的,代做错的代价是破坏性的,而模型 97% 的轨迹
                 不跑测试,发现不了被代做坏的东西。
  只在失败后介入 三个闸门都只挂在"已经失败/已经无效"的动作之后
                 (未注册的调用、空改动的 finish、没有接收者的纯文本),
                 不触碰任何成功路径,错误触发面结构性为零。
  每次触发记录   guard_record 打 [harness-guard] 日志;设 OH_GUARD_LOG=<路径> 时
                 另落一行 JSONL,事后按臂统计触发率 —— 触发率本身就是行为指标。
                 (教训:旧模糊层静默缺席一整轮,就是因为没有任何存在性记录。)

开关(缺省全开;不部署到正在跑的评测树,下一轮整轮生效):
  OH_TOOL_ALIAS=0     关工具名映射提示
  OH_FINISH_GATE=0    关 finish 空改动拦截
  OH_CHANNEL_NUDGE=0  关通道催动;OH_CHANNEL_NUDGE_K 连续纯文本阈值(缺省 2)
"""

import json
import os
import re
import time

from openhands.core.logger import openhands_logger as logger


def guard_record(event: str, **fields) -> None:
    """每次闸门触发记录一条;任何异常都吞掉,记录失败不能影响主流程。"""
    rec = {'ts': round(time.time(), 3), 'event': event, **fields}
    try:
        logger.info('[harness-guard] ' + json.dumps(rec, ensure_ascii=False))
    except Exception:
        pass
    log_path = os.environ.get('OH_GUARD_LOG', '').strip()
    if log_path:
        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        except Exception:
            pass


# ------------------------------------------------------------------ 工具名映射
# 普查:26 次未注册调用全部是两种混淆 —— 把 str_replace_editor 的子命令当独立工具
# (view 12 次),或把 shell 命令当工具(grep 14 次)。只改写错误消息,不转译调用。
_EDITOR_SUBCMDS = ('view', 'create', 'str_replace', 'insert', 'undo_edit')
_SHELL_WORDS = ('grep', 'find', 'ls', 'cat', 'sed', 'awk', 'head', 'tail',
                'python', 'python3', 'bash', 'sh', 'shell', 'terminal', 'run',
                'pytest', 'pip', 'git')


def _fmt_arg(v) -> str:
    s = repr(v)
    return s if len(s) <= 120 else s[:120] + '…'


def alias_hint(tool_name: str, arguments: dict) -> str | None:
    """未注册工具名若是已知混淆,返回带正确调用示例的错误消息;否则 None。

    示例里回显模型自己的参数 —— view 的参数本来就是 {'path': ...},grep 的参数
    本来就是 {'command': ...},形状已经是对的,重试只需照抄。
    """
    if os.environ.get('OH_TOOL_ALIAS', '1') == '0':
        return None
    name = (tool_name or '').strip().lower()
    args = arguments if isinstance(arguments, dict) else {}
    if name in _EDITOR_SUBCMDS:
        shown = ', '.join(f'{k}={_fmt_arg(v)}' for k, v in list(args.items())[:4])
        example = f"str_replace_editor(command='{name}'"
        if shown:
            example += f', {shown}'
        example += ')'
        return (
            f'Tool `{name}` is not registered. `{name}` is a command of the '
            f'str_replace_editor tool, not a standalone tool. Retry as: {example}'
        )
    if name in _SHELL_WORDS:
        cmd = args.get('command')
        if not isinstance(cmd, str) or not cmd.strip():
            cmd = f'{name} ...'
        return (
            f'Tool `{name}` is not registered. Shell commands must be run through '
            f'the execute_bash tool. Retry as: execute_bash(command={_fmt_arg(cmd)})'
        )
    return None


# ------------------------------------------------------------ finish 空改动拦截
# 普查:73 条轨迹零编辑尝试、空补丁,最后发言却宣称 "I've successfully implemented ..."。
# SWE-bench 里空改动永远不是正确答案,所以拦截条件客观;一次重试语义把对
# 如实弃题者的代价封顶在一轮。
#
# "有没有改动"用历史近似(控制器拿不到容器里的 git):宽松认定 —— 任何成功的
# 编辑器改动,或任何退出码为 0 且命令形似写文件的 bash,都算"有改动"。
# 宁可漏拦(bash 改的文件没识别出来 -> 不拦),不可错拦。
_MUT_RE = re.compile(
    r'(^|[;&|]\s*|\s)('
    r'sed\s+(-\S*\s+)*-i'          # sed -i / sed -E -i
    r'|tee\s|dd\s|truncate\s|rsync\s|touch\s'
    r'|cp\s|mv\s|ln\s'
    r'|patch\s|applypatch'
    r'|git\s+(apply|checkout|restore|stash|reset|cherry-pick|merge|revert|mv|rm)'
    r')'
)


def _cmd_mutates(cmd: str) -> bool:
    c = cmd or ''
    if '>' in c or '<<' in c:      # 重定向与 heredoc,宽松算写文件
        return True
    return bool(_MUT_RE.search(c))


def _edit_obs_success(content: str) -> bool:
    c = (content or '').lstrip()
    return c.startswith(('The file', 'File created'))


def history_has_file_mutation(history) -> bool:
    """历史里是否出现过任何"像是改了文件"的成功动作(宽松口径)。"""
    from openhands.events.observation import (
        CmdOutputObservation,
        FileEditObservation,
    )

    for ev in history or []:
        try:
            if isinstance(ev, FileEditObservation):
                if _edit_obs_success(getattr(ev, 'content', '')):
                    return True
            elif isinstance(ev, CmdOutputObservation):
                code = getattr(ev, 'exit_code', None)
                if code == 0 and _cmd_mutates(getattr(ev, 'command', '')):
                    return True
        except Exception:
            continue
    return False


FINISH_GATE_MSG = (
    'Your finish was not accepted: no file in the repository appears to have been '
    'modified in this session (no successful edit and no file-mutating command). '
    'If you believe you made changes, they did not succeed - re-check the results '
    'of your edit actions. Make the required code change first. If you are truly '
    'unable to solve the task, call finish again and it will be accepted.'
)


def finish_gate_should_block(history, already_fired: bool) -> bool:
    if already_fired or os.environ.get('OH_FINISH_GATE', '1') == '0':
        return False
    return not history_has_file_mutation(history)


# ------------------------------------------------------------ 掉出工具通道催动
# 普查:agent 无工具调用的纯文本消息在压缩臂是明文臂的 4 倍(hardlast1 394 次
# vs allhard 95 次),卡循环第二大循环体就是 message x4 + recall x2(183 条)。
# 评测无人值守,纯文本没有任何接收者,催动没有误伤面。
def channel_nudge(history) -> str | None:
    """末尾连续 K 条 agent 纯文本消息(允许与用户消息/观察交错)时返回催动文本。"""
    if os.environ.get('OH_CHANNEL_NUDGE', '1') == '0':
        return None
    try:
        k = int(os.environ.get('OH_CHANNEL_NUDGE_K', '2') or 2)
    except ValueError:
        k = 2
    if not history:
        return None
    from openhands.events.action import Action, MessageAction

    count = 0
    for ev in list(history)[-40:][::-1]:
        try:
            src = getattr(ev, 'source', None)
            if isinstance(ev, MessageAction) and src == 'agent':
                count += 1
                continue
            if isinstance(ev, Action) and src == 'agent':
                break              # 最近一次真动作之前的消息不算
        except Exception:
            continue
    if count < k:
        return None
    guard_record('channel-nudge', consecutive=count)
    return (
        f'You have sent {count} plain messages in a row without calling any tool. '
        'There is no human in this session to read them. You MUST proceed with '
        'exactly one tool call now (execute_bash or str_replace_editor): convert '
        'your plan into a concrete action.'
    )
