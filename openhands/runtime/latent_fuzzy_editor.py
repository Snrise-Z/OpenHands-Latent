"""str_replace 的模糊匹配:空白归一化唯一命中才自动改,其余只提示。

上游 OHEditor.str_replace 要求 old_str 逐字唯一出现。500 题 x 7 策略主实验实测
9140 次 str_replace 里 82% 被拒,其中 93% 是 old_str 逐字没匹配上;对 250 条真实
失败样本的标定:最像候选 94% 指向正确位置、6% 指错。提示指错的代价是线性的
(模型多试一轮),自动改错的代价是破坏性的(静默写进错误位置,而智能体只有 3% 的
轨迹跑过测试,发现不了)。据此定策:

  自动层  仅当 old_str 与文件某个连续行窗口在逐行去首尾空白后完全一致、且该窗口
          在文件中唯一(纯缩进/行尾差异)。写回前有三道保护:
            换行边界  替换文本的结尾换行与被替换窗口保持一致;
            缩进重排  对齐行沿用文件真实缩进,改动/新增行按锚点字面缩进增减
                      (制表符文件照抄制表符);
            语法闸门  .py 编辑前能编译、编辑后不能就不写(OH_FUZZY_SYNTAX_GUARD=0 关)。
          成功消息带 [fuzzy-auto] 标记并回显被替换的原文。

  提示层  其余一律不改文件,只在拒绝消息里给可操作的提示(带 [fuzzy-hint] 标记):
            唯一的相近候选      回显候选原文(带行号)+ 与 old_str 的逐行差异;
            多个难分候选        只给各候选行号,不给内容,避免二选一指错;
            找不到相近内容      提示重新查看文件 —— 引用的内容可能根本不在文件里。
          相近判据:半全局编辑距离(Sellers 动态规划) <= FUZZY_MAX_DISTANCE(缺省
          0.40,约相当于相似度 0.60);歧义判据:不相交次优与最优差 < FUZZY_MIN_MARGIN
          (缺省 0.05)。

  记录    每次决策(auto / hint-* / guard-blocked)都打一条 [fuzzy-editor] 日志;
          设 OH_FUZZY_LOG=<路径> 时另落一行 JSONL,事后可逐臂统计。
          这一条是被"兜底静默缺席"逼出来的:此前模糊层只接在 action_execution_server
          (Docker 运行时)上,CLIRuntime/UDockerRuntime 路径一直走上游裸 OHEditor,
          9140 次尝试没有任何日志能看出兜底不在场。

环境变量 OH_FUZZY_STR_REPLACE=0 整体关闭,行为回到上游。
"""

import difflib
import json
import os
import re
import time
from pathlib import Path

from openhands.core.logger import openhands_logger as logger
from openhands_aci.editor.editor import OHEditor
from openhands_aci.editor.exceptions import EditorToolParameterInvalidError
from openhands_aci.editor.exceptions import ToolError
from openhands_aci.editor.results import CLIResult

try:
    import numpy as _np
except ImportError:  # 第三层退化为不可用, 第一/二层不受影响
    _np = None

_MIN_PATTERN_CHARS = 40


def _substantive_len(s: str) -> int:
    return len(re.sub(r'\s+', '', s))


def _strip_lines(s: str) -> list[str]:
    return [ln.strip() for ln in s.splitlines()]


def _indent_of(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _indent_unit(lines: list[str]) -> str:
    """从一组行里推断"一级缩进"的字面写法。

    含制表符就用制表符(Makefile 之类靠制表符表达语义的文件必须照抄),
    否则取最小的正缩进宽度, 都没有就退回 4 个空格。
    """
    indents = [_indent_of(ln) for ln in lines if ln.strip()]
    if any('\t' in ind for ind in indents):
        return '\t'
    widths = sorted({len(ind) for ind in indents if ind})
    return ' ' * widths[0] if widths else '    '


def _reindent_to_window(window: str, old_str: str, new_str: str) -> str:
    """把 new_str 的缩进对齐到文件窗口的真实缩进。

    仅在窗口与引用行数一致时生效(第二层总是成立; 第三层的字符级窗口可能不成立,
    那时原样返回, 由语法闸门兜底)。
    """
    w_lines = window.splitlines()
    o_lines = old_str.splitlines()
    n_lines = new_str.splitlines()
    if not n_lines or len(w_lines) != len(o_lines):
        return new_str

    out: list[str | None] = [None] * len(n_lines)
    matcher = difflib.SequenceMatcher(
        a=[ln.strip() for ln in o_lines], b=[ln.strip() for ln in n_lines]
    )
    for tag, i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag != 'equal':
            continue  # 改动行留给下面按锚点定位
        for k in range(j2 - j1):
            src = n_lines[j1 + k]
            out[j1 + k] = (
                _indent_of(w_lines[i1 + k]) + src.strip() if src.strip() else src
            )

    for j, val in enumerate(out):
        if val is not None:
            continue
        src = n_lines[j]
        if not src.strip():
            out[j] = src
            continue
        anchor = next(
            (
                k
                for k in range(j - 1, -1, -1)
                if out[k] is not None and n_lines[k].strip()
            ),
            None,
        )
        if anchor is None:
            anchor = next(
                (
                    k
                    for k in range(j + 1, len(n_lines))
                    if out[k] is not None and n_lines[k].strip()
                ),
                None,
            )
        if anchor is None:
            out[j] = src
            continue
        # 改动行/新增行: 以锚点在文件里的**字面**缩进为基准, 按模型给的相对层级增减。
        # 早先这里合成空格(' ' * n), 在制表符缩进的文件里会造出制表符与空格混用
        # —— Python 抛 TabError(被闸门拦下, 合法编辑失败), Makefile 则直接静默改坏。
        base = _indent_of(out[anchor])
        delta = len(_indent_of(src)) - len(_indent_of(n_lines[anchor]))
        if delta == 0:
            out[j] = base + src.lstrip()
            continue
        model_unit = max(1, len(_indent_unit(n_lines)))
        file_unit = _indent_unit(w_lines)
        levels = int(round(delta / model_unit))
        if levels >= 0:
            out[j] = base + file_unit * levels + src.lstrip()
        else:
            keep = max(0, len(base) + levels * len(file_unit))
            out[j] = base[:keep] + src.lstrip()

    return '\n'.join(out) + ('\n' if new_str.endswith('\n') else '')


_IDENTICAL_HINT = (
    'No replacement was performed: the `old_str` and `new_str` you provided are '
    'byte-for-byte identical, so there is nothing to change. [fuzzy-hint] If you meant to '
    'change indentation or whitespace, note that BOTH strings you sent carry the same '
    'indentation. Re-read the region with `view` first, copy the file text verbatim into '
    '`old_str` (watch for tabs vs spaces), and write the corrected text into `new_str`. '
    'When re-indenting a block, include a few surrounding lines in both strings so that they '
    'actually differ. Do not fall back to shell commands such as `sed -i` or `rm` to edit '
    'source files: this workspace has no git history, so such edits cannot be undone.'
)


def _record(event: str, path='', **fields) -> None:
    """每次模糊层决策都记录:logger 一条,OH_FUZZY_LOG 设了再落一行 JSONL。

    任何异常都吞掉 —— 记录失败不能影响编辑本身。
    """
    rec = {'ts': round(time.time(), 3), 'event': event, 'path': str(path), **fields}
    try:
        logger.info('[fuzzy-editor] ' + json.dumps(rec, ensure_ascii=False))
    except Exception:
        pass
    log_path = os.environ.get('OH_FUZZY_LOG', '').strip()
    if log_path:
        try:
            with open(log_path, 'a') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        except Exception:
            pass


class FuzzyOHEditor(OHEditor):
    """精确优先、失败后分层回退的 str_replace。"""

    def __call__(self, *, command=None, path=None, old_str=None, new_str=None, **kwargs):
        """把上游"新旧串完全相同"的通用拒绝换成可操作指引。

        上游在参数分发处(调用 str_replace 之前)就抛 EditorToolParameterInvalidError,
        原文只说两者必须不同,没告诉模型下一步怎么办。Verified 20 题实测:模型连撞五次后
        转向 bash 的 sed -i / rm 硬改源文件,而 SWE-bench 容器剥掉了 .git,破坏不可逆
        (两题因此把源文件删空或削残)。这里给出具体做法,把它引回编辑器通道。
        """
        if command == "str_replace" and old_str is not None and new_str == old_str:
            _record("hint-identical-strs", path, chars=len(old_str or ""))
            raise EditorToolParameterInvalidError("new_str", new_str, _IDENTICAL_HINT)
        return super().__call__(
            command=command, path=path, old_str=old_str, new_str=new_str, **kwargs
        )

    def str_replace(
        self,
        path: Path,
        old_str: str,
        new_str: str | None,
        enable_linting: bool,
        encoding: str = 'utf-8',
    ) -> CLIResult:
        try:
            return super().str_replace(path, old_str, new_str, enable_linting)
        except ToolError as err:
            if 'did not appear verbatim' not in str(err):
                raise  # 歧义等其他错误原样透传
            if os.environ.get('OH_FUZZY_STR_REPLACE', '1') == '0':
                raise
            result = self._fuzzy_str_replace(
                path, old_str, new_str or '', enable_linting, err
            )
            if result is None:
                raise
            return result

    # ------------------------------------------------------------- 第二层
    def _match_normalized_lines(self, file_content: str, old_str: str):
        """唯一的行级空白归一化窗口 -> (start_idx, end_idx, indent_delta)."""
        f_lines = file_content.splitlines(keepends=True)
        f_stripped = [ln.strip() for ln in f_lines]
        p_stripped = _strip_lines(old_str)
        if not p_stripped:
            return None
        n, m = len(f_stripped), len(p_stripped)
        hits = [
            i for i in range(n - m + 1) if f_stripped[i : i + m] == p_stripped
        ]
        if len(hits) != 1:
            return None if not hits else ('AMBIGUOUS', hits)
        i = hits[0]
        start_idx = sum(len(ln) for ln in f_lines[:i])
        end_idx = start_idx + sum(len(ln) for ln in f_lines[i : i + m])
        # 窗口按整行计, 末行含换行; 引用末行不带换行时要把它留在原文里,
        # 否则替换会吞掉与下一行之间的换行符。
        if not old_str.endswith('\n') and file_content[end_idx - 1 : end_idx] == '\n':
            end_idx -= 1
        # 缩进偏移是否恒定 —— 只用于给模型的提示文案, 重排一律走 _reindent_to_window
        deltas = set()
        p_lines = old_str.splitlines()
        for k in range(m):
            if p_stripped[k]:
                deltas.add(
                    (len(_indent_of(f_lines[i + k])), len(_indent_of(p_lines[k])))
                )
        offsets = {fi - pi for fi, pi in deltas}
        delta = offsets.pop() if len(offsets) == 1 else None
        return start_idx, end_idx, delta

    # ------------------------------------------------------------- 第三层
    @staticmethod
    def _semiglobal_ends(pattern: bytes, text: bytes):
        """返回 last-row 距离数组(终点为 j 的最优编辑距离), numpy 向量化。"""
        pa = _np.frombuffer(pattern, dtype=_np.uint8)
        ta = _np.frombuffer(text, dtype=_np.uint8)
        n = len(ta)
        prev = _np.zeros(n + 1, dtype=_np.int32)
        ar = _np.arange(n + 1, dtype=_np.int32)
        for i in range(1, len(pa) + 1):
            base = _np.empty(n + 1, dtype=_np.int32)
            base[0] = i
            base[1:] = _np.minimum(prev[1:] + 1, prev[:-1] + (ta != pa[i - 1]))
            # cur[j] = min_k<=j base[k] + (j-k) —— 前缀最小值技巧
            prev = _np.minimum.accumulate(base - ar) + ar
        return prev[1:]

    def _match_edit_distance(self, file_content: str, old_str: str):
        """唯一(带间隔门)的最小编辑距离窗口 -> (start_idx, end_idx, d1, |P|)."""
        if _np is None:
            return None
        # 只用于提示层的接受门(自动层不经过这里); 0.40 约相当于相似度 0.60,
        # 低于它的候选连提示价值都没有(250 条真实失败标定里 16% 落在这一档)。
        tau = float(os.environ.get('FUZZY_MAX_DISTANCE', '0.40') or 0.40)
        min_margin = float(os.environ.get('FUZZY_MIN_MARGIN', '0.05') or 0.05)
        pb = old_str.encode('utf-8', 'ignore')
        tb = file_content.encode('utf-8', 'ignore')
        m = len(pb)
        if m == 0 or len(tb) < m // 2:
            return None
        last = self._semiglobal_ends(pb, tb)
        j1 = int(last.argmin())
        d1 = int(last[j1])
        if d1 > tau * m:
            return None
        lo, hi = max(0, j1 - m // 2), min(len(tb), j1 + m // 2)
        mask = _np.ones(len(tb), dtype=bool)
        mask[lo:hi] = False
        if mask.any():
            d2 = int(last[mask].min())
            if (d2 - d1) < min_margin * m:
                j2 = int(_np.flatnonzero(mask)[last[mask].argmin()])
                lines = sorted(
                    {tb[:j].decode("utf-8", "ignore").count("\n") + 1 for j in (j1, j2)}
                )
                return ('AMBIGUOUS', lines)
        # 反向对齐求起点
        rev = self._semiglobal_ends(pb[::-1], tb[::-1])
        i1 = len(tb) - 1 - int(rev.argmin())
        start, end = max(0, min(i1, j1 + 1 - 1)), j1 + 1
        s_char = len(tb[:start].decode('utf-8', 'ignore'))
        e_char = len(tb[:end].decode('utf-8', 'ignore'))
        return s_char, e_char, d1, m

    # -------------------------------------------------------------- 闸门
    @staticmethod
    def _guard_syntax(path, before: str, after: str) -> None:
        """编辑前能编译、编辑后不能, 就不要写回。"""
        if not str(path).endswith('.py'):
            return
        if os.environ.get('OH_FUZZY_SYNTAX_GUARD', '1') == '0':
            return
        try:
            compile(before, str(path), 'exec')
        except SyntaxError:
            return  # 本来就不可编译, 不做判断
        try:
            compile(after, str(path), 'exec')
        except SyntaxError as e:
            raise ToolError(
                f'No replacement was performed. old_str did not appear verbatim; the '
                f'closest match in {path} was found, but applying new_str would break '
                f'the file: {e.msg} (line {e.lineno}). Re-quote old_str exactly as it '
                f'appears in the file, keeping the original indentation.'
            )

    # -------------------------------------------------------------- 决策
    def _fuzzy_str_replace(self, path, old_str, new_str, enable_linting, orig_err):
        file_content = self.read_file(path)

        # ---- 自动层: 空白归一化后逐行完全一致且窗口唯一(纯缩进/行尾差异) ----
        m2 = self._match_normalized_lines(file_content, old_str)
        if isinstance(m2, tuple) and m2 and m2[0] == 'AMBIGUOUS':
            lines = [h + 1 for h in m2[1]]
            _record('hint-ambiguous-normalized', path, candidate_line_starts=lines)
            raise ToolError(
                f'No replacement was performed. old_str did not appear verbatim in '
                f'{path}. [fuzzy-hint] Whitespace-normalized matching found multiple '
                f'candidate windows starting at lines {lines}. Quote more surrounding '
                f'context to make old_str unique. No changes were made.'
            )
        if m2 is not None:
            return self._auto_apply(
                path, file_content, m2, old_str, new_str, enable_linting
            )

        # ---- 提示层: 一律不改文件, 只描述最接近的候选 ----
        if _substantive_len(old_str) < _MIN_PATTERN_CHARS:
            _record('hint-none-short', path,
                    substantive_chars=_substantive_len(old_str))
            raise ToolError(
                str(orig_err)
                + ' [fuzzy-hint] The quoted old_str is very short and has no '
                'whitespace-equivalent match in the file; view the file again and '
                'quote the target text exactly as it appears, with more surrounding '
                'context. No changes were made.'
            )
        m3 = self._match_edit_distance(file_content, old_str)
        if isinstance(m3, tuple) and m3 and m3[0] == 'AMBIGUOUS':
            _record('hint-ambiguous-distance', path, candidate_lines=m3[1])
            raise ToolError(
                f'No replacement was performed. old_str did not appear verbatim in '
                f'{path}. [fuzzy-hint] Multiple similarly-close regions exist near '
                f'lines {m3[1]}; quote more surrounding context to disambiguate. '
                f'No changes were made.'
            )
        if m3 is None:
            _record('hint-none', path)
            raise ToolError(
                str(orig_err)
                + ' [fuzzy-hint] No sufficiently similar text was found anywhere in '
                'this file - the quoted old_str may be remembered rather than read. '
                'View the file again before editing. No changes were made.'
            )
        start, end, d1, plen = m3
        # 候选扩到整行再回显, 半行开头的提示没法照抄
        disp_start = file_content.rfind('\n', 0, start) + 1
        disp_end = file_content.find('\n', end)
        disp_end = len(file_content) if disp_end == -1 else disp_end
        candidate = file_content[disp_start:disp_end]
        first_line = file_content.count('\n', 0, disp_start) + 1
        shown = candidate if len(candidate) <= 1500 else candidate[:1500] + '…'
        numbered = '\n'.join(
            f'{first_line + i:6d}\t{ln}' for i, ln in enumerate(shown.splitlines())
        )
        diff_lines = list(difflib.unified_diff(
            old_str.splitlines(), candidate.splitlines(),
            fromfile='your old_str', tofile=f'{path} (actual)', lineterm='', n=1,
        ))
        diff = '\n'.join(diff_lines[:60])
        _record('hint-candidate', path, distance_chars=d1, pattern_chars=plen,
                first_line=first_line,
                last_line=first_line + candidate.count('\n'))
        raise ToolError(
            f'No replacement was performed. old_str did not appear verbatim in '
            f'{path}. [fuzzy-hint] The closest region (edit distance {d1}/{plen} '
            f'chars) is:\n{numbered}\n'
            f'Differences between your old_str (-) and the actual file text (+):\n'
            f'{diff}\n'
            f'If this is the region you intended to edit, re-issue str_replace '
            f'quoting the actual file text above verbatim (keep its exact '
            f'indentation). No changes were made.'
        )

    # -------------------------------------------------------------- 自动层施加
    def _auto_apply(self, path, file_content, m2, old_str, new_str, enable_linting):
        start, end, delta = m2
        new_str = _reindent_to_window(file_content[start:end], old_str, new_str)
        replaced = file_content[start:end]
        # 换行边界必须与被替换的窗口一致: 模型少写结尾换行会把下一行接上来
        # (或吃掉一个空行), 多写则凭空插一个空行。
        if replaced.endswith('\n') and not new_str.endswith('\n'):
            new_str += '\n'
        elif not replaced.endswith('\n') and new_str.endswith('\n'):
            new_str = new_str[:-1]
        new_file_content = file_content[:start] + new_str + file_content[end:]
        first_line = file_content.count('\n', 0, start) + 1
        try:
            self._guard_syntax(path, file_content, new_file_content)
        except ToolError:
            _record('guard-blocked', path, first_line=first_line)
            raise
        _record('auto', path, first_line=first_line,
                replaced_lines=replaced.count('\n') + 1,
                uniform_indent_shift=delta)
        self.write_file(path, new_file_content)
        self._history_manager.add_history(path, file_content)

        from openhands_aci.editor.config import SNIPPET_CONTEXT_WINDOW

        start_line = max(0, first_line - SNIPPET_CONTEXT_WINDOW)
        end_line = first_line + SNIPPET_CONTEXT_WINDOW + new_str.count('\n')
        snippet = self.read_file(path, start_line=start_line + 1, end_line=end_line)

        shown = replaced if len(replaced) <= 800 else replaced[:800] + '…'
        success_message = (
            f'The file {path} has been edited. [fuzzy-auto] old_str matched after '
            f'whitespace normalization only (pure indentation/line-ending '
            f'difference, unique in file); new_str was re-indented to the file.\n'
            f'The following actual file text was replaced (verify it is what you '
            f'intended):\n---\n{shown}\n---\n'
        )
        success_message += self._make_output(
            snippet, f'a snippet of {path}', start_line + 1
        )
        if enable_linting:
            lint_results = self._run_linting(file_content, new_file_content, path)
            success_message += '\n' + lint_results + '\n'
        success_message += (
            'Review the changes and make sure they are as expected. '
            'Edit the file again if necessary.'
        )
        return CLIResult(
            output=success_message,
            path=str(path),
            prev_exist=True,
            old_content=file_content,
            new_content=new_file_content,
        )
