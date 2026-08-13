"""str_replace 的分层模糊匹配编辑器。

上游 OHEditor.str_replace 要求 old_str 逐字唯一出现(含一次去首尾空白重试)。
实测(LCLM 仓库 adapter_realign/v1_lastk2_failure_modes.md 的 22 次真实被拒):
被拒引用与文件真实内容的归一化编辑距离全部 <= 12.3%, 而最优窗口与"不相交次优
窗口"的距离间隔全部 >= 41%——即失败几乎都是缩进丢失/整块重构, 且正确位置在
编辑距离意义下唯一得毫无悬念。

本模块在上游精确匹配失败("did not appear verbatim")后追加两层回退:

  第二层  行级空白归一化: 按"逐行去首尾空白后的行序列"在文件中找唯一的连续
          行窗口; 命中后, 若各行缩进偏移恒定, 对 new_str 施加同一偏移。
  第三层  半全局编辑距离(Sellers 动态规划, O(|P|*|T|), numpy 向量化):
          接受条件 = 归一化距离 <= FUZZY_MAX_DISTANCE(缺省 0.15)
          且不相交次优间隔 >= FUZZY_MIN_MARGIN(缺省 0.05)。
          间隔门是唯一性的操作化判据: 代码自相似导致的多候选会在这里被拒,
          并把候选行号列表返回给模型(比"逐字不匹配"可操作得多)。

安全边界: 过短模式(有效字符 < 40)不进入回退层; 非精确层的成功消息中回显
实际被替换的原文, 模型可自查; "Multiple occurrences" 类错误(歧义)原样透传,
不做模糊消解。环境变量 OH_FUZZY_STR_REPLACE=0 整体关闭, 行为回到上游。
"""

import os
import re
from pathlib import Path

from openhands_aci.editor.editor import OHEditor
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


class FuzzyOHEditor(OHEditor):
    """精确优先、失败后分层回退的 str_replace。"""

    def str_replace(
        self,
        path: Path,
        old_str: str,
        new_str: str | None,
        enable_linting: bool,
        encoding: str = 'utf-8',
    ) -> CLIResult:
        try:
            return super().str_replace(path, old_str, new_str, enable_linting, encoding)
        except ToolError as err:
            if 'did not appear verbatim' not in str(err):
                raise  # 歧义等其他错误原样透传
            if os.environ.get('OH_FUZZY_STR_REPLACE', '1') == '0':
                raise
            if _substantive_len(old_str) < _MIN_PATTERN_CHARS:
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
        # 恒定缩进偏移: 所有非空行的 (文件缩进 - 引用缩进) 一致才应用
        deltas = set()
        p_lines = old_str.splitlines()
        for k in range(m):
            if p_stripped[k]:
                deltas.add(
                    (len(_indent_of(f_lines[i + k])), len(_indent_of(p_lines[k])))
                )
        delta = None
        offsets = {fi - pi for fi, pi in deltas}
        if len(offsets) == 1:
            delta = offsets.pop()
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
        tau = float(os.environ.get('FUZZY_MAX_DISTANCE', '0.15') or 0.15)
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

    # -------------------------------------------------------------- 施加
    def _fuzzy_str_replace(self, path, old_str, new_str, enable_linting, orig_err):
        file_content = self.read_file(path)

        note = None
        span = None
        m2 = self._match_normalized_lines(file_content, old_str)
        if isinstance(m2, tuple) and m2 and m2[0] == 'AMBIGUOUS':
            raise ToolError(
                f'No replacement was performed. old_str did not appear verbatim, and '
                f'whitespace-normalized matching found multiple candidate windows at '
                f'line starts {[h + 1 for h in m2[1]]} in {path}. Please disambiguate.'
            )
        if m2 is not None:
            start, end, delta = m2
            if delta:
                new_lines = []
                for ln in new_str.splitlines(keepends=True):
                    if ln.strip():
                        new_lines.append(
                            (' ' * max(0, len(_indent_of(ln)) + delta)) + ln.lstrip()
                        )
                    else:
                        new_lines.append(ln)
                new_str = ''.join(new_lines)
            span = (start, end)
            note = '[fuzzy-match: whitespace-normalized line match' + (
                f', re-indented new_str by {delta:+d}]' if delta else ']'
            )
        else:
            m3 = self._match_edit_distance(file_content, old_str)
            if isinstance(m3, tuple) and m3 and m3[0] == 'AMBIGUOUS':
                raise ToolError(
                    f'No replacement was performed. old_str did not appear verbatim, '
                    f'and edit-distance matching found multiple similar windows near '
                    f'lines {m3[1]} in {path}. Please quote more context to disambiguate.'
                )
            if m3 is None:
                return None
            start, end, d1, plen = m3
            span = (start, end)
            note = f'[fuzzy-match: edit-distance window, distance {d1}/{plen} chars]'

        start, end = span
        replaced = file_content[start:end]
        new_file_content = file_content[:start] + new_str + file_content[end:]
        self.write_file(path, new_file_content)
        self._history_manager.add_history(path, file_content)

        replacement_line = file_content.count('\n', 0, start) + 1
        from openhands_aci.editor.config import SNIPPET_CONTEXT_WINDOW

        start_line = max(0, replacement_line - SNIPPET_CONTEXT_WINDOW)
        end_line = replacement_line + SNIPPET_CONTEXT_WINDOW + new_str.count('\n')
        snippet = self.read_file(path, start_line=start_line + 1, end_line=end_line)

        shown = replaced if len(replaced) <= 800 else replaced[:800] + '…'
        success_message = (
            f'The file {path} has been edited. {note}\n'
            f'Note: old_str did not match verbatim; the following actual file text was '
            f'replaced (verify it is what you intended):\n---\n{shown}\n---\n'
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
