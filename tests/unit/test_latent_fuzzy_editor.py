"""分层模糊匹配编辑器的回归测试。

用例形状取自八组矩阵里的真实故障(django sqlmigrate):模型引用的前两行缩进
正确、后五行少了 4 个空格,老实现把这份缩进原样写回,把方法体末尾移出了函数,
文件变成 'return' outside function。
"""

import os

import pytest

from openhands.runtime.latent_fuzzy_editor import FuzzyOHEditor
from openhands_aci.editor.exceptions import ToolError

FILE = r'''class Command:
    def handle(self, *args, **options):
        targets = [(1, 2)]

        # Show begin/end around output only for atomic migrations
        self.output_transaction = migration.atomic

        # Make a plan that represents just the requested migrations and show SQL
        # for it
        plan = [(1, 2)]
        sql_statements = collect_sql(plan)
        return "\n".join(sql_statements)
'''

# 后五行少了 4 个空格 —— 上游必拒
OLD_PARTIAL = r'''        # Show begin/end around output only for atomic migrations
        self.output_transaction = migration.atomic

    # Make a plan that represents just the requested migrations and show SQL
    # for it
    plan = [(1, 2)]
    sql_statements = collect_sql(plan)
    return "\n".join(sql_statements)'''

NEW_PARTIAL = r'''        # Show begin/end around output only for atomic migrations
        self.output_transaction = migration.atomic and can_rollback_ddl

    # Make a plan that represents just the requested migrations and show SQL
    # for it
    plan = [(1, 2)]
    sql_statements = collect_sql(plan)
    return "\n".join(sql_statements)'''


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv('OH_FUZZY_STR_REPLACE', '1')
    monkeypatch.delenv('OH_FUZZY_SYNTAX_GUARD', raising=False)
    f = tmp_path / 'sqlmigrate.py'
    f.write_text(FILE)
    return tmp_path, f


def test_partial_indent_loss_keeps_file_compilable(workspace):
    root, f = workspace
    editor = FuzzyOHEditor(workspace_root=root)
    editor.str_replace(f, OLD_PARTIAL, NEW_PARTIAL, False)

    after = f.read_text()
    compile(after, str(f), 'exec')  # 老实现在这里抛 'return' outside function
    assert '        plan = [(1, 2)]' in after  # 未改动的行保持文件原有缩进
    assert '    plan = [(1, 2)]\n' not in after.replace('        plan', '')
    assert 'migration.atomic and can_rollback_ddl' in after  # 目标改动确实落了


def test_constant_indent_offset_is_reindented(workspace):
    root, f = workspace
    editor = FuzzyOHEditor(workspace_root=root)
    old = '\n'.join(ln[4:] for ln in OLD_PARTIAL.splitlines()[:2])
    new = old.replace('migration.atomic', 'migration.atomic and can_rollback_ddl')
    editor.str_replace(f, old, new, False)

    after = f.read_text()
    compile(after, str(f), 'exec')
    assert '        self.output_transaction = migration.atomic and can_rollback_ddl' in after


def test_syntax_guard_rejects_breaking_edit(workspace):
    root, f = workspace
    editor = FuzzyOHEditor(workspace_root=root)
    before = f.read_text()
    # 引用能模糊命中, 但替换文本本身语法就是坏的
    broken = NEW_PARTIAL.replace('collect_sql(plan)', 'collect_sql(plan')
    with pytest.raises(ToolError) as e:
        editor.str_replace(f, OLD_PARTIAL, broken, False)
    assert 'would break the file' in str(e.value)
    assert f.read_text() == before  # 没有写回


def test_disabled_falls_back_to_upstream(workspace, monkeypatch):
    root, f = workspace
    monkeypatch.setenv('OH_FUZZY_STR_REPLACE', '0')
    editor = FuzzyOHEditor(workspace_root=root)
    before = f.read_text()
    with pytest.raises(ToolError) as e:
        editor.str_replace(f, OLD_PARTIAL, NEW_PARTIAL, False)
    assert 'did not appear verbatim' in str(e.value)
    assert f.read_text() == before


TAB_FILE = (
    'class Handler:\n'
    '\tdef process(self, payload):\n'
    '\t\tvalidated = self.validate(payload)\n'
    '\t\tenriched = self.enrich(validated)\n'
    '\t\tstored = self.store(enriched)\n'
    '\t\treturn stored\n'
)
OLD_TAB = (
    '    def process(self, payload):\n'
    '        validated = self.validate(payload)\n'
    '        enriched = self.enrich(validated)\n'
    '        stored = self.store(enriched)\n'
    '        return stored'
)

MAKEFILE = (
    'all: build test\n'
    '\techo building the project now\n'
    '\techo running the test suite\n'
    '\techo finished everything\n'
    '\ntest:\n\techo test\n'
)
OLD_MK = (
    'all: build test\n'
    '    echo building the project now\n'
    '    echo running the test suite\n'
    '    echo finished everything'
)

NL_FILE = (
    'def calculate(alpha, beta):\n'
    '    first = alpha * 2\n'
    '    second = beta * 3\n'
    '    combined = first + second\n'
    '    return combined\n'
    '\n'
    'def other():\n'
    '    return 0\n'
)
OLD_NL = (
    '  first = alpha * 2\n  second = beta * 3\n'
    '  combined = first + second\n  return combined\n'
)


def _edit(tmp_path, monkeypatch, name, content, old, new):
    monkeypatch.setenv('OH_FUZZY_STR_REPLACE', '1')
    f = tmp_path / name
    f.write_text(content)
    FuzzyOHEditor(workspace_root=tmp_path).str_replace(f, old, new, False)
    return f.read_text()


def test_tab_indented_python_keeps_tabs(tmp_path, monkeypatch):
    new = OLD_TAB.replace('self.store(enriched)', 'self.store(enriched, retry=True)')
    after = _edit(tmp_path, monkeypatch, 'h.py', TAB_FILE, OLD_TAB, new)

    compile(after, 'h.py', 'exec')  # 合成空格会在这里抛 TabError
    assert '\t\tstored = self.store(enriched, retry=True)' in after
    assert not any(ln.startswith('    ') for ln in after.splitlines())


def test_tab_indented_python_nested_insert(tmp_path, monkeypatch):
    new = OLD_TAB.replace(
        '        return stored',
        '        if stored:\n            log(stored)\n        return stored')
    after = _edit(tmp_path, monkeypatch, 'h.py', TAB_FILE, OLD_TAB, new)

    compile(after, 'h.py', 'exec')
    assert '\t\tif stored:' in after
    assert '\t\t\tlog(stored)' in after  # 深一级要按文件的缩进单位加, 不是空格


def test_makefile_recipe_keeps_tabs(tmp_path, monkeypatch):
    new = OLD_MK.replace('finished everything', 'all done')
    after = _edit(tmp_path, monkeypatch, 'Makefile', MAKEFILE, OLD_MK, new)

    recipes = [ln for ln in after.splitlines() if ln.strip().startswith('echo')]
    assert recipes and all(ln.startswith('\t') for ln in recipes)
    assert '\techo all done' in after


def test_trailing_newline_boundary_preserved(tmp_path, monkeypatch):
    new = OLD_NL.replace('beta * 3', 'beta * 4').rstrip('\n')  # 模型漏掉结尾换行
    after = _edit(tmp_path, monkeypatch, 'c.py', NL_FILE, OLD_NL, new)

    compile(after, 'c.py', 'exec')
    assert '\n\ndef other():' in after  # 函数之间的空行不能被吃掉
    assert len(after.splitlines()) == len(NL_FILE.splitlines())


def test_guard_can_be_switched_off(workspace, monkeypatch):
    root, f = workspace
    monkeypatch.setenv('OH_FUZZY_SYNTAX_GUARD', '0')
    editor = FuzzyOHEditor(workspace_root=root)
    broken = NEW_PARTIAL.replace('collect_sql(plan)', 'collect_sql(plan')
    editor.str_replace(f, OLD_PARTIAL, broken, False)
    assert 'collect_sql(plan' in f.read_text()
    assert os.environ['OH_FUZZY_SYNTAX_GUARD'] == '0'


# ---------------- 策略改版(2026-08-19): 编辑距离层只提示不自动改 ----------------
# 依据 500x7 主实验 250 条真实失败的标定: 最像候选 94% 指对、6% 指错;
# 提示指错代价线性(多试一轮), 自动改错代价破坏性(静默写错位置且模型几乎不跑测试)。

OLD_TYPO = (
    '    first = alpha * 2\n'
    '    second = beta*3\n'          # 内部空格与文件不同 -> 归一化层不命中
    '    combined = first + second\n'
    '    return combined'
)


def test_close_match_is_hint_only(tmp_path, monkeypatch):
    monkeypatch.setenv('OH_FUZZY_STR_REPLACE', '1')
    f = tmp_path / 'c.py'
    f.write_text(NL_FILE)
    before = f.read_text()
    editor = FuzzyOHEditor(workspace_root=tmp_path)
    with pytest.raises(ToolError) as e:
        editor.str_replace(
            f, OLD_TYPO, OLD_TYPO.replace('alpha * 2', 'alpha * 9'), False
        )
    msg = str(e.value)
    assert '[fuzzy-hint]' in msg
    assert 'second = beta * 3' in msg          # 候选回显了文件真实内容
    assert 'your old_str' in msg               # 带差异
    assert 'No changes were made' in msg
    assert f.read_text() == before             # 一个字节都没改


def test_no_candidate_advises_reread(tmp_path, monkeypatch):
    monkeypatch.setenv('OH_FUZZY_STR_REPLACE', '1')
    f = tmp_path / 'c.py'
    f.write_text(NL_FILE)
    before = f.read_text()
    editor = FuzzyOHEditor(workspace_root=tmp_path)
    bogus = (
        'SELECT customer_id, SUM(amount) FROM ledger_entries\n'
        'WHERE posted_at >= :cutoff GROUP BY customer_id\n'
        'HAVING SUM(amount) > 10000 ORDER BY 2 DESC'
    )
    with pytest.raises(ToolError) as e:
        editor.str_replace(f, bogus, bogus + ' -- x', False)
    assert 'View the file again' in str(e.value)
    assert f.read_text() == before


def test_decisions_are_recorded(tmp_path, monkeypatch):
    import json as _json

    log = tmp_path / 'fuzzy.jsonl'
    monkeypatch.setenv('OH_FUZZY_STR_REPLACE', '1')
    monkeypatch.setenv('OH_FUZZY_LOG', str(log))
    f = tmp_path / 'c.py'
    f.write_text(NL_FILE)
    editor = FuzzyOHEditor(workspace_root=tmp_path)
    editor.str_replace(
        f, OLD_NL, OLD_NL.replace('beta * 3', 'beta * 4'), False
    )                                           # 自动层(纯缩进差异)
    with pytest.raises(ToolError):
        editor.str_replace(
            f, OLD_TYPO, OLD_TYPO.replace('alpha * 2', 'alpha * 9'), False
        )                                       # 提示层
    events = [_json.loads(l)['event'] for l in log.read_text().splitlines()]
    assert events[0] == 'auto'
    assert events[1].startswith('hint-')


def test_auto_success_message_is_marked(tmp_path, monkeypatch):
    monkeypatch.setenv('OH_FUZZY_STR_REPLACE', '1')
    f = tmp_path / 'c.py'
    f.write_text(NL_FILE)
    editor = FuzzyOHEditor(workspace_root=tmp_path)
    res = editor.str_replace(
        f, OLD_NL, OLD_NL.replace('beta * 3', 'beta * 4'), False
    )
    assert '[fuzzy-auto]' in res.output
    assert 'second = beta * 3' in res.output   # 回显被替换的原文
