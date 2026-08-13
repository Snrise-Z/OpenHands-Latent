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


def test_guard_can_be_switched_off(workspace, monkeypatch):
    root, f = workspace
    monkeypatch.setenv('OH_FUZZY_SYNTAX_GUARD', '0')
    editor = FuzzyOHEditor(workspace_root=root)
    broken = NEW_PARTIAL.replace('collect_sql(plan)', 'collect_sql(plan')
    editor.str_replace(f, OLD_PARTIAL, broken, False)
    assert 'collect_sql(plan' in f.read_text()
    assert os.environ['OH_FUZZY_SYNTAX_GUARD'] == '0'
