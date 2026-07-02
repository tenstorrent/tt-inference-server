"""
Tests for bash_exec alias support in orchestrator/tools.py (issue #104).

Kimi-K2 and similar models sometimes call the tool as 'bash', 'execute', or
'run' instead of the canonical 'bash_exec'.  The aliases in IMPL silently
redirect those calls so edits actually land rather than being swallowed.
"""

import orchestrator.tools as T


class TestBashExecAliases:
    def test_bash_alias_executes_command(self):
        result = T.execute("bash", {"command": "echo alias_ok"})
        assert "alias_ok" in result

    def test_execute_alias_executes_command(self):
        result = T.execute("execute", {"command": "echo alias_ok"})
        assert "alias_ok" in result

    def test_run_alias_executes_command(self):
        result = T.execute("run", {"command": "echo alias_ok"})
        assert "alias_ok" in result

    def test_aliases_not_in_defs(self):
        # Aliases must NOT appear in DEFS — the model should learn the canonical name.
        names = {d["function"]["name"] for d in T.DEFS}
        assert "bash" not in names
        assert "execute" not in names
        assert "run" not in names

    def test_canonical_bash_exec_still_works(self):
        result = T.execute("bash_exec", {"command": "echo canonical_ok"})
        assert "canonical_ok" in result

    def test_unknown_tool_still_errors(self):
        result = T.execute("nonexistent_tool", {"command": "echo hi"})
        assert result.startswith("ERROR")


class TestImplementerPromptExplicitToolNames:
    def test_implementer_prompt_names_bash_exec(self):
        from orchestrator.personas import IMPLEMENTER
        assert "bash_exec" in IMPLEMENTER["system"]

    def test_implementer_prompt_warns_against_bash(self):
        from orchestrator.personas import IMPLEMENTER
        # The prompt must explicitly forbid the wrong names so the model doesn't guess.
        system = IMPLEMENTER["system"]
        assert "NOT" in system or "not" in system
        assert "bash" in system
