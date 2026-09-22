"""Terminus 2 with a short recovery-hygiene addendum in its prompt.

Harbor's stock terminus-2 prompt documents ``C-c`` but gives no rule for when
to use it. On run 35667712636 (gemma-4-31B-it, QB2) the agent hung its own
shell on turn 0 with a foreground ``until ss ...; do sleep 1; done`` loop and
then spent 48 turns typing commands that were only echoed, never once sending
Ctrl+C. This subclass swaps the JSON prompt template for a copy that adds three
generic shell-recovery rules (prompt missing -> C-c; repeating output -> C-c;
never leave an unbounded loop in the foreground). Nothing task-specific.

Select it with ``agent_import_path`` on a TerminalBenchEvalConfig; Harbor
imports ``module:Class`` inside its own process, so the launcher puts the repo
root on PYTHONPATH. Results are "terminus-2 + recovery addendum", not stock
terminus-2, and must be labelled as such when compared with published scores.
"""

from pathlib import Path

from harbor.agents.terminus_2.terminus_2 import Terminus2

_TEMPLATES_DIR = Path(__file__).parent / "templates"


class Terminus2Recovery(Terminus2):
    @staticmethod
    def name() -> str:
        return "terminus-2-recovery"

    def _get_prompt_template_path(self) -> Path:
        if self._parser_name == "json":
            return _TEMPLATES_DIR / "terminus-json-plain-recovery.tmpl"
        return super()._get_prompt_template_path()
