import pytest
from orchestrator.personas import IMPLEMENTER


def test_implementer_system_requires_tool_use():
    # Regression guard for #147: models that default to chat mode must see an
    # explicit directive to call write_file/bash_exec rather than describing changes.
    system = IMPLEMENTER["system"]
    assert "automated agent" in system
    assert "write_file" in system
    assert "bash_exec" in system
    assert "text responses alone have no effect" in system
