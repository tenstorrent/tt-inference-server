"""
orchestrator package - debate-and-consensus multi-agent orchestrator.

Public API
----------
orchestrate(task, repo_path, max_debate_rounds=3, max_tool_rounds=100, verbose=True, api_key=None) -> bool
orchestrate_groom(task, repo_path, max_debate_rounds=3, max_tool_rounds=100, verbose=True, api_key=None) -> bool
MaxToolRoundsError  -- raised by agent.run() when the tool-round cap is hit
DEFAULT_MAX_TOOL_ROUNDS  -- the default hard cap on tool-call iterations (100)
"""

from orchestrator.orchestrator import orchestrate, orchestrate_groom
from orchestrator.agent import MaxToolRoundsError, DEFAULT_MAX_TOOL_ROUNDS

__all__ = ["orchestrate", "orchestrate_groom", "MaxToolRoundsError", "DEFAULT_MAX_TOOL_ROUNDS"]
