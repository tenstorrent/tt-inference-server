"""
orchestrator package - debate-and-consensus multi-agent orchestrator.

Public API
----------
orchestrate(task, repo_path, max_debate_rounds=3, verbose=True, api_key=None) -> bool
orchestrate_groom(task, repo_path, max_debate_rounds=3, verbose=True, api_key=None) -> bool
MaxToolRoundsError  -- raised by agent.run() when the tool-round cap is hit
"""

from orchestrator.orchestrator import orchestrate, orchestrate_groom
from orchestrator.agent import MaxToolRoundsError

__all__ = ["orchestrate", "orchestrate_groom", "MaxToolRoundsError"]
