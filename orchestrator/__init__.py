"""
orchestrator package - debate-and-consensus multi-agent orchestrator.

Public API
----------
orchestrate(task, repo_path, max_debate_rounds=3, verbose=True, api_key=None) -> bool
orchestrate_groom(task, repo_path, max_debate_rounds=3, verbose=True, api_key=None) -> bool
"""

from orchestrator.orchestrator import orchestrate, orchestrate_groom

__all__ = ["orchestrate", "orchestrate_groom"]
