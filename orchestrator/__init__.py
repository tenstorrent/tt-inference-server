"""
orchestrator package - debate-and-consensus multi-agent orchestrator.

Public API
----------
orchestrate(task, repo_path, max_debate_rounds=3, verbose=True, api_key=None) -> bool
"""

from orchestrator.orchestrator import orchestrate

__all__ = ["orchestrate"]
