"""Keep Harbor's agent/recipe, but settle its container workload before return."""
from harbor.agents.installed.mini_swe_agent import MiniSweAgent
from .execution_scope import installed_agent_scope


class OwnedMiniSweAgent(MiniSweAgent):
    async def run(self, *args, **kwargs):
        token = installed_agent_scope.set(True)
        try:
            return await super().run(*args, **kwargs)
        finally:
            installed_agent_scope.reset(token)
