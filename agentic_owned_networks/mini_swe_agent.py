"""Keep Harbor's agent/recipe, but settle its container workload before return."""
import asyncio
from harbor.agents.installed.mini_swe_agent import MiniSweAgent
from .execution_scope import installed_agent_scope


class OwnedMiniSweAgent(MiniSweAgent):
    async def run(self, *args, **kwargs):
        # Installation has already completed. Only the model-driven agent phase
        # loses external egress; verifier dependencies can be installed later.
        environment=kwargs.get('environment',args[1] if len(args)>1 else None)
        isolated=getattr(environment,'_owned_agent_isolation',False)
        if isolated:await environment.enter_agent_isolation()
        token = installed_agent_scope.set(True)
        try:
            return await super().run(*args, **kwargs)
        finally:
            installed_agent_scope.reset(token)
            if isolated:
                from .docker_environment import completion_barrier
                _, cancelled = await completion_barrier(environment.leave_agent_isolation())
                if cancelled:raise asyncio.CancelledError()
