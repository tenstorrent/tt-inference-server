"""Task-local execution ownership; no work is performed at import."""
from contextvars import ContextVar

installed_agent_scope = ContextVar('owned_installed_agent_scope', default=False)
compose_operation = ContextVar('owned_compose_operation', default='unknown')
