import os

LITELLM_BASE_URL = "https://litellm-proxy--tenstorrent.workload.tenstorrent.com"
KEY_FILE = "/workspace/global/.litellm.key"

def get_api_key(api_key: str | None = None) -> str:
    """Return a LiteLLM API key using the first source that supplies one.

    Priority (highest -> lowest):
      1. ``api_key`` argument passed explicitly by the caller (e.g. from --api-key CLI flag)
      2. ``TT_CHAT_API_KEY`` environment variable
      3. Key file at ``KEY_FILE``

    Raises:
        ValueError: If ``api_key`` is explicitly provided but is empty or
                    contains only whitespace, so that bad input is surfaced
                    immediately rather than silently falling through to a
                    different credential source.
    """
    if api_key is not None:
        if not api_key.strip():
            raise ValueError(
                "--api-key was provided but is empty or whitespace-only; "
                "pass a valid key or omit the flag to use the environment "
                "variable / key file fallback."
            )
        return api_key
    if "TT_CHAT_API_KEY" in os.environ:
        return os.environ["TT_CHAT_API_KEY"]
    with open(KEY_FILE) as f:
        return f.read().strip()

# Swap these out when Kimi/GLM land on the proxy
DEFAULT_MODEL = "anthropic/claude-sonnet-4-6"
FAST_MODEL    = "anthropic/claude-haiku-4-5"
