import os

LITELLM_BASE_URL = "https://litellm-proxy--tenstorrent.workload.tenstorrent.com"
KEY_FILE = "/workspace/global/.litellm.key"

def get_api_key() -> str:
    if "TT_CHAT_API_KEY" in os.environ:
        return os.environ["TT_CHAT_API_KEY"]
    with open(KEY_FILE) as f:
        return f.read().strip()

# Swap these out when Kimi/GLM land on the proxy
DEFAULT_MODEL = "anthropic/claude-sonnet-4-6"
FAST_MODEL    = "anthropic/claude-haiku-4-5"
