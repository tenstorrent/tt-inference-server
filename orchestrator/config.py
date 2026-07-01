import os

LITELLM_BASE_URL = "https://litellm-proxy--tenstorrent.workload.tenstorrent.com"
KEY_FILE = "/workspace/global/.litellm.key"

TT_CONSOLE_BASE_URL = "https://console.tenstorrent.com/v1"
_TT_CONSOLE_KEY_FILE = os.path.expanduser("~/.tt-console.key")


def get_api_key(api_key: str | None = None) -> str:
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


def get_tt_console_api_key() -> str:
    if "TT_CONSOLE_API_KEY" in os.environ:
        return os.environ["TT_CONSOLE_API_KEY"]
    try:
        with open(_TT_CONSOLE_KEY_FILE) as f:
            return f.read().strip()
    except FileNotFoundError:
        raise ValueError(
            "tt-console provider requires a key: set TT_CONSOLE_API_KEY env var "
            f"or write the key to {_TT_CONSOLE_KEY_FILE}"
        )


# Registry maps provider name -> (base_url, key_loader).
# key_loader is a zero-arg callable that raises ValueError on missing credentials.
PROVIDER_REGISTRY: dict[str, dict] = {
    "litellm": {
        "base_url": LITELLM_BASE_URL,
        # litellm key resolution delegates to get_api_key() which accepts an
        # optional explicit key; we store a sentinel callable here for validation.
        "get_key": get_api_key,
    },
    "tt-console": {
        "base_url": TT_CONSOLE_BASE_URL,
        "get_key": get_tt_console_api_key,
    },
}


def validate_provider_keys(providers: set[str]) -> None:
    """Raise ValueError at startup for any provider whose key is absent."""
    for name in providers:
        if name not in PROVIDER_REGISTRY:
            raise ValueError(
                f"Unknown provider {name!r}. Valid providers: {sorted(PROVIDER_REGISTRY)}"
            )
        if name == "tt-console":
            # Eagerly resolve; will raise descriptively if missing.
            get_tt_console_api_key()


# Swap these out when Kimi/GLM land on the proxy
DEFAULT_MODEL = "anthropic/claude-sonnet-4-6"
FAST_MODEL    = "anthropic/claude-haiku-4-5"
