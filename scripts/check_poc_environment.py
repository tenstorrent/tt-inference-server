#!/usr/bin/env python3
"""Check the GLM-5.3 PoC environment without changing it or printing secrets."""

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlsplit


def check_environment(server_url, cache_root, environ, dotenv):
    url = urlsplit(server_url)
    if (
        url.scheme not in ("http", "https")
        or not url.hostname
        or url.username is not None
        or url.password is not None
        or url.path
        or url.query
        or url.fragment
    ):
        raise ValueError("Use an HTTP(S) server origin without credentials or /v1")
    if not Path(cache_root).is_absolute():
        raise ValueError("CACHE_ROOT must be absolute")
    file_env = {}
    if dotenv.exists():
        for number, line in enumerate(dotenv.read_text().splitlines(), 1):
            if not line.strip() or line.startswith("#"):
                continue
            if "=" not in line:
                raise ValueError(f"Invalid .env assignment on line {number}")
            key, value = map(str.strip, line.split("=", 1))
            file_env[key] = value

    conflicts = []
    for source, values in (("environment", environ), (".env", file_env)):
        for key, value in values.items():
            if key in ("OPENAI_BASE_URL", "OPENAI_API_BASE"):
                if value != server_url + "/v1":
                    conflicts.append(f"{source}:{key}")
            elif key in ("SERVER_URL", "AIGATEWAY_URL"):
                if value != server_url:
                    conflicts.append(f"{source}:{key}")
            elif key == "CACHE_ROOT":
                if value != cache_root:
                    conflicts.append(f"{source}:{key}")
            elif key in (
                "MODEL_SPECS_ENV",
                "OVERRIDE_BENCHMARK_TARGETS",
            ) or key.startswith(("HARBOR_", "AIPERF_")):
                conflicts.append(f"{source}:{key}")
            elif source == ".env" and key in (
                "PYTHONPATH",
                "PYTHONHOME",
                "UV_OVERRIDE",
                "UV_PROJECT_ENVIRONMENT",
            ):
                conflicts.append(f"{source}:{key}")
    if conflicts:
        raise ValueError(
            "Remove conflicting PoC settings: " + ", ".join(sorted(conflicts))
        )
    return {"server_url": server_url, "cache_root": cache_root, "dotenv_checked": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--cache-root", required=True)
    args = parser.parse_args()
    try:
        receipt = check_environment(
            args.server_url,
            args.cache_root,
            os.environ,
            Path(__file__).resolve().parents[1] / ".env",
        )
    except ValueError as error:
        parser.exit(2, f"PoC environment check failed: {error}\n")
    cache = Path(args.cache_root)
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "poc-environment.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("PoC environment check passed")


if __name__ == "__main__":
    main()
