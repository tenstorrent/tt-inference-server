# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Lightweight readiness probe for remote OpenAI-compatible endpoints.

Kept in a separate module so callers (run_tests.py, run_benchmarks.py, …)
can import it without pulling in the heavy evals / media-client / scipy
dependency chain that lives in evals/run_evals.py.
"""

import logging
import os
import time

import requests

logger = logging.getLogger(__name__)


def _wait_for_remote_openai_ready(
    prompt_client, timeout: float = 300.0, interval: float = 10.0
) -> bool:
    """Readiness probe for a remote OpenAI-compatible endpoint.

    Remote / external endpoints (e.g. the Tenstorrent console) do not expose
    vLLM's ``/health`` route; instead poll the OpenAI ``/v1/models`` route,
    which returns 200 once the served model is reachable.
    """
    models_url = f"{prompt_client._get_api_base_url()}/models"
    # PromptClient.headers only carries VLLM_API_KEY / JWT auth; remote
    # endpoints authenticate with the literal OPENAI_API_KEY / API_KEY, so
    # build the bearer header explicitly for the probe.
    headers = dict(prompt_client.headers or {})
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    if api_key and "Authorization" not in headers:
        headers["Authorization"] = f"Bearer {api_key}"
    logger.info(
        "Waiting up to %ss for remote OpenAI endpoint %s (auth: %s)",
        timeout,
        models_url,
        "bearer set" if api_key else "NO KEY — 401 is likely",
    )
    start_time = time.time()
    attempt = 0
    while time.time() - start_time < timeout:
        attempt += 1
        elapsed = int(time.time() - start_time)
        try:
            response = requests.get(models_url, headers=headers, timeout=interval)
            if response.status_code == 200:
                logger.info(
                    "✅ Remote OpenAI endpoint ready at %s (attempt %d, %ss)",
                    models_url,
                    attempt,
                    elapsed,
                )
                return True
            # INFO, not DEBUG: this loop can run for the full timeout, and at
            # DEBUG the default INFO log level renders it as total silence —
            # indistinguishable from a hang. A repeating 401 here is the single
            # most common cause and must be visible immediately.
            logger.info(
                "Remote readiness attempt %d (%ss/%ss): HTTP %s from %s%s",
                attempt,
                elapsed,
                int(timeout),
                response.status_code,
                models_url,
                " — check the API key" if response.status_code in (401, 403) else "",
            )
        except requests.exceptions.RequestException as e:
            logger.info(
                "Remote readiness attempt %d (%ss/%ss) failed: %s",
                attempt,
                elapsed,
                int(timeout),
                e,
            )
        time.sleep(interval)
    logger.error(
        "⛔️ Remote OpenAI endpoint did not become ready within %ss at %s",
        timeout,
        models_url,
    )
    return False
