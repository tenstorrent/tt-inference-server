"""Preflight auth/canary probe for host-to-server bearer-token flow.

Catches the failure class where the host-side eval/benchmark client signs
requests with a different bearer than the in-container vLLM server validates.
Without this probe, a JWT_SECRET / VLLM_API_KEY divergence between .env
(read by docker --env-file) and the host os.environ (read by the eval
client) silently surfaces as lm-eval scoring 0% on every task ("Could not
parse generations: 'choices'") or vLLM benchmarks reporting "Total generated
tokens: 0".

The probe sends a tiny max_tokens=1 completion using the resolved bearer
and asserts that the response payload contains a non-empty choices list.
Anything else (401, 403, 500, HTML error page, 200 with missing/empty
choices) is fatal and aborts the workflow with a clear, fingerprint-tagged
RuntimeError before any expensive eval starts.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_BODY_PREVIEW_CHARS = 500


@dataclass
class ProbeResult:
    ok: bool
    status_code: Optional[int]
    body_preview: str
    parsed_choices: int
    bearer_fp: str
    url: str
    error: Optional[str] = None

    def summary(self) -> str:
        if self.ok:
            return (
                f"probe OK: {self.url} -> {self.status_code} "
                f"(choices={self.parsed_choices}, bearer sha256[:8]={self.bearer_fp})"
            )
        return (
            f"probe FAILED: {self.url} -> status={self.status_code} "
            f"choices={self.parsed_choices} bearer sha256[:8]={self.bearer_fp} "
            f"err={self.error!r} body[:{_BODY_PREVIEW_CHARS}]={self.body_preview!r}"
        )


def _fingerprint(bearer: Optional[str]) -> str:
    if not bearer:
        return "none"
    return hashlib.sha256(bearer.encode("utf-8")).hexdigest()[:8]


def _resolve_probe_url(base_url: str, *, chat: bool) -> str:
    suffix = "/chat/completions" if chat else "/completions"
    if base_url.endswith(suffix):
        return base_url
    if base_url.endswith("/v1"):
        return base_url + suffix
    if "/v1/" in base_url:
        return base_url
    return base_url.rstrip("/") + suffix


def probe_bearer(
    base_url: str,
    bearer: Optional[str],
    *,
    model: Optional[str] = None,
    chat: bool = False,
    timeout: float = 30.0,
) -> ProbeResult:
    """Send a tiny completion request and validate the response shape.

    Returns a ProbeResult. ok=True requires HTTP 200 AND a non-empty
    'choices' list in the JSON body. Everything else is ok=False with
    the most informative diagnostic available.
    """

    fp = _fingerprint(bearer)
    if not bearer:
        return ProbeResult(
            ok=False,
            status_code=None,
            body_preview="",
            parsed_choices=0,
            bearer_fp=fp,
            url=base_url,
            error="bearer is empty; refusing to send unauthenticated probe",
        )

    url = _resolve_probe_url(base_url, chat=chat)
    headers = {
        "Authorization": f"Bearer {bearer}",
        "Content-Type": "application/json",
    }

    if chat:
        payload = {
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
            "stream": False,
        }
    else:
        payload = {
            "prompt": "ping",
            "max_tokens": 1,
            "stream": False,
        }
    if model:
        payload["model"] = model

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        return ProbeResult(
            ok=False,
            status_code=None,
            body_preview="",
            parsed_choices=0,
            bearer_fp=fp,
            url=url,
            error=f"transport error: {type(exc).__name__}: {exc}",
        )

    body_preview = (resp.text or "")[:_BODY_PREVIEW_CHARS]

    if resp.status_code != 200:
        return ProbeResult(
            ok=False,
            status_code=resp.status_code,
            body_preview=body_preview,
            parsed_choices=0,
            bearer_fp=fp,
            url=url,
            error="non-200 status; likely auth or model mismatch",
        )

    try:
        body = resp.json()
    except (ValueError, json.JSONDecodeError) as exc:
        return ProbeResult(
            ok=False,
            status_code=resp.status_code,
            body_preview=body_preview,
            parsed_choices=0,
            bearer_fp=fp,
            url=url,
            error=f"200 but body is not JSON: {exc}",
        )

    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) == 0:
        return ProbeResult(
            ok=False,
            status_code=resp.status_code,
            body_preview=body_preview,
            parsed_choices=0,
            bearer_fp=fp,
            url=url,
            error="200 OK but response has no usable 'choices'; lm-eval would silently score 0%",
        )

    return ProbeResult(
        ok=True,
        status_code=resp.status_code,
        body_preview=body_preview,
        parsed_choices=len(choices),
        bearer_fp=fp,
        url=url,
    )


def assert_probe_ok(result: ProbeResult) -> None:
    """Raise a multi-line RuntimeError if the probe failed.

    The error message contains the full probe summary plus body preview;
    both are essential for diagnosing cross-process auth drift in CI logs.
    """
    if result.ok:
        logger.info(result.summary())
        return

    raise RuntimeError(
        "Preflight auth probe failed; aborting workflow before any "
        "expensive eval or benchmark.\n"
        f"  {result.summary()}\n"
        "Hint: compare this bearer sha256[:8] with the server's startup "
        "log. If they differ, the host's .env / os.environ / docker "
        "--env-file are out of sync."
    )
