# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""Opt-in, two-scrape suite counters; no profiler or per-request instrumentation.

Goal-12 controls.record_suite_metrics=true enables this only for Qwen/Qwen3.8-27B. Scrapes must
surround (not enter) all harness timing scopes. These are server-interval counts,
not proof that requests belong to the suite or that a particular engine survived.
"""

import json
import logging
import math
import re
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

MODEL = "Qwen/Qwen3.8-27B"
COUNTERS = (
    "vllm:num_preemptions_total",
    "vllm:request_success_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:prompt_tokens_total",
    "vllm:prompt_tokens_cached_total",
    "vllm:generation_tokens_total",
    "vllm:request_num_preemptions_sum",
    "vllm:request_num_preemptions_count",
)
GAUGES = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "process_start_time_seconds",
)
CREATED = tuple(sorted({n.rsplit("_", 1)[0] + "_created" for n in COUNTERS}))
SELECTED = set(COUNTERS + GAUGES + CREATED)
LABELS = {"model_name", "engine", "finish_reason", "finished_reason"}
MAX_BYTES = 2 * 1024 * 1024
TIMEOUT = 5.0
_SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+(\S+)(?:\s+\S+)?$")
_LABEL = re.compile(r'\s*([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\[\\"n])*)"\s*(?:,|$)')
logger = logging.getLogger(__name__)


def enabled(model, task_name):
    if model != MODEL:
        return False
    from reference_config.evals.qwen38_goal12 import controls

    c = controls()
    return c.get("record_suite_metrics", False) and task_name in c["evals"]


def metrics_url(base_url):
    """Keep deployment path prefixes; strip only an OpenAI /v1 suffix."""
    parts = urlsplit(base_url)
    if (
        parts.scheme not in ("http", "https")
        or not parts.hostname
        or parts.username
        or parts.password
        or parts.query
        or parts.fragment
    ):
        raise ValueError("invalid metrics base URL")
    path = parts.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunsplit((parts.scheme, parts.netloc, path + "/metrics", "", ""))


def parse_metrics(body):
    """Discard unselected exposition and fail closed for ambiguous selected data."""
    samples = {}
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = re.split(r"[\s{]", line, maxsplit=1)[0]
        if name not in SELECTED:
            continue
        match = _SAMPLE.fullmatch(line)
        if not match:
            raise ValueError("malformed selected sample")
        labels, pos = {}, 0
        raw = match.group(2) or ""
        while pos < len(raw):
            label = _LABEL.match(raw, pos)
            if not label or label.group(1) in labels:
                raise ValueError("malformed or duplicate selected label")
            key, val = label.group(1), label.group(2)
            if key not in LABELS:
                raise ValueError("unexpected selected label")
            # Prometheus label escapes are only backslash, quote and newline.
            val = re.sub(r'\\([\\"n])', lambda m: "\n" if m[1] == "n" else m[1], val)
            if len(val) > 256 or "\n" in val:
                raise ValueError("unsafe selected label")
            labels[key] = val
            pos = label.end()
        if labels.get("model_name", MODEL) != MODEL:
            continue
        if "engine" in labels and not labels["engine"].isdigit():
            raise ValueError("unexpected engine label")
        if "finish_reason" in labels and "finished_reason" in labels:
            raise ValueError("ambiguous finish reason labels")
        for reason_label in ("finish_reason", "finished_reason"):
            if reason_label in labels and labels[reason_label] not in {
                "stop",
                "length",
                "abort",
                "error",
                "repetition",
            }:
                raise ValueError("unexpected finish reason")
        value = float(match.group(3))
        if not math.isfinite(value) or value < 0:
            raise ValueError("invalid selected number")
        key = (name, tuple(sorted(labels.items())))
        if key in samples:
            raise ValueError("duplicate selected series")
        samples[key] = value
    return [
        {"name": n, "labels": dict(labels), "value": value}
        for (n, labels), value in sorted(samples.items())
    ]


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def fetch_metrics(base_url, auth_token):
    req = Request(
        metrics_url(base_url),
        headers={"Authorization": "Bearer " + auth_token} if auth_token else {},
    )
    deadline = time.monotonic() + TIMEOUT
    with build_opener(_NoRedirect).open(req, timeout=TIMEOUT) as response:
        data = bytearray()
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError("metrics deadline")
            chunk = response.read1(min(65536, MAX_BYTES + 1 - len(data)))
            data.extend(chunk)
            if len(data) > MAX_BYTES:
                raise ValueError("metrics body exceeds limit")
            if not chunk:
                break
    return bytes(data).decode("utf-8")


def snapshot(base_url, auth_token):
    start = time.monotonic()
    result = {
        "at": datetime.now(timezone.utc).isoformat(),
        "status": "missing",
        "samples": [],
    }
    try:
        result["samples"] = parse_metrics(fetch_metrics(base_url, auth_token))
        result["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 - optional diagnostics must preserve harness outcome
        # Exception text can contain URL credentials or response bodies.
        result["missing_reason"] = type(exc).__name__
    result["scrape_seconds"] = time.monotonic() - start
    return result


def summarize(before, after):
    def index(s):
        return {
            (x["name"], tuple(sorted(x["labels"].items()))): x["value"]
            for x in s["samples"]
        }

    b, a = index(before), index(after)
    reasons = []
    process_key = ("process_start_time_seconds", ())
    api_changed = (
        process_key in a and process_key in b and a[process_key] != b[process_key]
    )
    if api_changed:
        reasons.append("api_process_changed")
    elif process_key not in a or process_key not in b:
        reasons.append("api_process_start_unavailable")
    reasons.append("engine_identity_not_verified_by_api_process_guard")
    if before["status"] != "ok" or after["status"] != "ok":
        reasons.append("scrape_missing")
    for n in ("vllm:num_requests_running", "vllm:num_requests_waiting"):
        for label, values in (("before", b), ("after", a)):
            matches = [v for (name, _), v in values.items() if name == n]
            if not matches:
                reasons.append(f"{label}:{n}:missing")
            elif any(v != 0 for v in matches):
                reasons.append(f"{label}:{n}:nonzero_interval_attribution_uncertain")
    deltas = []
    for name in COUNTERS:
        keys = sorted(k for k in set(a) | set(b) if k[0] == name)
        if not keys:
            deltas.append(
                {
                    "name": name,
                    "labels": {},
                    "delta": None,
                    "missing_reason": "series_missing",
                }
            )
        for key in keys:
            row = {"name": name, "labels": dict(key[1]), "delta": None}
            created = (name.rsplit("_", 1)[0] + "_created", key[1])
            if before["status"] != "ok" or after["status"] != "ok":
                row["missing_reason"] = "scrape_missing"
            elif api_changed:
                row["missing_reason"] = "api_process_changed"
            elif key not in a or key not in b:
                row["missing_reason"] = "series_missing_at_boundary"
            elif (created in a) != (created in b):
                row["missing_reason"] = "counter_creation_guard_changed_presence"
            elif created in a and a[created] != b[created]:
                row["missing_reason"] = "counter_creation_changed"
            elif a[key] < b[key]:
                row["missing_reason"] = "counter_decreased"
            else:
                row["delta"] = a[key] - b[key]
            row["creation_guard_available"] = created in a and created in b
            deltas.append(row)
    return {
        "deltas": deltas,
        "limitations": reasons,
        "scope": "server_interval_not_request_joined",
        "engine_identity_verified": False,
    }


def _write(path, receipt):
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        tmp.replace(path)
    except Exception as exc:  # noqa: BLE001 - optional diagnostics must preserve harness outcome
        logger.warning("Suite metrics receipt unavailable (%s)", type(exc).__name__)


@contextmanager
def suite_metrics(path, task_name, base_url, auth_token="", *, enabled=False):
    if not enabled:
        yield
        return
    receipt = {
        "schema_version": 1,
        "task_name": task_name,
        "before": snapshot(base_url, auth_token),
    }
    _write(path, receipt)
    try:
        yield
    finally:
        receipt["after"] = snapshot(base_url, auth_token)
        try:
            receipt.update(summarize(receipt["before"], receipt["after"]))
        except Exception as exc:  # noqa: BLE001 - optional diagnostics must preserve harness outcome
            receipt["missing_reason"] = type(exc).__name__
        _write(path, receipt)
