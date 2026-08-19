# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Prometheus scrape helpers for speculative-decoding acceptance metrics.

Acceptance rate is invisible in the OpenAI wire protocol -- a response carries
the committed tokens, not whether a drafter proposed them -- so it has to come
off the server's ``/metrics`` endpoint. The driver snapshots the counters before
each aiperf invocation and deltas them afterwards, which is what makes the
figures per-run on a long-lived server.

TWO DIALECTS, because two different servers count this:

* vLLM exports ``vllm:spec_decode_num_{accepted,draft}_tokens_total`` on the
  OpenAI-API server itself.
* tt-media-server's cpp_server worker exports ``tt_worker_spec_accepts_total``
  and ``tt_worker_spec_rejects_total`` (see
  ``tt-media-server/cpp_server/src/runtime/worker/blaze_worker_metrics_renderer.cpp``),
  fed from the shared-memory counters the Blaze decode runner bumps per turn.

They are NOT interchangeable arithmetic. vLLM's denominator is draft tokens
offered; the worker's is accepts + rejects, and ``dflash_backend.cpp`` charges
the whole stale remainder of a block on the first reject -- so the worker's
denominator is speculative SLOTS CONSUMED and its rate reads lower than a
proposals-offered rate for identical device behavior. ``draft_tokens`` therefore
carries whichever denominator was used, and ``source`` says which dialect it was,
so a number is never silently compared against one computed the other way.

WHICH ENDPOINT. In a Dynamo deployment the load target (the frontend) has
neither dialect: it answers ``/metrics`` with ``dynamo_*`` families only, which
is a 200 carrying no acceptance at all. The worker's endpoint has to be scraped
directly -- pass it via ``--spec-decode-metrics-url`` (or
``$TT_SPEC_DECODE_METRICS_URL``), the same shape the prefix-cache benchmark uses
for its ``tt_prefix_cache_*`` counters.

Engine-agnostic on purpose: SGLang and other servers that expose Prometheus
text in the same shape can reuse :func:`fetch_prometheus_counters` directly.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

ACCEPTED_COUNTER = "vllm:spec_decode_num_accepted_tokens_total"
DRAFT_COUNTER = "vllm:spec_decode_num_draft_tokens_total"
NUM_DRAFTS_COUNTER = "vllm:spec_decode_num_drafts_total"
PER_POS_PREFIX = "vllm:spec_decode_num_accepted_tokens_per_pos"

SPEC_DECODE_PREFIX = "vllm:spec_decode_"

# cpp_server worker dialect. Both are per-worker (``worker_id`` label) and
# cumulative, so the same before/after delta works on them.
TT_WORKER_ACCEPTS_COUNTER = "tt_worker_spec_accepts_total"
TT_WORKER_REJECTS_COUNTER = "tt_worker_spec_rejects_total"
TT_WORKER_PREFIX = "tt_worker_spec_"

#: Every prefix worth keeping from a scrape. Snapshotting both dialects costs
#: nothing and means the caller does not have to know what it is talking to.
ACCEPTANCE_PREFIXES = (SPEC_DECODE_PREFIX, TT_WORKER_PREFIX)

#: Comma-separated worker ``/metrics`` endpoints, for callers with no CLI (the
#: dflash harness). ``--spec-decode-metrics-url`` takes precedence.
METRICS_URL_ENV = "TT_SPEC_DECODE_METRICS_URL"

#: Speculative block size, needed to turn a slot count into a round count for the
#: accepted-length ESTIMATE. DFlash's block_size is 8 (7 drafts + the anchor).
#: Unset means "don't guess" -- the column stays empty rather than showing a
#: number derived from a block size nobody confirmed.
BLOCK_SIZE_ENV = "TT_SPEC_DECODE_BLOCK_SIZE"


def configured_block_size(explicit: Optional[int] = None) -> Optional[int]:
    """Block size for the accepted-length estimate, or None to skip it."""
    if explicit is not None:
        return explicit
    raw = os.environ.get(BLOCK_SIZE_ENV, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; ignoring", BLOCK_SIZE_ENV, raw)
        return None
    if value < 2:
        logger.warning("%s=%d must be >= 2; ignoring", BLOCK_SIZE_ENV, value)
        return None
    return value


def estimate_accepted_length(
    accepted: float, slots: float, block_size: Optional[int]
) -> Tuple[Optional[float], Optional[float]]:
    """``(rounds, mean_accepted_length)`` estimated from a SLOT count.

    Neither the cpp_server worker's counters nor its log record how many
    verification ROUNDS ran, so the round count is inferred on the assumption
    that every round consumes a full block of speculative slots -- accepted ones,
    then the stale remainder charged on the first reject::

        rounds     = slots / (block_size - 1)
        accept_len = 1 + accepted / rounds

    The ``+1`` is the target's own token, which lands every round (the correction
    on a reject, the bonus on a full block) -- the same convention as vLLM's
    ``SpecDecodingLogging``, so the two dialects' numbers stay comparable.

    Returns ``(None, None)`` without a block size: an accepted-length derived
    from a guessed block size would be indistinguishable in the report from a
    measured one. THIS IS AN ESTIMATE either way; callers mark it as such.
    """
    if not block_size or block_size < 2 or slots <= 0:
        return None, None
    rounds = slots / (block_size - 1)
    return rounds, 1 + (accepted / rounds)


_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


def _parse_labels(label_str: str) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted(_LABEL_RE.findall(label_str)))


def _canonical_key(name: str, labels: Tuple[Tuple[str, str], ...]) -> str:
    if not labels:
        return name
    label_part = ",".join(f'{k}="{v}"' for k, v in labels)
    return f"{name}{{{label_part}}}"


def parse_prometheus_text(
    text: str, *, prefix: "str | Sequence[str]" = ACCEPTANCE_PREFIXES
) -> Dict[str, float]:
    """Parse Prometheus exposition text into ``{canonical_key: value}``.

    Only metric lines whose name starts with ``prefix`` (a single prefix, or any
    of several) are retained -- the default keeps both acceptance dialects, so a
    caller need not know whether it is scraping vLLM or a cpp_server worker. The
    canonical key includes labels sorted alphabetically so two snapshots against
    the same series always produce matching keys.
    """
    prefixes: Tuple[str, ...] = (prefix,) if isinstance(prefix, str) else tuple(prefix)
    result: Dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "{" in line:
            name, rest = line.split("{", 1)
            if "}" not in rest:
                continue
            label_str, value_part = rest.split("}", 1)
        else:
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            name, value_part = parts
            label_str = ""
        if not any(name.startswith(p) for p in prefixes):
            continue
        value_tokens = value_part.strip().split()
        if not value_tokens:
            continue
        try:
            value = float(value_tokens[0])
        except ValueError:
            continue
        result[_canonical_key(name, _parse_labels(label_str))] = value
    return result


def normalize_metrics_base(value: str) -> str:
    """A user-supplied metrics endpoint reduced to the base URL we append to.

    Accepts ``host:port``, ``http://host:port`` and ``http://host:port/metrics``
    (the three forms people actually paste) and returns the scheme-qualified base
    without the trailing ``/metrics``, because :func:`fetch_prometheus_counters`
    adds it. Mirrors ``aiperf_prefix_cache._normalize_metrics_url``, which
    normalizes the same three forms for AIPerf's ``--server-metrics``.
    """
    url = value.strip().rstrip("/")
    if "://" not in url:
        url = f"http://{url}"
    if url.endswith("/metrics"):
        url = url[: -len("/metrics")]
    return url.rstrip("/")


def configured_metrics_urls(
    explicit: Optional[Iterable[str]] = None,
) -> Tuple[str, ...]:
    """Worker ``/metrics`` endpoints to scrape, normalized and de-duplicated.

    ``explicit`` (``--spec-decode-metrics-url``, repeatable) wins over
    ``$TT_SPEC_DECODE_METRICS_URL`` (comma-separated) so a CLI run is never
    overridden by a stale export in the shell.
    """
    raw: Sequence[str]
    if explicit:
        raw = [u for u in explicit if u and u.strip()]
    else:
        raw = [u for u in os.environ.get(METRICS_URL_ENV, "").split(",") if u.strip()]
    seen: Dict[str, None] = {}
    for value in raw:
        seen.setdefault(normalize_metrics_base(value), None)
    return tuple(seen)


def fetch_prometheus_counters(
    base_url: str, *, timeout: float = 10.0
) -> Dict[str, float]:
    """GET ``{base_url}/metrics`` and return parsed acceptance counters."""
    # Imported here (not module top) so importing llm_module doesn't require
    # requests in venvs that never touch the spec-decode path.
    import requests

    url = normalize_metrics_base(base_url) + "/metrics"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return parse_prometheus_text(response.text)


def _sum_by_metric(deltas: Dict[str, float], metric_name: str) -> float:
    """Sum delta values whose canonical key matches ``metric_name`` (any labels)."""
    total = 0.0
    prefix_with_brace = metric_name + "{"
    for k, v in deltas.items():
        if k == metric_name or k.startswith(prefix_with_brace):
            total += v
    return total


def _extract_per_position(deltas: Dict[str, float]) -> Dict[int, float]:
    per_pos: Dict[int, float] = {}
    prefix_with_brace = PER_POS_PREFIX + "{"
    for k, v in deltas.items():
        if not (k == PER_POS_PREFIX or k.startswith(prefix_with_brace)):
            continue
        match = re.search(r'position="([^"]+)"', k)
        if not match:
            continue
        try:
            pos = int(match.group(1))
        except ValueError:
            continue
        per_pos[pos] = per_pos.get(pos, 0.0) + v
    return per_pos


def metrics_from_deltas(
    deltas: Dict[str, float], *, block_size: Optional[int] = None
) -> Dict[str, Any]:
    """Acceptance figures from already-differenced counters, either dialect.

    vLLM wins when it reported draft tokens; otherwise the cpp_server worker's
    accepts/rejects are used, with ``draft_tokens = accepts + rejects`` (slots
    consumed -- see the module docstring on why that denominator differs). With
    neither present every field is zero and ``source`` is None, which is the
    caller's signal that nothing measured this run rather than that the drafter
    lost every token.
    """
    accepted = _sum_by_metric(deltas, ACCEPTED_COUNTER)
    draft = _sum_by_metric(deltas, DRAFT_COUNTER)
    num_drafts = _sum_by_metric(deltas, NUM_DRAFTS_COUNTER)
    per_pos = sorted(_extract_per_position(deltas).items())

    source: Optional[str] = "prometheus:vllm" if draft > 0 else None
    estimated_rounds: Optional[float] = None
    estimated_length: Optional[float] = None
    if draft <= 0:
        worker_accepts = _sum_by_metric(deltas, TT_WORKER_ACCEPTS_COUNTER)
        worker_rejects = _sum_by_metric(deltas, TT_WORKER_REJECTS_COUNTER)
        if worker_accepts > 0 or worker_rejects > 0:
            accepted = worker_accepts
            draft = worker_accepts + worker_rejects
            num_drafts = 0.0  # the worker counts slots, not rounds
            source = "prometheus:tt_worker"
            # ...so accepted-length has to be estimated from the slot count, the
            # same way the worker-log source does it. Same helper, so the two can
            # never drift into reporting different lengths for the same run.
            estimated_rounds, estimated_length = estimate_accepted_length(
                accepted, draft, block_size
            )

    return {
        "acceptance_rate": (accepted / draft) if draft > 0 else 0.0,
        "accepted_tokens": accepted,
        "draft_tokens": draft,
        "num_drafts": num_drafts if num_drafts > 0 else None,
        # ``1 + accepted/rounds``: the +1 is the bonus token the target verifies
        # at the end of every draft round (vLLM's SpecDecodingLogging convention).
        # Needs a round count, which only the vLLM dialect reports.
        "mean_accepted_length": (
            1 + (accepted / num_drafts) if num_drafts > 0 else estimated_length
        ),
        # True when the length above came from the block-size assumption rather
        # than from a counted round, so a reader knows which number they have.
        "mean_accepted_length_is_estimate": num_drafts <= 0
        and estimated_length is not None,
        "estimated_rounds": estimated_rounds,
        "block_size": block_size if estimated_length is not None else None,
        "accepted_per_pos": per_pos,
        "source": source,
    }


def scrape_spec_decode_metrics(
    base_url: str, before: Dict[str, float], *, block_size: Optional[int] = None
) -> Dict[str, Any]:
    """Scrape ``/metrics`` and compute deltas vs ``before``.

    Returns the :func:`metrics_from_deltas` shape: ``acceptance_rate``,
    ``accepted_tokens`` / ``draft_tokens`` (deltas in this window),
    ``mean_accepted_length``, ``accepted_per_pos``, and ``source``.
    """
    after = fetch_prometheus_counters(base_url)
    all_keys = set(before) | set(after)
    return metrics_from_deltas(
        {k: after.get(k, 0.0) - before.get(k, 0.0) for k in all_keys},
        block_size=block_size,
    )


def scrape_worker_metrics(
    urls: Sequence[str],
    before: Dict[str, Dict[str, float]],
    *,
    block_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Scrape several worker endpoints and pool their deltas into one figure.

    Counters are summed ACROSS workers before the rate is computed, so a
    multi-worker deployment reports one pooled acceptance rate rather than an
    average of per-worker rates that would over-weight an idle worker. An
    endpoint that fails to answer is skipped with a warning: partial coverage is
    still a better number than none, and the failure is visible.
    """
    pooled: Dict[str, float] = {}
    reached = 0
    for url in urls:
        try:
            after = fetch_prometheus_counters(url)
        except Exception as exc:  # noqa: BLE001 -- scrape is best-effort
            logger.warning("Could not scrape worker metrics at %s: %s", url, exc)
            continue
        reached += 1
        snapshot = before.get(url, {})
        for key in set(snapshot) | set(after):
            # Keys carry a worker_id label, so two workers never collide here;
            # the sum is over distinct series.
            pooled[key] = pooled.get(key, 0.0) + (
                after.get(key, 0.0) - snapshot.get(key, 0.0)
            )
    if not reached:
        raise RuntimeError(f"no worker metrics endpoint answered: {list(urls)}")
    result = metrics_from_deltas(pooled, block_size=block_size)
    result["metrics_urls"] = list(urls)
    return result


__all__ = [
    "ACCEPTANCE_PREFIXES",
    "BLOCK_SIZE_ENV",
    "ACCEPTED_COUNTER",
    "DRAFT_COUNTER",
    "METRICS_URL_ENV",
    "NUM_DRAFTS_COUNTER",
    "PER_POS_PREFIX",
    "SPEC_DECODE_PREFIX",
    "TT_WORKER_ACCEPTS_COUNTER",
    "TT_WORKER_PREFIX",
    "TT_WORKER_REJECTS_COUNTER",
    "configured_block_size",
    "configured_metrics_urls",
    "estimate_accepted_length",
    "fetch_prometheus_counters",
    "metrics_from_deltas",
    "normalize_metrics_base",
    "parse_prometheus_text",
    "scrape_spec_decode_metrics",
    "scrape_worker_metrics",
]
