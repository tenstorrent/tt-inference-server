# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Unit tests for benchmarking.spec_decode_metrics."""

import pytest

from benchmarking import spec_decode_metrics
from benchmarking.spec_decode_metrics import (
    parse_prometheus_text,
    scrape_spec_decode_metrics,
)


PROMETHEUS_BEFORE = """
# HELP vllm:spec_decode_num_accepted_tokens_total Total accepted tokens
# TYPE vllm:spec_decode_num_accepted_tokens_total counter
vllm:spec_decode_num_accepted_tokens_total{model_name="llama-7b"} 600.0
vllm:spec_decode_num_draft_tokens_total{model_name="llama-7b"} 750.0
vllm:spec_decode_num_drafts_total{model_name="llama-7b"} 150.0
vllm:spec_decode_num_accepted_tokens_per_pos{model_name="llama-7b",position="0"} 130.0
vllm:spec_decode_num_accepted_tokens_per_pos{model_name="llama-7b",position="1"} 100.0
vllm:request_success_total 17.0
"""

PROMETHEUS_AFTER = """
vllm:spec_decode_num_accepted_tokens_total{model_name="llama-7b"} 800.0
vllm:spec_decode_num_draft_tokens_total{model_name="llama-7b"} 1000.0
vllm:spec_decode_num_drafts_total{model_name="llama-7b"} 200.0
vllm:spec_decode_num_accepted_tokens_per_pos{model_name="llama-7b",position="0"} 180.0
vllm:spec_decode_num_accepted_tokens_per_pos{model_name="llama-7b",position="1"} 150.0
vllm:spec_decode_num_accepted_tokens_per_pos{model_name="llama-7b",position="2"} 100.0
vllm:request_success_total 42.0
"""


def test_parse_keeps_only_spec_decode_keys():
    parsed = parse_prometheus_text(PROMETHEUS_AFTER)
    assert parsed, "expected at least one parsed counter"
    assert all(k.startswith("vllm:spec_decode_") for k in parsed)


def test_parse_handles_unlabeled_lines():
    parsed = parse_prometheus_text("vllm:spec_decode_num_accepted_tokens_total 42.0\n")
    assert parsed == {"vllm:spec_decode_num_accepted_tokens_total": 42.0}


def test_parse_canonicalizes_label_order():
    line_a = 'vllm:spec_decode_num_draft_tokens_total{a="1",b="2"} 10.0\n'
    line_b = 'vllm:spec_decode_num_draft_tokens_total{b="2",a="1"} 10.0\n'
    parsed_a = parse_prometheus_text(line_a)
    parsed_b = parse_prometheus_text(line_b)
    assert parsed_a == parsed_b


def test_parse_skips_malformed_lines():
    text = (
        "vllm:spec_decode_num_accepted_tokens_total{unterminated 10.0\n"
        "vllm:spec_decode_num_accepted_tokens_total not_a_number\n"
        "vllm:spec_decode_num_accepted_tokens_total 5.0\n"
    )
    parsed = parse_prometheus_text(text)
    assert parsed == {"vllm:spec_decode_num_accepted_tokens_total": 5.0}


def test_scrape_returns_deltas(monkeypatch):
    before = parse_prometheus_text(PROMETHEUS_BEFORE)
    after = parse_prometheus_text(PROMETHEUS_AFTER)

    monkeypatch.setattr(
        spec_decode_metrics, "fetch_prometheus_counters", lambda _url: after
    )
    metrics = scrape_spec_decode_metrics("http://stub", before)

    assert metrics["accepted_tokens"] == pytest.approx(200.0)
    assert metrics["draft_tokens"] == pytest.approx(250.0)
    assert metrics["num_drafts"] == pytest.approx(50.0)
    assert metrics["acceptance_rate"] == pytest.approx(200.0 / 250.0)
    assert metrics["mean_accepted_length"] == pytest.approx(1 + 200.0 / 50.0)


def test_scrape_per_position_includes_new_buckets(monkeypatch):
    before = parse_prometheus_text(PROMETHEUS_BEFORE)
    after = parse_prometheus_text(PROMETHEUS_AFTER)
    monkeypatch.setattr(
        spec_decode_metrics, "fetch_prometheus_counters", lambda _url: after
    )
    metrics = scrape_spec_decode_metrics("http://stub", before)
    per_pos = dict(metrics["accepted_per_pos"])
    # Position 0 grew by 50, position 1 by 50, position 2 is new (100).
    assert per_pos[0] == pytest.approx(50.0)
    assert per_pos[1] == pytest.approx(50.0)
    assert per_pos[2] == pytest.approx(100.0)


def test_scrape_handles_zero_draft(monkeypatch):
    monkeypatch.setattr(
        spec_decode_metrics, "fetch_prometheus_counters", lambda _url: {}
    )
    metrics = scrape_spec_decode_metrics("http://stub", {})
    assert metrics["acceptance_rate"] == 0.0
    assert metrics["mean_accepted_length"] is None
    assert metrics["num_drafts"] is None
    assert metrics["accepted_per_pos"] == []


def test_scrape_handles_missing_num_drafts_counter(monkeypatch):
    # Some vLLM releases don't expose vllm:spec_decode_num_drafts_total.
    after_no_drafts = parse_prometheus_text(
        "vllm:spec_decode_num_accepted_tokens_total 10.0\n"
        "vllm:spec_decode_num_draft_tokens_total 20.0\n"
    )
    monkeypatch.setattr(
        spec_decode_metrics,
        "fetch_prometheus_counters",
        lambda _url: after_no_drafts,
    )
    metrics = scrape_spec_decode_metrics("http://stub", {})
    assert metrics["acceptance_rate"] == pytest.approx(0.5)
    assert metrics["mean_accepted_length"] is None
    assert metrics["num_drafts"] is None
