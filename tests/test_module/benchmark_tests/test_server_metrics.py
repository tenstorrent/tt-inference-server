# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for reading the video service's own metrics during a benchmark."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from test_module._test_common import server_metrics as sm

BEFORE = """\
# HELP tt_video_queue_wait_seconds wait
# TYPE tt_video_queue_wait_seconds histogram
tt_video_queue_wait_seconds_bucket{le="1"} 0
tt_video_queue_wait_seconds_bucket{le="10"} 0
tt_video_queue_wait_seconds_bucket{le="+Inf"} 0
tt_video_queue_wait_seconds_sum 0
tt_video_queue_wait_seconds_count 0
tt_video_e2e_seconds_bucket{outcome="ok",le="100"} 0
tt_video_e2e_seconds_bucket{outcome="ok",le="+Inf"} 0
tt_video_e2e_seconds_sum{outcome="ok"} 0
tt_video_e2e_seconds_count{outcome="ok"} 0
other_metric 5
"""

# Four videos: waits of about 0.5 s, 0.5 s, 5 s and 5 s; e2e 80 s each.
AFTER = """\
tt_video_queue_wait_seconds_bucket{le="1"} 2
tt_video_queue_wait_seconds_bucket{le="10"} 4
tt_video_queue_wait_seconds_bucket{le="+Inf"} 4
tt_video_queue_wait_seconds_sum 11
tt_video_queue_wait_seconds_count 4
tt_video_e2e_seconds_bucket{outcome="ok",le="100"} 4
tt_video_e2e_seconds_bucket{outcome="ok",le="+Inf"} 4
tt_video_e2e_seconds_sum{outcome="ok"} 320
tt_video_e2e_seconds_count{outcome="ok"} 4
other_metric 9
"""


def test_only_the_video_series_are_read():
    snapshot = sm.parse_video_metrics(BEFORE)
    assert ("tt_video_queue_wait_seconds_count", "") in snapshot
    assert ("tt_video_e2e_seconds_count", '{outcome="ok"}') in snapshot
    assert not any(name == "other_metric" for name, _ in snapshot)


def test_histogram_delta_counts_only_the_new_observations():
    out = sm.histogram_delta(
        sm.parse_video_metrics(BEFORE),
        sm.parse_video_metrics(AFTER),
        "tt_video_queue_wait_seconds",
    )
    assert out["n"] == 4
    assert out["mean"] == pytest.approx(2.75)
    assert out["p50"] == pytest.approx(1.0)  # half of the videos waited up to 1 s
    assert 1.0 < out["p95"] <= 10.0


def test_a_phase_without_observations_is_empty():
    snapshot = sm.parse_video_metrics(AFTER)
    assert sm.histogram_delta(snapshot, snapshot, "tt_video_queue_wait_seconds") == {
        "n": 0
    }
    assert sm.histogram_delta(snapshot, snapshot, "tt_video_missing") == {"n": 0}


def test_the_label_filter_picks_one_series():
    out = sm.histogram_delta(
        sm.parse_video_metrics(BEFORE),
        sm.parse_video_metrics(AFTER),
        "tt_video_e2e_seconds",
        'outcome="ok"',
    )
    assert out["n"] == 4 and out["mean"] == pytest.approx(80.0)
    assert (
        sm.histogram_delta(
            sm.parse_video_metrics(BEFORE),
            sm.parse_video_metrics(AFTER),
            "tt_video_e2e_seconds",
            'outcome="error"',
        )["n"]
        == 0
    )


def test_server_side_lists_only_what_the_server_measured():
    out = sm.server_side_video_metrics(
        sm.parse_video_metrics(BEFORE), sm.parse_video_metrics(AFTER)
    )
    assert set(out) == {"queue_wait_s", "e2e_ok_s"}  # no generation or overhead data
    assert out["queue_wait_s"]["n"] == 4


def test_a_server_without_metrics_gives_nothing():
    assert sm.server_side_video_metrics({}, sm.parse_video_metrics(AFTER)) == {}
    assert sm.server_side_video_metrics(sm.parse_video_metrics(AFTER), {}) == {}


def test_scrape_survives_a_server_without_metrics(monkeypatch):
    def _refused(*args, **kwargs):
        raise ConnectionError("no /metrics")

    monkeypatch.setattr(sm.requests, "get", _refused)
    assert sm.scrape_video_metrics("http://localhost:8000") == {}

    monkeypatch.setattr(
        sm.requests, "get", lambda *a, **k: SimpleNamespace(status_code=404, text="")
    )
    assert sm.scrape_video_metrics("http://localhost:8000") == {}


def test_scrape_reads_the_body_of_a_server_with_metrics(monkeypatch):
    seen = {}

    def _get(url, timeout):
        seen["url"] = url
        return SimpleNamespace(status_code=200, text=AFTER)

    monkeypatch.setattr(sm.requests, "get", _get)
    snapshot = sm.scrape_video_metrics("http://localhost:8000")
    assert seen["url"] == "http://localhost:8000/metrics"
    assert snapshot[("tt_video_queue_wait_seconds_count", "")] == 4


PYTHON_AFTER = """\
tt_media_server_video_generation_duration_seconds_bucket{status="success",le="120"} 0
tt_media_server_video_generation_duration_seconds_bucket{status="success",le="+Inf"} 2
tt_media_server_video_generation_duration_seconds_sum{status="success"} 260
tt_media_server_video_generation_duration_seconds_count{status="success"} 2
tt_media_server_video_generation_duration_seconds_count{status="failed"} 1
tt_media_server_video_denoise_duration_seconds_bucket{le="+Inf"} 2
tt_media_server_video_denoise_duration_seconds_sum 230
tt_media_server_video_denoise_duration_seconds_count 2
tt_media_server_video_vae_decode_duration_seconds_bucket{le="+Inf"} 2
tt_media_server_video_vae_decode_duration_seconds_sum 24
tt_media_server_video_vae_decode_duration_seconds_count 2
"""

CPP_STAGES = """\
tt_video_denoise_seconds_bucket{le="+Inf"} 2
tt_video_denoise_seconds_sum 200
tt_video_denoise_seconds_count 2
tt_video_vae_decode_seconds_bucket{le="+Inf"} 2
tt_video_vae_decode_seconds_sum 20
tt_video_vae_decode_seconds_count 2
"""


def test_the_python_servers_metrics_are_read_under_the_same_keys():
    out = sm.server_side_video_metrics(
        {("tt_media_server_video_generation_duration_seconds_count", "x"): 0.0},
        sm.parse_video_metrics(PYTHON_AFTER),
    )
    assert out["e2e_ok_s"]["n"] == 2  # the failed generation is not counted
    assert out["e2e_ok_s"]["mean"] == pytest.approx(130.0)
    assert out["denoise_s"]["mean"] == pytest.approx(115.0)
    assert out["vae_decode_s"]["mean"] == pytest.approx(12.0)
    assert "queue_wait_s" not in out  # the Python server has no such series


def test_the_cpp_servers_stage_histograms_are_read():
    before = sm.parse_video_metrics("tt_video_denoise_seconds_count 0")
    out = sm.server_side_video_metrics(before, sm.parse_video_metrics(CPP_STAGES))
    assert out["denoise_s"]["mean"] == pytest.approx(100.0)
    assert out["vae_decode_s"]["mean"] == pytest.approx(10.0)
