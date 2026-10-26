# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: SIM117 - nested scopes make timer and exception boundaries explicit
import ast
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from llm_module import suite_metrics as m

LABEL = '{model_name="Qwen/Qwen3.8-27B",engine="0"}'


def body(value=3, created=100, process=50):
    return (
        f"vllm:num_preemptions_total{LABEL} {value}\n"
        f"vllm:num_preemptions_created{LABEL} {created}\n"
        f"process_start_time_seconds {process}\n"
        f"vllm:num_requests_running{LABEL} 0\n"
        f"vllm:num_requests_waiting{LABEL} 0\n"
    )


def snap(text):
    return {"status": "ok", "samples": m.parse_metrics(text)}


def preempt(result):
    return next(
        x for x in result["deltas"] if x["name"] == "vllm:num_preemptions_total"
    )


def test_selected_only_and_safe_labels():
    rows = m.parse_metrics(body() + 'unselected{prompt="secret"} 99\n')
    assert len(rows) == 5
    assert "secret" not in json.dumps(rows)
    assert m.parse_metrics('vllm:num_preemptions_total{model_name="other"} 4') == []


@pytest.mark.parametrize(
    "line",
    [
        f"vllm:num_preemptions_total{LABEL} NaN",
        f"vllm:num_preemptions_total{LABEL} +Inf",
        f"vllm:num_preemptions_total{LABEL} -1",
        'vllm:num_preemptions_total{engine="0",engine="1"} 1',
        'vllm:num_preemptions_total{request_id="secret"} 1',
        'vllm:num_preemptions_total{engine="0" 1',
        'vllm:num_preemptions_total{engine="x"} 1',
        'vllm:request_success_total{finish_reason="secret"} 1',
        body() + body(),
    ],
)
def test_bad_selected_data_fails_closed(line):
    with pytest.raises(ValueError):
        m.parse_metrics(line)


def test_delta_and_engine_identity_limit():
    r = m.summarize(snap(body(3)), snap(body(9)))
    assert preempt(r)["delta"] == 6
    assert preempt(r)["creation_guard_available"]
    assert not r["engine_identity_verified"]
    missing = next(
        x for x in r["deltas"] if x["name"] == "vllm:prefix_cache_hits_total"
    )
    assert missing["delta"] is None


@pytest.mark.parametrize(
    "after,reason",
    [
        (body(2), "counter_decreased"),
        (body(9, created=101), "counter_creation_changed"),
        (body(9, process=51), "api_process_changed"),
        ("", "series_missing_at_boundary"),
        (
            body(9).replace("vllm:num_preemptions_created" + LABEL + " 100\n", ""),
            "counter_creation_guard_changed_presence",
        ),
    ],
)
def test_counter_guards(after, reason):
    r = preempt(m.summarize(snap(body()), snap(after)))
    assert r["delta"] is None and r["missing_reason"] == reason


def test_active_boundaries_and_missing_api_guard_are_explicit():
    text = (
        body()
        .replace("process_start_time_seconds 50\n", "")
        .replace(
            "vllm:num_requests_running" + LABEL + " 0",
            "vllm:num_requests_running" + LABEL + " 1",
        )
    )
    r = m.summarize(snap(text), snap(text))
    assert "api_process_start_unavailable" in r["limitations"]
    assert any("attribution_uncertain" in v for v in r["limitations"])


def test_snapshots_exclude_timing_and_preserve_harness_exception(tmp_path):
    events = []

    def fetch(*_):
        events.append("scrape")
        return body()

    path = tmp_path / "suite.json"
    with patch.object(m, "fetch_metrics", fetch):
        with pytest.raises(RuntimeError, match="harness failed"):
            with m.suite_metrics(path, "swe", "http://server", enabled=True):
                events.extend(["timer_start", "harness", "timer_end"])
                raise RuntimeError("harness failed")
    assert events == ["scrape", "timer_start", "harness", "timer_end", "scrape"]
    assert json.loads(path.read_text())["after"]["status"] == "ok"


def test_scrape_and_write_failure_preserve_outcome(tmp_path):
    with patch.object(
        m, "fetch_metrics", side_effect=TimeoutError("credential-secret")
    ):
        with m.suite_metrics(tmp_path / "s.json", "tb", "http://server", enabled=True):
            rc = 124
    result = json.loads((tmp_path / "s.json").read_text())
    assert rc == 124 and result["before"]["missing_reason"] == "TimeoutError"
    assert "credential-secret" not in json.dumps(result)
    with patch.object(Path, "mkdir", side_effect=OSError("secret")):
        with patch.object(m, "fetch_metrics", return_value=body()):
            with pytest.raises(RuntimeError, match="harness"):
                with m.suite_metrics(
                    tmp_path / "x", "tb", "http://server", enabled=True
                ):
                    raise RuntimeError("harness")


def test_disabled_and_controls_opt_in(tmp_path):
    with patch.object(m, "fetch_metrics") as fetch:
        with m.suite_metrics(tmp_path / "x", "tb", "", enabled=False):
            pass
    fetch.assert_not_called()
    assert not (tmp_path / "x").exists()
    from reference_config.evals import qwen38_goal12 as c

    with patch.object(c, "controls", return_value={"evals": ["tb"]}):
        assert not m.enabled(m.MODEL, "tb")
    with patch.object(
        c, "controls", return_value={"evals": ["tb"], "record_suite_metrics": True}
    ):
        assert m.enabled(m.MODEL, "tb")
        assert not m.enabled(m.MODEL, "swe")
        assert not m.enabled("other", "tb")
    with patch.object(
        Path, "read_text", return_value='{"record_suite_metrics":"true"}'
    ):
        with pytest.raises(ValueError, match="boolean"):
            c.controls()


@pytest.mark.parametrize(
    "url,expected",
    [
        ("http://h:8000/v1", "http://h:8000/metrics"),
        ("https://h/proxy/v1/", "https://h/proxy/metrics"),
    ],
)
def test_url(url, expected):
    assert m.metrics_url(url) == expected


@pytest.mark.parametrize(
    "url", ["file:///tmp/x", "http://secret@h", "http://h?token=secret"]
)
def test_unsafe_url(url):
    with pytest.raises(ValueError):
        m.metrics_url(url)


def test_bounded_body_and_auth_not_persisted():
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read1(self, n):
            return b"x" * n

    with patch.object(m, "build_opener") as opener, patch.object(m, "MAX_BYTES", 16):
        opener.return_value.open.return_value = Response()
        result = m.snapshot("http://h/v1", "secret")
        req = opener.return_value.open.call_args.args[0]
        assert req.headers["Authorization"] == "Bearer secret"
        assert opener.return_value.open.call_args.kwargs["timeout"] == m.TIMEOUT
    assert result["missing_reason"] == "ValueError"
    assert "secret" not in json.dumps(result)
    assert m._NoRedirect().redirect_request(None) is None


def test_actual_call_sites_keep_both_timers_inside_snapshots():
    root = Path(__file__).resolve().parents[2]
    for relative in [
        "llm_module/drivers/agentic.py",
        "test_module/llm_tests/llm_eval_tests.py",
    ]:
        tree = ast.parse((root / relative).read_text())
        outer = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.With)
            and any(
                isinstance(i.context_expr, ast.Call)
                and isinstance(i.context_expr.func, ast.Name)
                and i.context_expr.func.id == "suite_metrics"
                for i in n.items
            )
        ]
        assert len(outer) == 1
        timers = [
            n
            for n in ast.walk(outer[0])
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "timed_attempt"
        ]
        assert len(timers) == 1
        if "llm_eval_tests" in relative:
            assigned = {
                target.id
                for n in ast.walk(outer[0])
                if isinstance(n, ast.Assign)
                for target in n.targets
                if isinstance(target, ast.Name)
            }
            assert {"started_at", "elapsed_seconds"} <= assigned


def test_installed_vllm_finished_reason_label():
    rows = m.parse_metrics(
        'vllm:request_success_total{engine="0",finished_reason="repetition",model_name="Qwen/Qwen3.8-27B"} 0'
    )
    assert rows[0]["labels"]["finished_reason"] == "repetition"


def test_real_timer_excludes_both_scrapes(tmp_path):
    from llm_module.attempt_timing import timed_attempt

    clock = [0.0]

    def fetch(*_):
        clock[0] += 5.0
        return body()

    with patch.object(m, "fetch_metrics", fetch), patch(
        "time.perf_counter", side_effect=lambda: clock[0]
    ):
        with m.suite_metrics(tmp_path / "metrics.json", "tb", "http://h", enabled=True):
            with timed_attempt(tmp_path / "timing.json", "tb", "harbor_harness"):
                clock[0] += 3.0
    assert clock[0] == 13.0
    assert json.loads((tmp_path / "timing.json").read_text())["elapsed_seconds"] == 3.0
