# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
import asyncio
import dataclasses
import json
from types import SimpleNamespace

import pytest

from llm_module import vllm_bench_stream_timing as timing


def _run(monkeypatch, events, *, handler_bytes=True):
    clock = [0.0]
    monkeypatch.setattr(timing.time, "perf_counter", lambda: clock[0])
    recorded = {}

    class Content:
        async def iter_any(self):
            for at, bodies in events:
                clock[0] = at
                if isinstance(bodies, dict):
                    bodies = [bodies]
                yield b"".join(("data: " + json.dumps(body) + "\n\n").encode() for body in bodies)
            yield b"data: [DONE]\n\n"

    class Response:
        content = Content()
        status = 200
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass

    class Session:
        def post(self, **kwargs):
            recorded.update(kwargs)
            return Response()

    endpoint = SimpleNamespace(
        _validate_api_url=lambda *a: None,
        _get_chat_content=lambda inp, **kw: [{"type": "text", "text": inp.prompt}],
        _update_payload_common=lambda payload, inp: payload.update(inp.extra_body),
        _update_headers_common=lambda *a: None,
        RequestFuncOutput=lambda: SimpleNamespace(success=False, error="", ttft=0.0, latency=0.0, output_tokens=0, itl=[]),
        StreamedResponseHandler=lambda: SimpleNamespace(add_chunk=lambda chunk: (chunk.strip().split(b"\n\n") if handler_bytes else chunk.decode().strip().split("\n\n"))),
    )
    inp = SimpleNamespace(api_url="http://host/v1/chat/completions", model_name=None,
                          model="Qwen/Qwen3.8-27B", prompt="unchanged", prompt_len=8,
                          output_len=8, extra_body={"cache_salt": "base", "seed": 42})
    return asyncio.run(timing.make_chat_request_func(endpoint)(inp, Session())), recorded


def _delta(**value):
    return {"choices": [{"delta": value}]}


def _events(key="reasoning_content"):
    return [(1, _delta(role="assistant", content="")), (3, _delta(**{key: "one"})),
            (4, _delta(content="")), (5, _delta(**{key: " two"})),
            (8, {"choices": [{"delta": {}, "finish_reason": "length"}]}),
            (10, {"choices": [], "usage": {"prompt_tokens": 60, "completion_tokens": 2}})]


@pytest.mark.parametrize("key", ["reasoning_content", "reasoning", "content"])
def test_nonempty_emissions_ignore_delayed_empty_role_finish_and_usage(monkeypatch, key):
    out, request = _run(monkeypatch, _events(key))
    assert out.success, out.error
    assert out.ttft == 3 and out.latency == 5 and out.itl == [2]
    assert out.output_tokens == 2 and out.prompt_len == 60
    assert out.generated_text == ("one two" if key == "content" else "")
    assert request["json"] == {"model": "Qwen/Qwen3.8-27B", "messages": [{"role": "user", "content": [{"type": "text", "text": "unchanged"}]}],
                               "temperature": 0.0, "max_completion_tokens": 8, "stream": True,
                               "stream_options": {"include_usage": True}, "cache_salt": "base", "seed": 42}


def test_coalesced_sse_events_are_one_observed_arrival(monkeypatch):
    events = _events()
    events[1] = (3, [_delta(reasoning_content="a"), _delta(reasoning_content="b")])
    events[3] = (5, [_delta(reasoning_content="c"), _delta(content="answer")])
    events[-1][1]["usage"]["completion_tokens"] = 7
    out, _ = _run(monkeypatch, events)
    assert out.success and out.ttft == 3 and out.latency == 5
    assert out.itl == [2] and out.output_tokens == 7 and out.generated_text == "answer"


@pytest.mark.parametrize("missing", ["emission", "usage", "finish"])
def test_missing_observable_emission_usage_or_finish_is_invalid(monkeypatch, missing):
    events = _events()
    if missing == "emission":
        events = [e for i, e in enumerate(events) if i not in (1, 3)]
    elif missing == "usage":
        events.pop()
    else:
        events.pop(4)
    out, _ = _run(monkeypatch, events)
    assert not out.success and "Missing observable" in out.error


def test_in_stream_error_invalidates_observations(monkeypatch):
    out, _ = _run(monkeypatch, _events() + [(11, {"error": {"message": "engine died"}})])
    assert not out.success and "engine died" in out.error


def test_metrics_use_usage_counts_without_changing_requests():
    @dataclasses.dataclass
    class Request:
        prompt_len: int
    req = [Request(8), Request(8)]
    outs = [SimpleNamespace(success=True, prompt_len=60), SimpleNamespace(success=False, prompt_len=0)]
    captured = timing.make_calculate_metrics(lambda actual, outputs: actual)(req, outs)
    assert [r.prompt_len for r in captured] == [60, 8]
    assert [r.prompt_len for r in req] == [8, 8]


def test_invalid_latency_placeholders_become_null_and_protocol_is_explicit(tmp_path):
    path = tmp_path / "raw.json"
    path.write_text(json.dumps({"backend": "openai-chat", "completed": 0, "failed": 1, "errors": ["no emission"], "ttfts": [0],
                                "mean_ttft_ms": 0, "p99_tpot_ms": 0, "mean_itl_ms": 0, "mean_e2el_ms": 0}))
    timing.finalize_result(["--save-result", "--result-filename", str(path)])
    raw = json.loads(path.read_text())
    assert raw["ttfts"] == [None] and raw["mean_ttft_ms"] is None
    assert raw["p99_tpot_ms"] is None and raw["mean_itl_ms"] is None and raw["mean_e2el_ms"] is None
    assert raw["failed"] == 1 and raw["tt_timing_protocol"] == timing.PROTOCOL
    assert "coalesced SSE" in raw["tt_timing_limitations"]


def test_valid_aggregate_survives_failed_request_placeholder(tmp_path):
    path = tmp_path / "raw.json"
    path.write_text(json.dumps({"backend": "openai-chat", "completed": 1, "failed": 1, "errors": ["", "failed"], "ttfts": [2, 0], "mean_ttft_ms": 2000}))
    timing.finalize_result(["--save-result", "--result-filename=" + str(path)])
    raw = json.loads(path.read_text())
    assert raw["ttfts"] == [2, None] and raw["mean_ttft_ms"] == 2000


def test_timing_semantics_reach_report_data_and_markdown():
    from llm_module.parsers.vllm import VLLMBenchParser
    from report_module.renderers import render_benchmarks
    block = VLLMBenchParser().parse({"completed": 1, "model_id": "Qwen/Qwen3.8-27B",
        "tt_timing_protocol": timing.PROTOCOL, "tt_timing_limitations": "Coalesced events are not model-token intervals."})
    assert block.data["timing_protocol"] == timing.PROTOCOL
    rendered = render_benchmarks(block, {})
    assert timing.PROTOCOL in rendered
    assert "Coalesced events are not model-token intervals." in rendered


def test_string_returning_sse_handler_is_supported(monkeypatch):
    out, _ = _run(monkeypatch, _events(), handler_bytes=False)
    assert out.success and out.ttft == 3 and out.latency == 5


@pytest.mark.parametrize("args,expected", [([], False), (["--backend", "openai"], False),
    (["--backend=openai-chat"], True), (["--backend", "openai-chat"], True),
    (["--backend=openai-chat", "--backend", "openai"], False)])
def test_only_explicit_effective_chat_backend_activates_timing(args, expected):
    assert timing.requested_chat_backend(args) is expected


def test_non_chat_results_are_not_relabelled(tmp_path):
    path = tmp_path / "raw.json"
    original = json.dumps({"backend": "openai", "completed": 0, "mean_ttft_ms": 0})
    path.write_text(original)
    timing.finalize_result(["--save-result", "--result-filename", str(path)])
    assert path.read_text() == original


@pytest.mark.parametrize("output_tokens", [1, 8])
def test_no_observable_interval_nulls_decode_metrics_but_keeps_ttft(tmp_path, output_tokens):
    path = tmp_path / "raw.json"
    path.write_text(json.dumps({"backend": "openai-chat", "completed": 1, "failed": 0,
        "errors": [""], "ttfts": [2], "itls": [[]], "output_lens": [output_tokens],
        "mean_ttft_ms": 2000, "mean_e2el_ms": 2000, "mean_tpot_ms": 0,
        "p99_itl_ms": 0, "max_output_tokens_per_s": 1, "request_goodput": 1}))
    timing.finalize_result(["--save-result", "--result-filename", str(path)])
    raw = json.loads(path.read_text())
    assert raw["mean_ttft_ms"] == 2000 and raw["mean_e2el_ms"] == 2000
    assert raw["mean_tpot_ms"] is None and raw["p99_itl_ms"] is None
    assert raw["max_output_tokens_per_s"] is None and raw["request_goodput"] is None
    assert raw["tt_tpot_valid_request_count"] == 0


def test_mixed_decode_summaries_exclude_unobservable_intervals(tmp_path):
    path = tmp_path / "raw.json"
    path.write_text(json.dumps({"backend": "openai-chat", "completed": 2, "failed": 0,
        "errors": ["", ""], "itls": [[], [2, 4]], "output_lens": [8, 4],
        "mean_tpot_ms": 1000, "p99_tpot_ms": 1980, "mean_itl_ms": 3000,
        "p99_itl_ms": 3980, "std_itl_ms": 1000}))
    timing.finalize_result(["--save-result", "--result-filename", str(path)])
    raw = json.loads(path.read_text())
    assert raw["mean_tpot_ms"] == 2000 and raw["p99_tpot_ms"] == 2000
    assert raw["mean_itl_ms"] == 3000 and raw["p99_itl_ms"] == 3980
    assert raw["std_itl_ms"] == 1000 and raw["tt_tpot_valid_request_count"] == 1
    assert "TPOT population: 1/2 successful requests; ITL population: 2" in raw["tt_timing_limitations"]


def test_failed_detailed_token_work_is_unknown(tmp_path):
    path = tmp_path / "raw.json"
    path.write_text(json.dumps({"backend": "openai-chat", "completed": 0, "failed": 1,
        "errors": ["connection lost"], "input_lens": [8], "output_lens": [0], "ttfts": [0]}))
    timing.finalize_result(["--save-result", "--result-filename", str(path)])
    raw = json.loads(path.read_text())
    assert raw["input_lens"] == [None] and raw["output_lens"] == [None]
    assert raw["failed"] == 1 and raw["errors"] == ["connection lost"]


def test_result_path_options_follow_argparse_last_wins(tmp_path):
    first, last = tmp_path / "a.json", tmp_path / "b.json"
    original = json.dumps({"backend": "openai-chat", "completed": 1})
    first.write_text(original)
    last.write_text(original)
    timing.finalize_result(["--save-result", "--result-filename", str(first), "--result-filename=" + str(last)])
    assert first.read_text() == original
    assert json.loads(last.read_text())["tt_timing_protocol"] == timing.PROTOCOL
