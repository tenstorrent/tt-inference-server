# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""TTS benchmarks: a sequential TTFT/RTR run, or for the tt-tts server
(``tts_api=tt_tts``) a chunk-aware closed-loop concurrency sweep.

The tt-tts sweep replays, by default, the client traffic captured on 2026-10-27
(datasets/tts_staged_2026-10-27.json): a text and a voice prompt length drawn
independently, weighted by how often each appeared (p50 ~740 tokens, 200-1600).
The capture kept only voice prompt lengths, not the voices, so
``--workflow-args voice_ids=<file>`` supplies real speech IDs: a JSON list (or
``{"speech_ids": [...]}``) of at least 1264 codec IDs (~25 s) of real speech; a
request drawing length N sends its first N IDs. Without that file each length
gets seeded random IDs: the ISL matches, but the model cannot clone a random
"voice", so many outputs run to the output cap. ``isl=N`` instead sends one
fixed text of ~N tokens and no voice.

``--workflow-args`` keys: ``concurrency=1,2,4`` ``isl=N`` ``voice_ids=<file>``
``warmup=S`` ``duration=S`` ``prewarm`` (discarded warmup burst before the
sweep) and ``prewarm_only`` (only that burst, no measurement).
"""

from __future__ import annotations

import array
import asyncio
import base64
import json
import logging
import math
import random
import shlex
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from workflow_module.context_helpers import get_num_calls

from report_module.schema import Block
from report_module.status import TestStatus

from .._test_common import (
    MetricSpec,
    ReportCheckTypes,
    block_id,
    run_tiered_check,
)
from ..context import (
    TEST_PAYLOADS_PATH,
    HardwareRequirement,
    MediaContext,
    require_health,
)
from ..eval_tests.tts_client import (
    TTS_API_TT_TTS,
    MediaServerTTSClient,
    TTSClient,
    VoiceClip,
    make_tts_client,
)
from ..test_status import TtsTestStatus
from . import tts_load_harness

logger = logging.getLogger(__name__)


DEFAULT_TTS_TEXT = "Hello, this is a test of the text to speech system."
# Reference clip for deployments that can only clone (Qwen3-TTS Base).
VOICE_CLIP_PAYLOAD = "image_client_audio_payload"


def _tts_num_calls(ctx: MediaContext, is_eval: bool = False) -> int:
    base = get_num_calls(ctx)
    if base != 2:
        logger.info(f"Using configured num_eval_runs: {base} calls")
        return base
    tts_default = 5 if is_eval else 10
    workflow_type = "eval" if is_eval else "benchmark"
    logger.info(
        f"Using TTS-specific {workflow_type} default: {tts_default} calls (was {base})"
    )
    return tts_default


def _tts_test_text(ctx: MediaContext) -> str:
    if (
        not isinstance(ctx.all_params, (list, tuple))
        and hasattr(ctx.all_params, "tasks")
        and len(ctx.all_params.tasks) > 0
    ):
        task = ctx.all_params.tasks[0]
        if hasattr(task, "text"):
            return task.text
        if hasattr(task, "task_name"):
            return task.task_name
    return DEFAULT_TTS_TEXT


def _tts_client_targets(ctx: MediaContext) -> dict:
    """Client settings (``tts_api`` etc.) from the model's eval tasks, else {}."""
    for task in getattr(ctx.all_params, "tasks", None) or []:
        kwargs = getattr(task, "model_kwargs", None) or {}
        if "tts_api" in kwargs:
            return dict(kwargs)
    return {}


def _benchmark_voice_clip() -> VoiceClip:
    """The audio benchmark's LibriSpeech clip (5.3 s), peak-normalized PCM16 WAV."""
    with open(_PROJECT_ROOT / TEST_PAYLOADS_PATH / VOICE_CLIP_PAYLOAD) as f:
        wav = base64.b64decode(json.load(f)["file"])
    chunks, pos = {}, 12
    while pos + 8 <= len(wav):
        (size,) = struct.unpack("<I", wav[pos + 4 : pos + 8])
        chunks[wav[pos : pos + 4]] = wav[pos + 8 : pos + 8 + size]
        pos += 8 + size + (size & 1)
    audio_format, channels, sample_rate, _, _, bits = struct.unpack(
        "<HHIIHH", chunks[b"fmt "][:16]
    )
    if (audio_format, channels, bits) != (3, 1, 32):
        raise ValueError(f"{VOICE_CLIP_PAYLOAD} is not mono float32")
    samples = array.array("f", chunks[b"data"])
    gain = 0.9 * 32767 / max(max(abs(x) for x in samples), 1e-9)
    pcm = array.array("h", (round(x * gain) for x in samples))
    if sys.byteorder == "big":
        pcm.byteswap()
    data = pcm.tobytes()
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(data),
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        sample_rate,
        sample_rate * 2,
        2,
        16,
        b"data",
        len(data),
    )
    return VoiceClip(header + data)


async def _generate_speech(
    ctx: MediaContext,
    client: TTSClient,
    text: str,
    voice: Optional[VoiceClip] = None,
) -> tuple[bool, float, Optional[float], Optional[float], Optional[float]]:
    logger.info("🔊 Calling TTS /v1/audio/speech endpoint")
    url = f"{ctx.base_url}/v1/audio/speech"
    try:
        result = await client.timed_synthesize(url, text, voice=voice, timeout_s=120)
        if not result.ok:
            return False, 0.0, None, None, None
        total_duration = result.elapsed_s
        ttft_ms = result.ttft_ms
        audio_duration = result.audio_seconds

        rtr = None
        if audio_duration is not None and total_duration > 0:
            rtr = audio_duration / total_duration
            logger.info(
                f"Calculated RTR: {rtr:.2f} (audio_duration={audio_duration}s, "
                f"processing_time={total_duration:.2f}s)"
            )
        else:
            logger.warning(
                "Could not calculate RTR: missing duration or invalid processing time"
            )

        rtr_str = f"{rtr:.2f}" if rtr is not None else "N/A"
        logger.info(
            f"✅ Done in {total_duration:.2f}s | TTFT={ttft_ms:.2f}ms | RTR={rtr_str}"
        )
        return True, total_duration, ttft_ms, rtr, audio_duration

    except Exception as e:
        logger.error(f"TTS generation failed: {type(e).__name__}: {e}")
        return False, 0.0, None, None, None


def _run_tts_benchmark(ctx: MediaContext, num_calls: int) -> list[TtsTestStatus]:
    logger.info(f"Running TTS benchmark with {num_calls} calls.")
    status_list: list[TtsTestStatus] = []
    client = make_tts_client(_tts_client_targets(ctx))
    # A Qwen3 deployment's eval tasks are not texts, so it reads the default sentence.
    if isinstance(client, MediaServerTTSClient):
        test_text = _tts_test_text(ctx)
    else:
        test_text = DEFAULT_TTS_TEXT
    voice = None if client.has_default_voice else _benchmark_voice_clip()
    logger.info(f"TTS client: {client.describe()}, cloned={voice is not None}")

    for i in range(num_calls):
        logger.info(f"Generating speech {i + 1}/{num_calls}...")
        status, elapsed, ttft_ms, rtr, audio_duration = asyncio.run(
            _generate_speech(ctx, client, test_text, voice)
        )
        logger.debug(f"Generated speech in {elapsed:.2f} seconds.")
        status_list.append(
            TtsTestStatus(
                status=status,
                elapsed=elapsed,
                ttft_ms=ttft_ms,
                rtr=rtr,
                text=test_text,
                audio_duration=audio_duration,
            )
        )
    return status_list


def _tts_avg(status_list: list[TtsTestStatus], attr: str) -> Optional[float]:
    valid = [getattr(s, attr) for s in status_list if getattr(s, attr) is not None]
    return sum(valid) / len(valid) if valid else None


def _tts_target_checks(
    ctx: MediaContext, ttft_ms_value: Optional[float], rtr_value: Optional[float]
) -> tuple[dict, ReportCheckTypes]:
    logger.info("Computing 3-tier target checks for TTFT, RTR")
    return run_tiered_check(
        ctx,
        [
            MetricSpec(
                "TTFT",
                ttft_ms_value,
                "ttft_ms",
                lower_is_better=True,
                field_name="ttft",
            ),
            MetricSpec(
                "RTR", rtr_value, "rtr", lower_is_better=False, field_name="rtr"
            ),
        ],
    )


def _tts_ttft_percentiles(
    status_list: list[TtsTestStatus],
) -> tuple[float, float, float]:
    """Return (P50, P90, P95) of TTFT in ms across successful requests."""
    logger.info("Calculating TTFT percentiles (P50, P90, P95)")
    valid = [s.ttft_ms for s in status_list if s.ttft_ms is not None]
    if not valid:
        return 0.0, 0.0, 0.0
    sorted_ttft = sorted(valid)
    n = len(sorted_ttft)

    def _percentile(fraction: float) -> float:
        index = min(math.ceil(n * fraction) - 1, n - 1)
        return sorted_ttft[max(index, 0)]

    return _percentile(0.5), _percentile(0.9), _percentile(0.95)


def _tts_throughput_rps(
    status_list: list[TtsTestStatus], wall_seconds: float
) -> Optional[float]:
    """Requests-per-second over the benchmark wall-clock (informational).

    The TTS benchmark issues requests sequentially, so this reflects
    serial end-to-end throughput, not peak concurrent throughput. There is
    no throughput target for TTS, so this metric is display-only.
    """
    successful = sum(1 for s in status_list if s.status)
    if wall_seconds <= 0 or successful == 0:
        return None
    return successful / wall_seconds


# ------------------------------------------------- tt-tts load sweep (tts_api=tt_tts)

TTS_LOAD_CONCURRENCY = (1, 2, 4, 8, 16)
TTS_LOAD_DURATION_SECONDS = 60.0
TTS_LOAD_WARMUP_SECONDS = 20.0
# Optional discarded burst before the sweep (``prewarm``) to burn off cold start
# (a fresh server's first request can take >10 s to first audio).
TTS_PREWARM_CONCURRENCY = 100
TTS_PREWARM_SECONDS = 60.0
TTS_LOAD_SETTLE_SECONDS = 5.0
# Fixed-text mode (``isl=N``): characters per text token.
TTS_CHARS_PER_TOKEN = 3.46
TTS_TRAFFIC_DATASET = (
    Path(__file__).resolve().parent / "datasets" / "tts_staged_2026-10-27.json"
)
TTS_TRAFFIC_SEED = 42
TTS_CODEBOOK_SIZE = 65536  # speech token IDs are <|s_0|>..<|s_65535|>
TTS_SPEECH_IDS_PER_SECOND = 50
TTS_LOAD_TIMEOUT_SECONDS = 300.0
TTS_RAW_RESULTS_FILENAME = "tts_load_results.json"

# (display name, report field); the field is also the PerformanceTargets attr.
_LATENCY_SPECS = (
    ("FC P50", "fc_p50_ms"),
    ("FC P95", "fc_p95_ms"),
    ("SC P50", "sc_p50_ms"),
    ("SC P95", "sc_p95_ms"),
    ("TC P50", "tc_p50_ms"),
    ("TC P95", "tc_p95_ms"),
    ("4C+ P50", "c4_p50_ms"),
    ("4C+ P95", "c4_p95_ms"),
)
# report prefix -> harness metric key
_HARNESS_KEYS = {"fc": "ttfb", "sc": "ttfs", "tc": "ttft", "c4": "ttf4"}


@dataclass(frozen=True)
class LoadParams:
    """The tt-tts sweep's settings, from ``--workflow-args`` over the defaults."""

    concurrency: tuple[int, ...] = TTS_LOAD_CONCURRENCY
    isl: Optional[int] = None  # None: replay the captured traffic
    voice_ids: Optional[str] = None
    warmup: float = TTS_LOAD_WARMUP_SECONDS
    duration: float = TTS_LOAD_DURATION_SECONDS
    prewarm: bool = False
    prewarm_only: bool = False


_INT_KEYS = ("isl", "warmup", "duration")
_FLAG_KEYS = ("prewarm", "prewarm_only")
_LOAD_ARG_KEYS = ("concurrency", "voice_ids") + _INT_KEYS + _FLAG_KEYS


def _positive_int(key: str, value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        parsed = 0
    if parsed <= 0:
        raise ValueError(
            f"TTS benchmark workflow arg {key}={value!r} must be a positive integer"
        )
    return parsed


def _flag(key: str, value: str) -> bool:
    if value.lower() in ("", "1", "true", "yes"):
        return True
    if value.lower() in ("0", "false", "no"):
        return False
    raise ValueError(f"TTS benchmark workflow arg {key}={value!r} must be a boolean")


def _load_params(ctx: MediaContext) -> LoadParams:
    """Parse ``--workflow-args`` (``key=value`` tokens; a bare flag means true)."""
    rc = getattr(ctx, "runtime_config", None)
    params = {}
    for token in shlex.split(getattr(rc, "workflow_args", None) or ""):
        key, _, value = token.partition("=")
        params[key.replace("-", "_")] = value
    # eval_* keys belong to the correctness eval (both run under --workflow release).
    unknown = sorted(
        k for k in set(params) - set(_LOAD_ARG_KEYS) if not k.startswith("eval_")
    )
    if unknown:
        logger.warning(
            f"TTS benchmark ignores workflow args {unknown}; "
            f"supported: {list(_LOAD_ARG_KEYS)}"
        )
    kwargs: dict = {}
    if "concurrency" in params:
        values = [v for v in params["concurrency"].split(",") if v.strip()]
        if not values:
            raise ValueError("TTS benchmark workflow arg concurrency is empty")
        # dict.fromkeys drops repeats but keeps the order the levels run in.
        kwargs["concurrency"] = tuple(
            dict.fromkeys(_positive_int("concurrency", v) for v in values)
        )
    for key in _INT_KEYS:
        if key in params:
            kwargs[key] = _positive_int(key, params[key])
    for key in _FLAG_KEYS:
        if key in params:
            kwargs[key] = _flag(key, params[key])
    if params.get("voice_ids"):
        kwargs["voice_ids"] = params["voice_ids"]
    return LoadParams(**kwargs)


def _finite(value) -> Optional[float]:
    """A JSON-safe float, or None for a missing (NaN) sample."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _report_rows(rows: list[dict]) -> list[dict]:
    """Harness rows in the report's FC/SC/TC/4C+ vocabulary."""
    report_rows = []
    for row in rows:
        out = {
            "concurrency": row["conc"],
            "average_concurrency": _finite(row["C"]),
            "num_successful": row["n_ok"],
            "error_request_count": row["n_err"],
            "throughput_rps": _finite(row["rps"]),
            "million_chars_per_hour": _finite(row["mchar_h"]),
            "average_chunks": _finite(row["chunks"]),
        }
        for prefix, key in _HARNESS_KEYS.items():
            for stat in ("p50", "p95", "max"):
                out[f"{prefix}_{stat}_ms"] = _finite(row[f"{key}_{stat}"])
        report_rows.append(out)
    return report_rows


def _results_are_complete(rows: list[dict], expected: tuple[int, ...]) -> bool:
    """Every level ran, had successes, no errors, and every latency sample."""
    return {row["concurrency"] for row in rows} == set(expected) and all(
        row["num_successful"] > 0
        and row["error_request_count"] == 0
        and all(row[field] is not None for _, field in _LATENCY_SPECS)
        for row in rows
    )


def _load_target_checks(
    ctx: MediaContext, rows: list[dict], expected: tuple[int, ...]
) -> tuple[dict, ReportCheckTypes]:
    """Grade the worst level of each latency percentile (so every level must pass),
    and fail outright on a partial sweep or any error."""
    worst = {
        field: max((row[field] for row in rows if row[field] is not None), default=None)
        for _, field in _LATENCY_SPECS
    }
    target_checks, target_check = run_tiered_check(
        ctx,
        [
            MetricSpec(
                name,
                worst[field],
                field,
                lower_is_better=True,
                field_name=field,
                inclusive=True,
            )
            for name, field in _LATENCY_SPECS
        ],
    )
    complete = _results_are_complete(rows, expected)
    integrity = ReportCheckTypes.PASS if complete else ReportCheckTypes.FAIL
    for tier in target_checks.values():
        tier["benchmark_integrity"] = 1
        tier["benchmark_integrity_ratio"] = 1.0 if complete else 0.0
        tier["benchmark_integrity_check"] = integrity
    if not complete:
        target_check = ReportCheckTypes.FAIL
    return target_checks, target_check


def _load_voice_ids(path: str) -> list[int]:
    with open(path) as f:
        data = json.load(f)
    ids = data["speech_ids"] if isinstance(data, dict) else data
    if not ids or not all(
        isinstance(i, int) and 0 <= i < TTS_CODEBOOK_SIZE for i in ids
    ):
        raise ValueError(
            f"TTS voice_ids file {path} must hold speech IDs in [0, {TTS_CODEBOOK_SIZE})"
        )
    return ids


def _traffic_draw(
    base: tts_load_harness.Target,
    voice_ids: Optional[list[int]] = None,
    dataset: Path = TTS_TRAFFIC_DATASET,
    seed: int = TTS_TRAFFIC_SEED,
) -> tts_load_harness.Draw:
    """Per-request prompts drawn like the captured traffic (see module docstring)."""
    with open(dataset) as f:
        ladder = json.load(f)["ladder"]
    texts = [t["text"] for t in ladder["texts"]]
    text_weights = [t["count"] for t in ladder["texts"]]
    sizes = [v["voice_speech_ids"] for v in ladder["voices"]]
    size_weights = [v["count"] for v in ladder["voices"]]
    if voice_ids is None:
        logger.warning(
            "TTS replay without voice_ids: voice prompts are seeded random speech "
            "IDs (right ISL, no real voice; expect more runaway outputs)"
        )
        id_rng = random.Random(seed)
        voices = {
            n: [id_rng.randrange(TTS_CODEBOOK_SIZE) for _ in range(n)]
            for n in sorted(set(sizes))
        }
    elif len(voice_ids) < max(sizes):
        raise ValueError(
            f"TTS voice_ids has {len(voice_ids)} IDs; the captured traffic needs up "
            f"to {max(sizes)} ({max(sizes) / TTS_SPEECH_IDS_PER_SECOND:.1f} s of speech)"
        )
    else:
        # A voice prompt of N IDs is the first N IDs of the same real speech.
        voices = {n: voice_ids[:n] for n in sizes}
    rng = random.Random(seed)

    def draw() -> tts_load_harness.Target:
        text = rng.choices(texts, text_weights)[0]
        size = rng.choices(sizes, size_weights)[0]
        return base.with_prompt(text, voices[size])

    return draw


def _prewarm_block(ctx: MediaContext, requests_done: int) -> Block:
    return Block(
        kind="benchmarks",
        task_type="text_to_speech",
        title="Text-to-Speech Prewarm",
        id=block_id(ctx) or None,
        targets={
            "warmup_concurrency": TTS_PREWARM_CONCURRENCY,
            "prewarm_seconds": TTS_PREWARM_SECONDS,
        },
        data={
            # Not a measurement: SKIP so acceptance does not grade it.
            "status": TestStatus.SKIP.value,
            "Prewarm": {
                "warmup_concurrency": TTS_PREWARM_CONCURRENCY,
                "prewarm_seconds": TTS_PREWARM_SECONDS,
                "requests_completed": requests_done,
                "measured": False,
            },
        },
    )


def run_tts_load_benchmark(ctx: MediaContext) -> Block:
    """Closed-loop FC/SC/TC/4C+ concurrency sweep against the tt-tts server."""
    # tt-tts is one worker owning the whole board, so the FULL_BOARD tier (which
    # expects max_concurrency ready workers) can never pass; ANY_CHIP is the board.
    require_health(ctx, HardwareRequirement.ANY_CHIP)
    params = _load_params(ctx)
    replay = params.isl is None
    text = (
        tts_load_harness.build_text(round(params.isl * TTS_CHARS_PER_TOKEN))
        if params.isl
        else DEFAULT_TTS_TEXT  # replay: only the preflight sends this
    )
    target = tts_load_harness.Target(
        url=f"{ctx.base_url}{tts_load_harness.SPEECH_PATH}",
        text=text,
        timeout=TTS_LOAD_TIMEOUT_SECONDS,
    )
    voice_ids = (
        _load_voice_ids(params.voice_ids) if replay and params.voice_ids else None
    )
    draw = _traffic_draw(target, voice_ids) if replay else None

    if params.prewarm_only:
        done = tts_load_harness.prewarm(
            target, TTS_PREWARM_CONCURRENCY, TTS_PREWARM_SECONDS, draw
        )
        return _prewarm_block(ctx, done)

    if replay:
        voice = f"{len(voice_ids)} real voice IDs" if voice_ids else "random voice IDs"
        workload = f"{TTS_TRAFFIC_DATASET.name}, {voice}"
    else:
        workload = f"isl={params.isl}"
    logger.info(
        f"TTS load sweep: concurrency={params.concurrency} {workload} "
        f"warmup={params.warmup:.0f}s duration={params.duration:.0f}s"
    )
    output_path = Path(ctx.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    document = tts_load_harness.sweep(
        tts_load_harness.SweepConfig(
            concurrency=params.concurrency,
            duration=params.duration,
            warmup=params.warmup,
            settle=TTS_LOAD_SETTLE_SECONDS,
            # Replayed responses vary in length; staggered starts keep workers
            # from completing and resending in lockstep.
            ramp=params.warmup if replay else 0.0,
            prewarm=TTS_PREWARM_SECONDS if params.prewarm else 0.0,
            prewarm_concurrency=TTS_PREWARM_CONCURRENCY,
            out=output_path / TTS_RAW_RESULTS_FILENAME,
        ),
        target,
        draw,
    )
    rows, chars_per_request = tts_load_harness.aggregate(document)
    report_rows = _report_rows(rows)
    target_checks, target_check = _load_target_checks(
        ctx, report_rows, params.concurrency
    )
    num_successful = sum(row["num_successful"] for row in report_rows)
    error_request_count = sum(row["error_request_count"] for row in report_rows)
    return Block(
        kind="benchmarks",
        task_type="text_to_speech",
        title="Text-to-Speech Load Benchmark",
        id=block_id(ctx) or None,
        targets={
            "concurrency": list(params.concurrency),
            "text_tokens": params.isl,
            "workload": workload,
        },
        data={
            "Benchmarks": {
                "num_requests": num_successful + error_request_count,
                "num_successful": num_successful,
                "error_request_count": error_request_count,
                "chars_per_request": chars_per_request,
                "arrival_model": "closed",
                "raw_results": TTS_RAW_RESULTS_FILENAME,
                "target_check": target_check,
                "target_checks": target_checks,
            },
            "Latency by Concurrency": report_rows,
        },
    )


def run_tts_benchmark(ctx: MediaContext) -> Block:
    """Run benchmarks for a TTS model (SpeechT5, Qwen3-TTS, tt-tts, ...)."""
    logger.info(
        f"Running benchmarks for model: {ctx.model_spec.model_name} on device: {ctx.device.name}"
    )
    if _tts_client_targets(ctx).get("tts_api") == TTS_API_TT_TTS:
        return run_tts_load_benchmark(ctx)
    require_health(ctx)

    try:
        num_calls = _tts_num_calls(ctx, is_eval=False)
        bench_start = time.monotonic()
        status_list = _run_tts_benchmark(ctx, num_calls)
        wall_seconds = time.monotonic() - bench_start
    except Exception as e:
        logger.error(f"Benchmark execution encountered an error: {e}")
        raise

    logger.info("Generating benchmark report...")
    ttft_value = _tts_avg(status_list, "ttft_ms")
    rtr_value = _tts_avg(status_list, "rtr")
    p50_ttft, p90_ttft, p95_ttft = _tts_ttft_percentiles(status_list)
    throughput_rps = _tts_throughput_rps(status_list, wall_seconds)
    target_checks, target_check = _tts_target_checks(ctx, ttft_value, rtr_value)

    return Block(
        kind="benchmarks",
        task_type="text_to_speech",
        title="Text-to-Speech Benchmark",
        id=block_id(ctx) or None,
        targets={"num_prompts": len(status_list)},
        data={
            "Benchmarks": {
                "num_requests": len(status_list),
                "ttft": ttft_value / 1000 if ttft_value is not None else None,
                "rtr": rtr_value,
                # ttft_p50 and throughput_rps are informational only: TTS has
                # no P50/throughput targets, so they carry no target check.
                "ttft_p50": p50_ttft / 1000,
                "ttft_p90": p90_ttft / 1000,
                "ttft_p95": p95_ttft / 1000,
                "throughput_rps": throughput_rps,
                "target_check": target_check,
                "target_checks": target_checks,
            },
        },
    )


__all__ = ["run_tts_benchmark"]
