# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC


from __future__ import annotations

import array
import asyncio
import base64
import json
import logging
import math
import struct
import sys
import time
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from workflow_module.context_helpers import get_num_calls

from report_module.renderers import NOTES_KEY
from report_module.schema import Block

from .._test_common import (
    MetricSpec,
    ReportCheckTypes,
    block_id,
    run_tiered_check,
)
from ..context import TEST_PAYLOADS_PATH, MediaContext, require_health
from ..eval_tests.tts_client import (
    MediaServerTTSClient,
    TimedSynthesis,
    TTSClient,
    VoiceClip,
    make_tts_client,
)
from ..eval_tests.tts_eval_tests import resolve_eval_tasks
from ..test_status import TtsTestStatus

logger = logging.getLogger(__name__)


DEFAULT_TTS_TEXT = "Hello, this is a test of the text to speech system."
# Reference clip for deployments that can only clone (Qwen3-TTS Base).
VOICE_CLIP_PAYLOAD = "image_client_audio_payload"
# Concurrency sweep prompts: about 2, 9 and 40 s of speech.
SWEEP_PROMPTS = {
    "short": "Good morning. Today is a beautiful day.",
    "medium": (
        "The quick brown fox jumps over the lazy dog. She sells sea shells by the "
        "sea shore, and the shells she sells are surely sea shells."
    ),
    "long": (
        "It was the best of times, it was the worst of times, it was the age of "
        "wisdom, it was the age of foolishness, it was the epoch of belief, it was "
        "the epoch of incredulity, it was the season of Light, it was the season of "
        "Darkness, it was the spring of hope, it was the winter of despair, we had "
        "everything before us, we had nothing before us, we were all going direct to "
        "Heaven, we were all going direct the other way. In short, the period was so "
        "far like the present period, that some of its noisiest authorities insisted "
        "on its being received, for good or for evil, in the superlative degree of "
        "comparison only."
    ),
}


SWEEP_LEGEND = (
    "**Legend**: each row sends one prompt C times at once (Concurrency), each "
    "request cloning the same 5.3 s LibriSpeech clip. Times are in seconds; min, "
    "mean and max are over the C requests.\n\n"
    "- **ttfa**: time to first audio, from sending the request to its first audio "
    "chunk.\n"
    "- **rtr**: real-time ratio, audio length / request time; above 1 is faster "
    "than real time.\n"
    "- **stream_rate_min**: audio length / time after the first chunk, slowest "
    "request: how fast audio arrives once it has started.\n"
    "- **playback_wait_total / playback_wait_longest**: a player starts at the "
    "first chunk and plays in real time; the total and the longest time it waits "
    "for audio, worst request. 0 means playback never stalls.\n"
    "- **audio_s_per_s**: audio produced by all C requests / time of the slowest "
    "one.\n"
    "- **failed_requests**: requests that returned no audio.\n"
    "- **Target checks**: C=1 only. RTR is the slowest request's rtr across "
    "prompts; a failed request anywhere fails it. TTFT, the slowest ttfa in ms, "
    "is checked only when the model sets a TTFT target, and is NA otherwise. "
    "Ratio is the measured value / the tier's target."
)


def _prompt_note(prompts: dict) -> str:
    """The sweep prompts with their length, quoted up to a line."""
    lines = []
    for name, text in prompts.items():
        quote = (
            text if len(text) <= 80 else text[:80].rsplit(" ", 1)[0].rstrip(",") + " …"
        )
        lines.append(f"- **{name}**: {len(text.split())} words. “{quote}”")
    return "**Prompts**\n\n" + "\n".join(lines)


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
    for task in resolve_eval_tasks(ctx):
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

    # A server's first request with a given voice clip can run far slower than the
    # rest (Qwen3-TTS builds speaker-encoder kernels per clip length), so one
    # untimed request keeps it out of the averages.
    logger.info("Warm-up request (not timed)...")
    asyncio.run(_generate_speech(ctx, client, test_text, voice))

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


def _playback_wait(arrivals: list[tuple[float, float]]) -> tuple[float, float]:
    """(total, longest) wait of a player that starts at the first audio chunk and
    plays in real time; ``arrivals`` is (seconds, audio seconds so far) per chunk."""
    clock = arrivals[0][0]
    total = longest = received = 0.0
    for at, cumulative in arrivals:
        wait = max(0.0, at - clock)
        total += wait
        longest = max(longest, wait)
        clock = max(clock, at) + cumulative - received
        received = cumulative
    return total, longest


def _sweep_row(prompt: str, concurrency: int, results: list[TimedSynthesis]) -> dict:
    """One row of the concurrency sweep: ``concurrency`` requests sent together."""
    done = [r for r in results if r.ok]
    row = {
        "prompt": prompt,
        "concurrency": concurrency,
        "failed_requests": len(results) - len(done),
    }
    if not done:
        return row
    if any(not r.arrivals for r in done):
        raise ValueError("the concurrency sweep needs a streaming TTS client")
    ttfa = [r.ttft_ms / 1000 for r in done]
    rtr = [r.audio_seconds / r.elapsed_s for r in done]
    # Audio per second once the first chunk is out: how fast a stream refills.
    stream = [r.audio_seconds / (r.elapsed_s - t) for r, t in zip(done, ttfa)]
    waits = [_playback_wait(r.arrivals) for r in done]
    row.update(
        ttfa_min=min(ttfa),
        ttfa_max=max(ttfa),
        rtr_min=min(rtr),
        rtr_mean=sum(rtr) / len(rtr),
        rtr_max=max(rtr),
        stream_rate_min=min(stream),
        playback_wait_total=max(total for total, _ in waits),
        playback_wait_longest=max(longest for _, longest in waits),
        audio_s_per_s=sum(r.audio_seconds for r in done)
        / max(r.elapsed_s for r in done),
    )
    return row


async def _speak_together(
    client: TTSClient, url: str, text: str, voice: Optional[VoiceClip], count: int
) -> list[TimedSynthesis]:
    return list(
        await asyncio.gather(
            *(
                client.timed_synthesize(url, text, voice=voice, timeout_s=300)
                for _ in range(count)
            )
        )
    )


def _run_tts_concurrency_sweep(ctx: MediaContext, levels: list[int]) -> Block:
    """Send each sweep prompt ``C`` times at once for every ``C`` in ``levels``; one
    row per (prompt, C). The gate is the slowest request at C=1; a failed request
    anywhere in the sweep counts as RTR 0."""
    client = make_tts_client(_tts_client_targets(ctx))
    voice = None if client.has_default_voice else _benchmark_voice_clip()
    url = f"{ctx.base_url}/v1/audio/speech"
    logger.info(f"TTS concurrency sweep: client={client.describe()}, C={levels}")

    # Same reason as in _run_tts_benchmark: the clip's first request is slow.
    asyncio.run(client.timed_synthesize(url, DEFAULT_TTS_TEXT, voice=voice))

    records = []
    for prompt, text in SWEEP_PROMPTS.items():
        for concurrency in levels:
            results = asyncio.run(
                _speak_together(client, url, text, voice, concurrency)
            )
            records.append(_sweep_row(prompt, concurrency, results))
            logger.info(f"Sweep row: {records[-1]}")

    failed = sum(row["failed_requests"] for row in records)
    single = [row for row in records if row["concurrency"] == 1 and "rtr_min" in row]
    if failed or not single:
        logger.error(f"{failed} sweep request(s) failed; gating on RTR 0")
        rtr_gate, ttfa_ms = 0.0, None
    else:
        rtr_gate = min(row["rtr_min"] for row in single)
        ttfa_ms = max(row["ttfa_max"] for row in single) * 1000
    target_checks, _ = _tts_target_checks(ctx, ttfa_ms, rtr_gate)

    return Block(
        kind="benchmarks",
        task_type="text_to_speech",
        title="Text-to-Speech Benchmark",
        id=block_id(ctx) or None,
        targets={
            "concurrency": list(levels),
            "prompts": list(SWEEP_PROMPTS),
            NOTES_KEY: [SWEEP_LEGEND, _prompt_note(SWEEP_PROMPTS)],
        },
        data={"records": records, "target_checks": target_checks},
    )


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


def run_tts_benchmark(ctx: MediaContext) -> Block:
    """Run benchmarks for a TTS model (SpeechT5, etc.)."""
    logger.info(
        f"Running benchmarks for model: {ctx.model_spec.model_name} on device: {ctx.device.name}"
    )
    require_health(ctx)

    levels = _tts_client_targets(ctx).get("benchmark_concurrency")
    if levels:
        return _run_tts_concurrency_sweep(ctx, levels)

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
