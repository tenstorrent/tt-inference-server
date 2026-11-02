# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC


from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import shlex
import sys
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from report_module.schema import Block

from workflow_module.context_helpers import get_num_calls

from .._test_common import ReportCheckTypes, block_id
from ..context import HardwareRequirement, MediaContext, require_health

logger = logging.getLogger(__name__)

# avg WER at or below this threshold passes the quality bar (20% WER).
DEFAULT_WER_THRESHOLD = 0.20
# Matches TTSQualityTest's own default so WER is averaged over enough utterances.
DEFAULT_QUALITY_SAMPLE_COUNT = 10
_NUM_CALLS_SENTINEL = 2
TTS_QUALITY_DEPS = ("numpy", "torch", "transformers", "librosa", "datasets")

# tt-tts correctness eval: one prompt sent many times, every response transcribed.
TT_TTS_IMPL = "tt-tts"
CORRECTNESS_REFS = 5
CORRECTNESS_REQUESTS = 100
CORRECTNESS_CONCURRENCY = 100
CORRECTNESS_ASR_MODEL = "distil-large-v3"
CORRECTNESS_MAX_WER = 0.05
CORRECTNESS_TIMEOUT_SECONDS = 300.0
CORRECTNESS_DIR = "tts_correctness"
# --workflow-args keys; prefixed so `--workflow release` can also pass the
# benchmark's own `concurrency=`/`isl=` in the same string.
_CORRECTNESS_ARG_PREFIX = "eval_"
_CORRECTNESS_ARG_KEYS = ("refs", "requests", "concurrency")


def _tts_sample_count(ctx: MediaContext) -> int:
    base = get_num_calls(ctx)
    if base != _NUM_CALLS_SENTINEL:
        return base
    return DEFAULT_QUALITY_SAMPLE_COUNT


def _missing_quality_deps() -> List[str]:
    """Return the subset of :data:`TTS_QUALITY_DEPS` that cannot be imported."""
    return [name for name in TTS_QUALITY_DEPS if not _can_import(name)]


def _can_import(module_name: str) -> bool:
    """Return True if ``module_name`` looks importable, without executing it."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def _intelligibility_score(avg_wer: float) -> float:
    """Map average WER (0..1, lower is better) to a 0..100 intelligibility
    score (higher is better), matching the audio eval's score convention."""
    return round(max(0.0, 1.0 - avg_wer) * 100.0, 2)


def _wer_accuracy_check(
    avg_wer: Optional[float], wer_threshold: float, valid_samples: int
) -> ReportCheckTypes:
    """PASS when measured WER meets the threshold, FAIL when it exceeds it,
    NA when no sample produced a usable measurement."""
    if valid_samples <= 0 or avg_wer is None:
        return ReportCheckTypes.NA
    if avg_wer <= wer_threshold:
        return ReportCheckTypes.PASS
    return ReportCheckTypes.FAIL


def _tts_eval_block(
    ctx: MediaContext,
    task,
    *,
    score: Optional[float],
    wer: Optional[float],
    accuracy_check: ReportCheckTypes,
    error: Optional[str] = None,
    extra: Optional[dict] = None,
) -> Block:
    """Build a canonical TTS eval Block (accuracy/quality fields only)."""
    data = {
        "task_name": task.task_name,
        "tolerance": task.score.tolerance,
        "published_score": task.score.published_score,
        "score": score,
        "published_score_ref": task.score.published_score_ref,
        "wer": wer,
        "accuracy_check": accuracy_check,
    }
    if extra:
        data.update(extra)
    if error is not None:
        data["error"] = error
    return Block(
        kind="evals",
        task_type="text_to_speech",
        title="Text-to-Speech Eval",
        id=block_id(ctx) or None,
        targets={
            "task_name": task.task_name,
            "tolerance": task.score.tolerance,
            "published_score": task.score.published_score,
            "published_score_ref": task.score.published_score_ref,
        },
        data=data,
    )


def _run_tts_quality_eval(ctx: MediaContext) -> dict:
    """Run ``TTSQualityTest`` against the live server and return its result data.

    Imported lazily so the module doesn't pull in numpy/torch at import time;
    callers must gate on :func:`_missing_quality_deps` first.

    The test runs through :meth:`BaseTest.run_tests`, so it inherits the shared
    retry/timeout/hardware-gate envelope; the WER metrics are
    read back from the returned Block's ``data``.
    """
    from .._test_common import TestConfig
    from .tts_quality_test import TTSQualityTest

    sample_count = _tts_sample_count(ctx)
    test = TTSQualityTest(
        TestConfig(
            {
                "timeout": 3600,
                "retry_attempts": 1,
                "retry_delay": 10,
                "break_on_failure": False,
            }
        ),
        targets={
            "sample_count": sample_count,
            "wer_threshold": DEFAULT_WER_THRESHOLD,
            "cleanup": True,
        },
        ctx=ctx,
    )
    logger.info(
        "Running TTSQualityTest: samples=%s wer_threshold=%s base_url=%s",
        sample_count,
        DEFAULT_WER_THRESHOLD,
        ctx.base_url,
    )
    return dict(test.run_tests().data)


def _correctness_params(ctx: MediaContext) -> dict:
    """``eval_refs``/``eval_requests``/``eval_concurrency`` from ``--workflow-args``.

    Keys without the ``eval_`` prefix belong to other workflows and are ignored.
    """
    params = {
        "refs": CORRECTNESS_REFS,
        "requests": CORRECTNESS_REQUESTS,
        "concurrency": CORRECTNESS_CONCURRENCY,
    }
    rc = getattr(ctx, "runtime_config", None)
    for token in shlex.split(getattr(rc, "workflow_args", None) or ""):
        key, _, value = token.partition("=")
        if not key.startswith(_CORRECTNESS_ARG_PREFIX):
            continue
        key = key[len(_CORRECTNESS_ARG_PREFIX) :]
        if key not in _CORRECTNESS_ARG_KEYS:
            logger.warning(
                f"TTS correctness eval ignores workflow arg {token!r}; supported: "
                f"{[_CORRECTNESS_ARG_PREFIX + k for k in _CORRECTNESS_ARG_KEYS]}"
            )
            continue
        try:
            params[key] = int(value)
        except ValueError:
            params[key] = -1
        if params[key] < (0 if key == "refs" else 1):
            raise ValueError(f"TTS correctness eval workflow arg {token!r} is invalid")
    return params


def _run_tts_correctness_eval(ctx: MediaContext, task) -> Block:
    """Same-prompt correctness check for the tt-tts server (any failure = FAIL).

    Sends ``refs`` sequential references, then ``requests`` copies of the same
    prompt at ``concurrency``; each response fails on no audio or Whisper WER
    above :data:`CORRECTNESS_MAX_WER`. WAVs and per-response transcripts land in
    ``<output_path>/tts_correctness``.
    """
    if not _can_import("faster_whisper"):
        reason = "TTS correctness eval needs faster-whisper, which is not installed"
        logger.error(reason)
        return _tts_eval_block(
            ctx,
            task,
            score=None,
            wer=None,
            accuracy_check=ReportCheckTypes.NA,
            error=reason,
        )
    from ..benchmark_tests import tts_correctness, tts_load_harness

    params = _correctness_params(ctx)
    parsed = urlparse(ctx.base_url)
    if not parsed.hostname:
        raise ValueError(f"TTS eval could not resolve a host from {ctx.base_url!r}")
    target = tts_load_harness.Target(
        host=parsed.hostname,
        port=ctx.server_port,
        timeout=CORRECTNESS_TIMEOUT_SECONDS,
        text=tts_correctness.DEFAULT_TEXT,
    )
    outdir = Path(ctx.output_path) / CORRECTNESS_DIR
    args = argparse.Namespace(
        outdir=str(outdir),
        no_asr=False,
        asr_model=CORRECTNESS_ASR_MODEL,
        asr_threads=os.cpu_count() or 8,
        max_wer=CORRECTNESS_MAX_WER,
    )
    logger.info(
        "Running TTS correctness eval: refs=%d requests=%d concurrency=%d asr=%s",
        params["refs"],
        params["requests"],
        params["concurrency"],
        CORRECTNESS_ASR_MODEL,
    )
    doc = tts_correctness.run_check(
        target, params["refs"], params["requests"], params["concurrency"], args
    )
    summary = doc["summary"]
    logger.info("\n%s", tts_correctness.report(summary, doc["records"]))

    graded = [r for r in doc["records"] if r["tag"] in ("ref", "load")]
    passed = sum(1 for r in graded if r["pass"])
    failed = [f"{r['tag']}_{r['i']:03d}" for r in graded if not r["pass"]]
    wers = [r["wer"] for r in graded if r.get("wer") is not None]
    accuracy_check = (
        ReportCheckTypes.FAIL
        if summary["load_failures"] or summary["reference_failures"]
        else ReportCheckTypes.PASS
    )
    return _tts_eval_block(
        ctx,
        task,
        score=round(100.0 * passed / len(graded), 2) if graded else None,
        wer=sum(wers) / len(wers) if wers else None,
        accuracy_check=accuracy_check,
        extra={
            "requests": summary["load_requests"],
            "concurrency": params["concurrency"],
            "load_failures": summary["load_failures"],
            "references": params["refs"],
            "reference_failures": summary["reference_failures"],
            "max_wer": CORRECTNESS_MAX_WER,
            "failed_responses": ", ".join(failed) or None,
            "results": str(outdir / tts_correctness.RESULTS_FILE),
        },
    )


def run_tts_eval(ctx: MediaContext) -> Block:
    """Run the WER quality eval for a TTS model (SpeechT5, etc.).

    The tt-tts server instead runs the same-prompt correctness check
    (:func:`_run_tts_correctness_eval`), where any malformed response fails.

    ``accuracy_check`` reflects Word Error Rate against the quality threshold;
    a missing transcription toolchain or a run that produced no usable samples
    yields an NA block (not a false FAIL) so an unprovisioned environment does
    not block acceptance.
    """
    logger.info(
        f"Running evals for model: {ctx.model_spec.model_name} on device: {ctx.device.name}"
    )
    require_health(ctx, HardwareRequirement.ANY_CHIP)
    task = ctx.all_params.tasks[0]
    impl_name = getattr(getattr(ctx.model_spec, "impl", None), "impl_name", None)
    if impl_name == TT_TTS_IMPL:
        return _run_tts_correctness_eval(ctx, task)

    missing = _missing_quality_deps()
    if missing:
        reason = f"TTS quality eval dependencies unavailable: {', '.join(missing)}"
        logger.error(reason)
        return _tts_eval_block(
            ctx,
            task,
            score=None,
            wer=None,
            accuracy_check=ReportCheckTypes.NA,
            error=reason,
        )

    try:
        result = _run_tts_quality_eval(ctx)
    except Exception as e:
        reason = f"TTS quality eval failed to run: {type(e).__name__}: {e}"
        logger.exception(reason)
        return _tts_eval_block(
            ctx,
            task,
            score=None,
            wer=None,
            accuracy_check=ReportCheckTypes.NA,
            error=reason,
        )

    avg_wer = result.get("avg_wer")
    valid_samples = int(result.get("valid_samples") or 0)
    wer_threshold = float(result.get("wer_threshold", DEFAULT_WER_THRESHOLD))
    accuracy_check = _wer_accuracy_check(avg_wer, wer_threshold, valid_samples)
    score = (
        _intelligibility_score(avg_wer)
        if accuracy_check is not ReportCheckTypes.NA
        else None
    )

    logger.info(
        "TTS eval: avg_wer=%s valid_samples=%s threshold=%s -> score=%s check=%s",
        avg_wer,
        valid_samples,
        wer_threshold,
        score,
        accuracy_check.name,
    )
    return _tts_eval_block(
        ctx,
        task,
        score=score,
        wer=avg_wer,
        accuracy_check=accuracy_check,
    )


__all__ = ["run_tts_eval"]
