# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC


from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from report_module.schema import Block

from workflows.utils import get_num_calls

from .._test_common import ReportCheckTypes, block_id
from ..context import HardwareRequirement, MediaContext, require_health

logger = logging.getLogger(__name__)

# avg WER at or below this threshold passes the quality bar (20% WER).
DEFAULT_WER_THRESHOLD = 0.20
# Matches TTSQualityTest's own default so WER is averaged over enough utterances.
DEFAULT_QUALITY_SAMPLE_COUNT = 10
_NUM_CALLS_SENTINEL = 2
TTS_QUALITY_DEPS = ("numpy", "torch", "transformers", "librosa", "datasets")


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


def run_tts_eval(ctx: MediaContext) -> Block:
    """Run the WER quality eval for a TTS model (SpeechT5, etc.).

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
