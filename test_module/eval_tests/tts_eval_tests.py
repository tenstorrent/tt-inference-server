# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC


from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from report_module.renderers import NOTES_KEY
from report_module.schema import Block

from workflow_module import accept_blocks
from workflow_module.context_helpers import get_num_calls
from workflow_module.target_pack import get_target_pack

from .._test_common import ReportCheckTypes, block_id, sweep_envelope
from ..context import HardwareRequirement, MediaContext, require_health

logger = logging.getLogger(__name__)

# avg WER at or below this threshold passes the quality bar (20% WER).
DEFAULT_WER_THRESHOLD = 0.20
# Matches TTSQualityTest's own default so WER is averaged over enough utterances.
DEFAULT_QUALITY_SAMPLE_COUNT = 10
_NUM_CALLS_SENTINEL = 2
TTS_QUALITY_DEPS = ("numpy", "torch", "transformers", "librosa", "datasets")
# SECS additionally decodes/resamples audio client-side.
TTS_SECS_DEPS = TTS_QUALITY_DEPS + ("soundfile",)
# Naturalness scores clips with a torch.hub MOS predictor (needs torchaudio).
TTS_NATURALNESS_DEPS = TTS_QUALITY_DEPS + ("soundfile", "torchaudio")

TASK_WER = "tts_generation"
TASK_SECS = "tts_speaker_similarity"
TASK_NATURALNESS = "tts_naturalness"


LIBRITTS_R_URL = "https://huggingface.co/datasets/blabble-io/libritts_r"
EVAL_LEGEND = (
    "**Legend**\n\n"
    "- **Score**: 0-100, higher is better. tts_generation: 100 × (1 − WER); "
    "tts_speaker_similarity: 100 × mean_secs; "
    "tts_naturalness: 100 × (mean_utmos − 1) / 4.\n"
    "- **Accuracy Check**: PASS when the task's gate, listed under Tasks, holds. "
    "Published Score and Tolerance are shown for reference and do not gate.\n"
    "- **WER**: word error rate, (substitutions + deletions + insertions) / words "
    "in the input text, of the openai/whisper-base transcript; mean over samples; "
    "0 is a perfect read.\n"
    "- **mean_secs**: speaker-embedding cosine similarity "
    "(microsoft/unispeech-sat-base-plus-sv) of each clone to its reference clip; "
    "1 is the same voice.\n"
    "- **mean_margin**: mean_secs minus the same score for a baseline, the same "
    "sentence cloned from another speaker's clip; how far a clone stands from a "
    "wrong voice.\n"
    "- **mean_utmos**: listener score (mean opinion score, 1-5) predicted by "
    "UTMOS22 strong.\n"
    "- **valid_samples / failed_samples / sample_count**: samples scored / "
    "requests that returned no audio / samples sent.\n\n"
    f"**Dataset**: [LibriTTS-R]({LIBRITTS_R_URL}), test.clean split, one utterance "
    "per speaker. For a cloning model, each sample clones one speaker's utterance "
    "and reads the next speaker's sentence.\n\n"
    "**Tasks**"
)


def _task_note(task_name: str, result: dict) -> Optional[str]:
    """One line on what ``task_name`` measured and how it gates; counts and gate
    values come from ``result``, so a task that did not run reads without them."""
    get = result.get
    count = get("sample_count", get("total_samples"))
    how = []
    if count:
        how.append(f"{count} samples")
    if get("concurrency"):
        how.append(f"{get('concurrency')} requests at a time")
    how = f" ({', '.join(how)})" if how else ""
    if task_name == TASK_WER:
        gate = get("wer_threshold") is not None and f"WER ≤ {get('wer_threshold')}"
        what = "intelligibility. Whisper transcribes each clone; WER against its text"
    elif task_name == TASK_SECS:
        gate = get("secs_floor") is not None and (
            f"mean_secs ≥ {get('secs_floor')} and "
            f"mean_margin ≥ {get('margin_threshold')}"
        )
        what = (
            "voice cloning. SECS of each clone to its reference clip, and of a "
            "baseline, the same sentence cloned from the previous speaker's clip, "
            "to that clip"
        )
    elif task_name == TASK_NATURALNESS:
        gate = get("utmos_threshold") is not None and (
            f"mean_utmos ≥ {get('utmos_threshold')}"
        )
        what = "naturalness. UTMOS scores each clone"
    else:
        return None
    gate = f" Pass: {gate}." if gate else ""
    return f"- **{task_name}**: {what}{how}.{gate}"


def _tts_sample_count(ctx: MediaContext) -> int:
    base = get_num_calls(ctx)
    if base != _NUM_CALLS_SENTINEL:
        return base
    return DEFAULT_QUALITY_SAMPLE_COUNT


class _PlaceholderScore:
    """Report-metadata stand-in used when no EvalConfig is registered."""

    published_score = None
    published_score_ref = ""
    tolerance = None


class _PlaceholderTask:
    """Minimal task shape _tts_eval_block needs to render an NA block."""

    task_name = "tts_generation"
    score = _PlaceholderScore()


_PLACEHOLDER_TASK = _PlaceholderTask()


def resolve_eval_tasks(ctx: MediaContext) -> list:
    """Return all eval tasks for this model with ``ctx.device``'s overrides applied
    ([] if none is registered)."""
    tasks = getattr(ctx.all_params, "tasks", None) or []
    pack = get_target_pack()
    device = getattr(ctx, "device", None)
    return [pack.resolve_eval_task_for_device(task, device) for task in tasks]


# EVALS_META injects add_bos_token for lm-eval; it is not a TTS eval target.
_NON_TARGET_MODEL_KWARGS = frozenset({"add_bos_token"})


def _task_targets(task) -> dict:
    """Eval targets for a SECS/naturalness task, from its ``model_kwargs``."""
    kwargs = getattr(task, "model_kwargs", None) or {}
    return {k: v for k, v in kwargs.items() if k not in _NON_TARGET_MODEL_KWARGS}


def _missing_deps(deps) -> List[str]:
    """Return the subset of ``deps`` that cannot be imported."""
    return [name for name in deps if not _can_import(name)]


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
    accuracy_check: ReportCheckTypes,
    error: Optional[str] = None,
    metrics: Optional[dict] = None,
    result: Optional[dict] = None,
) -> Block:
    """Build a canonical TTS eval Block (accuracy/quality fields only); ``result``,
    the eval test's data, fills in the task's line under the legend."""
    data = {
        "task_name": task.task_name,
        "tolerance": task.score.tolerance,
        "published_score": task.score.published_score,
        "score": score,
        "published_score_ref": task.score.published_score_ref,
        "accuracy_check": accuracy_check,
    }
    data.update(metrics or {})
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
            NOTES_KEY: [
                note
                for note in (EVAL_LEGEND, _task_note(task.task_name, result or {}))
                if note
            ],
        },
        data=data,
    )


def _run_tts_quality_eval(ctx: MediaContext, task) -> dict:
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
    targets = {
        "sample_count": sample_count,
        "wer_threshold": DEFAULT_WER_THRESHOLD,
        "cleanup": True,
        **_task_targets(task),
    }
    test = TTSQualityTest(
        TestConfig(
            {
                "timeout": 3600,
                "retry_attempts": 1,
                "retry_delay": 10,
                "break_on_failure": False,
            }
        ),
        targets=targets,
        ctx=ctx,
    )
    logger.info("Running TTSQualityTest: base_url=%s targets=%s", ctx.base_url, targets)
    return dict(test.run_tests().data)


def _run_wer_task(ctx: MediaContext, task) -> Block:
    """WER intelligibility eval (task_name=tts_generation) -> one Block."""
    missing = _missing_deps(TTS_QUALITY_DEPS)
    if missing:
        reason = f"TTS quality eval dependencies unavailable: {', '.join(missing)}"
        logger.error(reason)
        return _tts_eval_block(
            ctx,
            task,
            score=None,
            accuracy_check=ReportCheckTypes.NA,
            error=reason,
            metrics={"wer": None},
        )

    try:
        result = _run_tts_quality_eval(ctx, task)
    except Exception as e:
        reason = f"TTS quality eval failed to run: {type(e).__name__}: {e}"
        logger.exception(reason)
        return _tts_eval_block(
            ctx,
            task,
            score=None,
            accuracy_check=ReportCheckTypes.NA,
            error=reason,
            metrics={"wer": None},
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
        accuracy_check=accuracy_check,
        metrics={"wer": avg_wer},
        result=result,
    )


def _run_tts_secs_eval(ctx: MediaContext, task) -> dict:
    """Run ``TTSSpeakerSimilarityTest`` against the live server (lazy imports)."""
    from .tts_speaker_similarity_test import TTSSpeakerSimilarityTest

    return _run_eval_test(ctx, task, TTSSpeakerSimilarityTest)


def _run_eval_test(ctx: MediaContext, task, test_class) -> dict:
    """Run one eval ``test_class`` with the task's targets; return its result data."""
    from .._test_common import TestConfig

    targets = _task_targets(task)
    test = test_class(
        TestConfig(
            {
                "timeout": 3600,
                "retry_attempts": 1,
                "retry_delay": 10,
                "break_on_failure": False,
            }
        ),
        targets=targets,
        ctx=ctx,
    )
    logger.info(
        "Running %s: base_url=%s targets=%s",
        test_class.__name__,
        ctx.base_url,
        targets,
    )
    return dict(test.run_tests().data)


def _sample_counts(result: dict) -> dict:
    """Valid, failed and total sample counts of an eval result, for the report."""
    return {
        key: result.get(key)
        for key in ("valid_samples", "failed_samples", "sample_count")
    }


def _run_secs_task(ctx: MediaContext, task) -> Block:
    """Speaker-similarity eval (task_name=tts_speaker_similarity) -> one Block."""
    missing = _missing_deps(TTS_SECS_DEPS)
    if missing:
        reason = f"TTS speaker-similarity deps unavailable: {', '.join(missing)}"
        logger.error(reason)
        return _tts_eval_block(
            ctx,
            task,
            score=None,
            accuracy_check=ReportCheckTypes.NA,
            error=reason,
            metrics={"mean_secs": None},
        )

    try:
        result = _run_tts_secs_eval(ctx, task)
    except Exception as e:
        reason = f"TTS speaker-similarity eval failed to run: {type(e).__name__}: {e}"
        logger.exception(reason)
        return _tts_eval_block(
            ctx,
            task,
            score=None,
            accuracy_check=ReportCheckTypes.NA,
            error=reason,
            metrics={"mean_secs": None},
        )

    mean_secs = result.get("mean_secs")
    mean_margin = result.get("mean_margin")
    valid_samples = int(result.get("valid_samples") or 0)
    if valid_samples <= 0 or mean_secs is None or mean_margin is None:
        # Requests ran but none produced audio: that is a failed eval, not a missing one.
        sampled = int(result.get("sample_count") or 0) > 0
        accuracy_check = ReportCheckTypes.FAIL if sampled else ReportCheckTypes.NA
        score = None
    else:
        accuracy_check = (
            ReportCheckTypes.PASS if result.get("success") else ReportCheckTypes.FAIL
        )
        score = round(max(0.0, mean_secs) * 100.0, 2)
    logger.info(
        "TTS SECS eval: mean_secs=%s mean_margin=%s valid_samples=%s -> score=%s check=%s",
        mean_secs,
        mean_margin,
        valid_samples,
        score,
        accuracy_check.name,
    )
    return _tts_eval_block(
        ctx,
        task,
        score=score,
        accuracy_check=accuracy_check,
        metrics={
            "mean_secs": mean_secs,
            "mean_margin": mean_margin,
            **_sample_counts(result),
        },
        result=result,
    )


def _run_tts_naturalness_eval(ctx: MediaContext, task) -> dict:
    """Run ``TTSNaturalnessTest`` against the live server (lazy imports)."""
    from .._test_common import TestConfig
    from .tts_naturalness_test import TTSNaturalnessTest

    targets = _task_targets(task)
    test = TTSNaturalnessTest(
        TestConfig(
            {
                "timeout": 3600,
                "retry_attempts": 1,
                "retry_delay": 10,
                "break_on_failure": False,
            }
        ),
        targets=targets,
        ctx=ctx,
    )
    logger.info(
        "Running TTSNaturalnessTest: base_url=%s targets=%s", ctx.base_url, targets
    )
    return dict(test.run_tests().data)


def _run_naturalness_task(ctx: MediaContext, task) -> Block:
    """Naturalness eval (task_name=tts_naturalness) -> one Block."""
    missing = _missing_deps(TTS_NATURALNESS_DEPS)
    if missing:
        reason = f"TTS naturalness deps unavailable: {', '.join(missing)}"
        logger.error(reason)
        return _tts_eval_block(
            ctx,
            task,
            score=None,
            accuracy_check=ReportCheckTypes.NA,
            error=reason,
            metrics={"mean_utmos": None},
        )

    try:
        result = _run_tts_naturalness_eval(ctx, task)
    except Exception as e:
        reason = f"TTS naturalness eval failed to run: {type(e).__name__}: {e}"
        logger.exception(reason)
        return _tts_eval_block(
            ctx,
            task,
            score=None,
            accuracy_check=ReportCheckTypes.NA,
            error=reason,
            metrics={"mean_utmos": None},
        )

    mean_utmos = result.get("mean_utmos")
    valid_samples = int(result.get("valid_samples") or 0)
    if valid_samples <= 0 or mean_utmos is None:
        sampled = int(result.get("sample_count") or 0) > 0
        accuracy_check = ReportCheckTypes.FAIL if sampled else ReportCheckTypes.NA
        score = None
    else:
        accuracy_check = (
            ReportCheckTypes.PASS if result.get("success") else ReportCheckTypes.FAIL
        )
        # MOS is 1..5; map to the suite's 0..100 scale.
        score = round((mean_utmos - 1.0) / 4.0 * 100.0, 2)
    logger.info(
        "TTS naturalness eval: mean_utmos=%s valid_samples=%s -> score=%s check=%s",
        mean_utmos,
        valid_samples,
        score,
        accuracy_check.name,
    )
    return _tts_eval_block(
        ctx,
        task,
        score=score,
        accuracy_check=accuracy_check,
        metrics={"mean_utmos": mean_utmos, **_sample_counts(result)},
        result=result,
    )


_TASK_RUNNERS = {
    TASK_WER: _run_wer_task,
    TASK_SECS: _run_secs_task,
    TASK_NATURALNESS: _run_naturalness_task,
}


def run_tts_eval(ctx: MediaContext) -> Block:
    """Run every eval task the model's EvalConfig registers."""
    logger.info(
        f"Running evals for model: {ctx.model_spec.model_name} on device: {ctx.device.name}"
    )
    require_health(ctx, HardwareRequirement.ANY_CHIP)

    tasks = resolve_eval_tasks(ctx)
    if not tasks:
        reason = (
            f"No EvalConfig registered for {ctx.model_spec.model_name!r} "
            f"(hf_model_repo={ctx.model_spec.hf_model_repo!r}) in "
            "reference_config/evals/eval_config.py; cannot run the TTS quality eval."
        )
        logger.error(reason)
        return _tts_eval_block(
            ctx,
            _PLACEHOLDER_TASK,
            score=None,
            accuracy_check=ReportCheckTypes.NA,
            error=reason,
            metrics={"wer": None},
        )

    blocks: List[Block] = []
    for task in tasks:
        runner = _TASK_RUNNERS.get(task.task_name)
        if runner is None:
            reason = (
                f"Unknown TTS eval task {task.task_name!r}; "
                f"known: {sorted(_TASK_RUNNERS)}"
            )
            logger.error(reason)
            blocks.append(
                _tts_eval_block(
                    ctx,
                    task,
                    score=None,
                    accuracy_check=ReportCheckTypes.NA,
                    error=reason,
                )
            )
            continue
        blocks.append(runner(ctx, task))

    if len(blocks) > 1:
        accept_blocks(blocks[:-1], envelope=sweep_envelope(ctx))
    return blocks[-1]


__all__ = ["run_tts_eval"]
