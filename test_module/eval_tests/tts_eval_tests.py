# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC


from __future__ import annotations

import importlib.util
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from report_module.renderers import NOTES_KEY
from report_module.schema import Block

from workflow_module import accept_blocks
from workflow_module.context_helpers import get_num_calls
from workflow_module.engine_types import EvalLimitMode
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
TASK_SEED_TTS = "seed_tts_eval"
TASK_SEED_TTS_CC = "seed_tts_content_consistency"
TASK_SEED_TTS_SIM = "seed_tts_speaker_similarity"

SEED_TTS_RUNNER = Path(__file__).resolve().parent / "seed_tts" / "run_seed_tts.py"
# A full test-en pass is about 20 min of generation, with Whisper-large-v3
# transcribing on CPU at about 5 s a clip alongside, then SIM.
SEED_TTS_TIMEOUT_S = 4 * 3600


LIBRITTS_R_URL = "https://huggingface.co/datasets/blabble-io/libritts_r"
SEED_TTS_EVAL_URL = "https://github.com/BytedanceSpeech/seed-tts-eval"
EVAL_LEGEND = (
    "**Legend**\n\n"
    "- **Score**: 0-100, higher is better. tts_generation: 100 × (1 − WER); "
    "tts_speaker_similarity: 100 × mean_secs; "
    "tts_naturalness: 100 × (mean_utmos − 1) / 4; "
    "seed_tts_eval and seed_tts_content_consistency: 100 × (1 − WER); "
    "seed_tts_speaker_similarity: 100 × mean_sim.\n"
    "- **Accuracy Check**: PASS when the task's gate, listed under Tasks, holds. "
    "Published Score is shown for reference and does not gate.\n"
    "- **WER**: word error rate, (substitutions + deletions + insertions) / words "
    "in the input text, of the Whisper transcript; mean over samples; "
    "0 is a perfect read. tts_generation transcribes with openai/whisper-base. "
    "The seed_tts tasks share one openai/whisper-large-v3 transcript per clip: "
    "seed_tts_eval normalizes text with EvalScope's English normalizer; "
    "seed_tts_content_consistency as seed-tts-eval does, removing punctuation "
    "and lowercasing.\n"
    "- **mean_sim**: speaker-embedding cosine similarity (seed-tts-eval's "
    "WavLM-large speaker-verification model) of each clone to its prompt clip.\n"
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
    f"**Dataset**: tts_* tasks: [LibriTTS-R]({LIBRITTS_R_URL}), test.clean split, "
    "one utterance per speaker. For a cloning model, each sample clones one "
    "speaker's utterance and reads the next speaker's sentence. seed_tts tasks: "
    f"[Seed-TTS-Eval]({SEED_TTS_EVAL_URL}) test-en, Common Voice prompt clips; "
    "each sample clones its prompt clip, given the clip's transcript, and reads "
    "the target text. The three tasks score the same generated clips.\n\n"
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
    elif task_name == TASK_SEED_TTS:
        gate = get("wer_threshold") is not None and f"WER ≤ {get('wer_threshold')}"
        what = (
            "Seed-TTS-Eval test-en WER, by EvalScope, which also generates the "
            "clips. Published Score is the model card's test-en WER over all "
            "1,088 clips, 1.24%, as 100 × (1 − 0.0124)"
        )
    elif task_name == TASK_SEED_TTS_CC:
        gate = get("wer_threshold") is not None and f"WER ≤ {get('wer_threshold')}"
        what = "Seed-TTS-Eval test-en content consistency: WER by seed-tts-eval"
    elif task_name == TASK_SEED_TTS_SIM:
        gate = get("sim_floor") is not None and f"mean_sim ≥ {get('sim_floor')}"
        what = "Seed-TTS-Eval test-en speaker similarity: SIM by seed-tts-eval"
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


def _published(score) -> tuple:
    """Published Score and its ref, both N/A unless the score cites a source."""
    if score.published_score is None or not score.published_score_ref:
        return "N/A", "N/A"
    return score.published_score, score.published_score_ref


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
    published_score, published_score_ref = _published(task.score)
    data = {
        "task_name": task.task_name,
        "published_score": published_score,
        "score": score,
        "published_score_ref": published_score_ref,
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


# One Seed-TTS-Eval run per server, client and concurrency, shared by the three seed_tts
# tasks: each scores the same clips. A failed run is kept too, so the tasks
# after it report its error instead of starting it again.
_SEED_TTS_RUNS: dict = {}


def _seed_tts_limit(ctx: MediaContext, task):
    """The task's sample limit for the run's --limit-samples-mode (--ci-mode
    sets ci-nightly); None runs every sample."""
    rc = getattr(ctx, "runtime_config", None)
    mode = EvalLimitMode.from_string(getattr(rc, "limit_samples_mode", None))
    return task.limit_samples_map.get(mode) if mode else None


def _run_seed_tts_eval(ctx: MediaContext, targets: dict, limit=None) -> dict:
    """Run Seed-TTS-Eval test-en in its own venv; return the runner's summary.

    EvalScope and seed-tts-eval's scorer pin torch and transformers versions
    this venv cannot hold, so the runner is a subprocess of the
    EVALS_SEED_TTS venv's python.
    """
    from workflow_module.engine_types import WorkflowVenvType
    from workflow_module.venv_provisioner import get_venv_provisioner
    from workflows.workflow_venvs import (
        SEED_TTS_EVAL_DIR_NAME,
        SEED_TTS_SV_CKPT_NAME,
        SEED_TTS_TORCH_HOME,
    )

    from .tts_client import TTS_API_VLLM_OMNI, auth_headers

    if targets.get("tts_api") != TTS_API_VLLM_OMNI:
        raise ValueError(
            f"Seed-TTS-Eval sends vLLM-Omni requests; tts_api is "
            f"{targets.get('tts_api')!r}, not {TTS_API_VLLM_OMNI!r}"
        )
    provisioner = get_venv_provisioner()
    venv_type = WorkflowVenvType.EVALS_SEED_TTS
    if not provisioner.provision(venv_type, ctx.model_spec):
        raise RuntimeError(f"Failed to provision the {venv_type.name} venv")
    venv = Path(provisioner.venv_path(venv_type))
    token = auth_headers()["Authorization"].removeprefix("Bearer ")
    with tempfile.TemporaryDirectory(prefix="seed_tts_eval_") as tmp:
        out = Path(tmp) / "result.json"
        cmd = [
            provisioner.venv_python(venv_type),
            str(SEED_TTS_RUNNER),
            "--api-url",
            f"{ctx.base_url}/v1",
            "--work-dir",
            str(Path(tmp) / "work"),
            "--out",
            str(out),
            "--seed-tts-eval-dir",
            str(venv / SEED_TTS_EVAL_DIR_NAME),
            "--sv-checkpoint",
            str(venv / SEED_TTS_SV_CKPT_NAME),
            "--concurrency",
            str(targets.get("concurrency", 1)),
        ]
        if limit is not None:
            cmd += ["--limit", str(limit)]
        logger.info("Running Seed-TTS-Eval: %s", " ".join(cmd))
        subprocess.run(
            cmd,
            env={
                **os.environ,
                "TORCH_HOME": str(venv / SEED_TTS_TORCH_HOME),
                "TTS_API_KEY": token,
            },
            check=True,
            timeout=SEED_TTS_TIMEOUT_S,
        )
        return json.loads(out.read_text())


def _seed_tts_result(ctx: MediaContext, task) -> dict:
    """The shared Seed-TTS-Eval run's summary; raises the run's error."""
    targets = _task_targets(task)
    limit = _seed_tts_limit(ctx, task)
    key = (
        ctx.base_url,
        targets.get("tts_api"),
        targets.get("concurrency", 1),
        limit,
    )
    if key not in _SEED_TTS_RUNS:
        try:
            _SEED_TTS_RUNS[key] = _run_seed_tts_eval(ctx, targets, limit)
        except Exception as e:
            logger.exception("Seed-TTS-Eval failed to run")
            _SEED_TTS_RUNS[key] = e
    run = _SEED_TTS_RUNS[key]
    if isinstance(run, Exception):
        raise run
    return run


# task -> (summary key, report metric, gate target, whether higher passes)
_SEED_TTS_ROWS = {
    TASK_SEED_TTS: ("wer_evalscope", "wer", "wer_threshold", False),
    TASK_SEED_TTS_CC: ("wer_official", "wer", "wer_threshold", False),
    TASK_SEED_TTS_SIM: ("mean_sim", "mean_sim", "sim_floor", True),
}


def _run_seed_tts_task(ctx: MediaContext, task) -> Block:
    """One seed_tts task's row of the shared Seed-TTS-Eval run -> one Block."""
    summary_key, metric, gate_key, higher_passes = _SEED_TTS_ROWS[task.task_name]
    targets = _task_targets(task)
    try:
        summary = _seed_tts_result(ctx, task)
    except Exception as e:
        return _tts_eval_block(
            ctx,
            task,
            score=None,
            accuracy_check=ReportCheckTypes.NA,
            error=f"Seed-TTS-Eval failed to run: {type(e).__name__}: {e}",
            metrics={metric: None},
        )

    value = summary.get(summary_key)
    gate = targets.get(gate_key)
    if value is None or gate is None:
        accuracy_check = ReportCheckTypes.NA
    else:
        passed = value >= gate if higher_passes else value <= gate
        accuracy_check = ReportCheckTypes.PASS if passed else ReportCheckTypes.FAIL
    if value is None:
        score = None
    elif higher_passes:
        score = round(max(0.0, value) * 100.0, 2)
    else:
        score = _intelligibility_score(value)
    logger.info(
        "%s: %s=%s gate %s=%s samples=%s -> check=%s",
        task.task_name,
        metric,
        value,
        gate_key,
        gate,
        summary.get("sample_count"),
        accuracy_check.name,
    )
    return _tts_eval_block(
        ctx,
        task,
        score=score,
        accuracy_check=accuracy_check,
        metrics={metric: value, "sample_count": summary.get("sample_count")},
        result={**summary, gate_key: gate},
    )


_TASK_RUNNERS = {
    TASK_WER: _run_wer_task,
    TASK_SECS: _run_secs_task,
    TASK_NATURALNESS: _run_naturalness_task,
    **{name: _run_seed_tts_task for name in _SEED_TTS_ROWS},
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
