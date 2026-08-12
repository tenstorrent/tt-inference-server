# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Baseline-first quality evaluation for MiniMax-H3 generated videos.

This evaluator uses the documented MiniMax V2 create/query/delete lifecycle,
downloads each output immediately, and computes spatial, temporal, and optional
CLIP metrics. Quality is report-only in ``baseline`` mode; ``gate`` mode
requires explicit thresholds collected from real MiniMax-H3 runs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from report_module.schema import Block
from test_module._test_common import (
    BaseTest,
    HardwareRequirement,
    ReportCheckTypes,
    TestConfig,
    block_id,
)
from test_module._test_common.minimax_h3_client import (
    MiniMaxClientError,
    MiniMaxH3Client,
)
from test_module._test_common.video_quality_metrics import (
    BatchedCLIPScorer,
    MissingVideoQualityDependency,
    analyze_video_quality,
)

if TYPE_CHECKING:
    from test_module.context import MediaContext

logger = logging.getLogger(__name__)

MODEL_NAME = "MiniMax-H3"
RESOLUTION = "768P"
DURATION_SECONDS = 5
ASPECT_RATIO = "16:9"

DEFAULT_PROFILE = "smoke"
DEFAULT_MODE = "baseline"
DEFAULT_SAMPLE_COUNT = 8
DEFAULT_REQUEST_TIMEOUT_SECONDS = 60.0
DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 300.0
DEFAULT_POLL_INTERVAL_SECONDS = 5.0
DEFAULT_POLL_TIMEOUT_SECONDS = 900.0
DEFAULT_TEST_TIMEOUT_SECONDS = 7200
DEFAULT_OUTPUT_DIR = Path("/tmp/minimax_h3_video_quality")

Profile = Literal["smoke", "full", "calibration"]
EvaluationMode = Literal["baseline", "gate"]
_PROFILES = frozenset({"smoke", "full", "calibration"})
_MODES = frozenset({"baseline", "gate"})


@dataclass(frozen=True)
class VideoQualityPrompt:
    """One stable prompt and its concept/progression scoring metadata."""

    id: str
    category: str
    prompt: str
    required_concepts: tuple[str, ...]
    expected_start: str | None = None
    expected_end: str | None = None
    expects_motion: bool = True


@dataclass(frozen=True)
class QualityThresholds:
    """Explicit one-sided thresholds for opt-in gate mode."""

    min_generation_success_ratio: float = 1.0
    min_average_clip: float | None = None
    min_minimum_clip: float | None = None
    max_invalid_videos: int = 0
    max_frozen_videos: int = 0


QUALITY_PROMPTS = (
    VideoQualityPrompt(
        id="subject_identity",
        category="subject_identity",
        prompt=(
            "A small corgi wearing a bright yellow raincoat and round red "
            "glasses walks through a sunlit city plaza. Keep the same dog, "
            "raincoat, and glasses throughout the shot."
        ),
        required_concepts=(
            "a small corgi",
            "a bright yellow raincoat",
            "round red glasses",
        ),
    ),
    VideoQualityPrompt(
        id="object_action",
        category="object_action",
        prompt=(
            "A silver spoon falls into a clear glass of water on a white table, "
            "creating a visible splash and droplets that fall back down."
        ),
        required_concepts=(
            "a silver spoon",
            "a clear glass of water",
            "a visible water splash",
        ),
        expected_start="a silver spoon above a calm clear glass of water",
        expected_end="a spoon inside the glass with water droplets after a splash",
    ),
    VideoQualityPrompt(
        id="camera_movement",
        category="camera_movement",
        prompt=(
            "The camera slowly orbits clockwise around a glossy blue ceramic "
            "teapot on a white table in a bright studio, keeping the teapot "
            "centered while its viewpoint changes smoothly."
        ),
        required_concepts=(
            "a glossy blue ceramic teapot",
            "a white table in a bright studio",
        ),
    ),
    VideoQualityPrompt(
        id="multi_object_composition",
        category="multi_object_composition",
        prompt=(
            "On a clean white floor, a red cube remains on the left, a blue "
            "sphere remains in the center, and a yellow pyramid remains on the "
            "right while a small green ball rolls in front of all three."
        ),
        required_concepts=(
            "a red cube",
            "a blue sphere",
            "a yellow pyramid",
            "a small green ball",
        ),
    ),
    VideoQualityPrompt(
        id="temporal_progression",
        category="temporal_progression",
        prompt=(
            "A white paper airplane starts resting on a wooden desk, lifts into "
            "the air, glides smoothly across a bright room, and lands beside a "
            "sunlit window."
        ),
        required_concepts=(
            "a white paper airplane",
            "a wooden desk",
            "a sunlit window",
        ),
        expected_start="a white paper airplane resting on a wooden desk",
        expected_end="a white paper airplane beside a sunlit window",
    ),
)


def _resolve_api_key() -> str:
    for env_name in ("MINIMAX_API_KEY", "MINIMAX_MOCK_API_KEY"):
        value = os.getenv(env_name)
        if value:
            return value
    raise RuntimeError(
        "Set MINIMAX_API_KEY (real API) or MINIMAX_MOCK_API_KEY (mock API)"
    )


def _normalize_profile(value: Any) -> Profile:
    profile = str(value or DEFAULT_PROFILE).lower()
    if profile not in _PROFILES:
        raise ValueError(f"profile must be one of {sorted(_PROFILES)}, got {value!r}")
    return cast(Profile, profile)


def _normalize_mode(value: Any) -> EvaluationMode:
    mode = str(value or DEFAULT_MODE).lower()
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {sorted(_MODES)}, got {value!r}")
    return cast(EvaluationMode, mode)


def _selected_prompts(profile: Profile) -> tuple[VideoQualityPrompt, ...]:
    return QUALITY_PROMPTS[:1] if profile == "smoke" else QUALITY_PROMPTS


def _resolved_samples_per_prompt(
    profile: Profile,
    requested: int | None,
) -> int:
    samples = (
        requested if requested is not None else (3 if profile == "calibration" else 1)
    )
    if samples < 1:
        raise ValueError("samples_per_prompt must be at least 1")
    return samples


def _create_payload(prompt: str) -> dict[str, Any]:
    return {
        "model": MODEL_NAME,
        "content": [{"type": "text", "text": prompt}],
        "resolution": RESOLUTION,
        "duration": DURATION_SECONDS,
        "ratio": ASPECT_RATIO,
    }


def _validate_succeeded_task(task: dict[str, Any], *, task_id: str) -> str:
    if task.get("status") != "succeeded":
        raise MiniMaxClientError(
            f"task reached terminal status {task.get('status')!r}",
            task_id=task_id,
            response_body=json.dumps(task.get("error")),
        )
    expected = {
        "id": task_id,
        "model": MODEL_NAME,
        "resolution": RESOLUTION,
        "duration": DURATION_SECONDS,
        "ratio": ASPECT_RATIO,
        "task_type": "generation",
        "modality": "video",
    }
    mismatches = {
        field: {"expected": value, "actual": task.get(field)}
        for field, value in expected.items()
        if task.get(field) != value
    }
    if mismatches:
        raise MiniMaxClientError(
            f"succeeded task metadata mismatch: {mismatches}",
            task_id=task_id,
        )
    content = task.get("content")
    content_url = content.get("url") if isinstance(content, dict) else None
    if not isinstance(content_url, str) or not content_url.strip():
        raise MiniMaxClientError(
            "succeeded task has no non-empty content.url",
            task_id=task_id,
        )
    return content_url


async def _evaluate_sample(
    *,
    client: MiniMaxH3Client,
    prompt_case: VideoQualityPrompt,
    sample_index: int,
    output_dir: Path,
    sample_count: int,
    clip_scorer: BatchedCLIPScorer | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "prompt_id": prompt_case.id,
        "category": prompt_case.category,
        "sample_index": sample_index,
        "prompt": prompt_case.prompt,
        "task_id": None,
        "generation_success": False,
        "analysis_success": False,
        "cleanup_success": False,
    }
    task_id: str | None = None
    terminal_status: str | None = None

    try:
        task_id = await client.create_video(_create_payload(prompt_case.prompt))
        result["task_id"] = task_id
        terminal = await client.wait_for_terminal(task_id)
        result["observed_statuses"] = list(terminal.observed_statuses)
        result["task"] = terminal.task
        terminal_status = str(terminal.task.get("status"))

        content_url = _validate_succeeded_task(terminal.task, task_id=task_id)
        result["generation_success"] = True
        video_path = output_dir / f"{prompt_case.id}_{sample_index}_{task_id}.mp4"
        download = await client.download_video(content_url, video_path)
        result["video_path"] = str(download.path)
        result["bytes_downloaded"] = download.bytes_downloaded
        result["download_content_type"] = download.content_type

        metrics = await asyncio.to_thread(
            analyze_video_quality,
            video_path,
            prompt=prompt_case.prompt,
            required_concepts=prompt_case.required_concepts,
            expected_start=prompt_case.expected_start,
            expected_end=prompt_case.expected_end,
            expected_duration=DURATION_SECONDS,
            expected_ratio=ASPECT_RATIO,
            sample_count=sample_count,
            clip_scorer=clip_scorer,
        )
        result["metrics"] = metrics
        result["analysis_success"] = True
    except (MiniMaxClientError, MissingVideoQualityDependency, ValueError) as exc:
        result["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        if isinstance(exc, MiniMaxClientError):
            result["error"].update(exc.to_dict())
    except Exception as exc:  # noqa: BLE001 - preserve one failed sample, continue eval
        logger.exception(
            "Unexpected MiniMax quality error for %s sample %d",
            prompt_case.id,
            sample_index,
        )
        result["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    finally:
        if task_id is not None and terminal_status in {"succeeded", "failed"}:
            try:
                deletion = await client.delete_terminal_task(task_id)
                result["cleanup_success"] = True
                result["deletion"] = deletion
            except MiniMaxClientError as exc:
                result["cleanup_error"] = exc.to_dict()
        elif task_id is not None:
            result["cleanup_error"] = {
                "type": "TaskNotTerminal",
                "message": (
                    "task record was not deleted because its terminal status "
                    f"was not observed (last status={terminal_status!r})"
                ),
            }
    return result


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _aggregate_category_results(
    detailed_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    category_values: dict[str, list[float]] = {}
    category_valid: dict[str, int] = {}
    category_total: dict[str, int] = {}
    for result in detailed_results:
        category = str(result["category"])
        category_total[category] = category_total.get(category, 0) + 1
        metrics = result.get("metrics")
        if not isinstance(metrics, dict):
            continue
        if metrics.get("valid_video"):
            category_valid[category] = category_valid.get(category, 0) + 1
        clip = metrics.get("clip")
        if isinstance(clip, dict) and isinstance(
            clip.get("mean_prompt_score"), (int, float)
        ):
            category_values.setdefault(category, []).append(
                float(clip["mean_prompt_score"])
            )

    return [
        {
            "category": category,
            "samples": category_total[category],
            "valid_videos": category_valid.get(category, 0),
            "average_clip": _mean(category_values.get(category, [])),
        }
        for category in sorted(category_total)
    ]


def _quality_gate(
    *,
    summary: dict[str, Any],
    thresholds: QualityThresholds,
    clip_enabled: bool,
) -> tuple[bool, list[dict[str, Any]]]:
    checks = [
        {
            "metric": "generation_success_ratio",
            "actual": summary["generation_success_ratio"],
            "operator": ">=",
            "threshold": thresholds.min_generation_success_ratio,
            "passed": (
                summary["generation_success_ratio"]
                >= thresholds.min_generation_success_ratio
            ),
        },
        {
            "metric": "invalid_video_count",
            "actual": summary["invalid_video_count"],
            "operator": "<=",
            "threshold": thresholds.max_invalid_videos,
            "passed": summary["invalid_video_count"] <= thresholds.max_invalid_videos,
        },
        {
            "metric": "frozen_video_count",
            "actual": summary["frozen_video_count"],
            "operator": "<=",
            "threshold": thresholds.max_frozen_videos,
            "passed": summary["frozen_video_count"] <= thresholds.max_frozen_videos,
        },
    ]

    if not clip_enabled:
        checks.append(
            {
                "metric": "clip_enabled",
                "actual": False,
                "operator": "==",
                "threshold": True,
                "passed": False,
            }
        )
    if thresholds.min_average_clip is None:
        checks.append(
            {
                "metric": "min_average_clip_configured",
                "actual": None,
                "operator": "is not",
                "threshold": None,
                "passed": False,
            }
        )
    else:
        average_clip = summary.get("average_clip")
        checks.append(
            {
                "metric": "average_clip",
                "actual": average_clip,
                "operator": ">=",
                "threshold": thresholds.min_average_clip,
                "passed": (
                    average_clip is not None
                    and average_clip >= thresholds.min_average_clip
                ),
            }
        )

    if thresholds.min_minimum_clip is not None:
        minimum_clip = summary.get("minimum_clip")
        checks.append(
            {
                "metric": "minimum_clip",
                "actual": minimum_clip,
                "operator": ">=",
                "threshold": thresholds.min_minimum_clip,
                "passed": (
                    minimum_clip is not None
                    and minimum_clip >= thresholds.min_minimum_clip
                ),
            }
        )
    return all(check["passed"] for check in checks), checks


def _aggregate_results(
    *,
    detailed_results: list[dict[str, Any]],
    mode: EvaluationMode,
    thresholds: QualityThresholds,
    clip_enabled: bool,
) -> dict[str, Any]:
    requested_count = len(detailed_results)
    generation_success_count = sum(
        bool(result.get("generation_success")) for result in detailed_results
    )
    analyzed = [
        result
        for result in detailed_results
        if result.get("analysis_success") and isinstance(result.get("metrics"), dict)
    ]
    valid_video_count = sum(
        bool(result["metrics"].get("valid_video")) for result in analyzed
    )
    structural = [
        result["metrics"]["structural"]
        for result in analyzed
        if isinstance(result["metrics"].get("structural"), dict)
    ]
    frozen_video_count = sum(bool(metric.get("is_frozen")) for metric in structural)
    black_video_count = sum(bool(metric.get("is_black")) for metric in structural)
    flat_video_count = sum(bool(metric.get("is_flat")) for metric in structural)
    invalid_video_count = requested_count - valid_video_count

    clip_metrics = [
        result["metrics"]["clip"]
        for result in analyzed
        if isinstance(result["metrics"].get("clip"), dict)
    ]
    mean_clip_scores = [float(metric["mean_prompt_score"]) for metric in clip_metrics]
    minimum_clip_scores = [
        float(metric["minimum_prompt_score"]) for metric in clip_metrics
    ]
    progression_margins = [
        float(metric["progression"]["progression_margin"])
        for metric in clip_metrics
        if isinstance(metric.get("progression"), dict)
    ]
    motion_values = [
        float(metric["mean_frame_delta"])
        for metric in structural
        if isinstance(metric.get("mean_frame_delta"), (int, float))
    ]
    cleanup_failure_count = sum(
        bool(result.get("cleanup_error")) for result in detailed_results
    )

    summary: dict[str, Any] = {
        "requested_count": requested_count,
        "generation_success_count": generation_success_count,
        "generation_success_ratio": (
            generation_success_count / requested_count if requested_count else 0.0
        ),
        "analyzed_video_count": len(analyzed),
        "valid_video_count": valid_video_count,
        "invalid_video_count": invalid_video_count,
        "frozen_video_count": frozen_video_count,
        "black_video_count": black_video_count,
        "flat_video_count": flat_video_count,
        "cleanup_failure_count": cleanup_failure_count,
        "average_clip": _mean(mean_clip_scores),
        "minimum_clip": min(minimum_clip_scores) if minimum_clip_scores else None,
        "average_motion": _mean(motion_values),
        "average_progression_margin": _mean(progression_margins),
    }

    if mode == "gate":
        gate_passed, gate_checks = _quality_gate(
            summary=summary,
            thresholds=thresholds,
            clip_enabled=clip_enabled,
        )
        accuracy_check = ReportCheckTypes.from_result(gate_passed)
        success = gate_passed
    else:
        gate_checks = []
        accuracy_check = ReportCheckTypes.NA
        # Baseline mode evaluates quality without asserting an uncalibrated
        # threshold. At least one analyzed output is required for a useful run.
        success = bool(analyzed)

    return {
        "summary": summary,
        "category_results": _aggregate_category_results(detailed_results),
        "quality_gate_checks": gate_checks,
        "accuracy_check": int(accuracy_check),
        "quality_status": (
            "baseline" if mode == "baseline" else ("pass" if success else "fail")
        ),
        "success": success,
    }


async def run_video_quality_evaluation(
    *,
    base_url: str,
    api_key: str,
    output_dir: Path,
    profile: Profile = DEFAULT_PROFILE,
    mode: EvaluationMode = DEFAULT_MODE,
    samples_per_prompt: int | None = None,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    enable_clip: bool = True,
    thresholds: QualityThresholds = QualityThresholds(),
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    download_timeout: float = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
    poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
    poll_timeout: float = DEFAULT_POLL_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Generate, score, preserve, and clean up a MiniMax-H3 quality run."""

    normalized_profile = _normalize_profile(profile)
    normalized_mode = _normalize_mode(mode)
    resolved_samples = _resolved_samples_per_prompt(
        normalized_profile,
        samples_per_prompt,
    )
    if sample_count < 2:
        raise ValueError("sample_count must be at least 2")

    selected_prompts = _selected_prompts(normalized_profile)
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_scorer = BatchedCLIPScorer() if enable_clip else None
    detailed_results: list[dict[str, Any]] = []

    async with MiniMaxH3Client(
        base_url=base_url,
        api_key=api_key,
        request_timeout=request_timeout,
        download_timeout=download_timeout,
        poll_interval=poll_interval,
        poll_timeout=poll_timeout,
    ) as client:
        for prompt_case in selected_prompts:
            for sample_index in range(1, resolved_samples + 1):
                logger.info(
                    "Evaluating MiniMax quality prompt=%s sample=%d/%d",
                    prompt_case.id,
                    sample_index,
                    resolved_samples,
                )
                detailed_results.append(
                    await _evaluate_sample(
                        client=client,
                        prompt_case=prompt_case,
                        sample_index=sample_index,
                        output_dir=output_dir,
                        sample_count=sample_count,
                        clip_scorer=clip_scorer,
                    )
                )

    aggregate = _aggregate_results(
        detailed_results=detailed_results,
        mode=normalized_mode,
        thresholds=thresholds,
        clip_enabled=enable_clip,
    )
    return {
        "task_name": "minimax_h3_video_quality",
        "base_url": base_url.rstrip("/"),
        "model": MODEL_NAME,
        "profile": normalized_profile,
        "mode": normalized_mode,
        "samples_per_prompt": resolved_samples,
        "frame_sample_count": sample_count,
        "clip_enabled": enable_clip,
        "output_dir": str(output_dir),
        "thresholds": {
            "min_generation_success_ratio": thresholds.min_generation_success_ratio,
            "min_average_clip": thresholds.min_average_clip,
            "min_minimum_clip": thresholds.min_minimum_clip,
            "max_invalid_videos": thresholds.max_invalid_videos,
            "max_frozen_videos": thresholds.max_frozen_videos,
        },
        **aggregate,
        "detailed_results": detailed_results,
    }


class MiniMaxH3VideoQualityTest(BaseTest):
    """Workflow-compatible wrapper around the quality evaluator."""

    KIND = "minimax_h3_video_quality"
    TASK_TYPE = "video"
    HARDWARE_REQUIREMENT = HardwareRequirement.ANY_CHIP

    async def _run_specific_test_async(self) -> dict[str, Any]:
        output_dir_value = self.targets.get("output_dir")
        if output_dir_value:
            output_dir = Path(str(output_dir_value))
        elif self.ctx is not None:
            output_dir = Path(self.ctx.output_path) / "minimax_h3_video_quality"
        else:
            output_dir = DEFAULT_OUTPUT_DIR

        return await run_video_quality_evaluation(
            base_url=self.base_url,
            api_key=_resolve_api_key(),
            output_dir=output_dir,
            profile=_normalize_profile(self.targets.get("profile")),
            mode=_normalize_mode(self.targets.get("mode")),
            samples_per_prompt=self.targets.get("samples_per_prompt"),
            sample_count=int(self.targets.get("sample_count", DEFAULT_SAMPLE_COUNT)),
            enable_clip=bool(self.targets.get("enable_clip", True)),
            thresholds=QualityThresholds(
                min_generation_success_ratio=float(
                    self.targets.get("min_generation_success_ratio", 1.0)
                ),
                min_average_clip=_optional_float(self.targets.get("min_average_clip")),
                min_minimum_clip=_optional_float(self.targets.get("min_minimum_clip")),
                max_invalid_videos=int(self.targets.get("max_invalid_videos", 0)),
                max_frozen_videos=int(self.targets.get("max_frozen_videos", 0)),
            ),
            request_timeout=float(
                self.targets.get(
                    "request_timeout",
                    DEFAULT_REQUEST_TIMEOUT_SECONDS,
                )
            ),
            download_timeout=float(
                self.targets.get(
                    "download_timeout",
                    DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
                )
            ),
            poll_interval=float(
                self.targets.get(
                    "poll_interval",
                    DEFAULT_POLL_INTERVAL_SECONDS,
                )
            ),
            poll_timeout=float(
                self.targets.get(
                    "poll_timeout",
                    DEFAULT_POLL_TIMEOUT_SECONDS,
                )
            ),
        )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def run_minimax_h3_video_quality(
    ctx: MediaContext,
    targets: dict[str, Any] | None = None,
) -> Block:
    """Run the evaluator and return a workflow ``evals`` block."""

    resolved_targets = targets or {}
    test = MiniMaxH3VideoQualityTest(
        TestConfig(
            {
                "timeout": DEFAULT_TEST_TIMEOUT_SECONDS,
                # Retrying would create additional billable generations and
                # distort the requested sample count.
                "retry_attempts": 0,
                "retry_delay": 0,
                "break_on_failure": False,
            }
        ),
        resolved_targets,
        ctx=ctx,
    )
    internal_block = test.run_tests()
    return Block(
        kind="evals",
        id=block_id(ctx) or None,
        title="MiniMax-H3 Video Quality",
        task_type="video",
        targets=dict(resolved_targets),
        data=internal_block.data,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and score MiniMax-H3 videos against a fixed prompt set."
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--profile", choices=sorted(_PROFILES), default=DEFAULT_PROFILE)
    parser.add_argument("--mode", choices=sorted(_MODES), default=DEFAULT_MODE)
    parser.add_argument("--samples-per-prompt", type=int)
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--skip-clip", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT_SECONDS
    )
    parser.add_argument(
        "--download-timeout",
        type=float,
        default=DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL_SECONDS,
    )
    parser.add_argument(
        "--poll-timeout", type=float, default=DEFAULT_POLL_TIMEOUT_SECONDS
    )
    parser.add_argument("--min-generation-success-ratio", type=float, default=1.0)
    parser.add_argument("--min-average-clip", type=float)
    parser.add_argument("--min-minimum-clip", type=float)
    parser.add_argument("--max-invalid-videos", type=int, default=0)
    parser.add_argument("--max-frozen-videos", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = asyncio.run(
            run_video_quality_evaluation(
                base_url=args.base_url,
                api_key=_resolve_api_key(),
                output_dir=args.output_dir,
                profile=_normalize_profile(args.profile),
                mode=_normalize_mode(args.mode),
                samples_per_prompt=args.samples_per_prompt,
                sample_count=args.sample_count,
                enable_clip=not args.skip_clip,
                thresholds=QualityThresholds(
                    min_generation_success_ratio=args.min_generation_success_ratio,
                    min_average_clip=args.min_average_clip,
                    min_minimum_clip=args.min_minimum_clip,
                    max_invalid_videos=args.max_invalid_videos,
                    max_frozen_videos=args.max_frozen_videos,
                ),
                request_timeout=args.request_timeout,
                download_timeout=args.download_timeout,
                poll_interval=args.poll_interval,
                poll_timeout=args.poll_timeout,
            )
        )
    except Exception as exc:  # noqa: BLE001 - CLI must emit a structured report
        logger.exception("MiniMax-H3 video quality evaluation could not run")
        result = {
            "task_name": "minimax_h3_video_quality",
            "success": False,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }

    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(f"{rendered}\n", encoding="utf-8")
    return 0 if result.get("success") else 1


__all__ = [
    "MiniMaxH3VideoQualityTest",
    "QUALITY_PROMPTS",
    "QualityThresholds",
    "VideoQualityPrompt",
    "run_minimax_h3_video_quality",
    "run_video_quality_evaluation",
]


if __name__ == "__main__":
    sys.exit(main())
