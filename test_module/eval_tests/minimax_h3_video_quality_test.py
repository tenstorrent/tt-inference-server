# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Reference-driven quality evaluation for MiniMax-H3 text-to-video output.

This evaluator uses the inference server's V1 video job lifecycle, downloads
each output immediately, and computes spatial, temporal, and optional CLIP
metrics. When a matching reference is configured, the metrics are graded;
otherwise the result is informational with ``accuracy_check=NA``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
    resolve_server_api_key,
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
DURATION_SECONDS = 5
ASPECT_RATIO = "16:9"

DEFAULT_SAMPLE_COUNT = 8
DEFAULT_SAMPLES_PER_PROMPT = 1
DEFAULT_REQUEST_TIMEOUT_SECONDS = 60.0
DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 600.0
DEFAULT_POLL_INTERVAL_SECONDS = 5.0
DEFAULT_POLL_TIMEOUT_SECONDS = 1800.0
DEFAULT_TEST_TIMEOUT_SECONDS = 7200
DEFAULT_OUTPUT_DIR = Path("/tmp/minimax_h3_video_quality")
ACCURACY_REFERENCE_PATH = Path(
    "reference_config/evals/eval_targets/model_accuracy_reference.json"
)


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


T2V_PROMPT = VideoQualityPrompt(
    id="t2v",
    category="t2v",
    prompt=(
        "A small corgi wearing a bright yellow raincoat and round red "
        "glasses starts beside a red ball in a bright room. The camera "
        "smoothly tracks from left to right as the corgi nudges the ball, "
        "follows it across the floor, and stops with the ball beside a "
        "blue box near a sunlit window. Keep the same corgi, raincoat, "
        "and glasses throughout the shot."
    ),
    required_concepts=(
        "a small corgi",
        "a bright yellow raincoat",
        "round red glasses",
        "a red ball",
        "a blue box",
        "a sunlit window",
    ),
    expected_start=(
        "a corgi in a yellow raincoat and red glasses beside a red ball "
        "in a bright room"
    ),
    expected_end=("the same corgi and red ball beside a blue box near a sunlit window"),
)


def _resolved_samples_per_prompt(requested: int | None) -> int:
    samples = requested if requested is not None else DEFAULT_SAMPLES_PER_PROMPT
    if samples < 1:
        raise ValueError("samples_per_prompt must be at least 1")
    return samples


def _create_payload(prompt: str) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "aspect_ratio": ASPECT_RATIO,
        "duration_seconds": DURATION_SECONDS,
        "seed": 0,
    }


def _validate_completed_task(
    task: dict[str, Any],
    *,
    task_id: str,
    prompt: str,
) -> None:
    if task.get("status") != "completed":
        raise MiniMaxClientError(
            f"task reached terminal status {task.get('status')!r}",
            task_id=task_id,
            response_body=json.dumps(task.get("error")),
        )
    if task.get("id") != task_id or task.get("job_type") != "video":
        raise MiniMaxClientError(
            f"completed video job metadata mismatch: {task!r}",
            task_id=task_id,
        )
    request = task.get("request_parameters")
    if not isinstance(request, dict):
        raise MiniMaxClientError(
            "completed video job has no request_parameters object",
            task_id=task_id,
        )
    expected = _create_payload(prompt)
    mismatches = {
        field: {"expected": value, "actual": request.get(field)}
        for field, value in expected.items()
        if request.get(field) != value
    }
    if mismatches:
        raise MiniMaxClientError(
            f"completed task request metadata mismatch: {mismatches}",
            task_id=task_id,
        )


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
    }
    task_id: str | None = None

    try:
        task_id = await client.create_video(_create_payload(prompt_case.prompt))
        result["task_id"] = task_id
        terminal = await client.wait_for_terminal(task_id)
        result["observed_statuses"] = list(terminal.observed_statuses)
        result["task"] = terminal.task

        _validate_completed_task(
            terminal.task,
            task_id=task_id,
            prompt=prompt_case.prompt,
        )
        result["generation_success"] = True
        video_path = output_dir / f"{prompt_case.id}_{sample_index}_{task_id}.mp4"
        download = await client.download_video(task_id, video_path)
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


def _load_quality_reference(requested_count: int) -> dict[str, Any] | None:
    """Load the configured MiniMax reference for this evaluation size."""

    try:
        reference_data = json.loads(ACCURACY_REFERENCE_PATH.read_text())
    except FileNotFoundError:
        logger.warning("Accuracy reference file not found: %s", ACCURACY_REFERENCE_PATH)
        return None
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid accuracy reference JSON at {ACCURACY_REFERENCE_PATH}: {exc}"
        ) from exc

    model_reference = reference_data.get(MODEL_NAME)
    if not isinstance(model_reference, dict):
        return None
    accuracy = model_reference.get("accuracy")
    if not isinstance(accuracy, dict):
        return None
    reference = accuracy.get(str(requested_count))
    return reference if isinstance(reference, dict) else None


def _reference_checks(
    *,
    summary: dict[str, Any],
    reference: dict[str, Any],
    clip_enabled: bool,
) -> tuple[bool, list[dict[str, Any]]]:
    """Compare current metrics with a pre-recorded reference configuration."""

    checks = [
        {
            "metric": "generation_success_ratio",
            "actual": summary["generation_success_ratio"],
            "operator": ">=",
            "threshold": reference.get("generation_success_ratio_min", 1.0),
            "passed": (
                summary["generation_success_ratio"]
                >= float(reference.get("generation_success_ratio_min", 1.0))
            ),
        },
        {
            "metric": "invalid_video_count",
            "actual": summary["invalid_video_count"],
            "operator": "<=",
            "threshold": reference.get("max_invalid_videos", 0),
            "passed": summary["invalid_video_count"]
            <= int(reference.get("max_invalid_videos", 0)),
        },
        {
            "metric": "frozen_video_count",
            "actual": summary["frozen_video_count"],
            "operator": "<=",
            "threshold": reference.get("max_frozen_videos", 0),
            "passed": summary["frozen_video_count"]
            <= int(reference.get("max_frozen_videos", 0)),
        },
    ]

    clip_range = reference.get("clip_valid_range")
    if clip_range is not None and (
        not isinstance(clip_range, list) or len(clip_range) != 2
    ):
        raise ValueError("MiniMax clip_valid_range must contain [minimum, maximum]")
    if clip_range is not None and not clip_enabled:
        checks.append(
            {
                "metric": "clip_enabled",
                "actual": False,
                "operator": "==",
                "threshold": True,
                "passed": False,
            }
        )
    elif clip_range is not None:
        average_clip = summary.get("average_clip")
        checks.append(
            {
                "metric": "average_clip",
                "actual": average_clip,
                "operator": "within",
                "threshold": clip_range,
                "passed": (
                    average_clip is not None
                    and float(clip_range[0]) <= average_clip <= float(clip_range[1])
                ),
            }
        )

    minimum_clip_threshold = reference.get("minimum_clip_min")
    if minimum_clip_threshold is not None:
        minimum_clip = summary.get("minimum_clip")
        checks.append(
            {
                "metric": "minimum_clip",
                "actual": minimum_clip,
                "operator": ">=",
                "threshold": minimum_clip_threshold,
                "passed": (
                    minimum_clip is not None
                    and minimum_clip >= float(minimum_clip_threshold)
                ),
            }
        )
    return all(check["passed"] for check in checks), checks


def _aggregate_results(
    *,
    detailed_results: list[dict[str, Any]],
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
        "average_clip": _mean(mean_clip_scores),
        "minimum_clip": min(minimum_clip_scores) if minimum_clip_scores else None,
        "average_motion": _mean(motion_values),
        "average_progression_margin": _mean(progression_margins),
    }
    all_outputs_valid = (
        requested_count > 0
        and generation_success_count == requested_count
        and valid_video_count == requested_count
    )

    reference = _load_quality_reference(requested_count)
    if reference is not None:
        reference_passed, reference_checks = _reference_checks(
            summary=summary,
            reference=reference,
            clip_enabled=clip_enabled,
        )
        accuracy_check = ReportCheckTypes.from_result(reference_passed)
        success = all_outputs_valid and reference_passed
        quality_status = "pass" if reference_passed else "fail"
    else:
        reference_checks = []
        accuracy_check = ReportCheckTypes.NA
        success = all_outputs_valid
        quality_status = "na"

    return {
        "summary": summary,
        "category_results": _aggregate_category_results(detailed_results),
        "quality_reference": reference,
        "quality_reference_checks": reference_checks,
        "accuracy_check": int(accuracy_check),
        "quality_status": quality_status,
        "success": success,
    }


async def run_video_quality_evaluation(
    *,
    base_url: str,
    api_key: str,
    output_dir: Path,
    samples_per_prompt: int | None = None,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    enable_clip: bool = True,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    download_timeout: float = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
    poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
    poll_timeout: float = DEFAULT_POLL_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Generate, score, and preserve a MiniMax-H3 quality run."""

    resolved_samples = _resolved_samples_per_prompt(samples_per_prompt)
    if sample_count < 2:
        raise ValueError("sample_count must be at least 2")

    selected_prompts = (T2V_PROMPT,)
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
        clip_enabled=enable_clip,
    )
    return {
        "task_name": "minimax_h3_video_quality",
        "base_url": base_url.rstrip("/"),
        "model": MODEL_NAME,
        "evaluation_type": "calibration",
        "samples_per_prompt": resolved_samples,
        "frame_sample_count": sample_count,
        "clip_enabled": enable_clip,
        "output_dir": str(output_dir),
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
            api_key=resolve_server_api_key(),
            output_dir=output_dir,
            samples_per_prompt=self.targets.get("samples_per_prompt"),
            sample_count=int(self.targets.get("sample_count", DEFAULT_SAMPLE_COUNT)),
            enable_clip=bool(self.targets.get("enable_clip", True)),
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
        description=(
            "Generate calibration samples and score MiniMax-H3 videos "
            "against the T2V prompt."
        )
    )
    parser.add_argument("--base-url", required=True)
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = asyncio.run(
            run_video_quality_evaluation(
                base_url=args.base_url,
                api_key=resolve_server_api_key(),
                output_dir=args.output_dir,
                samples_per_prompt=args.samples_per_prompt,
                sample_count=args.sample_count,
                enable_clip=not args.skip_clip,
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
    "T2V_PROMPT",
    "VideoQualityPrompt",
    "run_minimax_h3_video_quality",
    "run_video_quality_evaluation",
]


if __name__ == "__main__":
    sys.exit(main())
