# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""End-to-end MiniMax-H3 lifecycle, download, video, and audio checks."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import shutil
import subprocess
import sys
from array import array
from pathlib import Path
from typing import TYPE_CHECKING, Any

from test_module._test_common import BaseTest, HardwareRequirement, TestConfig
from test_module._test_common.minimax_h3_client import (
    MiniMaxClientError,
    MiniMaxH3Client,
    resolve_server_api_key,
)
from test_module._test_common.video_quality_metrics import (
    MissingVideoQualityDependency,
    analyze_video_quality,
)

if TYPE_CHECKING:
    from report_module.schema import Block
    from test_module.context import MediaContext

logger = logging.getLogger(__name__)

PROMPT = (
    "A bright red kite flies across a clear blue daytime sky above a sunlit "
    "green field, with smooth visible motion and wind in the soundtrack."
)
ASPECT_RATIO = "16:9"
DURATION_SECONDS = 5
DEFAULT_OUTPUT_PATH = Path("/tmp/minimax_h3_lifecycle.mp4")
DEFAULT_REQUEST_TIMEOUT_SECONDS = 60.0
DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 600.0
DEFAULT_POLL_INTERVAL_SECONDS = 5.0
DEFAULT_POLL_TIMEOUT_SECONDS = 1800.0
DEFAULT_TEST_TIMEOUT_SECONDS = 2400
DEFAULT_FRAME_SAMPLE_COUNT = 8


def _create_payload() -> dict[str, Any]:
    return {
        "prompt": PROMPT,
        "aspect_ratio": ASPECT_RATIO,
        "duration_seconds": DURATION_SECONDS,
        "seed": 0,
    }


def _validate_completed_job(task: dict[str, Any], *, task_id: str) -> None:
    if task.get("status") != "completed":
        raise MiniMaxClientError(
            f"video job reached terminal status {task.get('status')!r}",
            task_id=task_id,
            response_body=json.dumps(task.get("error")),
        )
    if task.get("id") != task_id or task.get("job_type") != "video":
        raise MiniMaxClientError(
            f"unexpected completed video job metadata: {task!r}",
            task_id=task_id,
        )
    request = task.get("request_parameters")
    if not isinstance(request, dict):
        raise MiniMaxClientError(
            "completed video job has no request_parameters object",
            task_id=task_id,
        )
    expected = {
        "prompt": PROMPT,
        "aspect_ratio": ASPECT_RATIO,
        "duration_seconds": DURATION_SECONDS,
        "seed": 0,
    }
    mismatches = {
        key: {"expected": value, "actual": request.get(key)}
        for key, value in expected.items()
        if request.get(key) != value
    }
    if mismatches:
        raise MiniMaxClientError(
            f"completed video job request metadata mismatch: {mismatches}",
            task_id=task_id,
        )


def _ffmpeg_binary() -> str:
    binary = shutil.which("ffmpeg")
    if binary:
        return binary
    try:
        import imageio_ffmpeg  # pyright: ignore[reportMissingImports]

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise MissingVideoQualityDependency(
            "ffmpeg is required to verify the MiniMax-H3 soundtrack"
        ) from exc


def _probe_non_silent_audio(video_path: Path) -> dict[str, Any]:
    """Decode the first audio stream to mono PCM and verify non-silence."""

    result = subprocess.run(
        [
            _ffmpeg_binary(),
            "-v",
            "error",
            "-i",
            str(video_path),
            "-map",
            "0:a:0",
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            "16000",
            "pipe:1",
        ],
        check=False,
        capture_output=True,
        timeout=120,
    )
    if result.returncode != 0 or not result.stdout:
        detail = result.stderr.decode("utf-8", errors="replace")[-500:]
        raise ValueError(f"generated MP4 has no decodable audio stream: {detail}")

    samples = array("h")
    samples.frombytes(result.stdout[: len(result.stdout) // 2 * 2])
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        raise ValueError("generated MP4 audio stream decoded to zero samples")

    peak = max(abs(sample) for sample in samples)
    rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
    if peak == 0 or rms < 1.0:
        raise ValueError(
            f"generated MP4 soundtrack is silent (peak={peak}, rms={rms:.3f})"
        )
    return {
        "decoded_samples": len(samples),
        "decoded_sample_rate": 16000,
        "peak_pcm16": peak,
        "rms_pcm16": rms,
    }


async def run_lifecycle_download(
    *,
    base_url: str,
    api_key: str,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    download_timeout: float = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS,
    poll_interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
    poll_timeout: float = DEFAULT_POLL_TIMEOUT_SECONDS,
    sample_count: int = DEFAULT_FRAME_SAMPLE_COUNT,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    async with MiniMaxH3Client(
        base_url=base_url,
        api_key=api_key,
        request_timeout=request_timeout,
        download_timeout=download_timeout,
        poll_interval=poll_interval,
        poll_timeout=poll_timeout,
    ) as client:
        task_id = await client.create_video(_create_payload())
        terminal = await client.wait_for_terminal(task_id)
        _validate_completed_job(terminal.task, task_id=task_id)
        download = await client.download_video(task_id, output_path)

    video_metrics = await asyncio.to_thread(
        analyze_video_quality,
        output_path,
        prompt=PROMPT,
        required_concepts=(),
        expected_start=None,
        expected_end=None,
        expected_duration=DURATION_SECONDS,
        expected_ratio=ASPECT_RATIO,
        sample_count=sample_count,
        clip_scorer=None,
    )
    audio_metrics = await asyncio.to_thread(_probe_non_silent_audio, output_path)
    success = bool(video_metrics.get("valid_video")) and audio_metrics["rms_pcm16"] >= 1

    return {
        "task_name": "minimax_h3_lifecycle_download",
        "base_url": base_url.rstrip("/"),
        "task_id": task_id,
        "observed_statuses": list(terminal.observed_statuses),
        "task": terminal.task,
        "video_path": str(download.path),
        "bytes_downloaded": download.bytes_downloaded,
        "download_content_type": download.content_type,
        "video_metrics": video_metrics,
        "audio_metrics": audio_metrics,
        "success": success,
    }


class MiniMaxH3LifecycleDeleteTest(BaseTest):
    """Legacy-named wrapper for the inference server's lifecycle/download API."""

    KIND = "minimax_h3_lifecycle_download"
    TASK_TYPE = "functional"
    HARDWARE_REQUIREMENT = HardwareRequirement.ANY_CHIP

    async def _run_specific_test_async(self) -> dict[str, Any]:
        output_value = self.targets.get("output_path")
        if output_value:
            output_path = Path(str(output_value))
        elif self.ctx is not None:
            output_path = (
                Path(self.ctx.output_path) / "minimax_h3_lifecycle_download.mp4"
            )
        else:
            output_path = DEFAULT_OUTPUT_PATH

        return await run_lifecycle_download(
            base_url=self.base_url,
            api_key=resolve_server_api_key(),
            output_path=output_path,
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
            sample_count=int(
                self.targets.get("sample_count", DEFAULT_FRAME_SAMPLE_COUNT)
            ),
        )


def run_minimax_h3_lifecycle_delete(
    ctx: MediaContext,
    targets: dict[str, Any] | None = None,
) -> Block:
    """Compatibility entry point for the renamed lifecycle/download behavior."""

    return MiniMaxH3LifecycleDeleteTest(
        TestConfig(
            {
                "timeout": DEFAULT_TEST_TIMEOUT_SECONDS,
                "retry_attempts": 0,
                "retry_delay": 0,
                "break_on_failure": False,
            }
        ),
        targets or {},
        ctx=ctx,
    ).run_tests()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate MiniMax-H3 media through /v1/videos/generations and "
            "verify video plus non-silent audio."
        )
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
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
        "--poll-timeout",
        type=float,
        default=DEFAULT_POLL_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=DEFAULT_FRAME_SAMPLE_COUNT,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = asyncio.run(
            run_lifecycle_download(
                base_url=args.base_url,
                api_key=resolve_server_api_key(),
                output_path=args.output,
                request_timeout=args.request_timeout,
                download_timeout=args.download_timeout,
                poll_interval=args.poll_interval,
                poll_timeout=args.poll_timeout,
                sample_count=args.sample_count,
            )
        )
    except Exception as exc:  # noqa: BLE001 - CLI emits a structured failure
        logger.exception("MiniMax-H3 lifecycle/download test could not run")
        result = {
            "task_name": "minimax_h3_lifecycle_download",
            "success": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("success") else 1


__all__ = [
    "MiniMaxH3LifecycleDeleteTest",
    "run_lifecycle_download",
    "run_minimax_h3_lifecycle_delete",
]


if __name__ == "__main__":
    sys.exit(main())
