# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC


from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from report_module.schema import Block
from workflow_module import accept_blocks

from .._test_common import ReportCheckTypes, block_id, sweep_envelope
from .._test_common.video_output_check import save_clip
from ..context import HardwareRequirement, MediaContext, require_health

logger = logging.getLogger(__name__)


def _run_video_generation_eval(ctx: MediaContext) -> dict:
    """Delegate to VideoGenerationEvalsTest."""
    from .._test_common import TestConfig

    from .._test_common import i2v_eval_num_pairs, is_i2v_video_model
    from .video_generation_eval_test import (
        VideoGenerationEvalsTest,
        VideoGenerationEvalsTestRequest,
    )

    # Full HF repo id: keys the accuracy reference (full-repo keys). Routing
    # (is_i2v/endpoint) matches the "-I2V-" substring, which is preserved.
    model_name = ctx.model_spec.hf_model_repo
    generation_mode = "I2V" if is_i2v_video_model(model_name) else "T2V"
    logger.info(
        "Running video generation eval for %s using %s endpoint.",
        model_name,
        generation_mode,
    )

    num_prompts = i2v_eval_num_pairs() if generation_mode == "I2V" else 5
    num_inference_steps = 40
    start_from = 0
    frame_sample_rate = 8

    request = VideoGenerationEvalsTestRequest(
        model_name=model_name,
        num_prompts=num_prompts,
        start_from=start_from,
        num_inference_steps=num_inference_steps,
        server_url=ctx.base_url,
        frame_sample_rate=frame_sample_rate,
    )
    logger.info(f"Running VideoGenerationEvalsTest with request: {request}")

    config = TestConfig.create_default()
    test = VideoGenerationEvalsTest(config, {"request": request})

    logger.info("Starting VideoGenerationEvalsTest")
    result = test.run_tests()

    eval_results = result.data.get("eval_results", {})
    logger.info(f"VideoGenerationEvalsTest eval_results: {eval_results}")
    return eval_results


def _run_video_fvd_and_fvmd_eval() -> dict:
    """Run FVD + FVMD eval against reference and generated video directories."""
    from .._test_common import TestConfig

    from .video_fvd_eval_test import DATASET_DIR as FVD_DATASET_DIR
    from .video_fvd_eval_test import VideoFVDTest, VideoFVDTestRequest
    from .video_fvmd_eval_test import VideoFVMDTest, VideoFVMDTestRequest

    logger.info("Running video FVD and FVMD eval.")

    reference_videos_path = str(Path(FVD_DATASET_DIR) / "videos")
    generated_videos_path = str(Path("test_fixtures/datasets/videos").resolve())
    if not Path(generated_videos_path).exists():
        generated_videos_path = "/tmp/videos"
    logger.info(
        f"Reference path: {reference_videos_path}, generated path: {generated_videos_path}"
    )

    config = TestConfig.create_default()
    combined_results: dict = {
        "reference_videos_path": reference_videos_path,
        "generated_videos_path": generated_videos_path,
    }

    download_request = VideoFVDTestRequest(
        action="download", download_count=2, category="Sports"
    )
    download_test = VideoFVDTest(config, {"request": download_request})
    download_result = download_test.run_tests()
    if not download_result.data.get("success") or not download_result.data.get(
        "result", {}
    ).get("success"):
        logger.warning(
            "Reference video download failed: %s. "
            "FVD/FVMD may fail if reference dir is empty.",
            download_result,
        )
    else:
        logger.info("Reference videos downloaded successfully.")

    fvd_request = VideoFVDTestRequest(
        action="compute_fvd",
        reference_videos_path=reference_videos_path,
        generated_videos_path=generated_videos_path,
    )
    fvd_test = VideoFVDTest(config, {"request": fvd_request})
    fvd_result = fvd_test.run_tests()
    fvd_ok = fvd_result.data.get("success")
    if fvd_ok:
        combined_results["fvd"] = fvd_result.data.get("fvd_score")
        logger.info("FVD score: %s", combined_results["fvd"])
    else:
        combined_results["fvd"] = None
        logger.warning("FVD test failed or did not return score: %s", fvd_result)

    fvmd_request = VideoFVMDTestRequest(
        action="compute_fvmd",
        reference_videos_path=reference_videos_path,
        generated_videos_path=generated_videos_path,
    )
    fvmd_test = VideoFVMDTest(config, {"request": fvmd_request})
    fvmd_result = fvmd_test.run_tests()
    fvmd_ok = fvmd_result.data.get("success")
    if fvmd_ok:
        combined_results["fvmd"] = fvmd_result.data.get("fvmd_score")
        logger.info("FVMD score: %s", combined_results["fvmd"])
    else:
        combined_results["fvmd"] = None
        logger.warning("FVMD test failed or did not return score: %s", fvmd_result)

    return combined_results


# What an I2V clip's index entry carries beyond the prompt and image, so the
# VBench scorer can find its first frame and GPU reference.
I2V_CLIP_FIELDS = ("video_id", "width", "height", "seed")


def _save_eval_clips(
    ctx: MediaContext, videos: list[dict], rejected: list[dict] = ()
) -> None:
    """Keep every eval clip, with its prompt, in the uploaded run output.

    These are the clips the quality score is about, so they are worth looking at
    (and comparing with the GPU reference) on every run. Clips rejected for their
    shape go with the failed clips, out of the index the scorers read.
    """
    for video in rejected:
        save_clip(
            video.get("video_path"),
            ctx.output_path,
            "failed/evals",
            f"{video.get('video_id') or video.get('job_id') or 'clip'}.mp4",
        )
    index = []
    for i, video in enumerate(videos):
        name = f"{i:02d}_{video.get('video_id') or video.get('job_id') or 'clip'}.mp4"
        if save_clip(video.get("video_path"), ctx.output_path, "evals", name):
            entry = {"file": name, "prompt": video.get("prompt")}
            if video.get("image"):
                entry["conditioning_image"] = video["image"]
            if video.get("camera_motion"):
                entry["camera_motion"] = video["camera_motion"]
            entry.update({k: video[k] for k in I2V_CLIP_FIELDS if k in video})
            index.append(entry)
    if index:
        index_path = Path(ctx.output_path) / "videos" / "evals" / "index.json"
        index_path.write_text(json.dumps(index, indent=2))


# VBench runs on the host CPU after the clips are made: about 2 min per clip and
# dimension on a BH Galaxy host, so 20 clips fit with room to spare.
VBENCH_TIMEOUT_SECONDS = 90 * 60


def _strip_frames(record):
    """``record`` without the per-frame lists, which stay in vbench.json."""
    if isinstance(record, dict):
        return {k: _strip_frames(v) for k, v in record.items() if k != "frames"}
    return record


def _run_vbench(ctx: MediaContext) -> Block | None:
    """Score the saved eval clips with VBench, as a report section of its own.

    None when the eval saved no clips. The scorer sets up its own pinned venv,
    so it runs in a subprocess. A failure or timeout is reported in the section
    instead of failing the eval: nothing gates on VBench yet.
    """
    from .._test_common import vbench_i2v

    videos_dir = Path(ctx.output_path) / "videos" / "evals"
    if not (videos_dir / "index.json").is_file():
        return None
    command = [
        sys.executable,
        "-m",
        "test_module._test_common.vbench_i2v",
        "--videos-dir",
        str(videos_dir),
    ]
    logger.info("Scoring the eval clips with VBench: %s", " ".join(command))
    problem = None
    try:
        code = subprocess.run(
            command, cwd=_PROJECT_ROOT, timeout=VBENCH_TIMEOUT_SECONDS
        ).returncode
        if code:
            problem = f"VBench exited with code {code}"
    except subprocess.TimeoutExpired:
        problem = f"VBench stopped at its {VBENCH_TIMEOUT_SECONDS // 60} min limit"
    except Exception as e:
        problem = f"VBench did not run: {type(e).__name__}: {e}"
    if problem:
        logger.error(problem)

    # Written after each dimension, so a run cut short keeps what finished.
    out = videos_dir / "vbench.json"
    result = json.loads(out.read_text()) if out.is_file() else None
    title = f"{ctx.model_spec.hf_model_repo} on {ctx.device.name}"
    if result:
        markdown = vbench_i2v.to_markdown(result, title)
        if problem:
            markdown += f"\n{problem}; the scores above are what finished.\n"
    else:
        markdown = f"### VBench: {title}\n\n{problem or 'VBench scored nothing.'}\n"

    data: dict = {"markdown": markdown}
    if problem:
        data["error"] = problem
    if result:
        data.update(
            num_clips=result.get("num_clips"),
            device=result.get("device"),
            scores=vbench_i2v.summarize(result),
            dimensions=result.get("dimensions"),
            gpu_reference=_strip_frames(result.get("gpu_reference")),
        )
    return Block(
        kind="vbench",
        task_type="video",
        title="VBench",
        id=block_id(ctx) or None,
        targets={},
        data=data,
    )


def run_video_eval(ctx: MediaContext) -> Block:
    """Run evaluations for a video model (Mochi, WAN, etc.)."""
    logger.info(
        f"Running evals for model: {ctx.model_spec.model_name} on device: {ctx.device.name}"
    )
    require_health(ctx, HardwareRequirement.ANY_CHIP)

    try:
        eval_result = _run_video_generation_eval(ctx)
    except Exception as e:
        logger.error(f"Eval execution encountered an error: {e}")
        raise

    _save_eval_clips(
        ctx,
        (eval_result or {}).get("videos", []),
        (eval_result or {}).get("rejected_videos", []),
    )
    # Its own report section, next to the eval's (the runner returns one block).
    vbench = _run_vbench(ctx)
    if vbench is not None:
        accept_blocks([vbench], envelope=sweep_envelope(ctx))

    logger.info("Generating eval report...")
    task = ctx.all_params.tasks[0]
    data: dict = {
        "task_name": task.task_name,
        "tolerance": task.score.tolerance,
        "published_score": task.score.published_score,
        "published_score_ref": task.score.published_score_ref,
    }

    if eval_result:
        logger.info("Adding eval results from video generation test to benchmark data")
        data["num_prompts"] = eval_result.get("num_prompts", 0)
        data["num_generated"] = eval_result.get("num_generated", 0)
        rejected = eval_result.get("rejected_videos") or []
        if rejected:
            data["num_rejected"] = len(rejected)
            data["rejected"] = "; ".join(
                f"{v.get('video_id')}: {v.get('problem')}" for v in rejected
            )[:2000]
        data["num_inference_steps"] = eval_result.get("num_inference_steps", 0)

        clip_results = eval_result.get("clip_results", {})
        data["average_clip"] = clip_results.get("average_clip", 0.0)
        data["min_clip"] = clip_results.get("min_clip", 0.0)
        data["max_clip"] = clip_results.get("max_clip", 0.0)
        data["clip_standard_deviation"] = clip_results.get(
            "clip_standard_deviation", 0.0
        )
        data["accuracy_check"] = eval_result.get("accuracy_check", ReportCheckTypes.NA)
    else:
        logger.warning("No eval results from video generation test")

    data["fvd"] = None
    data["fvmd"] = None
    try:
        fvd_and_fvmd_result = _run_video_fvd_and_fvmd_eval()
        if fvd_and_fvmd_result:
            data["fvd"] = fvd_and_fvmd_result.get("fvd", 0)
            data["fvmd"] = fvd_and_fvmd_result.get("fvmd", 0)
    except Exception as e:
        logger.error(f"Error running video FVD and FVMD eval: {e}")

    return Block(
        kind="evals",
        task_type="video",
        title="Video Eval",
        id=block_id(ctx) or None,
        targets={
            "task_name": ctx.all_params.tasks[0].task_name,
            "tolerance": ctx.all_params.tasks[0].score.tolerance,
            "published_score": ctx.all_params.tasks[0].score.published_score,
            "published_score_ref": ctx.all_params.tasks[0].score.published_score_ref,
        },
        data=data,
    )


__all__ = ["run_video_eval"]
