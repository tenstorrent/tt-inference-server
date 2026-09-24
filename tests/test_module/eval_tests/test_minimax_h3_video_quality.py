# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""MiniMax-H3 quality eval: the CI run is graded against the checked-in reference, and a
failed sample says why in the job log (its result JSON only ships if the run survives)."""

from __future__ import annotations

import asyncio
import logging

from test_module.eval_tests import minimax_h3_video_quality_test as Q
from test_module.eval_tests.video_eval_tests import MINIMAX_H3_EVAL_TARGETS
from test_module._test_common.minimax_h3_client import MiniMaxClientError


def _analyzed(*, valid=True, frozen=False):
    return {
        "prompt_id": "t2v",
        "category": "t2v",
        "sample_index": 1,
        "task_id": "7b0c1a52-7c55-4b5f-9b43-5a8f0e0f7d11",
        "generation_success": True,
        "analysis_success": True,
        "metrics": {
            "probe": {
                "valid": valid,
                "width": 1344,
                "height": 768,
                "duration_seconds": 5.167 if valid else 2.0,
                "aspect_ratio_error": 0.0156,
            },
            "structural": {
                "total_decoded_frames": 125,
                "mean_frame_delta": 0.0 if frozen else 6.5,
                "is_black": False,
                "is_flat": False,
                "is_frozen": frozen,
            },
            "clip": None,
            "valid_video": valid,
        },
    }


def test_ci_eval_size_has_a_reference():
    # One prompt, samples_per_prompt clips of it: the reference key is the clip count.
    requested = len((Q.T2V_PROMPT,)) * MINIMAX_H3_EVAL_TARGETS["samples_per_prompt"]
    reference = Q._load_quality_reference(requested)
    assert reference is not None
    assert reference["generation_success_ratio_min"] == 1.0
    assert reference["max_invalid_videos"] == 0
    assert reference["max_frozen_videos"] == 0


def test_valid_clip_passes_the_reference():
    result = Q._aggregate_results(detailed_results=[_analyzed()], clip_enabled=False)
    assert result["success"] is True
    assert result["quality_status"] == "pass"
    assert result["quality_reference_checks"]


def test_frozen_clip_fails_the_reference_and_says_so(caplog):
    with caplog.at_level(logging.WARNING, logger=Q.logger.name):
        result = Q._aggregate_results(
            detailed_results=[_analyzed(frozen=True)], clip_enabled=False
        )
    assert result["success"] is False
    assert result["quality_status"] == "fail"
    assert "frozen_video_count=1 (want <= 0)" in caplog.text


def test_invalid_clip_logs_its_probe(caplog):
    with caplog.at_level(logging.WARNING, logger=Q.logger.name):
        Q._log_sample_outcome(_analyzed(valid=False))
        result = Q._aggregate_results(
            detailed_results=[_analyzed(valid=False)], clip_enabled=False
        )
    assert result["success"] is False
    assert "valid_video=False duration=2.0s size=1344x768" in caplog.text
    assert "0/1 valid videos" in caplog.text


def test_valid_clip_logs_at_info_only(caplog):
    with caplog.at_level(logging.INFO, logger=Q.logger.name):
        Q._log_sample_outcome(_analyzed())
    assert [r.levelno for r in caplog.records] == [logging.INFO]


def test_error_message_is_logged_and_truncated(caplog):
    result = {
        "prompt_id": "t2v",
        "sample_index": 1,
        "task_id": None,
        "error": {"type": "ValueError", "message": "x" * 2000},
    }
    with caplog.at_level(logging.WARNING, logger=Q.logger.name):
        Q._log_sample_outcome(result)
    (record,) = caplog.records
    assert "failed (ValueError)" in record.getMessage()
    assert len(record.getMessage()) < 700


class _FailingClient:
    """Stands in for MiniMaxH3Client: the job ends failed on the server."""

    async def create_video(self, payload):
        return "7b0c1a52-7c55-4b5f-9b43-5a8f0e0f7d11"

    async def wait_for_terminal(self, task_id):
        raise MiniMaxClientError(
            "task reached terminal status 'failed'", task_id=task_id
        )


def test_sample_error_reaches_the_log(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger=Q.logger.name):
        result = asyncio.run(
            Q._evaluate_sample(
                client=_FailingClient(),
                prompt_case=Q.T2V_PROMPT,
                sample_index=1,
                output_dir=tmp_path,
                sample_count=8,
                clip_scorer=None,
            )
        )
    assert result["generation_success"] is False
    assert "task reached terminal status 'failed'" in caplog.text
    assert "task=7b0c1a52-7c55-4b5f-9b43-5a8f0e0f7d11" in caplog.text
