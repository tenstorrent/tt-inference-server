# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from .health_tests import run_device_liveness

logger = logging.getLogger(__name__)

TEST_PAYLOADS_PATH = "utils/test_payloads"
DEVICE_LIVENESS_TEST_ALIVE = "alive"


@dataclass(frozen=True)
class MediaContext:
    """Immutable bundle of inputs needed by every eval/benchmark function."""

    all_params: Any
    model_spec: Any
    device: Any
    output_path: str
    service_port: int
    spec_tests_num_prompts_cap: Optional[int] = None
    runtime_config: Optional[Any] = None

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.service_port}"

    @property
    def test_payloads_path(self) -> str:
        return TEST_PAYLOADS_PATH


def get_health(ctx: MediaContext) -> tuple[bool, Optional[str]]:
    """Check server health via DeviceLivenessTest and return (ok, runner_in_use)."""
    logger.info("Checking server health using DeviceLivenessTest...")
    device_name = ctx.device.name if hasattr(ctx.device, "name") else str(ctx.device)
    num_devices = ctx.model_spec.device_model_spec.max_concurrency
    logger.info(f"Detected device: {device_name} with {num_devices} expected worker(s)")

    try:
        block = run_device_liveness(ctx)
    except SystemExit as e:
        logger.error(f"Health check failed with SystemExit: {e}")
        return (False, None)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return (False, None)

    data = block.data
    if data.get("success"):
        runner_in_use = data.get("runner_in_use")
        logger.info(
            f"✅ Health check passed after {data.get('attempts', 1)} attempt(s)"
        )
        return (True, runner_in_use)

    logger.error("Health check failed after all retry attempts")
    return (False, None)


@lru_cache(maxsize=None)
def get_tokenizer(hf_model_repo: str):
    """Return a cached ``AutoTokenizer`` for ``hf_model_repo`` or ``None`` on failure.

    Keyed by repo so multiple eval/benchmark calls for the same model reuse one
    tokenizer. Import is done lazily so modules that don't need tokenizers
    (e.g. image, video) don't pay the ``transformers`` import cost.
    """
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(hf_model_repo)
        logger.info(f"✅ Loaded tokenizer for {hf_model_repo}")
        return tokenizer
    except Exception as e:
        logger.warning(f"⚠️ Could not load tokenizer for {hf_model_repo}: {e}")
        logger.info("📝 Falling back to word-based token counting")
        return None


def count_tokens(hf_model_repo: str, text: str) -> int:
    """Count tokens using the cached tokenizer; fall back to word count."""
    if not text.strip():
        return 0

    tokenizer = get_tokenizer(hf_model_repo)
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception as e:
            logger.warning(f"Tokenizer encoding failed: {e}. Using word count.")

    return len(text.split())


def require_health(ctx: MediaContext) -> str:
    """Run the liveness check and raise on failure; return ``runner_in_use``."""
    health_status, runner_in_use = get_health(ctx)
    if not health_status:
        logger.error("Health check failed.")
        raise RuntimeError("Health check failed")
    logger.info(f"Health check passed. Runner in use: {runner_in_use}")
    return runner_in_use


def _now_timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def common_report_metadata(ctx: MediaContext, task_type: str) -> dict:
    """Metadata block shared by every benchmark report."""
    return {
        "model": ctx.model_spec.model_name,
        "device": ctx.device.name,
        "timestamp": _now_timestamp(),
        "task_type": task_type,
    }


def common_eval_metadata(ctx: MediaContext, task_type: str) -> dict:
    """Metadata block shared by every eval report.

    Adds ``task_name`` + ``tolerance`` from ``ctx.all_params.tasks[0]`` and
    lowercases ``device`` to match the legacy eval contract.
    """
    return {
        "model": ctx.model_spec.model_name,
        "device": ctx.device.name.lower(),
        "timestamp": _now_timestamp(),
        "task_type": task_type,
        "task_name": ctx.all_params.tasks[0].task_name,
        "tolerance": ctx.all_params.tasks[0].score.tolerance,
    }


__all__ = [
    "DEVICE_LIVENESS_TEST_ALIVE",
    "TEST_PAYLOADS_PATH",
    "MediaContext",
    "common_eval_metadata",
    "common_report_metadata",
    "count_tokens",
    "get_health",
    "get_tokenizer",
    "require_health",
]
