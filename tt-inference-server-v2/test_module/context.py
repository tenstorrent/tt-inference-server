# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Shared context + health check + tokenizer cache for the v2 test module.

``MediaContext`` is a frozen dataclass that bundles the immutable inputs every
eval/benchmark function needs (model spec, device, output path, service port).
Free functions take a ``MediaContext`` instead of living on a strategy class.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from server_tests.test_cases.device_liveness_test import DeviceLivenessTest
from server_tests.test_classes import TestConfig

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

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.service_port}"

    @property
    def test_payloads_path(self) -> str:
        return TEST_PAYLOADS_PATH


def get_health(ctx: MediaContext) -> tuple[bool, Optional[str]]:
    """Check server health via DeviceLivenessTest and return (ok, runner_in_use)."""
    logger.info("Checking server health using DeviceLivenessTest...")
    device_name = (
        ctx.device.name if hasattr(ctx.device, "name") else str(ctx.device)
    )
    num_devices = ctx.model_spec.device_model_spec.max_concurrency
    logger.info(
        f"Detected device: {device_name} with {num_devices} expected worker(s)"
    )

    test_config = TestConfig(
        {
            "test_timeout": 1200,
            "retry_attempts": 229,
            "retry_delay": 10,
            "break_on_failure": False,
        }
    )
    logger.info(f"TestConfig: {test_config}")

    targets = {
        "num_of_devices": num_devices if num_devices and num_devices > 0 else None
    }
    logger.info(f"Test targets: {targets}")

    liveness_test = DeviceLivenessTest(test_config, targets)
    liveness_test.service_port = ctx.service_port

    try:
        logger.info("Running DeviceLivenessTest...")
        test_result = liveness_test.run_tests()

        if isinstance(test_result, dict) and test_result.get("success"):
            result_data = test_result.get("result", {})
            runner_in_use = result_data.get("full_response", {}).get(
                "runner_in_use", None
            )

            logger.info(
                f"✅ Health check passed after {test_result.get('attempts', 1)} attempt(s)"
            )
            return (True, runner_in_use)

        logger.error("Health check failed after all retry attempts")
        return (False, None)

    except SystemExit as e:
        logger.error(f"Health check failed with SystemExit: {e}")
        return (False, None)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
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


__all__ = [
    "DEVICE_LIVENESS_TEST_ALIVE",
    "TEST_PAYLOADS_PATH",
    "MediaContext",
    "count_tokens",
    "get_health",
    "get_tokenizer",
]
