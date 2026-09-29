# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Run-context attribute helpers shared by (media) test modules.

These read knobs off a run context (``ctx.model_spec.cli_args`` /
``ctx.all_params``) and are engine-generic: they moved here from
``workflows/utils.py`` so test modules never import the Tenstorrent
adapter. ``workflows.utils`` re-exports them for pre-extraction callers.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# SDXL num prompts limits
SDXL_DEFAULT_NUM_PROMPTS = 100
SDXL_LOWER_BOUND_NUM_PROMPTS = 2
SDXL_UPPER_BOUND_NUM_PROMPTS = 5000


def is_streaming_enabled_for_whisper(self) -> bool:
    """Determine if streaming is enabled for the Whisper model based on CLI args. Default to True if not set."""
    logger.info("Checking if streaming is enabled for Whisper model")
    cli_args = getattr(self.model_spec, "cli_args", {})

    # Check if streaming arg exists and has a valid value
    streaming_value = cli_args.get("streaming")
    if streaming_value is None:
        return True

    # Convert to string and check if it's 'true'
    streaming_enabled = str(streaming_value).lower() == "true"

    return streaming_enabled


def is_preprocessing_enabled_for_whisper(self) -> bool:
    """Determine if preprocessing is enabled for the Whisper model based on CLI args. Default to True if not set."""
    logger.info("Checking if preprocessing is enabled for Whisper model")

    cli_args = getattr(self.model_spec, "cli_args", {})
    preprocessing_value = cli_args.get("preprocessing")
    if preprocessing_value is None:
        return True

    # Convert to string and check if it's 'true'
    preprocessing_enabled = str(preprocessing_value).lower() == "true"

    return preprocessing_enabled


def is_sdxl_num_prompts_enabled(self) -> int:
    """Determine the number of prompts to use for SDXL based on CLI args. Default to 100 if not set."""
    logger.info("Checking if sdxl_num_prompts is set")

    cli_args = getattr(self.model_spec, "cli_args", {})
    sdxl_num_prompts = cli_args.get("sdxl_num_prompts")
    if sdxl_num_prompts is None:
        return SDXL_DEFAULT_NUM_PROMPTS

    # Convert to int and return
    num_prompts = int(sdxl_num_prompts)
    if (
        num_prompts < SDXL_LOWER_BOUND_NUM_PROMPTS
        or num_prompts > SDXL_UPPER_BOUND_NUM_PROMPTS
    ):
        return SDXL_DEFAULT_NUM_PROMPTS

    return num_prompts


def get_num_calls(self) -> int:
    """Get number of calls from benchmark parameters."""
    logger.info("Extracting number of calls from benchmark parameters")

    # Guard clause: Handle single config object case (evals)
    if hasattr(self.all_params, "tasks") and not isinstance(
        self.all_params, (list, tuple)
    ):
        return 2  # hard coding for evals

    # Handle list/iterable case (benchmarks)
    if isinstance(self.all_params, (list, tuple)):
        return next(
            (
                getattr(param, "num_eval_runs", 2)
                for param in self.all_params
                if hasattr(param, "num_eval_runs")
            ),
            2,
        )

    return 2
