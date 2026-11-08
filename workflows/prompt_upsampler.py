# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Whether a media deployment's prompts are upsampled (FLUX.3 Image klein), for its server and its evals.

tt-media-server reads ``PROMPT_UPSAMPLER_FROZEN``: ``true`` uses every prompt
exactly as sent, without the pipeline's prompt upsampler; otherwise the
upsampler rewrites each prompt on the device.

It comes from the host's environment when it sets it, so a run chooses:
``PROMPT_UPSAMPLER_FROZEN=false python run.py ...`` runs end to end, through
the upsampler. Otherwise a --dev-mode run (local and CI) has frozen prompts:
its evals send the GPU run's upsampled captions, which tests the image model
alone. A prod run keeps the server's default, the upsampler.
"""

from __future__ import annotations

import os
from typing import Mapping

PROMPT_UPSAMPLER_ENV = ("PROMPT_UPSAMPLER_FROZEN",)


def prompt_upsampler_env_vars(
    dev_mode: bool, environ: Mapping[str, str] = os.environ
) -> dict:
    """The prompt upsampler setting to start the server with."""
    env = {key: environ[key] for key in PROMPT_UPSAMPLER_ENV if environ.get(key)}
    if dev_mode:
        env.setdefault("PROMPT_UPSAMPLER_FROZEN", "true")
    return env


def prompts_frozen(dev_mode: bool, environ: Mapping[str, str] = os.environ) -> bool:
    """Whether the server started for this run uses prompts exactly as sent."""
    value = prompt_upsampler_env_vars(dev_mode, environ).get(
        "PROMPT_UPSAMPLER_FROZEN", "false"
    )
    return value.strip().lower() in ("1", "true")
