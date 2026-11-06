# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""How a media deployment upsamples prompts (FLUX.3 Image klein), for its server and its evals.

tt-media-server reads two settings:

* ``PROMPT_UPSAMPLER``: ``remote`` (the default, the console's upsampler at
  ``PROMPT_UPSAMPLER_URL``) or ``local`` (the pipeline's own, on the device).
* ``PROMPT_UPSAMPLER_FROZEN``: ``true`` uses every prompt exactly as sent.

Both come from the host's environment when it sets them, so a run chooses:
``PROMPT_UPSAMPLER_FROZEN=false PROMPT_UPSAMPLER=local python run.py ...`` runs
end to end, through the live upsampler. Otherwise a --dev-mode run (local and
CI) has frozen prompts: its evals send the GPU run's upsampled captions, which
tests the image model alone. A prod run keeps the server's defaults. The
upsampler's URL and token are secrets, in .env with the others.
"""

from __future__ import annotations

import os
from typing import Mapping

PROMPT_UPSAMPLER_ENV = ("PROMPT_UPSAMPLER", "PROMPT_UPSAMPLER_FROZEN")


def prompt_upsampler_env_vars(
    dev_mode: bool, environ: Mapping[str, str] = os.environ
) -> dict:
    """The prompt upsampler settings to start the server with."""
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
