# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import pytest

from workflows.prompt_upsampler import prompt_upsampler_env_vars, prompts_frozen


@pytest.mark.parametrize(
    "dev_mode, environ, env, frozen",
    [
        # dev mode (local and CI) freezes prompts; prod keeps the server's defaults
        (True, {}, {"PROMPT_UPSAMPLER_FROZEN": "true"}, True),
        (False, {}, {}, False),
        # a run can choose live prompts, end to end, in dev mode too
        (
            True,
            {"PROMPT_UPSAMPLER_FROZEN": "false", "PROMPT_UPSAMPLER": "local"},
            {"PROMPT_UPSAMPLER_FROZEN": "false", "PROMPT_UPSAMPLER": "local"},
            False,
        ),
        (
            False,
            {"PROMPT_UPSAMPLER_FROZEN": "1"},
            {"PROMPT_UPSAMPLER_FROZEN": "1"},
            True,
        ),
        # other variables stay out; the URL and token come from .env
        (False, {"PROMPT_UPSAMPLER_URL": "http://u", "HOME": "/h"}, {}, False),
    ],
)
def test_the_run_chooses_how_prompts_are_upsampled(dev_mode, environ, env, frozen):
    assert prompt_upsampler_env_vars(dev_mode, environ) == env
    assert prompts_frozen(dev_mode, environ) is frozen
