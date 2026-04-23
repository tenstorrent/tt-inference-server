# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Unit tests for Settings._set_config_overrides runner resolution.

Covers both code paths:
- Explicit MODEL_RUNNER env var (info log path).
- Unset MODEL_RUNNER falling back to the default runner resolved from
  MODEL_RUNNER_TO_MODEL_NAMES_MAP (warning log path).

Uses SDXL on n150 because (TT_SDXL_TRACE, N150) sets is_galaxy=False, which
short-circuits _set_device_pairs_overrides and keeps the test hermetic (no
DeviceManager / tt-smi calls).
"""

from __future__ import annotations

import pytest

from config.constants import (
    MODEL_RUNNER_TO_MODEL_NAMES_MAP,
    ModelNames,
    ModelRunners,
)

# Env vars consumed by Settings._set_mesh_overrides; cleared per test so parent
# env cannot perturb the resolved device_mesh_shape.
_MESH_ENV_VARS = ("SD_3_5_FAST", "SD_3_5_BASE", "TP2", "SP_MESH_4X32")


def _make_settings(
    monkeypatch: pytest.MonkeyPatch,
    *,
    model: str,
    model_runner: str | None,
    device: str = "n150",
):
    from config.settings import Settings

    monkeypatch.setenv("MODEL", model)
    if model_runner is None:
        monkeypatch.delenv("MODEL_RUNNER", raising=False)
    else:
        monkeypatch.setenv("MODEL_RUNNER", model_runner)
    for env_var in _MESH_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)

    return Settings(device=device)


def test_explicit_model_runner_is_used_when_env_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _make_settings(
        monkeypatch,
        model=ModelNames.STABLE_DIFFUSION_XL_BASE.value,
        model_runner=ModelRunners.TT_SDXL_TRACE.value,
    )

    assert settings.model_runner == ModelRunners.TT_SDXL_TRACE.value
    # matching_config was applied (proves the full path ran, not just the env read).
    assert settings.is_galaxy is False
    assert settings.device_mesh_shape == (1, 1)


def test_default_model_runner_resolved_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_name = ModelNames.STABLE_DIFFUSION_XL_BASE
    # Sanity-check the precondition the test relies on: SDXL_BASE must map to
    # TT_SDXL_TRACE as the first matching runner in insertion order.
    first_matching_runner = next(
        runner
        for runner, names in MODEL_RUNNER_TO_MODEL_NAMES_MAP.items()
        if model_name in names
    )
    assert first_matching_runner == ModelRunners.TT_SDXL_TRACE

    settings = _make_settings(
        monkeypatch,
        model=model_name.value,
        model_runner=None,
    )

    assert settings.model_runner == ModelRunners.TT_SDXL_TRACE.value
    assert settings.is_galaxy is False
    assert settings.device_mesh_shape == (1, 1)
