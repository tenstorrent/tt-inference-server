# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""google/embeddinggemma-2 registration: Settings resolution and catalog parity."""

import contextlib
import os
import re
import sys
import types

import pytest

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VENV_DIR = os.path.join(_PROJECT_DIR, "python_env")
_SETTINGS_CPP = os.path.join(
    _PROJECT_DIR, "cpp_server", "src", "config", "settings.cpp"
)


@contextlib.contextmanager
def _fresh_project_modules():
    """Import project modules afresh, then put ``sys.modules`` back.

    Other test files replace modules such as ``config.settings`` with Mocks in
    ``sys.modules`` and leave them there.
    """
    saved = dict(sys.modules)
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None) or ""
        is_project_module = (
            path.startswith(_PROJECT_DIR + os.sep)
            and not path.startswith(_VENV_DIR + os.sep)
            and not name.startswith("tests")
        )
        if not isinstance(module, types.ModuleType) or is_project_module:
            del sys.modules[name]
    try:
        yield
    finally:
        sys.modules.clear()
        sys.modules.update(saved)


def _settings(monkeypatch, device):
    monkeypatch.setenv("MODEL", "embeddinggemma-2")
    monkeypatch.setenv("DEVICE", device)
    for var in (
        "MODEL_RUNNER",
        "MAX_BATCH_SIZE",
        "TRACE_REGION_SIZE",
        "DEVICE_MESH_SHAPE",
        "DEVICE_IDS",
    ):
        monkeypatch.delenv(var, raising=False)
    with _fresh_project_modules():
        from config.settings import Settings

        return Settings()


def test_p150_resolves_embedding_gemma2_runner(monkeypatch):
    settings = _settings(monkeypatch, "p150")
    assert settings.model_runner == "embedding_gemma2"
    assert settings.model_weights_path == "google/embeddinggemma-2"
    assert settings.device_mesh_shape == (1, 1)
    assert settings.max_batch_size == 32
    assert settings.trace_region_size == 512 * 1024 * 1024
    assert settings.vllm.max_model_length == 8192
    assert settings.vllm.max_num_seqs == 32


def test_embedding_service_owns_the_runner():
    from config.constants import (
        MODEL_SERVICE_RUNNER_MAP,
        ModelRunners,
        ModelServices,
    )

    assert (
        ModelRunners.EMBEDDING_GEMMA2
        in MODEL_SERVICE_RUNNER_MAP[ModelServices.EMBEDDING]
    )


@pytest.mark.parametrize("device", ["n150", "galaxy"])
def test_no_wormhole_config(device):
    from config.constants import DeviceTypes, ModelConfigs, ModelRunners

    assert (ModelRunners.EMBEDDING_GEMMA2, DeviceTypes(device)) not in ModelConfigs


def test_python_config_mirrors_cpp_catalog():
    """constants.py and the C++ embedding catalog must agree (no runtime handshake)."""
    from config.constants import DeviceTypes, ModelConfigs, ModelRunners

    with open(_SETTINGS_CPP) as f:
        src = f.read()
    row = re.search(
        r"\{ModelRunnerType::TT_EMBEDDING_GEMMA2,\s*\"([^\"]+)\",\s*(\d+),\s*(\w+),\s*"
        r"\{\{\"(\w+)\",\s*(\d+),\s*(\d+),\s*\{(\d+),\s*(\d+)\}\}\}\}",
        src,
    )
    assert row, "tt_embedding_gemma2 row not found in settings.cpp"
    hf_id, _ncq, _trace, device, batch, seq, rows, cols = row.groups()

    cfg = ModelConfigs[(ModelRunners.EMBEDDING_GEMMA2, DeviceTypes(device))]
    assert hf_id == cfg["vllm"]["model"]
    assert int(batch) == cfg["max_batch_size"]
    assert int(seq) == cfg["vllm"]["max_model_length"]
    assert (int(rows), int(cols)) == cfg["device_mesh_shape"]
