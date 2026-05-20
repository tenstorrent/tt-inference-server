# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# Smoke tests that exercise the chart's _helpers.tpl resolution by invoking
# `helm template` against the live values.yaml. Skipped if helm isn't on PATH.

import shutil
import subprocess
from pathlib import Path

import pytest

CHART = Path(__file__).resolve().parents[2] / "charts" / "tt-inference-server"

pytestmark = pytest.mark.skipif(
    shutil.which("helm") is None, reason="helm CLI not available"
)


def _render(*set_args):
    cmd = [
        "helm",
        "template",
        str(CHART),
        "--set",
        "hfToken=fake",
    ]
    for s in set_args:
        cmd.extend(["--set", s])
    return subprocess.run(cmd, capture_output=True, text=True)


def test_single_engine_resolves_without_flags():
    r = _render("model=Llama-3.1-8B-Instruct", "device=galaxy")
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "vllm-tt-metal-src" in out
    assert '"--model"' in out and '"--tt-device"' in out


def test_default_engine_picks_vllm_when_device_under_multiple_engines():
    r = _render("model=Llama-3.1-70B", "device=t3k")
    assert r.returncode == 0, r.stderr
    assert "vllm-tt-metal-src" in r.stdout


def test_explicit_engine_override_picks_media():
    r = _render("model=Llama-3.1-70B", "device=t3k", "engine=media")
    assert r.returncode == 0, r.stderr
    assert "tt-media-inference-server" in r.stdout


def test_explicit_impl_override_picks_non_default():
    r = _render(
        "model=Qwen3-32B",
        "device=galaxy",
        "impl=tt_transformers",
    )
    assert r.returncode == 0, r.stderr
    assert "e95ffa5-48eba14" in r.stdout, "expected tt_transformers' image tag"


def test_unknown_impl_fails_with_clear_error():
    r = _render(
        "model=Llama-3.1-8B-Instruct",
        "device=galaxy",
        "impl=does_not_exist",
    )
    assert r.returncode != 0
    assert "No impl 'does_not_exist'" in r.stderr


def test_unknown_device_fails_with_clear_error():
    r = _render("model=Llama-3.1-8B-Instruct", "device=mars")
    assert r.returncode != 0
    assert "has no engine that provides device 'mars'" in r.stderr


def test_unknown_model_fails_with_clear_error():
    r = _render("model=DoesNotExist", "device=galaxy")
    assert r.returncode != 0
    assert "Unknown model 'DoesNotExist'" in r.stderr


def test_env_list_renders_into_container_env():
    r = _render("model=Llama-3.1-8B-Instruct", "device=galaxy")
    assert r.returncode == 0, r.stderr
    assert "VLLM_CONFIGURE_LOGGING" in r.stdout
    assert "MESH_DEVICE" in r.stdout


def test_hfcachedir_sets_weights_env_and_mount():
    r = _render(
        "model=Llama-3.1-8B-Instruct",
        "device=galaxy",
        "hfCacheDir=/data/weights",
    )
    assert r.returncode == 0, r.stderr
    assert "MODEL_WEIGHTS_DIR" in r.stdout
    assert "MODEL_WEIGHTS_PATH" in r.stdout
    assert "/mnt/hf-cache" in r.stdout


def test_cache_hostpath_includes_impl():
    r = _render("model=Qwen3-32B", "device=galaxy", "impl=tt_transformers")
    assert r.returncode == 0, r.stderr
    assert "/opt/cache/Qwen3-32B-galaxy-tt_transformers" in r.stdout
