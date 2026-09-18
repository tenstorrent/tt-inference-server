# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# Chart-side tests: these exercise the chart's _helpers.tpl resolver
# (engine + impl lookup, error messaging, hfCacheDir wiring) by invoking
# `helm template` against the live values.yaml.
#
# TODO: investigate replacing these with the `helm unittest` plugin
# (https://github.com/helm-unittest/helm-unittest)

import shutil
import subprocess
import pytest

from workflows.utils import get_repo_root_path

CHART = get_repo_root_path() / "charts" / "tt-inference-server"
FIXTURES = get_repo_root_path() / "tests" / "test_chart" / "fixtures"

pytestmark = pytest.mark.skipif(
    shutil.which("helm") is None, reason="helm CLI not available"
)


def _render(*set_args, values_files=()):
    cmd = [
        "helm",
        "template",
        str(CHART),
        "--set",
        "hfToken=fake",
        # media/forge rows fail closed without an auth choice.
        "--set",
        "auth.apiKey=fake",
    ]
    for f in values_files:
        cmd.extend(["-f", str(FIXTURES / f)])
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
    # No catalogue model has one device under two engines, so overlay a
    # synthetic fixture whose t3k is served by both vllm and media.
    r = _render(
        "model=resolver-multi-engine-fixture",
        "device=t3k",
        values_files=["resolver_multi_engine.yaml"],
    )
    assert r.returncode == 0, r.stderr
    assert "vllm-tt-metal-src" in r.stdout


def test_explicit_engine_override_picks_media():
    r = _render(
        "model=resolver-multi-engine-fixture",
        "device=t3k",
        "engine=media",
        values_files=["resolver_multi_engine.yaml"],
    )
    assert r.returncode == 0, r.stderr
    assert "tt-media-inference-server" in r.stdout


def test_explicit_impl_override_picks_named_impl():
    r = _render(
        "model=Qwen3-32B",
        "device=galaxy",
        "impl=qwen3_32b_galaxy",
    )
    assert r.returncode == 0, r.stderr
    assert "0.21.0-bc4c4df-be7d805" in r.stdout


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
    r = _render("model=Qwen3-32B", "device=galaxy", "impl=qwen3_32b_galaxy")
    assert r.returncode == 0, r.stderr
    assert "/opt/cache/Qwen3-32B-galaxy-qwen3_32b_galaxy" in r.stdout
