# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Served model identity: /v1/models must distinguish variants that share weights.

``settings`` is a module-level singleton built at import time from the environment,
so each case runs in a subprocess with its own env rather than mutating a shared
object.
"""

import json
import os
import subprocess
import sys

import pytest

from config.constants import ModelNames, SupportedModels

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_PROBE = """
import json
from config.settings import settings
from open_ai_api.models import list_models

data = list_models()["data"]
print("__PROBE__" + json.dumps({
    "model_id": data[0]["id"] if data else None,
    "model_name": settings.model_name,
    "served_model_name": settings.served_model_name,
    "model_weights_path": settings.model_weights_path,
    "model_service": settings.model_service,
}))
"""


def probe(**env):
    """Resolve settings in a subprocess with `env` applied, and return the result.

    IS_GALAXY=false keeps this off the hardware: it is the only guard in front of
    ``Settings._set_device_pairs_overrides``'s ``DeviceManager()``, which would
    otherwise enumerate devices once per probe. Nothing asserted here depends on
    device layout.
    """
    child_env = {
        **os.environ,
        "IS_GALAXY": "false",
        **{k: str(v) for k, v in env.items()},
    }
    # Inherited values would defeat the point of a per-case environment.
    for leaked in ("SERVED_MODEL_NAME", "MODEL_WEIGHTS_PATH", "MODEL_WEIGHTS_DIR"):
        if leaked not in env:
            child_env.pop(leaked, None)

    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=REPO_ROOT,
        env=child_env,
        capture_output=True,
        text=True,
    )
    marker = "__PROBE__"
    for line in result.stdout.splitlines():
        if line.startswith(marker):
            return json.loads(line[len(marker) :])
    raise AssertionError(
        f"probe did not report settings.\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


# (MODEL_RUNNER, MODEL, DEVICE) for each Wan2.2 variant. All I2V entries share one
# HF repo, and their SupportedModels members are Enum aliases of each other.
WAN_VARIANTS = [
    ("tt-wan2.2", "Wan2.2-T2V-A14B-Diffusers", "galaxy"),
    ("tt-wan2.2-t2v-prodia", "Wan2.2-T2V-A14B-Prodia", "galaxy"),
    ("tt-wan2.2-i2v", "Wan2.2-I2V-A14B-Diffusers", "bh-galaxy"),
    ("tt-wan2.2-i2v-prodia", "Wan2.2-I2V-A14B-Prodia", "bh-galaxy"),
    ("tt-wan2.2-i2v-anisora", "Wan2.2-I2V-AniSora-V3.2", "bh-galaxy"),
    ("tt-wan2.2-i2v-distill", "Wan2.2-I2V-Distill-LightX2V", "bh-galaxy"),
    ("tt-wan2.2-i2v-lora", "Wan2.2-I2V-LoRA", "bh-galaxy"),
    ("tt-wan2.2-i2v-lightning", "Wan2.2-I2V-Lightning", "bh-galaxy"),
]


def test_supported_models_cannot_distinguish_wan_variants():
    """The premise: this is why the weights path can't serve as an identity."""
    i2v_names = [
        "WAN_2_2_I2V",
        "WAN_2_2_I2V_PRODIA",
        "WAN_2_2_I2V_ANISORA",
        "WAN_2_2_I2V_DISTILL",
        "WAN_2_2_I2V_LORA",
        "WAN_2_2_I2V_LIGHTNING",
    ]
    assert len({SupportedModels[n].value for n in i2v_names}) == 1
    # ...while ModelNames, which MODEL carries, is unique per variant.
    assert len({ModelNames[n].value for n in i2v_names}) == len(i2v_names)


@pytest.mark.parametrize("runner,model,device", WAN_VARIANTS)
def test_variant_reports_its_own_name(runner, model, device):
    got = probe(MODEL_RUNNER=runner, MODEL=model, DEVICE=device)
    assert got["model_id"] == model
    assert got["served_model_name"] == model


def test_wan_variants_report_distinct_ids():
    ids = [
        probe(MODEL_RUNNER=runner, MODEL=model, DEVICE=device)["model_id"]
        for runner, model, device in WAN_VARIANTS
    ]
    assert len(set(ids)) == len(WAN_VARIANTS), f"colliding ids: {ids}"


def test_served_model_name_env_overrides_derived_name():
    got = probe(
        MODEL_RUNNER="tt-wan2.2-i2v-lightning",
        MODEL="Wan2.2-I2V-Lightning",
        DEVICE="bh-galaxy",
        SERVED_MODEL_NAME="Wan2.2 I2V Lightning",
    )
    assert got["model_id"] == "Wan2.2 I2V Lightning"
    # An explicit label must not be mistaken for a weights location.
    assert got["model_weights_path"] == "Wan-AI/Wan2.2-I2V-A14B-Diffusers"


def test_display_name_in_model_weights_path_does_not_leak_into_weights():
    """Deployments pass a display name in MODEL_WEIGHTS_PATH; MODEL now supplies the
    id, and the weights path must still resolve to something loadable."""
    got = probe(
        MODEL_RUNNER="tt-wan2.2-i2v-lightning",
        MODEL="Wan2.2-I2V-Lightning",
        DEVICE="bh-galaxy",
        MODEL_WEIGHTS_PATH="Wan2.2 I2V Lightning",
    )
    assert got["served_model_name"] == "Wan2.2-I2V-Lightning"
    assert got["model_weights_path"] == "Wan-AI/Wan2.2-I2V-A14B-Diffusers"


@pytest.mark.parametrize("hash_seed", ["0", "1", "2", "3"])
def test_runner_only_weights_fallback_is_deterministic(hash_seed):
    """SP_RUNNER maps to several models; the pick must not vary with hash order.

    Enum members hash by identity, so iterating a set here previously resolved to
    T2V or I2V weights depending on PYTHONHASHSEED.
    """
    got = probe(MODEL_RUNNER="sp_runner", DEVICE="bh-galaxy", PYTHONHASHSEED=hash_seed)
    # First entry of the ordered runner->names map is the canonical model.
    assert got["model_weights_path"] == "Wan-AI/Wan2.2-T2V-A14B-Diffusers"


def test_model_wins_over_runner_guess_for_multi_model_runner():
    got = probe(
        MODEL_RUNNER="sp_runner", MODEL="Wan2.2-I2V-Lightning", DEVICE="bh-galaxy"
    )
    assert got["served_model_name"] == "Wan2.2-I2V-Lightning"
    assert got["model_weights_path"] == "Wan-AI/Wan2.2-I2V-A14B-Diffusers"


def test_llm_id_is_unchanged():
    got = probe(MODEL_RUNNER="vllm_forge", MODEL="Qwen3-4B", DEVICE="n150")
    assert got["model_service"] == "llm"
    assert got["model_id"] == "Qwen/Qwen3-4B"


def test_image_id_is_still_the_runner_slug():
    got = probe(MODEL_RUNNER="tt-flux.1-dev", MODEL="FLUX.1-dev", DEVICE="galaxy")
    assert got["model_service"] == "image"
    assert got["model_id"] == "tt-flux.1-dev"


# Non-video services are deliberately out of scope: their /v1/models id must be
# byte-identical to before, so clients that echo it back keep working.
NON_VIDEO_SERVICES = [
    ("bge_large_en_v1_5", "bge-large-en-v1.5", "n150", "BAAI/bge-large-en-v1.5"),
    ("tt-whisper", "whisper-large-v3", "n150", "openai/whisper-large-v3"),
    ("tt-speecht5-tts", "speecht5_tts", "n150", "microsoft/speecht5_tts"),
]


@pytest.mark.parametrize("runner,model,device,expected_id", NON_VIDEO_SERVICES)
def test_non_video_services_still_report_the_weights_path(
    runner, model, device, expected_id
):
    got = probe(MODEL_RUNNER=runner, MODEL=model, DEVICE=device)
    assert got["model_id"] == expected_id
    assert got["served_model_name"] == ""


def test_served_model_name_env_works_for_a_non_video_service():
    """The explicit override stays available everywhere, even where we don't scope."""
    got = probe(
        MODEL_RUNNER="bge_large_en_v1_5",
        MODEL="bge-large-en-v1.5",
        DEVICE="n150",
        SERVED_MODEL_NAME="my-embedder",
    )
    assert got["model_id"] == "my-embedder"


def test_weights_dir_mount_does_not_become_the_model_id(tmp_path):
    """A mounted cache is a path, not a name — it must not leak into /v1/models."""
    cache = tmp_path / "hf-cache"
    cache.mkdir()
    (cache / "config.json").write_text("{}")

    got = probe(
        MODEL_RUNNER="tt-wan2.2-i2v-lightning",
        MODEL="Wan2.2-I2V-Lightning",
        DEVICE="bh-galaxy",
        MODEL_WEIGHTS_DIR=str(cache),
    )
    assert got["model_weights_path"] == str(cache)
    assert got["model_id"] == "Wan2.2-I2V-Lightning"
