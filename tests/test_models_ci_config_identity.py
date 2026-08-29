# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import json
from pathlib import Path

from scripts.validate_models_ci_config import validate_implementation_identities

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_implementation_image_must_be_an_immutable_oci_digest():
    base = {
        "models": {
            "m": {
                "implementations": [
                    {
                        "inference_engine": "vLLM",
                        "impl": "quetzal",
                        "image": "ghcr.io/tenstorrent/ttis-quetzal@sha256:" + "a" * 64,
                        "ci": {"nightly": {"devices": ["P300X2"]}},
                    }
                ]
            }
        }
    }
    assert validate_implementation_identities(base) == []

    mutable = json.loads(json.dumps(base))
    mutable["models"]["m"]["implementations"][0]["image"] = (
        "ghcr.io/tenstorrent/ttis-quetzal:latest"
    )
    assert (
        "image must be an immutable" in validate_implementation_identities(mutable)[0]
    )

    adversarial = json.loads(json.dumps(base))
    adversarial["models"]["m"]["implementations"][0]["image"] = (
        "!/" * 10_000 + "!:@sha256:" + "a" * 64
    )
    assert (
        "image must be an immutable"
        in validate_implementation_identities(adversarial)[0]
    )


def test_duplicate_engine_requires_impl_and_unique_identity():
    missing = {
        "models": {
            "m": {
                "implementations": [
                    {"inference_engine": "vLLM", "ci": {}},
                    {"inference_engine": "vLLM", "impl": "quetzal", "ci": {}},
                ]
            }
        }
    }
    assert "impl is required" in validate_implementation_identities(missing)[0]

    duplicate = {
        "models": {
            "m": {
                "implementations": [
                    {"inference_engine": "vLLM", "impl": "quetzal", "ci": {}},
                    {"inference_engine": "vLLM", "impl": "quetzal", "ci": {}},
                ]
            }
        }
    }
    assert (
        "duplicate Models CI identity"
        in validate_implementation_identities(duplicate)[0]
    )


def test_only_qualified_qwen_is_enrolled_for_quetzal_nightly_and_release():
    config = json.loads(
        (REPO_ROOT / ".github/workflows/models-ci-config.json").read_text()
    )
    assert validate_implementation_identities(config) == []

    expected_native = {
        "Qwen3.6-27B": "qwen36-blackhole",
        "gemma-4-31B-it": "tt-transformers",
        "gpt-oss-120b": "gpt-oss",
    }
    for model, native_impl in expected_native.items():
        rows = config["models"][model]["implementations"]
        by_impl = {row["impl"]: row for row in rows}
        assert native_impl in by_impl
        if model == "Qwen3.6-27B":
            assert "quetzal" in by_impl
        else:
            assert "quetzal" not in by_impl

    qwen = {
        row["impl"]: row for row in config["models"]["Qwen3.6-27B"]["implementations"]
    }["quetzal"]
    expected_args = (
        "--quetzal-models-root "
        "/mnt/models/quetzal/immutable/v1/packages/sha256-f1d6cebaf6cd432c78721ec3b81101ab86493f387b37f63bc11aca2fc6f6d8d8-0a8efa103ee378c7cd0e2fa25b0426cbb82752e270f8927bdf44eb2cfe68ce66"
    )
    for schedule in ("nightly", "release"):
        lane = qwen["ci"][schedule]
        assert lane["devices"] == ["P300X2"]
        args = lane["device-args"]["P300X2"]["additional-args"]
        assert "--impl" not in args
        assert args == expected_args
