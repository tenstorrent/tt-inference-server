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


def test_qualified_generated_models_are_enrolled_without_replacing_native_rows():
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
        assert "quetzal" in by_impl

    qwen = {
        row["impl"]: row for row in config["models"]["Qwen3.6-27B"]["implementations"]
    }["quetzal"]
    expected_args = (
        "--quetzal-models-root "
        "/mnt/MLPerf/huggingface/quetzal/nkapre/packages/sha256-f1d6cebaf6cd432c78721ec3b81101ab86493f387b37f63bc11aca2fc6f6d8d8-0a8efa103ee378c7cd0e2fa25b0426cbb82752e270f8927bdf44eb2cfe68ce66"
    )
    for schedule in ("nightly", "release"):
        lane = qwen["ci"][schedule]
        assert lane["devices"] == ["P300X2"]
        args = lane["device-args"]["P300X2"]["additional-args"]
        assert "--impl" not in args
        assert args == expected_args

    gemma = {
        row["impl"]: row
        for row in config["models"]["gemma-4-31B-it"]["implementations"]
    }["quetzal"]
    expected_gemma_args = (
        "--quetzal-models-root "
        "/mnt/models/quetzal/immutable/v1/packages/"
        "sha256-8373c1467294ed11e00ac791392eaa80c9cd1a1366f15200469bbdb4bc410522-"
        "259ee130f4f1e259980f2dde67415f8692f4c16086f34edc1cdb98c496b68edc"
    )
    for schedule in ("nightly", "release"):
        lane = gemma["ci"][schedule]
        assert lane["devices"] == ["P300X2"]
        args = lane["device-args"]["P300X2"]["additional-args"]
        assert "--impl" not in args
        assert args == expected_gemma_args

    gpt = {
        row["impl"]: row for row in config["models"]["gpt-oss-120b"]["implementations"]
    }["quetzal"]
    expected_gpt_args = (
        "--quetzal-models-root "
        "/mnt/models/quetzal/immutable/v1/packages/"
        "sha256-v2-dacc0476febcaf6fb237d1446908e553b16a122b8d6392c933034bd9984c618b-"
        "3086bdd6e0b5aaccaedfe5bdaa514c74409a15da96503395738b3bbee9ed35e2-"
        "2cf6ad2acd9ca99e07ae3fd5dce462dd7ede7695529bfc5894893c82a85a0fc9 "
        "--quetzal-runtime-attestation "
        "/mnt/models/quetzal/immutable/v1/runtime-attestations/"
        "5f12696cdd958028dca60f87cd5fc1ff0e2add41d86129785b253efd5d0ea3db.json "
        "--quetzal-auxiliary-root openai_gpt-oss-120b-streamed-cache="
        "/mnt/models/quetzal/immutable/v1/auxiliary/"
        "openai_gpt-oss-120b-streamed-cache/"
        "sha256-2b2e528a75cae51a53db4a3e309f075553fe5f5f7fec7d2a29480f6572f2e416"
    )
    for schedule in ("nightly", "release"):
        lane = gpt["ci"][schedule]
        assert lane["devices"] == ["P300X2"]
        args = lane["device-args"]["P300X2"]["additional-args"]
        assert "--impl" not in args
        assert args == expected_gpt_args
