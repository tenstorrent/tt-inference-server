import json
from pathlib import Path

from scripts.validate_models_ci_config import validate_implementation_identities


REPO_ROOT = Path(__file__).resolve().parents[1]


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


def test_quetzal_nightly_rows_preserve_native_vllm_rows():
    config = json.loads(
        (REPO_ROOT / ".github/workflows/models-ci-config.json").read_text()
    )
    assert validate_implementation_identities(config) == []

    expected_native = {
        "Qwen3.6-27B": "qwen36-blackhole",
        "gemma-4-31B-it": "tt-transformers",
        "gpt-oss-120b": "gpt-oss",
    }
    expected_revision = {
        "Qwen3.6-27B": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
        "gemma-4-31B-it": "842da3794eaa0b77d5f08bae87a17459d91ff475",
        "gpt-oss-120b": "b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
    }
    quetzal_commit = "49f103ad8f80523ba0d35c5825aee908507f196b"
    for model, native_impl in expected_native.items():
        rows = config["models"][model]["implementations"]
        by_impl = {row["impl"]: row for row in rows}
        assert native_impl in by_impl
        assert "quetzal" in by_impl
        qz_nightly = by_impl["quetzal"]["ci"]["nightly"]
        assert qz_nightly["devices"] == ["P300X2"]
        args = qz_nightly["device-args"]["P300X2"]["additional-args"]
        # tt-shield forwards matrix.config.impl exactly once; duplicating it in
        # additional-args would create two selectors at the run.py boundary.
        assert "--impl" not in args
        assert "--quetzal-models-root" in args
        assert quetzal_commit in args
        assert expected_revision[model] in args
