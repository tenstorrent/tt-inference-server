# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

from scripts.release.promote_dev_spec_to_prod import (
    ReleaseCombo,
    collect_release_combos,
    iter_implementations,
    model_name_from_weight,
    template_identity,
    template_matches,
)
from workflows.workflow_types import DeviceTypes, InferenceEngine


def test_model_name_from_weight_strips_org_prefix():
    assert model_name_from_weight("meta-llama/Llama-3.1-8B-Instruct") == (
        "Llama-3.1-8B-Instruct"
    )
    assert model_name_from_weight("openai/gpt-oss-20b") == "gpt-oss-20b"


def test_iter_implementations_flat_shape():
    entry = {"inference_engine": "FORGE", "ci": {"nightly": {"devices": ["P150"]}}}
    assert list(iter_implementations(entry)) == [entry]


def test_iter_implementations_array_shape():
    impl_a = {"inference_engine": "vLLM", "ci": {}}
    impl_b = {"inference_engine": "FORGE", "ci": {}}
    entry = {"implementations": [impl_a, impl_b]}
    assert list(iter_implementations(entry)) == [impl_a, impl_b]


def test_collect_release_combos_array_and_flat_shapes():
    ci_config = {
        "models": {
            "Llama-3.1-8B-Instruct": {
                "implementations": [
                    {
                        "inference_engine": "vLLM",
                        "ci": {
                            "nightly": {"devices": ["N150"]},
                            "release": {"devices": ["GALAXY", "P300X2"]},
                        },
                    },
                    {
                        "inference_engine": "FORGE",
                        "ci": {"nightly": {"devices": ["P150"]}},
                    },
                ]
            },
            "whisper-large-v3": {
                "inference_engine": "MEDIA",
                "ci": {"release": {"devices": ["P150"]}},
            },
            "Falcon3-7B-Instruct": {
                "inference_engine": "FORGE",
                "ci": {"nightly": {"devices": ["P150"]}},
            },
        }
    }
    combos = collect_release_combos(ci_config)
    assert combos == {
        ReleaseCombo("Llama-3.1-8B-Instruct", InferenceEngine.VLLM, DeviceTypes.GALAXY),
        ReleaseCombo("Llama-3.1-8B-Instruct", InferenceEngine.VLLM, DeviceTypes.P300X2),
        ReleaseCombo("whisper-large-v3", InferenceEngine.MEDIA, DeviceTypes.P150),
    }


def test_collect_release_combos_ignores_nightly_and_weekly():
    ci_config = {
        "models": {
            "m": {
                "inference_engine": "vLLM",
                "ci": {
                    "nightly": {"devices": ["GALAXY"]},
                    "weekly": {"devices": ["GALAXY"]},
                },
            }
        }
    }
    assert collect_release_combos(ci_config) == set()


def _llama_template():
    return {
        "weights": ["meta-llama/Llama-3.1-8B-Instruct"],
        "impl": "tt_transformers",
        "inference_engine": "VLLM",
        "device_model_specs": [
            {"device": "GALAXY", "max_concurrency": 32},
            {"device": "N150", "max_concurrency": 1},
            {"device": "P300X2", "max_concurrency": 8},
        ],
    }


def test_template_matches_on_basename_engine_and_device():
    combo = ReleaseCombo(
        "Llama-3.1-8B-Instruct", InferenceEngine.VLLM, DeviceTypes.GALAXY
    )
    assert template_matches(_llama_template(), combo) is True


def test_template_does_not_match_wrong_device():
    combo = ReleaseCombo("Llama-3.1-8B-Instruct", InferenceEngine.VLLM, DeviceTypes.T3K)
    assert template_matches(_llama_template(), combo) is False


def test_template_does_not_match_wrong_engine():
    combo = ReleaseCombo(
        "Llama-3.1-8B-Instruct", InferenceEngine.FORGE, DeviceTypes.GALAXY
    )
    assert template_matches(_llama_template(), combo) is False


def test_template_identity_is_impl_engine_weights():
    assert template_identity(_llama_template()) == (
        "tt_transformers",
        InferenceEngine.VLLM,
        frozenset({"meta-llama/Llama-3.1-8B-Instruct"}),
    )
