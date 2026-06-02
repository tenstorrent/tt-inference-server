# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import textwrap

from ruamel.yaml.comments import CommentedSeq

from scripts.release.promote_dev_spec_to_prod import (
    ReleaseCombo,
    collect_release_combos,
    find_matches,
    iter_implementations,
    model_name_from_weight,
    template_identity,
    template_matches,
    upsert_template,
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


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text))


def test_find_matches_picks_whole_template_and_reports_unmatched(tmp_path):
    dev = tmp_path / "dev"
    _write(
        dev / "llm.yaml",
        """
        templates:
        - weights:
            - meta-llama/Llama-3.1-8B-Instruct
          impl: tt_transformers
          inference_engine: VLLM
          device_model_specs:
            - device: GALAXY
              max_concurrency: 32
            - device: N150
              max_concurrency: 1
        """,
    )
    combos = {
        ReleaseCombo("Llama-3.1-8B-Instruct", InferenceEngine.VLLM, DeviceTypes.GALAXY),
        ReleaseCombo("nonexistent", InferenceEngine.VLLM, DeviceTypes.GALAXY),
    }
    matches_by_file, unmatched = find_matches(dev, combos)

    assert list(matches_by_file.keys()) == ["llm.yaml"]
    picked = matches_by_file["llm.yaml"]
    assert len(picked) == 1
    # whole template: the non-release N150 device is still present
    devices = [d["device"] for d in picked[0]["device_model_specs"]]
    assert devices == ["GALAXY", "N150"]
    assert unmatched == {
        ReleaseCombo("nonexistent", InferenceEngine.VLLM, DeviceTypes.GALAXY)
    }


def test_find_matches_dedups_template_matched_by_two_combos(tmp_path):
    dev = tmp_path / "dev"
    _write(
        dev / "llm.yaml",
        """
        templates:
        - weights:
            - meta-llama/Llama-3.1-8B-Instruct
          impl: tt_transformers
          inference_engine: VLLM
          device_model_specs:
            - device: GALAXY
            - device: P300X2
        """,
    )
    combos = {
        ReleaseCombo("Llama-3.1-8B-Instruct", InferenceEngine.VLLM, DeviceTypes.GALAXY),
        ReleaseCombo("Llama-3.1-8B-Instruct", InferenceEngine.VLLM, DeviceTypes.P300X2),
    }
    matches_by_file, unmatched = find_matches(dev, combos)
    assert len(matches_by_file["llm.yaml"]) == 1
    assert unmatched == set()


def test_upsert_appends_new_template():
    prod = CommentedSeq()
    tmpl = {
        "weights": ["meta-llama/Llama-3.1-8B-Instruct"],
        "impl": "tt_transformers",
        "inference_engine": "VLLM",
        "device_model_specs": [{"device": "GALAXY"}],
    }
    action = upsert_template(prod, tmpl)
    assert action == "appended"
    assert len(prod) == 1


def test_upsert_replaces_same_identity_and_leaves_others():
    other = {
        "weights": ["openai/gpt-oss-20b"],
        "impl": "gpt_oss",
        "inference_engine": "VLLM",
        "device_model_specs": [{"device": "T3K"}],
    }
    old = {
        "weights": ["meta-llama/Llama-3.1-8B-Instruct"],
        "impl": "tt_transformers",
        "inference_engine": "VLLM",
        "version": "0.1.0",
        "device_model_specs": [{"device": "GALAXY"}],
    }
    prod = CommentedSeq()
    prod.append(other)
    prod.append(old)

    new = dict(old)
    new["version"] = "0.2.0"
    action = upsert_template(prod, new)

    assert action == "updated"
    assert len(prod) == 2
    assert prod[0] is other  # untouched
    assert prod[1]["version"] == "0.2.0"
