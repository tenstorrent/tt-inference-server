# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import json

import pytest

from workflows.model_spec import load_templates_from_yaml, resolve_model_spec
from workflows.runtime_config import RuntimeConfig
from workflows.utils import get_repo_root_path

MODEL = "google/gemma-4-31B-it"
IMPL = "gemma4-31b-autoport-v3"
REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
ADAPTER = "models.autoports.google_gemma_4_31b_it.tt.generator_vllm:Gemma4ForCausalLM"


def _spec():
    templates = load_templates_from_yaml(
        get_repo_root_path() / "workflows/model_specs/dev/llm.yaml"
    )
    specs = [s for t in templates for s in t.expand_to_specs()]
    return resolve_model_spec(
        specs, model=MODEL, device="p300x2", impl=IMPL, catalog_name="dev"
    )


def test_autoport_resolves_to_pinned_text_only_adapter():
    spec = _spec()
    device = spec.device_model_spec
    assert spec.impl.code_path == "models/autoports/google_gemma_4_31b_it"
    assert spec.supported_modalities == ["text"]
    assert spec.docker_image is None  # Supplied by the source-build dispatch.
    assert device.max_context == 262144
    assert device.max_concurrency == 8
    assert device.vllm_args["max_num_seqs"] == "8"
    assert device.vllm_args["revision"] == REVISION
    assert device.vllm_args["tokenizer_revision"] == REVISION
    assert device.vllm_args["block_size"] == "32"
    assert device.vllm_args["max_num_batched_tokens"] == "2048"
    assert json.loads(device.vllm_args["override_generation_config"]) == {"top_k": 32}
    assert device.env_vars["TT_MODEL_CLASS_OVERRIDES"] == (
        f"TTGemma4ForConditionalGeneration={ADAPTER},TTGemma4ForCausalLM={ADAPTER}"
    )
    assert device.vllm_args["enable_prefix_caching"] is True
    assert device.vllm_args["enable_chunked_prefill"] is True
    assert device.override_tt_config == {
        "sample_on_device_mode": "all",
        "trace_region_size": 134217728,
        "fabric_config": "FABRIC_1D_RING",
    }


@pytest.mark.parametrize("batch", [1, 8, 16, 32])
def test_supported_batch_is_selectable_by_stock_runtime_override(batch):
    spec = _spec()
    assert batch in spec.metadata["validated_decode_batches"]
    spec.apply_overrides(
        RuntimeConfig(
            model=MODEL,
            workflow="agentic",
            device="p300x2",
            vllm_override_args=json.dumps({"max_num_seqs": batch}),
        )
    )
    assert int(spec.device_model_spec.vllm_args["max_num_seqs"]) == batch
    assert spec.device_model_spec.vllm_args["max_model_len"] == "262144"


def test_autoport_release_benchmark_matrix_is_isolated(monkeypatch):
    from llm_module.benchmark_configs import get_llm_configs

    # Existing stock switch selects reference rows without adding generic sweeps.
    monkeypatch.setenv("ONLY_BENCHMARK_TARGETS", "1")
    spec = _spec()
    rows = get_llm_configs(spec, spec.device_type)
    assert [
        (row.isl, row.osl, row.max_concurrency, row.num_prompts) for row in rows
    ] == [(128, 128, 1, 8)]
    assert not rows[0].targets


def test_autoport_native_thinking_and_tool_protocol_flags():
    args = _spec().device_model_spec.vllm_args
    assert args["reasoning_parser"] == "gemma4"
    assert args["tool_call_parser"] == "gemma4"
    assert args["enable_auto_tool_choice"] is True
