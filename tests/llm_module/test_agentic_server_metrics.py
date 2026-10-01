# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from llm_module.agentic.server_metrics import metrics_url, select_vllm_metrics


def test_metrics_url_replaces_openai_path():
    assert metrics_url("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000/metrics"


def test_select_vllm_metrics_keeps_histograms_and_occupancy():
    payload = """
# HELP ignored help
vllm:num_requests_running{model_name="qwen"} 5
vllm:time_to_first_token_seconds_bucket{le="1.0",model_name="qwen"} 3
vllm:time_to_first_token_seconds_sum{model_name="qwen"} 1.25
vllm:request_generation_tokens_sum{model_name="qwen"} 16384
vllm:prefix_cache_hits{model_name="qwen"} 0
python_gc_objects_collected_total{generation="0"} 99
malformed
"""

    selected = select_vllm_metrics(payload)

    assert selected == {
        'vllm:num_requests_running{model_name="qwen"}': 5.0,
        'vllm:prefix_cache_hits{model_name="qwen"}': 0.0,
        'vllm:request_generation_tokens_sum{model_name="qwen"}': 16384.0,
        'vllm:time_to_first_token_seconds_bucket{le="1.0",model_name="qwen"}': 3.0,
        'vllm:time_to_first_token_seconds_sum{model_name="qwen"}': 1.25,
    }
