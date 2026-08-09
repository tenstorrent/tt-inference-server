# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

import pytest

from llm_module.parsers.vllm import VLLMBenchParser


def _parse_block_metrics(
    *,
    completed,
    total_output_tokens,
    request_throughput,
    mean_e2el_ms,
    output_lens=None,
):
    return (
        VLLMBenchParser()
        .parse(
            {
                "completed": completed,
                "total_input_tokens": 128 * completed,
                "total_output_tokens": total_output_tokens,
                "request_throughput": request_throughput,
                "mean_e2el_ms": mean_e2el_ms,
                # vLLM divides inter-event time by token count. For block-emitting
                # models this is near zero when one event contains 128 token IDs.
                "mean_tpot_ms": 0.00025,
                "output_throughput": total_output_tokens / 100.0,
                "tt_output_block_size": 256,
                **({"output_lens": output_lens} if output_lens is not None else {}),
            },
            device="P300X2",
        )
        .data
    )


def test_partial_output_block_uses_request_latency_not_token_tpot():
    data = _parse_block_metrics(
        completed=8,
        total_output_tokens=1024,
        request_throughput=0.0461580898,
        mean_e2el_ms=21664.4408,
    )

    assert data["output_blocks_per_request"] == 1
    assert data["output_blocks_per_second"] == pytest.approx(0.0462)
    assert data["mean_block_latency_ms"] == pytest.approx(21664.4408)


def test_multi_block_latency_amortizes_request_e2el_over_emitted_blocks():
    data = _parse_block_metrics(
        completed=4,
        total_output_tokens=4096,
        request_throughput=0.0129386199,
        mean_e2el_ms=77287.7441,
    )

    assert data["output_blocks_per_request"] == 4
    assert data["output_blocks_per_second"] == pytest.approx(0.0518)
    assert data["mean_block_latency_ms"] == pytest.approx(19321.936)


def test_block_count_uses_each_actual_emitted_output_length():
    data = _parse_block_metrics(
        completed=3,
        total_output_tokens=514,
        request_throughput=0.3,
        mean_e2el_ms=4000.0,
        output_lens=[1, 256, 257],
    )

    # ceil(1/256) + ceil(256/256) + ceil(257/256) = 4 actual
    # scheduling blocks across three requests.
    assert data["output_blocks_per_request"] == pytest.approx(4 / 3, abs=1e-4)
    assert data["output_blocks_per_second"] == pytest.approx(0.4)
    assert data["mean_block_latency_ms"] == pytest.approx(3000.0)
