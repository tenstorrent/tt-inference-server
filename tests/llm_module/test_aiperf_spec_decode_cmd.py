# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The spec-decode aiperf command sends a temperature only when the run sets
one; otherwise requests get the server's default, as the GPU reference sweeps
did."""

from pathlib import Path

from llm_module.drivers import aiperf_spec_decode as driver_mod
from llm_module.spec_decode import SpecDecodeRun


def _extra_inputs(run):
    cmd = driver_mod._build_aiperf_cmd(
        run=run,
        venv_python=Path("/venv/bin/python"),
        model_name="org/model",
        tokenizer="org/model",
        url="127.0.0.1:8000",
        artifact_dir="/tmp/artifacts",
        auth_token="",
    )
    return [cmd[i + 1] for i, arg in enumerate(cmd) if arg == "--extra-inputs"]


def _payload(run):
    return driver_mod._build_payload(
        run=run, metrics={}, spec_decode_metrics=None, model_repo="org/model"
    )


def test_no_temperature_by_default():
    run = SpecDecodeRun(public_dataset="speed_bench_coding", max_concurrency=1)
    assert not any(x.startswith("temperature") for x in _extra_inputs(run))
    assert _payload(run)["temperature"] is None


def test_a_set_temperature_is_sent_and_recorded():
    run = SpecDecodeRun(
        public_dataset="speed_bench_throughput_32k",
        max_concurrency=16,
        max_completion_tokens=8192,
        temperature=0.0,
    )
    assert "temperature:0" in _extra_inputs(run)
    assert _payload(run)["temperature"] == 0.0
