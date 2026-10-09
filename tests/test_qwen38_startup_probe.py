# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import json

import pytest

from scripts.release.probe_qwen38_startup import argument_values, verify_handoff


@pytest.fixture
def launch():
    return (
        {
            "device_model_spec": {
                "vllm_args": {
                    "model": "/weights",
                    "data_parallel_size": 8,
                    "additional_config": {"tt": {"groups": ["0,1,2,3"]}},
                    "async-scheduling": True,
                    "enable-prefix-caching": False,
                },
                "env_vars": {"QWEN_PRECISION_CONFIG": "old.json"},
            },
            "env_vars": {"QWEN_PRECISION_CONFIG": "qualified.json"},
        },
        [
            "wrapper.py",
            "--model=/weights",
            "--data-parallel-size",
            "8",
            "--additional_config",
            json.dumps({"tt": {"groups": ["0,1,2,3"]}}),
            "--async-scheduling",
        ],
        {"QWEN_PRECISION_CONFIG": "qualified.json"},
    )


def test_equivalent_argument_spellings_and_environment_precedence(launch):
    report = verify_handoff(*launch)
    assert report["arguments"]["data-parallel-size"] == "8"
    assert report["declared_environment"]["QWEN_PRECISION_CONFIG"] == "qualified.json"


@pytest.mark.parametrize(
    "extra",
    [
        ["--seed", "123"],
        ["--enable-prefix-caching"],
        ["--data_parallel_size", "4"],
        ["unexpected-position"],
    ],
)
def test_changed_or_duplicate_launch_rejected(launch, extra):
    spec, argv, environment = launch
    with pytest.raises(ValueError):
        verify_handoff(spec, argv + extra, environment)


def test_changed_physical_partition_rejected(launch):
    spec, argv, environment = launch
    argv[5] = json.dumps({"tt": {"groups": ["4,5,6,7"]}})
    with pytest.raises(ValueError, match="argument differs.*additional-config"):
        verify_handoff(spec, argv, environment)


def test_missing_required_flag_rejected(launch):
    spec, argv, environment = launch
    with pytest.raises(ValueError, match="async-scheduling"):
        verify_handoff(spec, argv[:-1], environment)


@pytest.mark.parametrize(
    "change",
    [
        {"QWEN_PRECISION_CONFIG": "wrong.json"},
        {"TT_METAL_OPERATION_TIMEOUT_SECONDS": "5"},
    ],
)
def test_changed_runtime_environment_rejected(launch, change):
    spec, argv, environment = launch
    with pytest.raises(ValueError):
        verify_handoff(spec, argv, {**environment, **change})


def test_alias_duplicate_detected_before_value_comparison():
    with pytest.raises(ValueError, match="Duplicate startup argument"):
        argument_values(["wrapper", "--max_num_seqs=16", "--max-num-seqs", "32"])
