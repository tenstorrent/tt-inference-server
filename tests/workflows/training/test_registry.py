# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import pytest
import yaml

from workflows.training.loss_check import evaluate, parse_config
from workflows.training.registry import expected_config_path


def test_known_mapping_resolves_to_existing_file():
    path = expected_config_path("Llama-3.1-8B-Instruct", "p150")
    assert path.is_file()


@pytest.mark.parametrize("device", ["p300", "p300x2"])
def test_multichip_blackhole_boards_share_the_p150_expectation(device):
    assert expected_config_path("Llama-3.1-8B-Instruct", device) == (
        expected_config_path("Llama-3.1-8B-Instruct", "p150")
    )


def test_device_is_case_insensitive():
    lower = expected_config_path("Llama-3.1-8B-Instruct", "p150")
    upper = expected_config_path("Llama-3.1-8B-Instruct", "P150")
    assert lower == upper


def test_unknown_mapping_raises():
    with pytest.raises(KeyError):
        expected_config_path("Llama-3.1-8B-Instruct", "n150")


def test_shipped_expectation_parses():
    path = expected_config_path("Llama-3.1-8B-Instruct", "p150")
    cfg = parse_config(yaml.safe_load(path.read_text()))
    # The request block is what the launcher POSTs; it must carry the dataset.
    assert cfg.request["dataset"]["id"] == "sst2"
    assert cfg.request["model"]["lora"]["target_modules"] == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]
    assert cfg.require_decreasing
    assert cfg.final_train_loss_max is not None


def test_shipped_expectation_rejects_rising_loss():
    path = expected_config_path("Llama-3.1-8B-Instruct", "p150")
    cfg = parse_config(yaml.safe_load(path.read_text()))
    metrics = [
        {"global_step": step, "metric_name": "train_loss", "value": value}
        for step, value in ((5, 1.0), (10, 1.1), (15, 1.19))
    ]
    assert evaluate(metrics, cfg, model="m", device="p150").passed is False
