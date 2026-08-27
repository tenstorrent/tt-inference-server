# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import pytest
import yaml

from workflows.training.loss_check import parse_config
from workflows.training.registry import expected_config_path


def test_known_mapping_resolves_to_existing_file():
    path = expected_config_path("meta-llama/Llama-3.1-8B", "p150")
    assert path.is_file()


def test_device_is_case_insensitive():
    lower = expected_config_path("meta-llama/Llama-3.1-8B", "p150")
    upper = expected_config_path("meta-llama/Llama-3.1-8B", "P150")
    assert lower == upper


def test_unknown_mapping_raises():
    with pytest.raises(KeyError):
        expected_config_path("meta-llama/Llama-3.1-8B", "n150")


def test_shipped_expectation_parses():
    path = expected_config_path("meta-llama/Llama-3.1-8B", "p150")
    cfg = parse_config(yaml.safe_load(path.read_text()))
    # The request block is what the launcher POSTs; it must carry the dataset.
    assert cfg.request.get("dataset_loader") == "SST2"
    assert cfg.expectations, "expected_losses must not be empty"
