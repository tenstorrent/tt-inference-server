# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import json
from pathlib import Path

import jsonschema
import pytest

from scripts.validate_models_ci_config import validate_implementation_identities


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads(
    (REPO_ROOT / ".github/workflows/models-ci-config-schema.json").read_text()
)
SCHEDULE = {"nightly": {"devices": ["P300X2"]}}


def _config(model_entry):
    return {"models": {"model": model_entry}}


@pytest.mark.parametrize(
    "model_entry",
    [
        {"inference_engine": "vLLM", "impl": "quetzal", "ci": SCHEDULE},
        {
            "implementations": [
                {
                    "inference_engine": "vLLM",
                    "impl": "quetzal",
                    "ci": SCHEDULE,
                }
            ]
        },
    ],
)
def test_schema_accepts_optional_impl_for_flat_and_implementation_rows(model_entry):
    jsonschema.validate(_config(model_entry), SCHEMA)


@pytest.mark.parametrize(
    "model_entry",
    [
        {"inference_engine": "vLLM", "impl": "", "ci": SCHEDULE},
        {"implementations": [{"inference_engine": "vLLM", "impl": "", "ci": SCHEDULE}]},
    ],
)
def test_schema_rejects_empty_impl(model_entry):
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(_config(model_entry), SCHEMA)


def test_semantics_allow_unqualified_distinct_engines():
    config = _config(
        {
            "implementations": [
                {"inference_engine": "vLLM", "ci": SCHEDULE},
                {"inference_engine": "FORGE", "ci": SCHEDULE},
            ]
        }
    )

    assert validate_implementation_identities(config) == []


def test_semantics_allow_distinct_impls_for_same_engine():
    config = _config(
        {
            "implementations": [
                {
                    "inference_engine": "vLLM",
                    "impl": "tt-transformers",
                    "ci": SCHEDULE,
                },
                {
                    "inference_engine": "vLLM",
                    "impl": "quetzal",
                    "ci": SCHEDULE,
                },
            ]
        }
    )

    assert validate_implementation_identities(config) == []


def test_semantics_reject_ambiguous_unqualified_same_engine_row():
    config = _config(
        {
            "implementations": [
                {"inference_engine": "vLLM", "ci": SCHEDULE},
                {
                    "inference_engine": "vLLM",
                    "impl": "quetzal",
                    "ci": SCHEDULE,
                },
            ]
        }
    )

    errors = validate_implementation_identities(config)

    assert len(errors) == 1
    assert "impl is required" in errors[0]


def test_semantics_reject_duplicate_same_engine_impl_identity():
    row = {"inference_engine": "vLLM", "impl": "quetzal", "ci": SCHEDULE}
    config = _config({"implementations": [row, row]})

    errors = validate_implementation_identities(config)

    assert len(errors) == 1
    assert "duplicate Models CI identity" in errors[0]


def test_repository_config_has_unambiguous_implementation_identities():
    config = json.loads(
        (REPO_ROOT / ".github/workflows/models-ci-config.json").read_text()
    )

    assert validate_implementation_identities(config) == []
