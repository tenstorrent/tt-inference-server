# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for the --requirements-json CLI wiring in run_workflows.parse_args."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import run_workflows

_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "requirements"
    / "acme-llm-serving.json"
)


def _argv(*args: str):
    return ["run_workflows.py", *args]


def test_requirements_mode_defaults_model_and_device(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        _argv("--workflow", "benchmarks", "--requirements-json", str(_FIXTURE)),
    )
    args = run_workflows.parse_args()
    assert args.requirements_doc is not None
    assert args.model == "openai/gpt-oss-120b"
    assert args.device == "super_cluster"  # SC8 -> SUPER_CLUSTER


def test_requirements_mode_allows_off_catalog_model_override(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(
            "--workflow",
            "benchmarks",
            "--requirements-json",
            str(_FIXTURE),
            "--model",
            "acme/not-in-catalog",
            "--device",
            "super_cluster",
        ),
    )
    # No catalog gate in requirements mode: an off-catalog model is accepted.
    args = run_workflows.parse_args()
    assert args.model == "acme/not-in-catalog"


def test_requirements_mode_defaults_device_when_no_hardware(monkeypatch, tmp_path):
    # A document with no deployment.hardware still runs: the device falls back
    # to the requirements default (the endpoint is chosen by --server-url, not
    # the device).
    doc = {
        "schemaVersion": "2.1.0",
        "id": "no-hw",
        "model": {"name": "acme/tiny-llm", "contextLength": 4096},
        "deployment": {"maxConcurrencyPerInstance": 8},
        "accuracyEvals": [],
        "scenarios": [],
    }
    path = tmp_path / "no-hw.json"
    path.write_text(json.dumps(doc))
    monkeypatch.setattr(
        sys,
        "argv",
        _argv("--workflow", "benchmarks", "--requirements-json", str(path)),
    )
    args = run_workflows.parse_args()
    assert args.device == run_workflows._DEFAULT_REQUIREMENTS_DEVICE
    assert args.model == "acme/tiny-llm"


def test_non_requirements_mode_still_gates_unknown_model(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(
            "--workflow",
            "benchmarks",
            "--model",
            "acme/not-in-catalog",
            "--device",
            "super_cluster",
        ),
    )
    with pytest.raises(SystemExit):
        run_workflows.parse_args()
