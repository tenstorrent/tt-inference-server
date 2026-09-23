# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for the --requirements-json CLI wiring at both entry points.

``run_workflows.py`` drives the engine standalone; ``run.py`` additionally
brings up the server and dispatches into it. A document handed to either has
to produce the same model, device, and validation content, so both are
covered here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import run_workflows
from workflows.requirements_cli import (
    DEFAULT_REQUIREMENTS_DEVICE,
    register_requirements_providers,
)

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
    assert args.device == DEFAULT_REQUIREMENTS_DEVICE
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


def test_unknown_accuracy_eval_fails_at_parse_time(monkeypatch, tmp_path):
    """A mistyped eval name is a CLI usage error, not a mid-run crash."""
    doc = {
        "schemaVersion": "2.1.0",
        "id": "bad-eval",
        "model": {"name": "acme/tiny-llm", "contextLength": 4096},
        "deployment": {"maxConcurrencyPerInstance": 8},
        "accuracyEvals": [{"name": "GPQA Diamnd"}],  # typo
        "scenarios": [],
    }
    path = tmp_path / "bad-eval.json"
    path.write_text(json.dumps(doc))
    monkeypatch.setattr(
        sys,
        "argv",
        _argv("--workflow", "evals", "--requirements-json", str(path)),
    )
    with pytest.raises(SystemExit):
        run_workflows.parse_args()


# The document's model is deliberately off-catalog: the point of a
# requirements-driven run is validating a model the repo was never onboarded
# with, which is exactly what every catalog gate below would otherwise refuse.
_DOC_MODEL = "openai/gpt-oss-120b"


@pytest.fixture
def restore_seams():
    """Undo the process-wide provider/pack registration a requirements run does."""
    from workflow_module import model_catalog, target_pack

    provider, pack = model_catalog._provider, target_pack._target_pack
    yield
    model_catalog._provider, target_pack._target_pack = provider, pack


def _run_py_args(monkeypatch, *extra: str):
    import run

    monkeypatch.setattr(
        sys,
        "argv",
        ["run.py", "--requirements-json", str(_FIXTURE), *extra],
    )
    args = run.parse_arguments()
    register_requirements_providers(args.requirements_doc)
    return args


def test_run_py_defaults_model_and_device_from_document(monkeypatch, restore_seams):
    args = _run_py_args(monkeypatch, "--workflow", "agentic")
    assert args.model == _DOC_MODEL
    assert args.device == "super_cluster"  # SC8 -> SUPER_CLUSTER
    # --requirements-json is pinned absolute so children resolve it from any cwd.
    assert Path(args.requirements_json).is_absolute()


def test_run_py_without_requirements_still_gates_unknown_model(monkeypatch):
    import run

    monkeypatch.setattr(
        sys,
        "argv",
        ["run.py", "--workflow", "agentic", "--model", "acme/not-in-catalog"],
    )
    with pytest.raises(SystemExit):
        run.parse_arguments()


def test_run_py_synthesizes_off_catalog_spec(monkeypatch, restore_seams):
    import run

    args = _run_py_args(monkeypatch, "--workflow", "agentic")
    runtime_config, model_spec = run.resolve_runtime(args)
    assert model_spec.model_name == _DOC_MODEL
    assert model_spec.hf_model_repo == _DOC_MODEL
    assert runtime_config.requirements_json == args.requirements_json


@pytest.mark.parametrize("workflow", ["agentic", "evals", "release", "benchmarks"])
def test_dispatch_forwards_requirements_to_children(
    monkeypatch, restore_seams, tmp_path, workflow
):
    import run
    from workflows.workflow_dispatch import build_engine_commands

    args = _run_py_args(monkeypatch, "--workflow", workflow)
    runtime_config, model_spec = run.resolve_runtime(args)
    commands = build_engine_commands(
        model_spec, runtime_config, tmp_path / "runtime_model_spec.json"
    )
    argvs = [c.argv for c in commands if hasattr(c, "argv")]
    assert argvs
    for argv in argvs:
        assert "--requirements-json" in argv
        assert argv[argv.index("--requirements-json") + 1] == args.requirements_json


def test_release_provisions_venvs_for_the_documents_evals(monkeypatch, restore_seams):
    """The document's evals decide the venvs, not the (absent) catalog entry.

    The fixture asks for GPQA-Diamond plus two agentic harnesses, so a release
    run has to build both the standard and the agentic eval venvs even though
    the catalog has no eval config for this model at all.
    """
    import run
    from workflows.workflow_dispatch import _engine_dependency_venv_types
    from workflows.workflow_types import WorkflowType, WorkflowVenvType

    args = _run_py_args(monkeypatch, "--workflow", "release")
    runtime_config, model_spec = run.resolve_runtime(args)
    venv_types = _engine_dependency_venv_types(
        model_spec, WorkflowType.RELEASE, runtime_config
    )
    assert WorkflowVenvType.EVALS_COMMON in venv_types
    assert WorkflowVenvType.EVALS_AGENTIC in venv_types


def test_validation_accepts_the_off_catalog_requirements_model(
    monkeypatch, restore_seams
):
    """Neither catalog gate may reject a run the document fully describes."""
    import run
    from workflows.validate_setup import validate_runtime_args

    args = _run_py_args(monkeypatch, "--workflow", "release")
    runtime_config, model_spec = run.resolve_runtime(args)
    validate_runtime_args(model_spec, runtime_config)


# --- the llm-gauntlet misfile guard ----------------------------------------
#
# CI synthesizes the document path from the model under test
# (specs/<customer>/<org>/<model> or specs/<customer>/<model>), so a document
# filed under the wrong folder would gate one model's endpoint on another
# model's criteria.


def _doc_named(name: str):
    from types import SimpleNamespace

    return SimpleNamespace(model=SimpleNamespace(name=name))


@pytest.mark.parametrize(
    "folder, model_name",
    [
        ("Qwen/Qwen3-32B", "Qwen/Qwen3-32B"),
        ("moonshotai/Kimi-K2.7-Code", "moonshotai/Kimi-K2.7-Code"),
        # The bare model name, without the org folder, also matches.
        ("Qwen3-32B", "Qwen/Qwen3-32B"),
        # Case drift in the folder is not a misfile.
        ("qwen/qwen3-32b", "Qwen/Qwen3-32B"),
        ("qwen3-32b", "Qwen/Qwen3-32B"),
        # A document model with no org prefix still matches.
        ("gpt-oss-120b", "gpt-oss-120b"),
    ],
)
def test_folder_matching_the_document_model_is_not_misfiled(folder, model_name):
    from workflows.requirements_cli import misfiled_gauntlet_document

    path = f"/clone/specs/tt-internal/{folder}/x.json"

    assert misfiled_gauntlet_document(path, _doc_named(model_name)) is None


def test_folder_disagreeing_with_the_document_model_is_reported():
    from workflows.requirements_cli import misfiled_gauntlet_document

    path = "/clone/specs/tt-internal/Qwen/Qwen3-32B/x.json"

    message = misfiled_gauntlet_document(path, _doc_named("moonshotai/Kimi-K2.7-Code"))

    # Both halves must be named: the message is the only signal an operator gets.
    assert message is not None
    assert "moonshotai/Kimi-K2.7-Code" in message
    assert "Qwen/Qwen3-32B" in message


def test_a_plain_path_is_never_checked_for_misfiling(monkeypatch):
    """The folder convention belongs to llm-gauntlet, not to every path.

    The fixture lives in a `requirements/` directory and names gpt-oss-120b, so
    it would fail the folder check -- loading it must not apply one.
    """
    monkeypatch.setattr(
        sys,
        "argv",
        _argv("--workflow", "benchmarks", "--requirements-json", str(_FIXTURE)),
    )

    assert run_workflows.parse_args().requirements_doc is not None
