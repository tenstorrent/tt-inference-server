"""Observed pytest acquisition is checked without executing benchmark tests."""

import json
from pathlib import Path

import pytest

from llm_module.agentic import offline_prerequisite_probe as probe


def test_exact_observed_verifier_requirement_and_disposable_install(
    tmp_path, monkeypatch
):
    calls = []

    def command(args, **kwargs):
        calls.append((args, kwargs))
        return "pytest 8.4.1\n" if "-c" in args else ""

    monkeypatch.setattr(probe, "command", command)
    result = probe.confirm_verifier_dependency(tmp_path / "cache", tmp_path)
    assert result["benchmark_tests_executed"] is False
    assert calls[0][0][-1] == "pytest==8.4.1"
    assert "--no-index" in calls[0][0] and "--target" in calls[0][0]
    assert calls[1][1]["env"]["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert "pytest.main(['--version'])" in calls[1][0][3]


def test_wrong_version_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "command", lambda *a, **k: "pytest 7.0.0")
    with pytest.raises(AssertionError, match="Unexpected verifier"):
        probe.confirm_verifier_dependency(tmp_path, tmp_path)


def test_cache_covers_py313_linux_pytest_closure():
    lock = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "reference_config/evals/gemma4_terminal_prerequisites.json"
        ).read_text()
    )
    versions = {
        entry["name"].lower(): entry["version"] for entry in lock["dependencies"]
    }
    assert versions["pytest"] == "8.4.1"
    assert {"iniconfig", "packaging", "pluggy", "pygments"} <= versions.keys()
    assert not {"exceptiongroup", "tomli", "typing_extensions"} & versions.keys()
