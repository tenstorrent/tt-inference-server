# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from types import SimpleNamespace
from unittest.mock import Mock
import subprocess

import pytest

from workflows import workflow_venvs as mod

PIN = "e" * 40


@pytest.fixture
def setup_calls(monkeypatch, tmp_path):
    checkout = Mock(return_value=True)
    resolve = Mock(return_value=SimpleNamespace(stdout=PIN + "\n"))
    install = Mock(return_value=0)
    monkeypatch.setattr(mod, "checkout_pinned_repo", checkout)
    monkeypatch.setattr(mod.subprocess, "run", resolve)
    monkeypatch.setattr(mod, "run_command", install)
    monkeypatch.setattr(mod, "_write_harbor_adapters_pth", Mock(return_value=True))
    config = SimpleNamespace(venv_path=tmp_path, venv_python=tmp_path / "bin/python")
    return config, checkout, resolve, install


@pytest.mark.parametrize(
    "metadata", [{}, {"harbor_git_ref": PIN}, {"harbor_git_ref": PIN.upper()}]
)
def test_setup_selects_exact_ref_and_logs_resolved_commit(
    setup_calls, metadata, monkeypatch
):
    config, checkout, resolve, install = setup_calls
    log_info = Mock()
    monkeypatch.setattr(mod.logger, "info", log_info)
    assert mod.setup_evals_agentic(config, SimpleNamespace(metadata=metadata))
    checkout.assert_called_once_with(
        config.venv_path / "harbor",
        mod.HARBOR_REPO,
        PIN if metadata else mod.HARBOR_REF,
    )
    assert resolve.call_args.args[0] == [
        "git",
        "-C",
        str(config.venv_path / "harbor"),
        "rev-parse",
        "HEAD",
    ]
    log_info.assert_any_call(
        "Harbor source: requested=%s resolved=%s",
        PIN if metadata else mod.HARBOR_REF,
        PIN,
    )
    install.assert_called_once()


@pytest.mark.parametrize(
    "ref",
    [
        None,
        "",
        "main",
        "tt-inference-server",
        "a" * 39,
        "a" * 41,
        "g" * 40,
        " " + PIN,
        123,
        [PIN],
    ],
)
def test_mutable_or_invalid_ref_is_rejected_before_checkout(setup_calls, ref):
    config, checkout, resolve, install = setup_calls
    with pytest.raises(ValueError, match="full 40-character commit SHA"):
        mod.setup_evals_agentic(
            config, SimpleNamespace(metadata={"harbor_git_ref": ref})
        )
    checkout.assert_not_called()
    resolve.assert_not_called()
    install.assert_not_called()


@pytest.mark.parametrize("actual", ["f" * 40, "", "main"])
def test_checkout_mismatch_never_installs(setup_calls, actual):
    config, _, resolve, install = setup_calls
    resolve.return_value.stdout = actual
    assert not mod.setup_evals_agentic(
        config, SimpleNamespace(metadata={"harbor_git_ref": PIN})
    )
    install.assert_not_called()


def test_checkout_failure_never_resolves_or_installs(setup_calls):
    config, checkout, resolve, install = setup_calls
    checkout.return_value = False
    assert not mod.setup_evals_agentic(
        config, SimpleNamespace(metadata={"harbor_git_ref": PIN})
    )
    resolve.assert_not_called()
    install.assert_not_called()


@pytest.mark.parametrize(
    "error",
    [
        OSError("git unavailable"),
        subprocess.TimeoutExpired("git", 30),
        subprocess.CalledProcessError(1, "git"),
    ],
)
def test_verification_failure_never_installs(setup_calls, error):
    config, _, resolve, install = setup_calls
    resolve.side_effect = error
    assert not mod.setup_evals_agentic(
        config, SimpleNamespace(metadata={"harbor_git_ref": PIN})
    )
    install.assert_not_called()
