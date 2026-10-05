# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from workflows.workflow_venvs import (
    EVALS_COMMON_LM_EVAL_COMMIT,
    REQUIREMENTS_DIR,
    verify_evals_common_lm_eval,
)


def test_evals_common_requirement_uses_the_reviewed_full_commit():
    requirements = (REQUIREMENTS_DIR / "evals-common.txt").read_text()

    assert (
        f"lm-evaluation-harness.git@{EVALS_COMMON_LM_EVAL_COMMIT}#egg=" in requirements
    )
    assert "lm-evaluation-harness.git@evals-common#egg=" not in requirements


def test_evals_common_setup_verifies_and_logs_installed_commit():
    venv = SimpleNamespace(venv_python=Path("/venv/bin/python"))

    with patch("workflows.workflow_venvs.run_command", return_value=0) as run:
        assert verify_evals_common_lm_eval(venv, model_spec=SimpleNamespace())

    command = run.call_args.args[0]
    assert command[:2] == ["/venv/bin/python", "-c"]
    assert EVALS_COMMON_LM_EVAL_COMMIT in command[2]
    assert "direct_url.json" in command[2]


def test_evals_common_setup_refuses_unverified_install():
    venv = SimpleNamespace(venv_python=Path("/venv/bin/python"))

    with patch("workflows.workflow_venvs.run_command", return_value=1):
        assert not verify_evals_common_lm_eval(venv, model_spec=SimpleNamespace())


def test_evals_common_rejects_corrupt_punkt_archive_before_installing():
    import json

    import pytest

    venv = SimpleNamespace(venv_python=Path("/venv/bin/python"))
    with patch("workflows.workflow_venvs.run_command", return_value=0) as run:
        assert verify_evals_common_lm_eval(venv, model_spec=SimpleNamespace())
    probe = run.call_args.args[0][2]
    distribution = SimpleNamespace(
        metadata={"Name": "lm-eval"},
        read_text=lambda _: json.dumps(
            {"vcs_info": {"commit_id": EVALS_COMMON_LM_EVAL_COMMIT}}
        ),
    )
    with patch("importlib.metadata.distribution", return_value=distribution):
        with patch("subprocess.check_output", return_value=b"corrupt archive"):
            with patch("pathlib.Path.mkdir") as mkdir:
                with pytest.raises(SystemExit, match="checksum mismatch"):
                    exec(probe)
    mkdir.assert_not_called()
