# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Deterministic, silicon-free proof that the mini-swe-agent patch-capture shim
recovers an agent's in-container edit when the runner crashes before submit.

Reproduces the exact upstream bug shape (``mini-swe-agent==2.2.8``
``process_instance``: ``exit_status, result = type(e).__name__, ""``) with a
faithful in-memory fake of ``minisweagent.run.benchmarks.swebench`` whose
``process_instance`` looks ``get_sb_environment`` / ``update_preds_file`` up as
module globals at call time -- so ``install()``'s monkeypatch is exercised end to
end. No container, no model, no device; runs in milliseconds.
"""

from __future__ import annotations

import sys
import types

import pytest

from llm_module.agentic import mini_swe_capture as cap


class _FakeEnv:
    """Stands in for a live DockerEnvironment. ``execute`` returns the same dict
    shape the real environment does (``output`` / ``returncode``)."""

    def __init__(self, diff: str, returncode: int = 0, raise_exc: Exception | None = None):
        self._diff = diff
        self._rc = returncode
        self._raise = raise_exc
        self.calls: list[dict] = []

    def execute(self, action, cwd: str = "", *, timeout=None):
        self.calls.append(action)
        if self._raise is not None:
            raise self._raise
        return {"output": self._diff, "returncode": self._rc, "exception_info": ""}


@pytest.fixture(autouse=True)
def _reset_capture_state():
    """Each test starts from a clean, uninstalled shim."""
    cap._INSTALLED = False
    with cap._REGISTRY_LOCK:
        cap._ENV_BY_INSTANCE.clear()
    yield
    cap._INSTALLED = False
    with cap._REGISTRY_LOCK:
        cap._ENV_BY_INSTANCE.clear()


# --- unit level: the pure recovery helper --------------------------------------


def test_recover_patch_returns_diff_on_success():
    env = _FakeEnv(diff="diff --git a/x b/x\n+edit\n")
    assert cap.recover_patch_from_env(env) == "diff --git a/x b/x\n+edit\n"
    assert env.calls == [{"command": "git add -A && git diff --cached"}]


def test_recover_patch_none_env_is_empty():
    assert cap.recover_patch_from_env(None) == ""


def test_recover_patch_nonzero_returncode_is_empty():
    assert cap.recover_patch_from_env(_FakeEnv(diff="junk", returncode=1)) == ""


def test_recover_patch_swallows_execute_exception():
    assert cap.recover_patch_from_env(_FakeEnv(diff="", raise_exc=RuntimeError("boom"))) == ""


# --- integration level: full crash path through a faithful fake runner ---------

# Bodies are exec'd into the fake module's namespace so the bare-name calls to
# ``get_sb_environment`` / ``update_preds_file`` resolve to *its* globals at call
# time -- exactly like the real runner -- which is what makes monkeypatching the
# module attributes take effect.
_FAKE_RUNNER_SRC = '''
def get_sb_environment(config, instance):
    return config["_env"]


def update_preds_file(output_path, instance_id, model_name, result):
    _preds[instance_id] = {
        "model_name_or_path": model_name,
        "instance_id": instance_id,
        "model_patch": result,
    }


def process_instance(instance, config):
    """Mirror the upstream crash path: edit happened in the container, then an
    exception fires before submit, and the handler discards the result."""
    instance_id = instance["instance_id"]
    result = None
    try:
        env = get_sb_environment(config, instance)
        raise RuntimeError(config.get("_crash", "InputTokenBudgetExceeded"))
    except Exception as e:  # noqa: BLE001 -- reproduces upstream behaviour
        exit_status, result = type(e).__name__, ""
    finally:
        update_preds_file("preds.json", instance_id, "model", result)
'''


def _install_fake_runner(monkeypatch):
    pkg_names = [
        "minisweagent",
        "minisweagent.run",
        "minisweagent.run.benchmarks",
    ]
    for name in pkg_names:
        module = types.ModuleType(name)
        module.__path__ = []  # mark as package
        monkeypatch.setitem(sys.modules, name, module)
    runner = types.ModuleType("minisweagent.run.benchmarks.swebench")
    runner._preds = {}
    exec(compile(_FAKE_RUNNER_SRC, "<fake_runner>", "exec"), runner.__dict__)
    monkeypatch.setitem(sys.modules, "minisweagent.run.benchmarks.swebench", runner)
    sys.modules["minisweagent.run.benchmarks"].swebench = runner
    return runner


def test_install_recovers_patch_on_crash(monkeypatch):
    runner = _install_fake_runner(monkeypatch)
    assert cap.install() is True

    env = _FakeEnv(diff="diff --git a/app.py b/app.py\n+real edit\n")
    runner.process_instance({"instance_id": "django__django-11299"}, {"_env": env})

    record = runner._preds["django__django-11299"]
    # The bug would leave this "" (discarded). The shim backfills the live diff.
    assert record["model_patch"] == "diff --git a/app.py b/app.py\n+real edit\n"
    assert record["model_patch"].strip()  # non-empty -> ttis _valid_prediction passes
    # Environment registry is drained so containers are not held alive.
    assert cap._ENV_BY_INSTANCE == {}


def test_install_leaves_genuine_empty_diff_empty(monkeypatch):
    runner = _install_fake_runner(monkeypatch)
    assert cap.install() is True

    env = _FakeEnv(diff="")  # crashed with no edits made
    runner.process_instance({"instance_id": "no__edits-1"}, {"_env": env})

    assert runner._preds["no__edits-1"]["model_patch"] == ""


def test_install_does_not_touch_successful_submission(monkeypatch):
    runner = _install_fake_runner(monkeypatch)
    assert cap.install() is True

    # A run that submits normally calls update_preds_file with a real patch and
    # never enters the crash handler; the shim must pass it through untouched even
    # though it also pops the env registry.
    env = _FakeEnv(diff="should NOT be used")
    cap.register_environment("ok-1", env)
    runner.update_preds_file("preds.json", "ok-1", "model", "submitted-patch")

    assert runner._preds["ok-1"]["model_patch"] == "submitted-patch"
    assert env.calls == []  # recovery not attempted on a non-empty result
    assert cap._ENV_BY_INSTANCE == {}


def test_install_is_idempotent(monkeypatch):
    runner = _install_fake_runner(monkeypatch)
    assert cap.install() is True
    first_get, first_update = runner.get_sb_environment, runner.update_preds_file
    assert cap.install() is True
    # No double-wrapping on a second install.
    assert runner.get_sb_environment is first_get
    assert runner.update_preds_file is first_update


def test_install_returns_false_without_runner(monkeypatch):
    for name in [
        "minisweagent",
        "minisweagent.run",
        "minisweagent.run.benchmarks",
        "minisweagent.run.benchmarks.swebench",
    ]:
        monkeypatch.setitem(sys.modules, name, None)  # force ImportError
    assert cap.install() is False
