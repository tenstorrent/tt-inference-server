# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The tt-lab-glm image's entrypoint: download the weights repo, then run run_glm_tt.py."""

import importlib.util
import subprocess
from pathlib import Path

_SRC = (
    Path(__file__).resolve().parents[1]
    / "vllm-tt-metal"
    / "src"
    / "glm_tt_entrypoint.py"
)
_spec = importlib.util.spec_from_file_location("glm_tt_entrypoint", _SRC)
entrypoint = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(entrypoint)

REPO = "tenstorrent/GLM-5.3-Flash-TTQ"


def _env(tmp_path, **extra):
    env = {
        "CACHE_ROOT": str(tmp_path),
        "MODEL": "GLM-5.3-Flash",
        "GLM_TT_WEIGHTS_REPO": REPO,
        "GLM_TT_PACKAGE": "glm-tt-x",
        "GLM_TT_TTQ": "w.ttq",
        "GLM_TT_VOCAB": "v.gguf",
    }
    env.update(extra)
    return env


def _complete(target):
    (target / "glm-tt-x" / "bin").mkdir(parents=True)
    (target / "glm-tt-x" / "bin" / "glm-tt").write_bytes(b"x")
    (target / "w.ttq.manifest").write_text("part w.ttq.00001 0 1\n")
    (target / "v.gguf").write_bytes(b"x")


class _Run:
    def __init__(self, status):
        self.status = status
        self.commands = []

    def __call__(self, command, env):
        self.commands.append(command)
        return subprocess.CompletedProcess(command, self.status)


def test_downloads_the_repo_into_the_weights_directory(tmp_path):
    run = _Run(0)
    assert entrypoint.fetch_weights(_env(tmp_path), run=run) == 0
    target = tmp_path / "weights" / "GLM-5.3-Flash"
    assert run.commands == [
        ["hf", "download", REPO, "--local-dir", str(target), "--exclude", "original/**"]
    ]
    assert target.is_dir()


def test_offline_serves_complete_weights(tmp_path):
    target = tmp_path / "weights" / "GLM-5.3-Flash"
    _complete(target)
    assert entrypoint.fetch_weights(_env(tmp_path), run=_Run(1)) == 0


def test_offline_fails_on_incomplete_weights(tmp_path):
    target = tmp_path / "weights" / "GLM-5.3-Flash"
    _complete(target)
    staging = target / ".cache" / "huggingface" / "download"
    staging.mkdir(parents=True)
    (staging / "w.ttq.00001.incomplete").write_bytes(b"x")
    assert entrypoint.fetch_weights(_env(tmp_path), run=_Run(1)) == 64
    (staging / "w.ttq.00001.incomplete").unlink()
    (target / "v.gguf").unlink()
    assert entrypoint.fetch_weights(_env(tmp_path), run=_Run(1)) == 64


def test_no_download_with_mounted_weights_or_when_skipped(tmp_path):
    for extra in (
        {"MODEL_WEIGHTS_DIR": "/mnt/hf-cache"},
        {"GLM_TT_SKIP_DOWNLOAD": "1"},
    ):
        run = _Run(1)
        assert entrypoint.fetch_weights(_env(tmp_path, **extra), run=run) == 0
        assert run.commands == []


def test_needs_a_repo(tmp_path):
    env = _env(tmp_path)
    del env["GLM_TT_WEIGHTS_REPO"]
    assert entrypoint.fetch_weights(env, run=_Run(0)) == 64


def test_execs_the_launcher_beside_it():
    assert Path(entrypoint.LAUNCHER) == _SRC.parent / "run_glm_tt.py"
    assert Path(entrypoint.LAUNCHER).is_file()
