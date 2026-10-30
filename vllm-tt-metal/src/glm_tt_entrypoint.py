#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Entrypoint of the tt-lab-glm image: fetch the weights repo, then run run_glm_tt.py.

run.py's host setup downloads the weights repo (the TTQ, its MTP sidecar, the vocabulary
and the glm-tt package) into the persistent volume before the container starts. Under the
Helm chart nothing does, so this does it in the container, into the same place:
$CACHE_ROOT/weights/$MODEL, the chart's cache hostPath. ``hf download`` resumes a partial
download and skips files already present, so a warm start only checks the repo; when
Hugging Face cannot be reached but the files run_glm_tt.py needs are present, it serves
from them. Then it replaces itself with run_glm_tt.py, passing its arguments on.

Environment:
  GLM_TT_WEIGHTS_REPO  the Hugging Face repo to download (the image sets the spec's
                       hf_weights_repo)
  MODEL_WEIGHTS_DIR    set (the chart's hfCacheDir): the weights are pre-downloaded there,
                       read-only; no download
  GLM_TT_SKIP_DOWNLOAD 1: no download (the weights are already in the volume)
  CACHE_ROOT, MODEL, HF_TOKEN, and everything run_glm_tt.py reads.
"""

import os
import subprocess
import sys

LAUNCHER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_glm_tt.py")
SETUP_FAILED = 64


def log(message):
    print(f"glm_tt_entrypoint: {message}", file=sys.stderr, flush=True)


def _truthy(value) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes")


def weights_target(env):
    """Where to download the weights repo, or None when no download is wanted."""
    if env.get("MODEL_WEIGHTS_DIR") or _truthy(env.get("GLM_TT_SKIP_DOWNLOAD")):
        return None
    if not (env.get("CACHE_ROOT") and env.get("MODEL")):
        return None
    return os.path.join(env["CACHE_ROOT"], "weights", env["MODEL"])


def weights_present(target, env) -> bool:
    """True when the files run_glm_tt.py needs are in target, with no download in flight."""
    ttq = env.get("GLM_TT_TTQ", "")
    package = env.get("GLM_TT_PACKAGE") or "glm-tt"
    needed = [
        os.path.join(target, package, "bin", "glm-tt"),
        os.path.join(target, env.get("GLM_TT_VOCAB", "")),
    ]
    if not ttq:
        return False
    ttq_path = os.path.join(target, ttq)
    if not (os.path.isfile(ttq_path) or os.path.isfile(f"{ttq_path}.manifest")):
        return False
    if any(not os.path.isfile(path) for path in needed):
        return False
    for root, _dirs, files in os.walk(target):
        if any(name.endswith(".incomplete") for name in files):
            return False
    return True


def download_command(repo, target):
    return ["hf", "download", repo, "--local-dir", target, "--exclude", "original/**"]


def fetch_weights(env, run=subprocess.run) -> int:
    """0 when the weights are in place (downloaded, or present while offline), else 64."""
    target = weights_target(env)
    if target is None:
        log("no download: MODEL_WEIGHTS_DIR or GLM_TT_SKIP_DOWNLOAD is set")
        return 0
    repo = env.get("GLM_TT_WEIGHTS_REPO")
    if not repo:
        log("GLM_TT_WEIGHTS_REPO is not set: no repo to download the weights from")
        return SETUP_FAILED
    os.makedirs(target, exist_ok=True)
    command = download_command(repo, target)
    log(f"{' '.join(command)}")
    status = run(command, env=env).returncode
    if status == 0:
        return 0
    if weights_present(target, env):
        log(f"hf download exited {status}; serving the weights already in {target}")
        return 0
    log(f"hf download exited {status} and {target} is incomplete")
    return SETUP_FAILED


def main(argv=None, environ=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    env = dict(os.environ if environ is None else environ)
    env.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    status = fetch_weights(env)
    if status:
        return status
    os.execve(sys.executable, [sys.executable, LAUNCHER, *argv], env)
    return SETUP_FAILED  # not reached


if __name__ == "__main__":
    sys.exit(main())
