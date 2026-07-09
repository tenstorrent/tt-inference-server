# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""run_uvicorn.sh must guarantee HF_HOME is writable before launching the
server. If the launcher-provided HF_HOME (a persistent cache_root path) isn't
writable by this uid — e.g. a bind-mounted cache_root in CI, or an image
without the root-entrypoint permission fixup — it falls back to a per-user
cache so model download/load never crashes at startup."""

import os
import subprocess
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "run_uvicorn.sh"

# A path on the read-only procfs: mkdir fails for every uid (incl. root), so it
# reliably exercises the "not writable" branch regardless of test environment.
UNWRITABLE_HF_HOME = "/proc/tt_hf_nowrite"


def _source_and_report(hf_home, home):
    """Source run_uvicorn.sh with uvicorn stubbed, return the resulting HF_HOME.

    The stub replaces the real server launch with an echo, so sourcing the
    script only runs the startup guards. --skip-venv avoids venv activation.
    """
    stub = (
        f'uvicorn() {{ echo "RESULT_HF_HOME=$HF_HOME"; }}; '
        f'source "{SCRIPT}" --skip-venv'
    )
    env = {**os.environ, "HOME": home}
    env.pop("TT_KV_POOL_GB", None)  # skip the unrelated worker-patch block
    if hf_home is None:
        env.pop("HF_HOME", None)
    else:
        env["HF_HOME"] = hf_home
    result = subprocess.run(
        ["bash", "-c", stub], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, result.stderr
    for line in result.stdout.splitlines():
        if line.startswith("RESULT_HF_HOME="):
            return line[len("RESULT_HF_HOME=") :]
    raise AssertionError(f"stub did not run; output:\n{result.stdout}\n{result.stderr}")


def test_unwritable_hf_home_falls_back_to_user_cache():
    with tempfile.TemporaryDirectory() as home:
        hf_home = _source_and_report(UNWRITABLE_HF_HOME, home)
        assert hf_home == str(Path(home) / ".cache" / "huggingface")
        assert Path(hf_home).is_dir()  # fallback dir was created


def test_writable_hf_home_is_kept():
    with tempfile.TemporaryDirectory() as home:
        target = str(Path(home) / "persistent" / "huggingface")
        hf_home = _source_and_report(target, home)
        assert hf_home == target
        assert not (Path(target) / ".hf_write_test").exists()  # probe cleaned up


def test_unset_hf_home_is_left_unset():
    with tempfile.TemporaryDirectory() as home:
        assert _source_and_report(None, home) == ""
