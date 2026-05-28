# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC

import os
import shutil
import subprocess
from pathlib import Path


def test_tt_console_auth_uses_valid_vllm_api_key_when_openai_key_is_non_tt(tmp_path):
    repo_root = Path(__file__).resolve().parent.parent
    script = repo_root / "evals" / "scripts" / "helper_external_lm_eval.sh"
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path / "home"),
        "BASE_URL": "https://console.tenstorrent.com",
        "OPENAI_API_KEY": "sk-openai-example",
        "VLLM_API_KEY": "sk-tt-valid-example",
        "LM_EVAL_BIN": shutil.which("true") or "/usr/bin/true",
        "OUTPUT_DIR": str(tmp_path / "out"),
    }
    Path(env["HOME"]).mkdir()

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--task",
            "r1_aime24",
            "--chat-api",
            "--print-command",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "No valid TT Console API key found" not in result.stderr
    assert "local-chat-completions" in result.stdout
