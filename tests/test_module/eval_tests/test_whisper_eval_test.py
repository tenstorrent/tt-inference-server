# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import asyncio
import io
import json
from pathlib import Path
from types import SimpleNamespace

from test_module.eval_tests.audio_eval_tests import _extract_wer
from test_module.eval_tests.whisper_eval_test import WhisperEvalTest


def test_whisper_retry_does_not_reuse_previous_score(tmp_path, monkeypatch):
    monkeypatch.setattr(
        WhisperEvalTest, "_find_lmms_eval_executable", lambda _: "lmms-eval"
    )
    monkeypatch.setenv("OPENAI_API_BASE", "http://localhost:8000")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    test = WhisperEvalTest(
        config={"output_path": str(tmp_path), "timeout": 10}, targets={}
    )
    outputs = []

    def run(cmd, **kwargs):
        output = Path(cmd[cmd.index("--output_path") + 1])
        outputs.append(output)
        result_dir = output / test.hf_model_repo.replace("/", "__")
        result_dir.mkdir(parents=True, exist_ok=True)
        if len(outputs) == 1:
            (result_dir / "old_results.json").write_text(
                json.dumps({"results": {test.task_name: {"wer,none": 5.0}}})
            )
        else:
            (result_dir / "new_results.json").write_text("{")
        return SimpleNamespace(
            stdout=io.StringIO(""),
            stderr=io.StringIO(""),
            returncode=0,
            wait=lambda **kwargs: None,
        )

    monkeypatch.setattr("subprocess.Popen", run)
    first = asyncio.run(test._run_specific_test_async())
    second = asyncio.run(test._run_specific_test_async())

    assert _extract_wer(first, test.task_name) == 5.0
    assert _extract_wer(second, test.task_name) is None
    assert outputs[0] != outputs[1]
    assert list(outputs[0].rglob("old_results.json"))
