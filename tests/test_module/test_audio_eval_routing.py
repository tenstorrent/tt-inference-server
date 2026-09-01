# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Routing guards for the AUDIO eval and benchmark paths.

Both paths used to branch on ``impl_name == "whisper"``. That silently sent
every other ASR model down the fallback branches, where the eval reports a TTFT
in the score column with a hardcoded NA accuracy check, and the benchmark skips
the 60s clip and gates on the lighter 30s pass.
"""

from types import SimpleNamespace

import pytest

from reference_config.evals.eval_config import _eval_config_map
from test_module.benchmark_tests.audio_benchmark_tests import (
    _supports_transcription_sweep,
)
from test_module.eval_tests.audio_eval_tests import _uses_lmms_eval


def _ctx(eval_class, impl_name):
    tasks = [SimpleNamespace(eval_class=eval_class)] if eval_class else []
    return SimpleNamespace(
        all_params=SimpleNamespace(tasks=tasks),
        model_spec=SimpleNamespace(impl=SimpleNamespace(impl_name=impl_name)),
    )


@pytest.mark.parametrize("impl_name", ["whisper", "qwen3-asr"])
def test_whisper_tt_eval_class_routes_to_lmms_eval(impl_name):
    assert _uses_lmms_eval(_ctx("whisper_tt", impl_name))


def test_non_asr_eval_class_does_not_route_to_lmms_eval():
    assert not _uses_lmms_eval(_ctx("local-completions", "speecht5-tts"))


def test_missing_tasks_does_not_route_to_lmms_eval():
    assert not _uses_lmms_eval(_ctx(None, "qwen3-asr"))


@pytest.mark.parametrize("impl_name", ["whisper", "qwen3-asr"])
def test_asr_impls_get_the_30s_60s_sweep(impl_name):
    assert _supports_transcription_sweep(_ctx("whisper_tt", impl_name))


def test_non_asr_impl_does_not_get_the_sweep():
    assert not _supports_transcription_sweep(_ctx("local-completions", "speecht5-tts"))


@pytest.mark.parametrize(
    "hf_model_repo",
    [
        "Qwen/Qwen3-ASR-1.7B",
        "openai/whisper-large-v3",
        "distil-whisper/distil-large-v3",
    ],
)
def test_registered_asr_models_declare_the_lmms_eval_class(hf_model_repo):
    """The routing above is only reachable if the EvalConfig declares it.

    Keyed on the raw registration rather than EVAL_CONFIGS, which is filtered by
    the active catalog and so omits models still only in the dev specs.
    """
    config = _eval_config_map.get(hf_model_repo)
    assert config is not None, f"{hf_model_repo} missing from the eval config list"
    assert config.tasks[0].eval_class == "whisper_tt"
