# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""MiMo-V2.6-Flash multimodal serving and evals: the mimo_v2_6_flash_mm spec, the
EVALS_MULTIMODAL / EVALS_OPENBENCH tasks and their commands, the lmms-eval launcher,
and openbench_eval.py's output as the eval workflow reads it."""

import json
import sys
import types
from types import SimpleNamespace

import pytest

from llm_module import openbench_eval
from llm_module.eval_command import build_eval_command
from reference_config.evals.eval_config import _eval_config_map
from test_module.llm_tests import llm_eval_tests as llm_eval
from workflows.model_spec import load_templates_from_yaml
from workflows.utils import get_repo_root_path
from workflows.workflow_types import ModelType, WorkflowVenvType
from workflows.workflow_venvs import REQUIREMENTS_DIR, _venv_config_list

REPO = "XiaomiMiMo/MiMo-V2.6-Flash-MOPD"
DEV_LLM = get_repo_root_path() / "workflows" / "model_specs" / "dev" / "llm.yaml"
MEDIA_TASKS = {
    "mmmu_pro_standard_cot": "image",
    "ocrbench": "image",
    "videomme_short": "video",
    "mmau_test_mini": "audio",
    "librispeech_test_clean": "audio",
}


def _mimo_templates():
    return {
        t.impl.impl_id: t
        for t in load_templates_from_yaml(DEV_LLM)
        if t.impl.impl_id.startswith("mimo_v2_6_flash")
    }


def test_mm_spec_serves_the_multimodal_class_and_leaves_text_entries_alone():
    templates = _mimo_templates()
    (spec,) = templates["mimo_v2_6_flash_mm"].expand_to_specs()
    dms = spec.device_model_spec
    assert spec.model_type == ModelType.VLM
    assert set(spec.supported_modalities) == {"text", "image", "audio", "video"}
    assert not dms.default_impl and dms.submesh_engines
    assert dms.env_vars["TT_MODEL_CLASS_OVERRIDES"] == (
        "TTMiMoV2OmniForCausalLM="
        "models.experimental.mimo_v2_6_flash.tt.generator_vllm:MiMoV2OmniForCausalLM"
    )
    assert dms.env_vars["TT_PROXY_FETCH_MEDIA"] == 1
    args = dms.vllm_args
    assert json.loads(args["limit-mm-per-prompt"]) == {
        "image": 8,
        "video": 1,
        "audio": 4,
    }
    assert json.loads(args["media-io-kwargs"]) == {"video": {"num_frames": 32}}
    assert "hf_overrides" not in args and "hf-overrides" not in args
    assert args["max_num_batched_tokens"] == 4096 and args["data_parallel_size"] == 4
    # The text entries: no multimodal class, no media downloads, MTP the default.
    for impl in ("mimo_v2_6_flash", "mimo_v2_6_flash_mtp"):
        (text,) = templates[impl].expand_to_specs()
        assert "TT_MODEL_CLASS_OVERRIDES" not in text.device_model_spec.env_vars
        assert "TT_PROXY_FETCH_MEDIA" not in text.device_model_spec.env_vars
        assert text.supported_modalities == ["text"]
    (mtp,) = templates["mimo_v2_6_flash_mtp"].expand_to_specs()
    assert mtp.device_model_spec.default_impl


def test_the_serving_image_installs_torchaudio():
    dockerfile = (
        get_repo_root_path() / "vllm-tt-metal" / "vllm.tt-metal.src.dev.Dockerfile"
    )
    assert "torchaudio==" in dockerfile.read_text()


def test_new_venvs_are_provisioned_from_their_requirements():
    configs = {c.venv_type: c for c in _venv_config_list}
    for venv_type, req in (
        (WorkflowVenvType.EVALS_MULTIMODAL, "evals-multimodal.txt"),
        (WorkflowVenvType.EVALS_OPENBENCH, "evals-openbench.txt"),
    ):
        assert configs[venv_type].requirements_file == req
        assert (REQUIREMENTS_DIR / req).exists()
    text = (REQUIREMENTS_DIR / "evals-openbench.txt").read_text()
    assert "openbench==0.5.3" in text and "mcp<2" in text
    assert configs[WorkflowVenvType.EVALS_OPENBENCH].setup_function is not None
    assert "lmms-eval==0.7.4" in (REQUIREMENTS_DIR / "evals-multimodal.txt").read_text()


def _tasks():
    return {t.task_name: t for t in _eval_config_map[REPO].tasks}


def test_mimo_eval_tasks():
    tasks = _tasks()
    for name, modality in MEDIA_TASKS.items():
        task = tasks[name]
        assert task.workflow_venv_type == WorkflowVenvType.EVALS_MULTIMODAL
        assert task.eval_class == "mimo_openrouter" and task.modalities == [modality]
        assert task.thinking is False and task.priority == "should"
    for name in ("gpqa_diamond_openrouter", "tau2_airline"):
        assert tasks[name].workflow_venv_type == WorkflowVenvType.EVALS_OPENBENCH
    assert tasks["gpqa_diamond_openrouter"].model_kwargs["epochs"] == "10"
    # Scores already in percent are not scaled twice.
    results = {"videomme_short": {"videomme_perception_score,none": 62.5}}
    score = tasks["videomme_short"].score
    assert score.score_func(results, "videomme_short", score.score_func_kwargs) == 62.5


def _spec():
    return SimpleNamespace(
        model_id="id_mimo",
        model_name="mimo",
        hf_model_repo=REPO,
        device_model_spec=SimpleNamespace(
            max_context=1048576, max_concurrency=16, eval_max_retries=None
        ),
    )


def _arg(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def test_multimodal_command(tmp_path):
    task = _tasks()["mmau_test_mini"]
    smoke = SimpleNamespace(limit_samples_mode="smoke-test", eval_samples=None)
    cmd = build_eval_command(
        task, _spec(), "galaxy", tmp_path, 8000, runtime_config=smoke
    )
    assert cmd[0].endswith(".venv_evals_multimodal/bin/python")
    assert cmd[1].endswith("llm_module/lmms_eval_multimodal.py")
    assert _arg(cmd, "--model") == "mimo_openrouter"
    args = dict(kv.split("=", 1) for kv in _arg(cmd, "--model_args").split(","))
    assert args["base_url"] == "http://127.0.0.1:8000/v1"
    assert args["thinking"] == "off" and args["target"] == "vllm"
    assert args["model_version"] == REPO and args["num_concurrent"] == "16"
    assert args["journal"] == str(
        tmp_path / "eval_id_mimo" / "mmau_test_mini_journal.jsonl"
    )
    assert _arg(cmd, "--include_path").endswith("lmms_plugins/mimo_lmms/tasks")
    assert _arg(cmd, "--output_path") == str(tmp_path / "eval_id_mimo")
    assert "--limit" in cmd and "--gen_kwargs" not in cmd  # stream=False is not sent


def test_openbench_commands(tmp_path):
    tasks = _tasks()
    cmd = build_eval_command(
        tasks["gpqa_diamond_openrouter"], _spec(), "g", tmp_path, 8000
    )
    assert cmd[1].endswith("llm_module/openbench_eval.py")
    assert _arg(cmd, "--task") == "gpqa_diamond_openrouter"
    assert _arg(cmd, "--epochs") == "10" and _arg(cmd, "--concurrency") == "16"
    assert _arg(cmd, "--base-url") == "http://127.0.0.1:8000/v1"
    cmd = build_eval_command(tasks["tau2_airline"], _spec(), "g", tmp_path, 8000)
    assert _arg(cmd, "--user-model") == "openrouter/openai/gpt-4.1"
    assert "--limit" not in cmd


def test_lmms_launcher_loads_the_plugin(monkeypatch):
    seen = {}
    fake_main = types.ModuleType("lmms_eval.__main__")

    def cli_evaluate():
        import mimo_lmms.models

        seen["argv"] = list(sys.argv)
        seen["plugins"] = __import__("os").environ["LMMS_EVAL_PLUGINS"]
        seen["models"] = mimo_lmms.models.AVAILABLE_MODELS

    fake_main.cli_evaluate = cli_evaluate
    monkeypatch.setitem(sys.modules, "lmms_eval", types.ModuleType("lmms_eval"))
    monkeypatch.setitem(sys.modules, "lmms_eval.__main__", fake_main)
    monkeypatch.setenv("LMMS_EVAL_PLUGINS", "other")
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "argv", ["launcher", "--tasks", "ocrbench"])
    from llm_module import lmms_eval_multimodal

    assert lmms_eval_multimodal.main() == 0
    assert seen["argv"] == ["lmms-eval", "--tasks", "ocrbench"]
    assert seen["plugins"] == "other,mimo_lmms"
    assert seen["models"] == {"mimo_openrouter": "OpenRouterChat"}
    tasks = lmms_eval_multimodal.PLUGIN_TASKS / "videomme_short" / "videomme_short.yaml"
    assert tasks.exists()


def test_openbench_output_is_read_and_graded_by_the_eval_workflow(tmp_path):
    assert openbench_eval.resolve_limit(None, 198) is None
    assert openbench_eval.resolve_limit("0.01", 198) == 2
    assert openbench_eval.resolve_limit("5", 198) == 5
    error = SimpleNamespace(message="ConnectionError: refused")
    samples = [
        SimpleNamespace(
            id=i,
            epoch=1,
            target="B",
            error=error if i == 2 else None,
            output=SimpleNamespace(completion="Answer: B"),
            scores={"mcq_scorer": SimpleNamespace(value="C")},
        )
        for i in range(3)
    ]
    accuracy = SimpleNamespace(value=0.5)
    log = SimpleNamespace(
        results=SimpleNamespace(
            scores=[SimpleNamespace(metrics={"accuracy": accuracy, "stderr": accuracy})]
        ),
        samples=samples,
    )
    metrics, rows, n = openbench_eval.summarize(log)
    assert metrics == {"accuracy": 0.5} and n == 3
    out = tmp_path / "eval_id_mimo" / REPO.replace("/", "__")
    path = openbench_eval.write_outputs(
        out, "gpqa_diamond_openrouter", metrics, rows, n, 198, {"epochs": 10}
    )
    assert path is not None
    files = llm_eval.discover_eval_results(
        tmp_path, SimpleNamespace(model_id="id_mimo", hf_model_repo=REPO)
    )
    results, counts = llm_eval.load_eval_results(files)
    assert results["gpqa_diamond_openrouter"] == {"accuracy,none": 0.5}
    assert counts["gpqa_diamond_openrouter"] == 3
    assert llm_eval.count_inference_errors(files) == (1, 3)  # the errored sample
    task = _tasks()["gpqa_diamond_openrouter"]
    kwargs = task.score.score_func_kwargs
    assert task.score.score_func(results, "gpqa_diamond_openrouter", kwargs) == 50.0
    # An unfinished run leaves its samples but no score.
    assert (
        openbench_eval.write_outputs(out, "tau2_airline", {}, rows, n, 50, {}) is None
    )


@pytest.mark.parametrize("name", sorted(MEDIA_TASKS))
def test_media_tasks_skip_text_only_profiles(name):
    from llm_module.eval_configs import _task_modalities

    assert _task_modalities(_tasks()[name]) == {MEDIA_TASKS[name]}
