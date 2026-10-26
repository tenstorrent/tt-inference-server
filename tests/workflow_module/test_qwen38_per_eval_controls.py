"""Bounded release/combined selectors through actual command and Harbor builders."""
import copy
import hashlib
import json
import random
from pathlib import Path
from unittest.mock import patch

import pytest

from llm_module import DriverContext, ServerConnection
from llm_module.agentic.harbor import _write_harbor_config
from llm_module.drivers.agentic import build_harbor_config, resolve_n_tasks
from llm_module.eval_command import build_eval_command
from reference_config.evals.eval_config import EVAL_CONFIGS, EvalLimitMode
from reference_config.evals.qwen38_goal12 import MODEL, configure, controls
from tests.workflow_module.test_qwen38_full_suite_path import runtime, spec

ROOT = Path("reference_config/evals")


def template(label, *, ready=True):
    c = json.loads((ROOT / f"qwen38_goal12_{label}.template.json").read_text())
    if ready:
        for name, values in c["per_eval"].items():
            values.update(max_concurrent=4 if name == "r1_gpqa_diamond" else 8,
                          llm_timeout_sec=4200 if name == "r1_gpqa_diamond" else 3600,
                          suite_timeout_sec=9000 if name == "r1_gpqa_diamond" else 15000)
            if name != "r1_gpqa_diamond":
                values["agent_timeout_sec"] = 5400
    return c


def configured(c):
    with patch("reference_config.evals.qwen38_goal12.controls", return_value=c):
        return configure(EVAL_CONFIGS[MODEL], EvalLimitMode)


def harbor_body(task, tmp_path, *, ci):
    rc = runtime(ci)
    h = build_harbor_config(
        task, ServerConnection(base_url="http://127.0.0.1", service_port=8000, model=MODEL),
        DriverContext(output_dir=tmp_path, device="P300X2"), runtime_config=rc,
        n_tasks=resolve_n_tasks(task, rc), run_stamp="host-only-proof")
    return h, json.loads(_write_harbor_config(h).read_text())


def test_absent_and_empty_overrides_preserve_legacy_configuration():
    c = copy.deepcopy(controls())
    c.pop("per_eval", None)
    old = configured(c)
    c["per_eval"] = {}
    assert configured(c) == old


def test_legacy_count_does_not_rewrite_a_frozen_harbor_filter():
    c = copy.deepcopy(controls())
    c.pop("per_eval", None)
    c["sample_count"] = 1
    for task in configured(c).tasks:
        if task.agentic_eval_config is not None:
            assert task.agentic_eval_config.task_names_map[EvalLimitMode.CI_NIGHTLY] == c["task_names"][task.task_name]


@pytest.mark.parametrize("label", ["release32_16", "combined8"])
def test_effective_subset_bounds_and_generation_preserved(label, tmp_path):
    c = template(label)
    baseline = {t.task_name: t for t in configured(controls()).tasks}
    for task in configured(c).tasks:
        name = task.task_name
        e = c["per_eval"][name]
        assert task.seed == c["seed"] == 1234
        assert task.limit_samples_map[EvalLimitMode.CI_NIGHTLY] == e["sample_count"]
        assert task.score.published_score is None and task.score.gpu_reference_score is None
        if name == "r1_gpqa_diamond":
            assert task.doc_ids == c["gpqa_handoff"]["doc_ids"]
            assert len(task.doc_ids) == e["sample_count"]
            assert task.max_concurrent == 4
            assert task.model_kwargs["timeout"] == 4200
            assert task.wall_clock_timeout_seconds == 9000
            assert task.gen_kwargs == baseline[name].gen_kwargs
            assert task.gen_kwargs["max_gen_toks"] == 16384
            assert task.request_body == baseline[name].request_body
            continue
        h, body = harbor_body(task, tmp_path / name, ci=True)
        assert h.n_tasks == e["sample_count"]
        assert body["datasets"][0]["n_tasks"] == e["sample_count"]
        assert body["datasets"][0]["task_names"] == c["task_names"][name]
        assert body["n_concurrent_trials"] == 8
        assert body["agents"][0]["override_timeout_sec"] == 5400
        assert body["agent_timeout_multiplier"] == 1.0
        assert h.llm_timeout_sec == 3600 and h.harbor_timeout_sec == 15000
        assert "verifier" not in body  # Original enabled/default verifier is untouched.
        assert task.agentic_eval_config.verifier_env == baseline[name].agentic_eval_config.verifier_env
        kwargs = body["agents"][0]["kwargs"]
        if name == "terminal_bench_2_1":
            assert body["datasets"][0]["name"] == "terminal-bench/terminal-bench-2-1"
            assert body["datasets"][0]["ref"] == c["terminal_dataset_ref"]
            assert kwargs["model_info"]["max_output_tokens"] == 16384
            assert kwargs["llm_kwargs"]["max_tokens"] == 16384
            assert kwargs["llm_kwargs"]["timeout"] == 3600
            assert kwargs["llm_kwargs"]["seed"] == 1234
            assert kwargs["llm_kwargs"]["extra_body"] == baseline[name].agentic_eval_config.agent_kwargs["llm_kwargs"]["extra_body"]
        else:
            assert kwargs["max_tokens"] == 16384
            assert kwargs["config"]["model"]["model_kwargs"]["timeout"] == 3600
            assert kwargs["config"]["model"]["model_kwargs"]["seed"] == 1234


def test_full_run_disables_ci_count_and_identity_filters(tmp_path):
    c = template("combined8")
    c["full_suite"] = True
    for task in configured(c).tasks:
        if task.agentic_eval_config is None:
            assert task.doc_ids is None
            with patch("huggingface_hub.snapshot_download", side_effect=AssertionError("network forbidden")), \
                 patch("llm_module.eval_command.get_venv_provisioner") as venv:
                venv.return_value.venv_path.return_value = tmp_path
                command = build_eval_command(task, spec(), None, tmp_path, 8000,
                                             runtime_config=runtime(False), tokenizer_path="/local/tokenizer")
            assert "--samples" not in command and "--limit" not in command
        else:
            h, body = harbor_body(task, tmp_path / task.task_name, ci=False)
            assert h.n_tasks is None and h.task_names == []
            assert "n_tasks" not in body["datasets"][0]
            assert "task_names" not in body["datasets"][0]
            assert h.n_concurrent_trials == 8 and h.agent_timeout_sec == 5400


@pytest.mark.parametrize("label", ["release32_16", "combined8"])
def test_prepared_templates_fail_until_numeric_bounds_are_chosen(label):
    with pytest.raises(ValueError, match="positive integers"):
        configured(template(label, ready=False))


@pytest.mark.parametrize("override", [
    {"sample_count": 999}, {"max_concurrent": 0}, {"agent_timeout_sec": True},
    {"llm_timeout_sec": None}, {"suite_timeout_sec": -1},
    {"output_tokens": 1}, {"seed": 2}, {"verifier": False},
])
def test_invalid_or_quality_changing_overrides_fail_closed(override):
    c = copy.deepcopy(controls())
    c["per_eval"] = {"terminal_bench_2_1": override}
    with pytest.raises(ValueError):
        configured(c)


def test_frozen_manifests_reproduce_seeded_nested_subsets_and_source_hashes():
    c = controls()
    source_path = ROOT / "qwen38_goal12_terminal21_occupancy32.json"
    source = json.loads(source_path.read_text())
    swe = json.loads((ROOT / "qwen38_goal12_swe_registry.json").read_text())[0]
    swe_sources = {row["name"]: row for row in swe["tasks"]}
    selected = {}
    for label in ("release32_16", "combined8"):
        candidate = template(label, ready=False)
        payload = Path(candidate["selection_manifest"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == candidate["selection_manifest_sha256"]
        manifest = json.loads(payload)
        assert manifest["source_manifest_sha256"] == hashlib.sha256(source_path.read_bytes()).hexdigest()
        selected[label] = manifest["evals"]
        for name, row in manifest["evals"].items():
            original = source["evals"][name]
            key = "doc_ids" if name == "r1_gpqa_diamond" else "task_names"
            expected = sorted(random.Random(1234).sample(original[key], 32)[:row["sample_count"]])
            assert row[key] == expected
            if key == "doc_ids":
                hashes = dict(zip(original[key], original["document_sha256"]))
                assert row["document_sha256"] == [hashes[i] for i in expected]
                assert candidate["gpqa_handoff"]["doc_ids"] == expected
                assert candidate["gpqa_handoff"]["document_sha256"] == row["document_sha256"]
            else:
                assert candidate["task_names"][name] == expected
                sources = original["selected_sources"] if name == "terminal_bench_2_1" else swe_sources
                assert row["selected_sources"] == {i: sources[i] for i in expected}
                assert row["selected_sources_sha256"] == hashlib.sha256(
                    json.dumps(row["selected_sources"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    for name in c["evals"]:
        key = "doc_ids" if name == "r1_gpqa_diamond" else "task_names"
        assert set(selected["combined8"][name][key]) <= set(selected["release32_16"][name][key])
