# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""Goal-12 publication must select the autoport and all requested harnesses."""
from unittest.mock import MagicMock, patch

from reference_config.evals.eval_config import EVAL_CONFIGS
from reference_config.evals.qwen38_goal12 import MODEL, controls
from report_module.schema import Block
from workflow_module.execution import OrchestratorMetadata
from workflow_module.workflows import AgenticWorkflow
from workflows.model_spec import MODEL_SPECS


def test_autoport_default_and_seeded_frozen_tasks():
    specs = [s for s in MODEL_SPECS.values() if s.hf_model_repo == MODEL
             and s.device_type.name == "P300X2" and s.device_model_spec.default_impl]
    assert len(specs) == 1
    s = specs[0]
    assert s.impl.code_path == "models/autoports/qwen_qwen3_8_27b"
    assert s.device_model_spec.max_concurrency == 32
    assert s.device_model_spec.vllm_args["enable_prefix_caching"] is True
    tasks = EVAL_CONFIGS[MODEL].tasks
    assert {t.task_name for t in tasks} == set(controls()["evals"])
    gpqa = next((t for t in tasks if t.agentic_eval_config is None), None)
    if gpqa is not None:
        assert gpqa.doc_ids == (None if controls()["full_suite"] else controls()["gpqa_handoff"]["doc_ids"][:controls()["sample_count"]])
    for t in tasks:
        assert t.seed == controls()["seed"]
        assert t.score.published_score is None
    tb = next((t for t in tasks if t.task_name in ("terminal_bench_2", "terminal_bench_2_1")), None)
    if tb is not None:
        assert tb.agentic_eval_config.dataset == ("terminal-bench/terminal-bench-2-1" if tb.task_name == "terminal_bench_2_1" else "terminal-bench/terminal-bench-2")
        assert tb.agentic_eval_config.dataset_ref == controls()["terminal_dataset_ref"]
        assert tb.agentic_eval_config.agent_kwargs["session_id_headers"] == ["X-Session-ID"]


def test_goal12_agentic_route_provisions_and_runs_standard_harness_first():
    ctx = MagicMock()
    ctx.model_spec.hf_model_repo = MODEL
    block = Block(kind="evals", task_type="llm", title="measured", data={"accuracy": 0.0})
    events = []
    # This regression exercises mixed routing, independent of the active arm.
    mixed_controls = {**controls(), "include_standard_evals": True,
                      "evals": ["r1_gpqa_diamond", "swe_bench_verified"]}
    with patch("reference_config.evals.qwen38_goal12.controls", return_value=mixed_controls), \
         patch("test_module.llm_tests.agentic_eval_tests._configure_openai_env"), \
         patch("workflow_module.venv_provisioner.get_venv_provisioner") as provisioner, \
         patch("test_module.llm_tests.llm_eval_tests.run_llm_eval", side_effect=lambda *a, **kw: events.append("gpqa") or [block]), \
         patch("test_module.llm_tests.agentic_eval_tests.run_llm_agentic_eval", side_effect=lambda *a: events.append("harbor") or [block]):
        provisioner.return_value.provision.return_value = True
        workflow = AgenticWorkflow(ctx, orchestrator_metadata=OrchestratorMetadata())
        outcomes = workflow.run_tasks()
    assert events == ["gpqa", "harbor"]
    assert outcomes[0].exit_code == 0


def test_active_manifest_and_handoff_exact_identities():
    import hashlib
    import json
    from pathlib import Path
    c = controls()
    manifest_path = Path(c['selection_manifest'])
    payload = manifest_path.read_bytes()
    m = json.loads(payload)
    assert hashlib.sha256(payload).hexdigest() == c['selection_manifest_sha256']
    if 'swe25' in m:
        assert c['evals'] == ['swe_bench_verified']
        assert c['sample_count'] in (1, 25, 50, 100)
        if c['sample_count'] == 1:
            expected_tasks = ['astropy__astropy-14309']
            assert c['max_concurrent'] == 1
            assert set(expected_tasks) <= set(m['swe25'])
        elif c['sample_count'] == 25:
            expected_tasks = m['swe25']
            assert c['max_concurrent'] in (8, 16)
        else:
            remaining = [x for x in m['swe500'] if x not in set(m['swe100'])]
            cohorts = [m['swe100']] + [remaining[i:i+100] for i in range(0, 400, 100)]
            assert len(cohorts) == 5 and all(len(set(x)) == 100 for x in cohorts)
            assert len(set(sum(cohorts, []))) == 500
            assert set(sum(cohorts, [])) == set(m['swe500'])
            if c['sample_count'] == 50:
                partitions = [cohort[i:i+50] for cohort in cohorts for i in (0, 50)]
                assert len(partitions) == 10
                assert all(len(set(part)) == 50 for part in partitions)
                assert len(set(sum(partitions, []))) == 500
                assert set(sum(partitions[:2], [])) == set(m['swe100'])
                expected_tasks = c['task_names']['swe_bench_verified']
                assert expected_tasks in partitions
                assert c['max_concurrent'] == 8
            else:
                expected_tasks = c['task_names']['swe_bench_verified']
                assert expected_tasks in cohorts
                assert c['max_concurrent'] == 16
            assert c['suite_timeout_sec'] == 45900
            assert c['suite_timeout_sec'] + 4500 == 840*60
        assert len(set(m['swe25'])) == 25
        assert len(set(m['swe100'])) == 100
        assert len(set(m['swe500'])) == 500
        assert m['swe25'] == m['swe100'][::4]
        assert set(m['swe100']) <= set(m['swe500'])
        assert c['task_names']['swe_bench_verified'] == expected_tasks
        assert c['agent_timeout_sec'] == 6300
        assert c['llm_timeout_sec'] == 3600
        assert c['output_tokens'] == 16384
        assert c['swe_agent_isolated_network'] is True
        assert c['suite_timeout_sec'] == ((c['sample_count']+c['max_concurrent']-1)//c['max_concurrent'])*6300+(900 if c['sample_count']==1 else 1800)
        task = EVAL_CONFIGS[MODEL].tasks[0]
        from reference_config.evals.eval_config import EvalLimitMode
        agent = task.agentic_eval_config
        assert agent.task_names_map[EvalLimitMode.CI_NIGHTLY] == expected_tasks
        assert agent.n_concurrent_trials == c['max_concurrent']
        assert agent.agent_timeout_sec == 6300
        assert agent.harbor_timeout_sec == c['suite_timeout_sec']
        return
    assert c['sample_count'] == c['max_concurrent'] == 32
    assert isinstance(c['full_suite'], bool)
    h = c['gpqa_handoff']
    g = m['evals']['r1_gpqa_diamond']
    assert h['manifest'] == c['selection_manifest']
    assert h['manifest_sha256'] == c['selection_manifest_sha256']
    assert h['doc_ids'] == g['doc_ids']
    assert h['document_sha256'] == g['document_sha256']
    assert h['population_sha256'] == g['population_sha256']
    assert len(set(h['doc_ids'])) == len(h['document_sha256']) == 32
    hashes = dict(zip(h['doc_ids'], h['document_sha256']))
    assert [hashes[i] for i in g['frozen10_doc_ids']] == g['frozen10_document_sha256']
    assert set(g['doc_ids']) == set(g['frozen10_doc_ids']) | set(g['additional22_doc_ids'])
    for name in ('swe_bench_verified', 'terminal_bench_2_1'):
        e = m['evals'][name]
        assert c['task_names'][name] == e['task_names']
        assert len(set(e['task_names'])) == 32
        assert set(e['task_names']) == set(e['frozen10_task_names']) | set(e['additional22_task_names'])
