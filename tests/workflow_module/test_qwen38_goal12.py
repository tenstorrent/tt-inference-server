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
    gpqa = next(t for t in tasks if t.agentic_eval_config is None)
    assert gpqa.doc_ids == controls()["gpqa_handoff"]["doc_ids"][:controls()["sample_count"]]
    for t in tasks:
        assert t.seed == controls()["seed"]
        assert t.score.published_score is None
    tb = next(t for t in tasks if t.task_name == "terminal_bench_2")
    assert tb.agentic_eval_config.dataset == "terminal-bench/terminal-bench-2"
    assert tb.agentic_eval_config.agent_kwargs["session_id_headers"] == ["X-Session-ID"]


def test_goal12_agentic_route_provisions_and_runs_standard_harness_first():
    ctx = MagicMock()
    ctx.model_spec.hf_model_repo = MODEL
    block = Block(kind="evals", task_type="llm", title="measured", data={"accuracy": 0.0})
    events = []
    with patch("test_module.llm_tests.agentic_eval_tests._configure_openai_env"), \
         patch("workflow_module.venv_provisioner.get_venv_provisioner") as provisioner, \
         patch("test_module.llm_tests.llm_eval_tests.run_llm_eval", side_effect=lambda *a, **kw: events.append("gpqa") or [block]), \
         patch("test_module.llm_tests.agentic_eval_tests.run_llm_agentic_eval", side_effect=lambda *a: events.append("harbor") or [block]):
        provisioner.return_value.provision.return_value = True
        workflow = AgenticWorkflow(ctx, orchestrator_metadata=OrchestratorMetadata())
        outcomes = workflow.run_tasks()
    assert events == ["gpqa", "harbor"]
    assert outcomes[0].exit_code == 0
