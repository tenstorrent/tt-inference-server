# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for the requirements-driven model-spec provider and target pack."""

from __future__ import annotations

from pathlib import Path

import pytest

from workflow_module.requirements_schema import Slo, load_requirements
from workflows.model_spec import MODEL_SPECS
from workflows.model_spec_provider import (
    TenstorrentModelSpecProvider,
    hardware_to_device_name,
)
from workflows.requirements_target_pack import (
    _EVAL_NAME_TO_TASK,
    RequirementsModelSpecProvider,
    RequirementsTargetPack,
    _normalize_eval_name,
    unknown_eval_names,
)
from llm_module.goodput import GoodputSlo
from workflows.target_pack_provider import TenstorrentTargetPack
from workflows.workflow_types import DeviceTypes

_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "requirements"
    / "acme-llm-serving.json"
)


@pytest.fixture
def doc():
    return load_requirements(_FIXTURE)


@pytest.fixture
def pack(doc):
    return RequirementsTargetPack(doc, TenstorrentTargetPack())


# --- hardware mapping --------------------------------------------------------


@pytest.mark.parametrize("hardware", ["SC8", "SC12", "SC20", "sc8", "SC128"])
def test_hardware_super_cluster_family_maps_to_super_cluster(hardware):
    # The Super Cluster ships in several node counts; every SC<N> is the same
    # SUPER_CLUSTER device from the engine's perspective.
    assert hardware_to_device_name(hardware) == "SUPER_CLUSTER"


def test_hardware_accepts_device_name_directly():
    assert hardware_to_device_name("galaxy") == "GALAXY"


@pytest.mark.parametrize(
    ("hardware", "device"),
    [
        ("GLX", "GALAXY"),
        ("WH GALAXY", "GALAXY"),
        ("WH GLX", "GALAXY"),
        ("wh glx", "GALAXY"),
        ("BH GALAXY", "BLACKHOLE_GALAXY"),
        ("BH GLX", "BLACKHOLE_GALAXY"),
    ],
)
def test_hardware_galaxy_aliases(hardware, device):
    # Customer-facing galaxy names ("WH GLX") map to the DeviceTypes member.
    assert hardware_to_device_name(hardware) == device


@pytest.mark.parametrize("hardware", ["mystery-box", "SC", "SC8X", "8SC"])
def test_hardware_unknown_raises(hardware):
    with pytest.raises(ValueError, match="Unknown deployment.hardware"):
        hardware_to_device_name(hardware)


# --- off-catalog synthesis ---------------------------------------------------


def test_catalog_provider_accepts_full_repo_and_basename():
    provider = TenstorrentModelSpecProvider()
    spec = next(iter(MODEL_SPECS.values()))

    assert spec.hf_model_repo in provider.model_names()
    assert spec.model_name in provider.model_names()
    assert spec in provider.resolve_candidates(
        spec.hf_model_repo, spec.device_type.name.lower()
    )


def test_synthesize_off_catalog_spec():
    provider = TenstorrentModelSpecProvider()
    spec = provider.synthesize(
        model_name="acme/tiny-llm",
        hf_model_repo="acme/tiny-llm",
        device="super_cluster",
        max_context=8192,
        max_concurrency=16,
    )
    assert spec.model_name == "acme/tiny-llm"
    assert spec.device_type == DeviceTypes.SUPER_CLUSTER
    assert spec.device_model_spec.max_context == 8192
    assert spec.device_model_spec.max_concurrency == 16


def test_synthesized_model_id_is_a_single_path_component():
    """An HF org prefix must not turn model_id into a nested directory.

    model_id names the runtime-spec JSON, the run log, and the per-eval output
    dirs; a raw "org/model" silently writes into an "org/" subdirectory that
    nothing has created, which fails the run before the workflow starts.
    """
    spec = TenstorrentModelSpecProvider().synthesize(
        model_name="acme/tiny-llm",
        hf_model_repo="acme/tiny-llm",
        device="super_cluster",
        max_context=8192,
        max_concurrency=16,
    )
    assert "/" not in spec.model_id
    assert spec.model_id.startswith("acme__tiny-llm")


def test_requirements_provider_synthesizes_when_off_catalog(doc):
    provider = RequirementsModelSpecProvider(TenstorrentModelSpecProvider(), doc)
    # The document's model name is the HF repo path and is not a catalog name,
    # so resolution falls back to synthesis from the doc's metadata.
    spec = provider.resolve(doc.model.name, "super_cluster")
    assert spec.model_name == doc.model.name
    assert spec.device_model_spec.max_context == doc.model.context_length
    assert (
        spec.device_model_spec.max_concurrency
        == doc.deployment.max_concurrency_per_instance
    )


# --- eval config synthesis ---------------------------------------------------


def test_eval_config_maps_names_and_regates(pack, doc):
    cfg = pack.eval_config(doc.model.name)
    by_task = {t.task_name: t for t in cfg.tasks}
    assert set(by_task) == {
        "gpqa_diamond_cot_zeroshot",
        "swe_bench_verified",
        "terminal_bench_2",
    }
    # Reference score + tolerance re-gated from the document.
    gpqa = by_task["gpqa_diamond_cot_zeroshot"]
    assert gpqa.score.gpu_reference_score == 79.2
    assert gpqa.score.tolerance == 0.05
    assert gpqa.priority == "must"
    # Terminal-Bench 2.0 is a "should" in the document.
    assert by_task["terminal_bench_2"].priority == "should"
    # Borrowed tasks keep their runnable harness config.
    assert by_task["swe_bench_verified"].agentic_eval_config is not None
    assert by_task["terminal_bench_2"].agentic_eval_config is not None


def test_eval_config_unknown_name_raises(doc):
    from dataclasses import replace

    from workflow_module.requirements_schema import AccuracyEval

    bad = replace(doc, accuracy_evals=[AccuracyEval(name="Made-Up-Bench")])
    pack = RequirementsTargetPack(bad, TenstorrentTargetPack())
    with pytest.raises(ValueError, match="no known catalog task mapping"):
        pack.eval_config(doc.model.name)


def test_eval_config_none_when_no_evals(doc):
    from dataclasses import replace

    empty = replace(doc, accuracy_evals=[])
    pack = RequirementsTargetPack(empty, TenstorrentTargetPack())
    assert pack.eval_config(doc.model.name) is None


def test_eval_task_synthesized_with_neutral_defaults(doc, monkeypatch):
    from dataclasses import replace

    # Fully off-catalog path: no catalog model defines the task, so the pack
    # synthesizes a neutral EvalTask — max_length from the document's
    # contextLength, streaming on, no sampling overrides, doc-gated score.
    # Narrow the doc to GPQA only: the harness-backed evals (SWE-bench,
    # Terminal-Bench) cannot be synthesized and are covered by the test below.
    gpqa_only = replace(doc, accuracy_evals=[doc.accuracy_evals[0]])
    pack = RequirementsTargetPack(gpqa_only, TenstorrentTargetPack())
    monkeypatch.setattr(
        pack, "_find_task_template", lambda candidates: (None, candidates[0])
    )
    cfg = pack.eval_config("acme/off-catalog-model")
    gpqa = next(t for t in cfg.tasks if t.task_name == "gpqa_diamond_cot_zeroshot")

    assert gpqa.model_kwargs == {"timeout": "3600", "max_length": 131072}
    assert gpqa.gen_kwargs == {"stream": "True"}  # synthesized default: stream on
    assert gpqa.score.gpu_reference_score == 79.2
    assert gpqa.score.published_score == 80.9
    assert gpqa.score.tolerance == 0.05
    assert gpqa.score.score_func_kwargs == {
        "result_keys": ["exact_match,flexible-extract"],
        "unit": "percent",
    }
    assert gpqa.priority == "must"


def test_document_gen_kwargs_override_the_template_per_parameter(doc, caplog):
    """The document states sampling; the borrowed template keeps the rest.

    A template is borrowed from whichever catalog model defines the task, and
    carries sampling tuned for that model. A document stating a temperature
    must not silently discard a token budget it never chose.
    """
    from dataclasses import replace

    from workflow_module.requirements_schema import AccuracyEval

    tuned = AccuracyEval.from_dict(
        {
            "name": "GPQA-Diamond",
            "gpuReferenceScore": 79.2,
            "genKwargs": {"temperature": 0.6, "topP": 0.95},
        }
    )
    pack = RequirementsTargetPack(
        replace(doc, accuracy_evals=[tuned]), TenstorrentTargetPack()
    )
    template, _ = pack._find_task_template(
        ("gpqa_diamond_cot_zeroshot", "r1_gpqa_diamond")
    )
    assert template is not None, "this test needs a catalog template to borrow"

    task = pack.eval_config(doc.model.name).tasks[0]

    # Stated: the document wins, whether or not the template had an opinion.
    assert task.gen_kwargs["temperature"] == 0.6
    assert task.gen_kwargs["top_p"] == 0.95
    # Unstated: every other parameter survives exactly as the template had it.
    for key, value in template.gen_kwargs.items():
        if key in ("temperature", "stream"):
            continue
        assert task.gen_kwargs[key] == value, key
    # And streaming is still forced on afterwards (the gateway deadline).
    assert task.gen_kwargs["stream"] == "true"


def test_document_gen_kwargs_reach_a_synthesized_task(doc, monkeypatch):
    """Off-catalog: there is no template, so the document is all there is."""
    from dataclasses import replace

    from workflow_module.requirements_schema import AccuracyEval

    tuned = AccuracyEval.from_dict(
        {
            "name": "GPQA-Diamond",
            "gpuReferenceScore": 79.2,
            "genKwargs": {"temperature": 0.6, "maxGenToks": 32768},
        }
    )
    pack = RequirementsTargetPack(
        replace(doc, accuracy_evals=[tuned]), TenstorrentTargetPack()
    )
    monkeypatch.setattr(
        pack, "_find_task_template", lambda candidates: (None, candidates[0])
    )
    task = pack.eval_config("acme/off-catalog-model").tasks[0]

    assert task.gen_kwargs == {
        "stream": "True",
        "temperature": 0.6,
        "max_gen_toks": 32768,
    }


def _agentic_task_for(doc, name, task, gen_kwargs):
    """Build the eval task a document with ``gen_kwargs`` produces for ``task``."""
    from dataclasses import replace

    from workflow_module.requirements_schema import AccuracyEval

    ae = AccuracyEval.from_dict(
        {"name": name, "gpuReferenceScore": 50.0, "genKwargs": gen_kwargs}
    )
    pack = RequirementsTargetPack(
        replace(doc, accuracy_evals=[ae]), TenstorrentTargetPack()
    )
    template, _ = pack._find_task_template((task,))
    assert template is not None, f"this test needs a catalog template for {task}"
    return pack.eval_config(doc.model.name).tasks[0], template


def test_agentic_sampling_is_repointed_where_the_agent_nests_it(doc):
    """Each agent keeps sampling somewhere different; re-point it in place.

    mini-swe-agent reads it from ``config.model.model_kwargs``, terminus-2
    takes ``temperature`` at the top level but ``top_p`` inside ``llm_kwargs``.
    Both values are the *donor's* — ``top_p`` is 0.95 for some catalog models
    and 1.0 for others — so a borrowed config otherwise samples as whoever the
    donor was.
    """
    swe, swe_template = _agentic_task_for(
        doc,
        "SWE-bench Verified",
        "swe_bench_verified",
        {"temperature": 0.3, "topP": 0.9},
    )
    model_kwargs = swe.agentic_eval_config.agent_kwargs["config"]["model"][
        "model_kwargs"
    ]
    assert model_kwargs["temperature"] == 0.3
    assert model_kwargs["top_p"] == 0.9
    # The donor's shared, module-level config is untouched.
    donor = swe_template.agentic_eval_config.agent_kwargs["config"]["model"]
    assert donor["model_kwargs"]["temperature"] == 1.0

    tb, _ = _agentic_task_for(
        doc, "Terminal-Bench 2.0", "terminal_bench_2", {"temperature": 0.3, "topP": 0.9}
    )
    agent_kwargs = tb.agentic_eval_config.agent_kwargs
    assert agent_kwargs["temperature"] == 0.3
    assert agent_kwargs["llm_kwargs"]["top_p"] == 0.9


def test_top_k_is_repointed_inside_extra_body(doc):
    """``top_k`` is not an OpenAI parameter, so it rides in ``extra_body``.

    LiteLLM runs with ``drop_params=True`` and throws away anything the OpenAI
    provider does not recognise; ``extra_body`` is merged into the request body
    verbatim, so that is the only place top_k survives. It is nested two levels
    below the sampling it belongs to, which is why re-pointing walks the whole
    config instead of reading fixed paths — and why the thinking toggle sitting
    beside it is left alone.
    """
    from workflows.requirements_target_pack import _repoint_sampling

    mini_swe_agent = {
        "version": "2.2.8",
        "max_tokens": 32 * 1024,
        "config": {
            "model": {
                "model_kwargs": {
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "extra_body": {
                        "top_k": 20,
                        "chat_template_kwargs": {"enable_thinking": True},
                    },
                }
            }
        },
    }
    found: set = set()

    out = _repoint_sampling(
        mini_swe_agent, {"temperature": 0.3, "top_p": 0.9, "top_k": 64}, found
    )

    model_kwargs = out["config"]["model"]["model_kwargs"]
    assert model_kwargs["temperature"] == 0.3
    assert model_kwargs["top_p"] == 0.9
    assert model_kwargs["extra_body"]["top_k"] == 64
    assert found == {"temperature", "top_p", "top_k"}
    # The agent's own budget and its thinking toggle are not sampling.
    assert out["max_tokens"] == 32 * 1024
    assert model_kwargs["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": True
    }
    # The donor's shared, module-level config is untouched.
    assert (
        mini_swe_agent["config"]["model"]["model_kwargs"]["extra_body"]["top_k"] == 20
    )


def _repoint_agent(doc, agent, agent_kwargs, gen_kwargs, task="tau3_bench_banking"):
    """Run the agentic re-point over one agent's real agent_kwargs shape."""
    import copy
    from dataclasses import dataclass
    from typing import Any, Dict

    from workflow_module.requirements_schema import AccuracyEval

    @dataclass(frozen=True)
    class FakeHarness:
        agent: str
        agent_kwargs: Dict[str, Any]

    cfg = FakeHarness(agent=agent, agent_kwargs=copy.deepcopy(agent_kwargs))
    ae = AccuracyEval.from_dict(
        {"name": "an eval", "gpuReferenceScore": 50.0, "genKwargs": gen_kwargs}
    )
    changes: dict = {}
    RequirementsTargetPack(doc, TenstorrentTargetPack())._repoint_agent_sampling(
        cfg, task, ae, changes
    )
    return changes, cfg


def test_tau3_json_args_are_constructed_when_the_donor_has_none(doc):
    """Whether a donor carries llm_args_json is arbitrary, so build it.

    The MiniMax tau3 entry has none and the Kimi one does, so without this the
    same document has its top_p honoured or dropped depending on which model
    the task happened to be borrowed from.
    """
    import json

    changes, _ = _repoint_agent(
        doc,
        "tau3_llm_agent",
        {"tau2_trial_index": 0, "temperature": 1.0, "max_steps": 200},
        {"temperature": 0.3, "topP": 0.9, "topK": 20},
    )
    agent_kwargs = changes["agent_kwargs"]

    assert agent_kwargs["temperature"] == 0.3
    raw = agent_kwargs["llm_args_json"]
    assert isinstance(raw, str), "a dict here fails in the adapter's shlex.quote"
    args = json.loads(raw)
    assert args["top_p"] == 0.9
    # top_k is not an OpenAI parameter, so drop_params discards it unless it
    # rides in extra_body.
    assert args["extra_body"] == {"top_k": 20}


def test_tau3_json_args_keep_what_the_donor_already_carried(doc):
    import json

    changes, cfg = _repoint_agent(
        doc,
        "tau3_llm_agent",
        {"temperature": 1.0, "llm_args_json": '{"top_p": 0.95, "seed": 7}'},
        {"topP": 0.9},
    )

    args = json.loads(changes["agent_kwargs"]["llm_args_json"])
    assert args["top_p"] == 0.9
    assert args["seed"] == 7
    # The donor's shared config object is untouched.
    assert cfg.agent_kwargs["llm_args_json"] == '{"top_p": 0.95, "seed": 7}'


def test_extra_body_is_built_where_the_agent_reads_it(doc):
    """top_k is not an OpenAI parameter, so drop_params discards it elsewhere.

    Six live terminus-2 entries carry it at ``llm_kwargs.extra_body``; a donor
    that happens not to is no reason to drop the document's value.
    """
    changes, _ = _repoint_agent(
        doc,
        "terminus-2",
        {"temperature": 1.0, "llm_kwargs": {"top_p": 1.0, "timeout": 3600}},
        {"topK": 20},
        task="terminal_bench_2_1",
    )

    llm_kwargs = changes["agent_kwargs"]["llm_kwargs"]
    assert llm_kwargs["extra_body"] == {"top_k": 20}
    assert llm_kwargs["timeout"] == 3600


def test_reasoning_effort_reaches_extra_body_even_when_set_first_class(doc):
    """The first-class argument alone is what drop_params discards.

    LiteLLM drops reasoning_effort for an openai-provider model it does not
    recognise as reasoning-capable, so updating the donor's first-class value
    in place would change a number that never reaches the server.
    """
    changes, _ = _repoint_agent(
        doc,
        "terminus-2",
        {"temperature": 1.0, "reasoning_effort": "low", "llm_kwargs": {"top_p": 1.0}},
        {"reasoningEffort": "high"},
        task="terminal_bench_2_1",
    )
    agent_kwargs = changes["agent_kwargs"]

    assert agent_kwargs["reasoning_effort"] == "high"
    assert agent_kwargs["llm_kwargs"]["extra_body"]["reasoning_effort"] == "high"


def test_reasoning_effort_already_in_extra_body_is_not_duplicated(doc):
    changes, _ = _repoint_agent(
        doc,
        "terminus-2",
        {"llm_kwargs": {"extra_body": {"reasoning_effort": "low"}}},
        {"reasoningEffort": "high"},
        task="terminal_bench_2_1",
    )

    assert changes["agent_kwargs"] == {
        "llm_kwargs": {"extra_body": {"reasoning_effort": "high"}}
    }


def test_agentic_parameters_with_nowhere_to_go_are_flagged(doc, caplog):
    """An unknown agent has no documented place to build one, so say so.

    The eval still runs and still produces a score, which without the warning
    reads as measured under a parameter that never reached the server.
    """
    with caplog.at_level("WARNING"):
        changes, _ = _repoint_agent(
            doc, "some-new-agent", {"temperature": 1.0}, {"topK": 20}
        )

    assert "no knob" in caplog.text and "top_k" in caplog.text
    assert "agent_kwargs" not in changes


def test_agentic_budget_splits_the_document_context_window(doc):
    """One requested budget sets the output cap and the input cap it implies.

    The two must sum below the server's context, so raising the output cap
    without shrinking the input one pushes requests past it mid-trial.
    """
    changes, _ = _repoint_agent(
        doc,
        "terminus-2",
        {
            "temperature": 1.0,
            "model_info": {"max_input_tokens": 16384, "max_output_tokens": 4096},
            "llm_kwargs": {"top_p": 1.0, "max_tokens": 4096},
        },
        {"maxGenToks": 65536},
        task="terminal_bench_2_1",
    )
    agent_kwargs = changes["agent_kwargs"]

    context = doc.model.context_length
    assert agent_kwargs["model_info"]["max_output_tokens"] == 65536
    assert agent_kwargs["model_info"]["max_input_tokens"] == context - 65536
    assert agent_kwargs["llm_kwargs"]["max_tokens"] == 65536


def test_agentic_budget_is_refused_when_it_leaves_no_room_for_a_prompt(doc, caplog):
    with caplog.at_level("WARNING"):
        changes, _ = _repoint_agent(
            doc,
            "terminus-2",
            {"model_info": {"max_input_tokens": 16384, "max_output_tokens": 4096}},
            {"maxGenToks": doc.model.context_length + 1},
            task="terminal_bench_2_1",
        )

    assert "no room for a prompt" in caplog.text
    assert "agent_kwargs" not in changes


def test_an_eval_without_gen_kwargs_only_gets_streaming(doc, pack):
    """No genKwargs is no sampling claim — the pre-existing behaviour stands."""
    template, _ = pack._find_task_template(
        ("gpqa_diamond_cot_zeroshot", "r1_gpqa_diamond")
    )
    assert doc.accuracy_evals[0].gen_kwargs is None

    task = pack.eval_config(doc.model.name).tasks[0]

    assert task.gen_kwargs == {**template.gen_kwargs, "stream": "true"}


def test_harness_concurrency_comes_from_the_document(doc, pack):
    """The borrowed template's trial count must not decide this deployment's.

    Which catalog model a harness template is borrowed from is decided by
    EVAL_CONFIGS iteration order, so its n_concurrent_trials is arbitrary here
    (Terminal-Bench 2 in particular borrows a serial, n=1 template). The
    document states what the instance under test serves concurrently, so that
    is what the harnesses run at.
    """
    harness_tasks = {
        task.task_name: task.agentic_eval_config
        for task in pack.eval_config(doc.model.name).tasks
        if task.agentic_eval_config is not None
    }
    assert set(harness_tasks) == {"swe_bench_verified", "terminal_bench_2"}
    for cfg in harness_tasks.values():
        assert cfg.n_concurrent_trials == doc.deployment.max_concurrency_per_instance


def test_harness_concurrency_falls_back_when_document_is_silent(doc):
    """No maxConcurrencyPerInstance means nothing to override with."""
    from dataclasses import replace

    silent = replace(
        doc, deployment=replace(doc.deployment, max_concurrency_per_instance=None)
    )
    pack = RequirementsTargetPack(silent, TenstorrentTargetPack())
    task = next(
        t
        for t in pack.eval_config(silent.model.name).tasks
        if t.task_name == "terminal_bench_2"
    )
    borrowed, name = pack._find_task_template(("terminal_bench_2",))
    assert name == "terminal_bench_2"
    assert (
        task.agentic_eval_config.n_concurrent_trials
        == borrowed.agentic_eval_config.n_concurrent_trials
    )


def test_eval_task_synthesis_rejects_harness_backed_task(pack, monkeypatch):
    # SWE-bench needs its SWEbenchEvalConfig harness wiring, which cannot be
    # synthesized — with no catalog template it must fail loudly.
    monkeypatch.setattr(
        pack, "_find_task_template", lambda candidates: (None, candidates[0])
    )
    with pytest.raises(ValueError, match="No catalog template or built-in profile"):
        pack.eval_config("acme/off-catalog-model")


# --- benchmark config synthesis ----------------------------------------------


def test_benchmark_config_builds_sweep_from_document(pack, doc):
    provider = RequirementsModelSpecProvider(TenstorrentModelSpecProvider(), doc)
    spec = provider.resolve(doc.model.name, "super_cluster")
    cfg = pack.benchmark_config(spec)

    assert len(cfg.tasks) == 1
    points = cfg.tasks[0].param_map[DeviceTypes.SUPER_CLUSTER]
    # One point per sweep entry in the document.
    assert len(points) == len(doc.scenarios[0].sweep)

    # A point with no scenario-level gate attached gates on its own
    # reference measurements only.
    ref_point = next(
        p for p in points if p.isl == 256 and p.osl == 128 and p.max_concurrency == 1
    )
    ref_target = ref_point.targets["target"]
    assert ref_target.ttft_ms == 135  # ttftMeanMs reference
    assert ref_target.tpot_ms == 5
    assert ref_target.e2el_ms == 775
    assert ref_target.tput == 200  # decodeThroughputTps
    assert ref_target.tput_total == 600  # totalThroughputTps
    assert ref_target.goodput == 100  # goodputPct
    assert ref_point.priority == "must"
    # The doc declares no benchmark tolerance: everything grades exact.
    assert ref_target.tolerance == 0.0

    # Scenario-level gates are capability gates: each attaches at the sweep
    # point whose reference is best for that metric, overriding it. Latency
    # SLOs and request_goodput are best at the lightest point (ISL 128,
    # c=1); system_throughput is best at ISL 65536, c=64.
    light = next(
        p for p in points if p.isl == 128 and p.osl == 128 and p.max_concurrency == 1
    )
    light_target = light.targets["target"]
    assert light_target.ttft_ms == 2000  # SLO, not the 128 reference
    assert light_target.tpot_ms == 20
    assert light_target.e2el_ms == 20000
    assert light_target.goodput == 99  # request_goodput scalar
    assert light_target.tput == 200  # reference: no scenario gate for it
    assert light_target.tolerance == 0.0

    capable = next(
        p for p in points if p.isl == 65536 and p.osl == 128 and p.max_concurrency == 64
    )
    assert capable.targets["target"].tput_total == 12000  # system_throughput

    # Nothing broadcasts: another peak-concurrency point keeps its own
    # references and sees neither the SLOs nor the scalar targets.
    loaded = next(
        p for p in points if p.isl == 128 and p.osl == 128 and p.max_concurrency == 64
    )
    loaded_target = loaded.targets["target"]
    assert loaded_target.ttft_ms == 2144  # reference, not the 2000 SLO
    assert loaded_target.tput_total == 4740  # reference, not the 12000 target
    assert loaded_target.goodput == 36  # reference, not the 99 target


def test_benchmark_config_per_metric_priorities(pack, doc):
    """A point mixing must/should targets keeps the per-metric severities."""
    provider = RequirementsModelSpecProvider(TenstorrentModelSpecProvider(), doc)
    spec = provider.resolve(doc.model.name, "super_cluster")
    points = pack.benchmark_config(spec).tasks[0].param_map[DeviceTypes.SUPER_CLUSTER]

    # The should-priority request_goodput scalar attaches at the lightest
    # point (best goodputPct reference), alongside the must-priority SLOs.
    light = next(
        p for p in points if p.isl == 128 and p.osl == 128 and p.max_concurrency == 1
    )
    assert light.priority == "must"  # block severity: any must => must
    assert light.target_priorities == {
        "ttft_ms": "must",
        "tpot_ms": "must",
        "e2el_ms": "must",
        "tput": "must",
        "tput_total": "must",
        "goodput": "should",
    }

    # Every other point is reference-gated only, all must.
    others = [p for p in points if p is not light]
    assert all(set(p.target_priorities.values()) == {"must"} for p in others)


def test_scenario_slo_alone_is_not_broadcast_as_goodput_bars(pack, doc, caplog):
    """A scenario-level SLO is a capability gate, not a bar for every point.

    One set of bars cannot hold across a sweep: e2el that is comfortable at
    128 output tokens is unreachable at 1024. The fixture declares a scenario
    SLO and no row SLOs, so nothing is measured -- and it says so.
    """
    provider = RequirementsModelSpecProvider(TenstorrentModelSpecProvider(), doc)
    spec = provider.resolve(doc.model.name, "super_cluster")
    with caplog.at_level("WARNING"):
        points = (
            pack.benchmark_config(spec).tasks[0].param_map[DeviceTypes.SUPER_CLUSTER]
        )

    assert {p.goodput for p in points} == {None}
    assert "capability gate" in caplog.text
    # It still does its other job: gating its own capability point's targets.
    light = next(
        p for p in points if p.isl == 128 and p.osl == 128 and p.max_concurrency == 1
    )
    assert light.targets["target"].ttft_ms == 2000


def test_row_slos_become_the_goodput_bars_per_point(doc):
    """Bars come from the row, so they can differ per (ISL, OSL)."""
    from dataclasses import replace

    scenario = doc.scenarios[0]
    rows = [
        replace(
            p,
            slo=Slo(ttft_ms=2000, tpot_ms=20, e2el_ms=10000 if p.osl == 128 else 30000),
        )
        for p in scenario.sweep
    ]
    with_rows = replace(doc, scenarios=[replace(scenario, sweep=rows, slo=None)])
    pack = RequirementsTargetPack(with_rows, TenstorrentTargetPack())
    provider = RequirementsModelSpecProvider(TenstorrentModelSpecProvider(), with_rows)
    spec = provider.resolve(with_rows.model.name, "super_cluster")

    points = pack.benchmark_config(spec).tasks[0].param_map[DeviceTypes.SUPER_CLUSTER]

    by_osl = {p.osl: p.goodput for p in points}
    assert by_osl[128] == GoodputSlo(ttft_ms=2000, tpot_ms=20, e2el_ms=10000)
    assert by_osl[1024] == GoodputSlo(ttft_ms=2000, tpot_ms=20, e2el_ms=30000)


def test_a_partial_row_slo_inherits_the_scenario_default(doc):
    """Row-wins is field-wise, matching effectiveSlo upstream."""
    from dataclasses import replace

    scenario = doc.scenarios[0]
    rows = [replace(p, slo=Slo(e2el_ms=30000)) for p in scenario.sweep]
    merged = replace(doc, scenarios=[replace(scenario, sweep=rows)])
    pack = RequirementsTargetPack(merged, TenstorrentTargetPack())
    provider = RequirementsModelSpecProvider(TenstorrentModelSpecProvider(), merged)
    spec = provider.resolve(merged.model.name, "super_cluster")

    points = pack.benchmark_config(spec).tasks[0].param_map[DeviceTypes.SUPER_CLUSTER]

    # e2el from the row; ttft/tpot inherited from the scenario.
    assert {p.goodput for p in points} == {
        GoodputSlo(ttft_ms=2000, tpot_ms=20, e2el_ms=30000)
    }


def test_benchmark_config_goodput_unmeasurable_without_slos(doc, caplog):
    """No SLOs => no --goodput constraints; the targets grade as NA."""
    from dataclasses import replace

    no_slo = replace(
        doc,
        scenarios=[replace(doc.scenarios[0], slo=None)],
    )
    pack = RequirementsTargetPack(no_slo, TenstorrentTargetPack())
    provider = RequirementsModelSpecProvider(TenstorrentModelSpecProvider(), no_slo)
    spec = provider.resolve(no_slo.model.name, "super_cluster")
    with caplog.at_level("WARNING"):
        points = (
            pack.benchmark_config(spec).tasks[0].param_map[DeviceTypes.SUPER_CLUSTER]
        )
    assert {p.goodput for p in points} == {None}
    assert "no sweep point declares its own SLOs" in caplog.text
    # The goodput expectation stays on its capability point's targets (it
    # grades NA, visibly).
    light = next(
        p for p in points if p.isl == 128 and p.osl == 128 and p.max_concurrency == 1
    )
    assert light.targets["target"].goodput == 99


def test_smoke_test_benchmark_config_narrows_to_one_point(pack, doc):
    provider = RequirementsModelSpecProvider(TenstorrentModelSpecProvider(), doc)
    spec = provider.resolve(doc.model.name, "super_cluster")
    cfg = pack.benchmark_config(spec)
    smoke = pack.smoke_test_benchmark_config(cfg, DeviceTypes.SUPER_CLUSTER)
    points = smoke.tasks[0].param_map[DeviceTypes.SUPER_CLUSTER]
    assert len(points) == 1


def test_target_pack_delegates_unspecified_content(pack):
    # Anything the document does not define falls through to the TT pack.
    assert pack.agentic_traces_min_profile_seconds() == (
        TenstorrentTargetPack().agentic_traces_min_profile_seconds()
    )
    assert pack.performance_targets_path() == (
        TenstorrentTargetPack().performance_targets_path()
    )
    assert pack.extra_spec_metadata_fields() == (
        TenstorrentTargetPack().extra_spec_metadata_fields()
    )


# --- agentic traces: template fallback for off-catalog models ----------------

_KIMI_TEMPLATE_MODEL_ID = "id_tt-transformers_Kimi-K2.7-Code_super_cluster"


def _synthesized_spec():
    return TenstorrentModelSpecProvider().synthesize(
        model_name="acme/tiny-llm",
        hf_model_repo="acme/tiny-llm",
        device="super_cluster",
        max_context=8192,
        max_concurrency=16,
    )


def test_agentic_traces_borrows_kimi_template_for_synthesized_spec():
    from reference_config.agentic_traces.agentic_traces_config import (
        AGENTIC_TRACES_CONFIGS,
        get_agentic_traces_config,
        get_agentic_traces_config_or_template,
    )

    spec = _synthesized_spec()
    # Strict lookup still refuses: the synthesized model_id has no entry.
    assert get_agentic_traces_config(spec) is None

    config = get_agentic_traces_config_or_template(spec)
    template = AGENTIC_TRACES_CONFIGS[_KIMI_TEMPLATE_MODEL_ID]
    assert config is not None
    assert config.model_id == spec.model_id  # retargeted, not Kimi's id
    assert config.runs == template.runs
    assert config.inferencex_git_ref == template.inferencex_git_ref


def test_agentic_traces_template_fallback_leaves_catalog_models_strict():
    from types import SimpleNamespace

    from reference_config.agentic_traces.agentic_traces_config import (
        get_agentic_traces_config_or_template,
    )

    # A catalog spec with no entry still gets None outside requirements mode:
    # a plain --workflow agentic_traces refuses rather than silently measuring
    # against a borrowed run shape. (pytest's argv has no --requirements-json.)
    spec = SimpleNamespace(
        model_id="id_tt-transformers_Some-Model_t3k",
        impl=SimpleNamespace(impl_id="tt-transformers"),
    )
    assert get_agentic_traces_config_or_template(spec) is None


def test_agentic_traces_borrows_template_for_catalog_spec_in_requirements_mode(
    monkeypatch,
):
    import sys
    from types import SimpleNamespace

    from reference_config.agentic_traces.agentic_traces_config import (
        get_agentic_traces_config_or_template,
    )

    # Being in the catalog says the model can be served, not that it is
    # onboarded to agentic traces -- a model added for evals has no entry. A
    # requirements document that asks for an agentic sweep has already said it
    # wants one, so it borrows rather than refusing.
    spec = SimpleNamespace(
        model_id="id_tt-transformers_Some-Model_t3k",
        impl=SimpleNamespace(impl_id="tt-transformers"),
    )
    monkeypatch.setattr(sys, "argv", ["run.py", "--requirements-json", "doc.json"])
    config = get_agentic_traces_config_or_template(spec)
    assert config is not None
    assert config.model_id == spec.model_id  # retargeted, not Kimi's id


def test_agentic_traces_template_fallback_preserves_own_entry():
    from types import SimpleNamespace

    from reference_config.agentic_traces.agentic_traces_config import (
        AGENTIC_TRACES_CONFIGS,
        get_agentic_traces_config_or_template,
    )

    spec = SimpleNamespace(model_id=_KIMI_TEMPLATE_MODEL_ID, impl=None)
    assert (
        get_agentic_traces_config_or_template(spec)
        is AGENTIC_TRACES_CONFIGS[_KIMI_TEMPLATE_MODEL_ID]
    )


def test_requirements_pack_agentic_traces_config_uses_template(pack):
    spec = _synthesized_spec()
    config = pack.agentic_traces_config(spec)
    assert config is not None
    assert config.model_id == spec.model_id


# --- agentic sweep -----------------------------------------------------------


def _agentic_pack(concurrencies, slo=None, sweep=None):
    """A pack whose document sweeps ``concurrencies`` for an agentic workload."""
    from workflow_module.requirements_schema import RequirementsDoc

    doc_dict = {
        "schemaVersion": "2.6.0",
        "document": {
            "id": "d",
            "model": {"name": "google/gemma-4-31B-it"},
            "deployment": {"hardware": "SC24"},
        },
        "workloads": [
            {
                "kind": "agentic",
                "id": "w1",
                "slo": slo or {},
                "agenticSweep": (
                    sweep
                    if sweep is not None
                    else [{"concurrency": c} for c in concurrencies]
                ),
            }
        ],
    }
    return RequirementsTargetPack(
        RequirementsDoc.from_dict(doc_dict), TenstorrentTargetPack()
    )


def _base_config():
    from reference_config.agentic_traces.agentic_traces_config import (
        AGENTIC_TRACES_CONFIGS,
        _REQUIREMENTS_TEMPLATE_MODEL_ID,
    )

    return AGENTIC_TRACES_CONFIGS[_REQUIREMENTS_TEMPLATE_MODEL_ID]


def test_replace_agentic_runs_sweeps_every_concurrency():
    from reference_config.agentic_traces.agentic_traces_config import (
        replace_agentic_runs,
    )

    base = _base_config()

    swept = replace_agentic_runs(base, [1, 8, 64])

    assert [r.concurrency for r in swept.runs] == [
        c for _ in base.runs for c in (1, 8, 64)
    ]


def test_replace_agentic_runs_leaves_config_alone_without_a_sweep():
    """A document with no agentic sweep keeps the catalog's operating point."""
    base = _base_config()
    from reference_config.agentic_traces.agentic_traces_config import (
        replace_agentic_runs,
    )

    assert replace_agentic_runs(base, []) is base


def test_agentic_concurrencies_are_deduplicated_and_ordered():
    pack = _agentic_pack([16, 1, 8, 1])

    assert pack._agentic_concurrencies() == [1, 8, 16]


def test_agentic_goodput_needs_slos():
    """goodputPct targets alone cannot be graded: nothing defines 'good'."""
    assert _agentic_pack([1])._agentic_goodput_by_concurrency() == {}


def test_agentic_goodput_uses_aiperf_tag_names():
    """AIPerf spells the bars out; vLLM's ttft/tpot/e2el keys are rejected."""
    pack = _agentic_pack(
        [],
        sweep=[
            {"concurrency": 1, "slo": {"ttftMs": 2000, "tpotMs": 20, "e2elMs": 20000}}
        ],
    )

    assert pack._agentic_goodput_by_concurrency() == {
        1: "time_to_first_token:2000 inter_token_latency:20 request_latency:20000"
    }


def test_vllm_goodput_keys_are_unchanged_by_the_aiperf_mapping():
    """The benchmark sweep keeps naming the bars after the metrics themselves."""
    (scenario,) = load_requirements(_FIXTURE).scenarios

    assert _vllm_bars(scenario.sweep[0], scenario) == "ttft:2000 tpot:20 e2el:20000"


def test_replace_agentic_runs_attaches_the_expected_sweep_to_every_run():
    """Every run carries the whole sweep, so the report can call out the
    points a truncated sweep never measured."""
    from reference_config.agentic_traces.agentic_traces_config import (
        replace_agentic_runs,
    )

    base = _base_config()
    sweep = [{"concurrency": 1, "ttftMeanMs": 800.0}, {"concurrency": 8}]

    swept = replace_agentic_runs(base, [1, 8], expected_sweep=sweep)

    assert all(run.expected_sweep == sweep for run in swept.runs)


def test_agentic_config_carries_the_documents_expected_sweep():
    """The pack hands the run specs the document's expected points, so the
    report can grade measured against expected field for field."""
    from types import SimpleNamespace

    sweep = [
        {"concurrency": 1, "ttftMeanMs": 800.0, "goodputPct": 90.0},
        {"concurrency": 8, "ttftMeanMs": 700.0, "goodputPct": 90.0},
    ]
    pack = _agentic_pack([], sweep=sweep)
    spec = SimpleNamespace(
        model_id="id_off-catalog",
        impl=SimpleNamespace(impl_id="requirements_synthesized"),
    )

    config = pack.agentic_traces_config(spec)

    assert config.runs
    assert all(run.expected_sweep == sweep for run in config.runs)


def test_agentic_expected_sweep_dedupes_first_workload_wins():
    """Two workloads sharing an operating point grade against the first,
    matching the concurrency dedupe that keeps the run from replaying twice."""
    sweep = [
        {"concurrency": 8, "ttftMeanMs": 700.0},
        {"concurrency": 1, "ttftMeanMs": 800.0},
        {"concurrency": 1, "ttftMeanMs": 999.0},
    ]
    pack = _agentic_pack([], sweep=sweep)

    assert pack._agentic_expected_sweep() == [
        {"concurrency": 1, "ttftMeanMs": 800.0},
        {"concurrency": 8, "ttftMeanMs": 700.0},
    ]


@pytest.mark.parametrize(
    "spelling",
    [
        "Tau^3-Banking Benchmark",
        "tau^3-banking benchmark",
        "  Tau^3-Banking   Benchmark  ",
        "Tau^3-Banking",
        "Tau3-Banking Benchmark",
        "tau3-banking",
        "Tau3-Bench Banking",
    ],
)
def test_tau3_banking_maps_to_its_catalog_task(spelling):
    """The document's human name has to reach a runnable catalog task.

    Without a mapping, ``unknown_eval_names`` rejects the whole document at
    parse time — the eval cannot be skipped, it aborts the run.
    """
    assert _EVAL_NAME_TO_TASK[_normalize_eval_name(spelling)] == ("tau3_bench_banking",)


def test_merged_document_evals_are_all_mapped():
    """Every accuracy eval the Kimi K2.7-Code document names must be known."""
    from workflow_module.requirements_schema import AccuracyEval, RequirementsDoc

    names = [
        "GPQA-Diamond",
        "Terminal-Bench 2.1",
        "SWE-bench Verified",
        "Tau^3-Banking Benchmark",
    ]
    doc = RequirementsDoc.from_dict(
        {
            "schemaVersion": "2.7.0",
            "model": {"name": "moonshotai/Kimi-K2.7-Code"},
            "accuracyEvals": [{"name": n, "gpuReferenceScore": 1.0} for n in names],
        }
    )
    assert isinstance(doc.accuracy_evals[0], AccuracyEval)
    assert unknown_eval_names(doc) == []


# --- per-point goodput -------------------------------------------------------


def _vllm_bars(point, scenario):
    """The bars in force at ``point``, in vLLM's spelling, via the real path."""
    from llm_module.goodput import VLLM_GOODPUT_KEYS, render_goodput
    from workflows.requirements_target_pack import _goodput_slo

    return render_goodput(
        _goodput_slo(point.effective_slo(scenario.slo)), VLLM_GOODPUT_KEYS
    )


def _per_point_pack(sweep, scenario_slo=None):
    from workflow_module.requirements_schema import RequirementsDoc

    doc = RequirementsDoc.from_dict(
        {
            "schemaVersion": "2.7.0",
            "id": "d",
            "model": {"name": "google/gemma-4-31B-it", "contextLength": 131072},
            "deployment": {"hardware": "SC24", "maxConcurrencyPerInstance": 32},
            "scenarios": [
                {
                    "kind": "text",
                    "id": "s1",
                    "oslValues": [128],
                    "slo": scenario_slo or {},
                    "sweep": sweep,
                }
            ],
        }
    )
    return RequirementsTargetPack(doc, TenstorrentTargetPack()), doc.scenarios[0]


def test_goodput_is_graded_per_point_not_per_scenario():
    """A row override must move the bars for its own point only."""
    _, scenario = _per_point_pack(
        [
            {"isl": 128, "osl": 128, "concurrency": 1},
            {"isl": 128, "osl": 128, "concurrency": 32, "slo": {"ttftMs": 9000}},
        ],
        scenario_slo={"ttftMs": 4100, "tpotMs": 22.2, "e2elMs": 10000},
    )

    assert [_vllm_bars(p, scenario) for p in scenario.sweep] == [
        "ttft:4100 tpot:22.2 e2el:10000",
        # tpot/e2el inherit; only ttft moved.
        "ttft:9000 tpot:22.2 e2el:10000",
    ]


def test_row_slos_apply_without_any_scenario_default():
    """A scenario can declare no SLOs and still have every row supply its own."""
    _, scenario = _per_point_pack(
        [
            {"isl": 128, "osl": 128, "concurrency": 1, "slo": {"ttftMs": 500}},
            {"isl": 128, "osl": 128, "concurrency": 32},
        ]
    )

    assert _vllm_bars(scenario.sweep[0], scenario) == "ttft:500"
    assert _vllm_bars(scenario.sweep[1], scenario) is None


def test_benchmark_params_carry_the_bars_tool_neutrally(doc):
    """One carried SLO, renderable into either tool's vocabulary."""
    from llm_module.goodput import (
        AIPERF_GOODPUT_KEYS,
        VLLM_GOODPUT_KEYS,
        render_goodput,
    )

    from dataclasses import replace

    scenario = doc.scenarios[0]
    rows = [
        replace(p, slo=Slo(ttft_ms=2000, tpot_ms=20, e2el_ms=20000))
        for p in scenario.sweep
    ]
    doc = replace(doc, scenarios=[replace(scenario, sweep=rows)])
    provider = RequirementsModelSpecProvider(TenstorrentModelSpecProvider(), doc)
    spec = provider.resolve(doc.model.name, "super_cluster")
    pack = RequirementsTargetPack(doc, TenstorrentTargetPack())
    points = pack.benchmark_config(spec).tasks[0].param_map[DeviceTypes.SUPER_CLUSTER]

    slo = points[0].goodput
    assert render_goodput(slo, VLLM_GOODPUT_KEYS) == "ttft:2000 tpot:20 e2el:20000"
    assert render_goodput(slo, AIPERF_GOODPUT_KEYS) == (
        "time_to_first_token:2000 inter_token_latency:20 request_latency:20000"
    )


def test_duplicate_sweep_shapes_are_flagged(caplog):
    """Downstream configs are keyed by shape, so a duplicate is last-wins."""
    pack, scenario = _per_point_pack(
        [
            {"isl": 128, "osl": 128, "concurrency": 1, "slo": {"ttftMs": 500}},
            {"isl": 128, "osl": 128, "concurrency": 1, "slo": {"ttftMs": 900}},
        ]
    )
    from workflows.requirements_target_pack import _warn_on_duplicate_shapes

    with caplog.at_level("WARNING"):
        _warn_on_duplicate_shapes(scenario)

    assert "same (isl=128, osl=128, concurrency=1)" in caplog.text


# --- agentic per-concurrency goodput ----------------------------------------


def test_agentic_rows_do_not_inherit_the_workload_slo(caplog):
    """An agentic document targets goodput per operating point, not service-wide.

    A row that states no SLOs of its own gets no bars, even when another row
    in the same workload does -- inheriting a workload default would grade
    that point against a contract the document never made for it.
    """
    pack = _agentic_pack(
        [],
        slo={"ttftMs": 9999, "tpotMs": 99, "e2elMs": 99999},
        sweep=[
            {"concurrency": 1, "slo": {"ttftMs": 1000, "tpotMs": 10, "e2elMs": 20000}},
            {"concurrency": 16},
        ],
    )

    with caplog.at_level("WARNING"):
        result = pack._agentic_goodput_by_concurrency()

    assert result == {
        1: "time_to_first_token:1000 inter_token_latency:10 request_latency:20000"
    }
    assert 16 not in result


def test_agentic_workload_slo_without_row_slos_is_ignored_loudly(caplog):
    """Bars put only at workload level are the old broadcast bug; say so."""
    pack = _agentic_pack(
        [],
        slo={"ttftMs": 1000, "tpotMs": 10, "e2elMs": 20000},
        sweep=[{"concurrency": 1}, {"concurrency": 16}],
    )

    with caplog.at_level("WARNING"):
        result = pack._agentic_goodput_by_concurrency()

    assert result == {}
    assert "graded per operating point" in caplog.text


def test_agentic_goodput_omits_concurrencies_with_no_bars():
    """No bars anywhere => leave that run on its own run spec's goodput."""
    pack = _agentic_pack(
        [], sweep=[{"concurrency": 1, "slo": {"ttftMs": 900}}, {"concurrency": 8}]
    )

    assert pack._agentic_goodput_by_concurrency() == {1: "time_to_first_token:900"}


def test_agentic_goodput_collision_warns_and_keeps_the_first(caplog):
    """An SLO set is a contract, not a lattice: do not reconcile, do say so."""
    from workflow_module.requirements_schema import RequirementsDoc

    doc = RequirementsDoc.from_dict(
        {
            "schemaVersion": "2.7.0",
            "id": "d",
            "model": {"name": "google/gemma-4-31B-it"},
            "deployment": {"hardware": "SC24"},
            "workloads": [
                {
                    "kind": "agentic",
                    "id": "w1",
                    "agenticSweep": [{"concurrency": 8, "slo": {"ttftMs": 1000}}],
                },
                {
                    "kind": "agentic",
                    "id": "w2",
                    "agenticSweep": [{"concurrency": 8, "slo": {"ttftMs": 4000}}],
                },
            ],
        }
    )
    pack = RequirementsTargetPack(doc, TenstorrentTargetPack())

    with caplog.at_level("WARNING"):
        result = pack._agentic_goodput_by_concurrency()

    assert result == {8: "time_to_first_token:1000"}
    assert "'w1' and 'w2'" in caplog.text
    assert "concurrency 8" in caplog.text


def test_replace_agentic_runs_accepts_a_per_concurrency_mapping():
    from reference_config.agentic_traces.agentic_traces_config import (
        replace_agentic_runs,
    )

    base = _base_config()
    swept = replace_agentic_runs(base, [1, 8], goodput={1: "a:1", 8: "b:2"})

    assert {(r.concurrency, r.goodput) for r in swept.runs} == {(1, "a:1"), (8, "b:2")}


def test_replace_agentic_runs_leaves_unmapped_concurrencies_alone():
    """A concurrency the mapping omits keeps its run spec's own goodput."""
    from reference_config.agentic_traces.agentic_traces_config import (
        replace_agentic_runs,
    )

    base = _base_config()
    swept = replace_agentic_runs(base, [1, 8], goodput={8: "b:2"})

    by_concurrency = {r.concurrency: r.goodput for r in swept.runs}
    assert by_concurrency[8] == "b:2"
    assert by_concurrency[1] == base.runs[0].goodput


def test_replace_agentic_runs_still_broadcasts_a_plain_string():
    """The single-SLO-set form stays supported for catalog callers."""
    from reference_config.agentic_traces.agentic_traces_config import (
        replace_agentic_runs,
    )

    swept = replace_agentic_runs(_base_config(), [1, 8], goodput="a:1")

    assert {r.goodput for r in swept.runs} == {"a:1"}


def test_harness_overrides_do_not_mutate_the_borrowed_config(doc):
    """A borrowed harness config belongs to the process-wide eval catalog.

    dataclasses.replace is shallow, so the template's agentic_eval_config is
    that shared object: re-pointing it in place would change the donor's own
    task for every later lookup in the process.
    """
    from dataclasses import dataclass, field
    from typing import Dict

    @dataclass(frozen=True)
    class FakeHarness:
        n_concurrent_trials: int = 4
        environment_env: Dict[str, str] = field(default_factory=dict)
        verifier_env: Dict[str, str] = field(default_factory=dict)

    @dataclass(frozen=True)
    class FakeTask:
        agentic_eval_config: FakeHarness

    shared = FakeHarness(
        environment_env={"TAU2_USER_MODEL": "openai/donor/Donor-1"},
        verifier_env={"TAU2_NL_ASSERTIONS_MODEL": "openai/donor/Donor-1"},
    )
    pack = RequirementsTargetPack(doc, TenstorrentTargetPack())

    overrides = pack._harness_overrides(
        FakeTask(shared), "tau3_bench_banking", doc.accuracy_evals[0]
    )

    fixed = overrides["agentic_eval_config"]
    want = f"openai/{doc.model.name}"
    assert fixed.environment_env["TAU2_USER_MODEL"] == want
    assert fixed.verifier_env["TAU2_NL_ASSERTIONS_MODEL"] == want
    assert fixed.n_concurrent_trials == doc.deployment.max_concurrency_per_instance
    # The donor's own config is untouched.
    assert shared.environment_env == {"TAU2_USER_MODEL": "openai/donor/Donor-1"}
    assert shared.verifier_env == {"TAU2_NL_ASSERTIONS_MODEL": "openai/donor/Donor-1"}
