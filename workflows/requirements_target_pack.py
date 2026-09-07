# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Requirements-driven adapters over the Tenstorrent validation content.

An LLM-serving requirements document (parsed by
``workflow_module.requirements_schema``) declares *what* to validate and the
targets that gate it. These adapters map that declaration onto the concrete
Tenstorrent content via the engine seams, wrapping the stock Tenstorrent
implementations so anything the document does not specify falls through to the
catalog:

- :class:`RequirementsModelSpecProvider` — resolves the document's model from
  the catalog when present, otherwise synthesizes an off-catalog spec from the
  document's model + deployment metadata.
- :class:`RequirementsTargetPack` — builds the eval config (from the document's
  ``accuracyEvals``, borrowing each task's runnable harness definition from the
  catalog and re-gating it with the document's reference score/tolerance) and
  the benchmark config (the document's sweep points + scalar targets / SLOs).
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, List, Mapping, Optional

from workflow_module.model_catalog import ModelSpecProvider
from workflow_module.requirements_schema import (
    PRIORITY_MUST,
    PRIORITY_SHOULD,
    AccuracyEval,
    RequirementsDoc,
    Scenario,
)
from workflow_module.target_pack import TargetPack

logger = logging.getLogger(__name__)

# Human-facing accuracy-eval names (from the document) -> catalog task_name.
# Matched case-insensitively after stripping whitespace. Extend as new evals
# appear in requirement documents.
_EVAL_NAME_TO_TASK = {
    "gpqa-diamond": "gpqa_diamond_cot_zeroshot",
    "gpqa diamond": "gpqa_diamond_cot_zeroshot",
    "swe-bench verified": "swe_bench_verified",
    "swe-bench-verified": "swe_bench_verified",
    "terminal-bench 2.0": "terminal_bench_2",
    "terminal-bench 2": "terminal_bench_2",
    "terminal-bench 2.1": "terminal_bench_2_1",
}

# Scenario scalar-target metric -> PerformanceTarget attribute. Only these
# aggregate metrics are graded; others are ignored with a warning.
_SCALAR_METRIC_TO_ATTR = {
    "system_throughput": "tput_total",
    "request_goodput": "goodput",
}

# Sweep-point reference-measurement key -> PerformanceTarget attribute. These
# are the document's per-point expectations (measured on the customer's
# reference stack); each gates its own sweep point. Keys without a graded
# counterpart (reqThroughputRps, kvCacheHitRatePct, p50/p99 latencies) stay
# provenance-only in SweepPoint.reference.
_REFERENCE_KEY_TO_ATTR = {
    "ttftMeanMs": "ttft_ms",
    "tpotMs": "tpot_ms",
    "e2elMs": "e2el_ms",
    "decodeThroughputTps": "tput",
    "totalThroughputTps": "tput_total",
    "goodputPct": "goodput",
}

# The requirements schema declares no benchmark tolerance, so every
# requirements-driven target — per-point reference or scenario-level gate —
# grades exact (tolerance 0), matching the document's gte/lte comparators.
# (Evals are separate: accuracyEvals carry their own tolerance field.)

# Minimal harness profiles for known evals, used only when NO catalog model
# defines the task (fully off-catalog model). A profile carries just the
# task-specific scoring wiring; everything model-specific stays neutral —
# EvalTask defaults supply the rest (EVALS_COMMON venv, gen_kwargs
# {"stream": "False"}, smoke limit), and max_length comes from the document's
# model.contextLength. Tasks backed by a harness config (SWE-bench,
# Terminal-Bench) have no profile: they cannot be synthesized and always
# require a catalog template.
_TASK_PROFILES = {
    "gpqa_diamond_cot_zeroshot": {
        "score_func_kwargs": {
            "result_keys": ["exact_match,flexible-extract"],
            "unit": "percent",
        },
    },
}

# Requests per sweep point = concurrency * this multiple, floored, so each
# point issues enough requests to characterize steady state.
_NUM_PROMPTS_CONCURRENCY_MULTIPLE = 8
_MIN_NUM_PROMPTS = 16


def _normalize_eval_name(name: str) -> str:
    return " ".join(name.strip().lower().split())


def unknown_eval_names(doc: RequirementsDoc) -> List[str]:
    """Accuracy-eval names in the document with no known catalog task mapping.

    Used by the CLI entry points to reject a mistyped/unsupported eval at
    parse time instead of mid-run when the eval config is built.
    """
    return [
        ae.name
        for ae in doc.accuracy_evals
        if _normalize_eval_name(ae.name) not in _EVAL_NAME_TO_TASK
    ]


def _num_prompts_for(concurrency: int) -> int:
    return max(_MIN_NUM_PROMPTS, concurrency * _NUM_PROMPTS_CONCURRENCY_MULTIPLE)


class RequirementsModelSpecProvider(ModelSpecProvider):
    """Model-spec provider that synthesizes off-catalog specs from a document.

    Delegates to the wrapped Tenstorrent provider for catalog models; when the
    catalog has no entry for the requested ``(model, device)`` it synthesizes a
    spec from the requirements document's model context length and the
    deployment's per-instance concurrency.
    """

    def __init__(self, delegate: Any, doc: RequirementsDoc) -> None:
        self._delegate = delegate
        self._doc = doc

    def model_names(self) -> List[str]:
        return self._delegate.model_names()

    def resolve(self, model: str, device: str) -> Any:
        try:
            return self._delegate.resolve(model=model, device=device)
        except ValueError:
            logger.info(
                "Model %r not in catalog for device %r; synthesizing spec from "
                "requirements document.",
                model,
                device,
            )
            return self._synthesize(model, device)

    def resolve_candidates(self, model: str, device: str) -> List[Any]:
        # Pure data access (no fallback policy): an off-catalog model has no
        # catalog candidates by definition — resolve() already synthesizes a
        # spec for it without raising, so callers only reach this when the
        # model is genuinely unknown.
        return self._delegate.resolve_candidates(model=model, device=device)

    def load_runtime_spec(self, path: str) -> Optional[Any]:
        return self._delegate.load_runtime_spec(path)

    def synthesize(self, **kwargs: Any) -> Any:
        return self._delegate.synthesize(**kwargs)

    def _synthesize(self, model: str, device: str) -> Any:
        context_length = self._doc.model.context_length
        if not context_length:
            raise ValueError(
                f"Cannot synthesize spec for off-catalog model {model!r}: the "
                "requirements document has no model.contextLength."
            )
        max_concurrency = self._doc.deployment.max_concurrency_per_instance
        if not max_concurrency:
            raise ValueError(
                f"Cannot synthesize spec for off-catalog model {model!r}: the "
                "requirements document has no deployment.maxConcurrencyPerInstance."
            )
        return self._delegate.synthesize(
            model_name=model,
            hf_model_repo=self._doc.model.name,
            device=device,
            max_context=context_length,
            max_concurrency=max_concurrency,
        )


class RequirementsTargetPack(TargetPack):
    """Target pack whose eval + benchmark content comes from a requirements doc.

    Everything the document does not define (agentic traces, reference-target
    paths, report-metadata policy, score acceptance math) is delegated to the
    wrapped Tenstorrent target pack.
    """

    def __init__(self, doc: RequirementsDoc, delegate: Any) -> None:
        self._doc = doc
        self._delegate = delegate

    # --- eval configs ---
    def eval_config(self, hf_model_repo: str) -> Optional[Any]:
        from reference_config.evals.eval_config import EvalConfig

        if not self._doc.accuracy_evals:
            return None
        tasks = [self._build_eval_task(ae) for ae in self._doc.accuracy_evals]
        return EvalConfig(hf_model_repo=self._doc.model.name, tasks=tasks)

    def _build_eval_task(self, ae: AccuracyEval) -> Any:
        task_name = _EVAL_NAME_TO_TASK.get(_normalize_eval_name(ae.name))
        if task_name is None:
            available = sorted(set(_EVAL_NAME_TO_TASK))
            raise ValueError(
                f"Requirements accuracy eval {ae.name!r} has no known catalog "
                f"task mapping. Known eval names: {available}."
            )
        template = self._find_task_template(task_name)
        if template is None:
            return self._synthesize_eval_task(ae, task_name)
        if template.score is None:
            raise ValueError(
                f"Catalog template for task {task_name!r} has no score definition; "
                "cannot re-gate it from the requirements document."
            )
        new_score = replace(
            template.score,
            gpu_reference_score=(
                ae.gpu_reference_score
                if ae.gpu_reference_score is not None
                else template.score.gpu_reference_score
            ),
            gpu_reference_score_ref=(
                ae.published_score_url or f"requirements:{self._doc.id}"
            ),
            published_score=(
                ae.published_score
                if ae.published_score is not None
                else template.score.published_score
            ),
            published_score_ref=(
                ae.published_score_url or template.score.published_score_ref
            ),
            tolerance=ae.tolerance,
        )
        return replace(
            template,
            score=new_score,
            priority=ae.priority,
            **self._harness_concurrency_overrides(template, task_name),
        )

    def _harness_concurrency_overrides(
        self, template: Any, task_name: str
    ) -> Mapping[str, Any]:
        """Re-point a borrowed harness config at the document's concurrency.

        The template is borrowed from whichever catalog model happens to define
        the task, so its ``n_concurrent_trials`` describes *that* model's
        deployment -- and which model is borrowed is decided by catalog
        iteration order, so inheriting it would make the trial count arbitrary.
        ``deployment.maxConcurrencyPerInstance`` is the document's own statement
        of what the instance under test serves concurrently, so it is the
        honest trial count here.

        Deliberately unclamped: the document is authoritative about the
        deployment. A trial count the host cannot afford is a property of the
        document, not something to silently correct.
        """
        concurrency = self._doc.deployment.max_concurrency_per_instance
        if not concurrency:
            return {}
        overrides = {}
        for field_name in ("agentic_eval_config", "swebench_eval_config"):
            cfg = getattr(template, field_name, None)
            if cfg is None or cfg.n_concurrent_trials == concurrency:
                continue
            logger.info(
                "Task %s: overriding borrowed n_concurrent_trials %s -> %s from "
                "the requirements document's deployment.maxConcurrencyPerInstance.",
                task_name,
                cfg.n_concurrent_trials,
                concurrency,
            )
            overrides[field_name] = replace(cfg, n_concurrent_trials=concurrency)
        return overrides

    def _find_task_template(self, task_name: str) -> Optional[Any]:
        """Borrow a runnable EvalTask for ``task_name`` from the catalog.

        Prefer the document model's own catalog entry (so any model-specific
        harness tuning is preserved), then fall back to any model that defines
        the task.
        """
        from reference_config.evals.eval_config import EVAL_CONFIGS

        preferred = EVAL_CONFIGS.get(self._doc.model.name)
        if preferred is not None:
            for task in preferred.tasks:
                if task.task_name == task_name:
                    return task
        for cfg in EVAL_CONFIGS.values():
            for task in cfg.tasks:
                if task.task_name == task_name:
                    return task
        return None

    def _synthesize_eval_task(self, ae: AccuracyEval, task_name: str) -> Any:
        """Build a neutral EvalTask for a known eval with no catalog template.

        Used for fully off-catalog models: the task-specific scoring wiring
        comes from ``_TASK_PROFILES``, ``max_length`` from the document's
        ``model.contextLength``, and everything else from ``EvalTask``'s
        neutral defaults (no sampling overrides — the server/model defaults
        apply). Scores, tolerance, and priority are gated by the document.
        """
        from reference_config.evals.eval_config import (
            EvalTask,
            EvalTaskScore,
            score_task_single_key,
        )

        profile = _TASK_PROFILES.get(task_name)
        if profile is None:
            raise ValueError(
                f"No catalog template or built-in profile for task {task_name!r} "
                f"(requirements eval {ae.name!r}). Harness-backed tasks "
                "(SWE-bench, Terminal-Bench) require a catalog template; plain "
                f"lm-eval tasks need a _TASK_PROFILES entry. Known profiles: "
                f"{sorted(_TASK_PROFILES)}."
            )
        model_kwargs: dict = {"timeout": "3600"}
        if self._doc.model.context_length:
            model_kwargs["max_length"] = self._doc.model.context_length
        logger.warning(
            "No catalog template for task %r; synthesizing with neutral "
            "defaults (max_length=%s from model.contextLength, streaming on, "
            "no sampling overrides).",
            task_name,
            model_kwargs.get("max_length"),
        )
        return EvalTask(
            task_name=task_name,
            # Streaming on by default for synthesized tasks (long generations
            # against an OpenAI-compatible server); no sampling overrides.
            gen_kwargs={"stream": "True"},
            score=EvalTaskScore(
                published_score=(
                    ae.published_score
                    if ae.published_score is not None
                    else (ae.gpu_reference_score or 0.0)
                ),
                published_score_ref=(
                    ae.published_score_url or f"requirements:{self._doc.id}"
                ),
                gpu_reference_score=ae.gpu_reference_score,
                gpu_reference_score_ref=(
                    ae.published_score_url or f"requirements:{self._doc.id}"
                ),
                tolerance=ae.tolerance,
                score_func=score_task_single_key,
                score_func_kwargs=dict(profile["score_func_kwargs"]),
            ),
            model_kwargs=model_kwargs,
            priority=ae.priority,
        )

    def resolve_eval_reference(self, score: Any, limit_mode: Any) -> Mapping[str, Any]:
        return self._delegate.resolve_eval_reference(score, limit_mode)

    def accept_eval_score(
        self,
        ref: Mapping[str, Any],
        score: float,
        n_total: Optional[int] = None,
    ) -> Optional[bool]:
        return self._delegate.accept_eval_score(ref, score, n_total=n_total)

    def resolve_eval_task_for_device(self, task: Any, device: Any) -> Any:
        return self._delegate.resolve_eval_task_for_device(task, device)

    # --- benchmark configs ---
    def benchmark_config(self, model_spec: Any) -> Any:
        from reference_config.benchmarking.benchmark_config import (
            select_vllm_benchmark_venv,
        )
        from reference_config.benchmarking.benchmark_config import (
            BenchmarkConfig,
            BenchmarkTask,
        )

        device = model_spec.device_type
        params = []
        for scenario in self._doc.scenarios:
            if scenario.kind and scenario.kind != "text":
                logger.info(
                    "Skipping non-text scenario %r (kind=%s) for LLM benchmark.",
                    scenario.id,
                    scenario.kind,
                )
                continue
            params.extend(self._scenario_params(scenario))

        task = BenchmarkTask(
            param_map={device: params},
            workflow_venv_type=select_vllm_benchmark_venv(model_spec),
        )
        return BenchmarkConfig(model_id=model_spec.model_id, tasks=[task])

    def _scenario_params(self, scenario: Scenario) -> List[Any]:
        from workflows.utils_report import BenchmarkTaskParams, PerformanceTarget

        if not scenario.sweep:
            return []
        goodput_constraints = _goodput_constraints(scenario)
        if goodput_constraints is None and _scenario_targets_goodput(scenario):
            logger.warning(
                "Scenario %r declares goodput expectations but no SLOs; "
                "goodput is only measured when SLOs provide the --goodput "
                "constraints, so those targets will grade as NA.",
                scenario.id,
            )

        # Scenario-level gates (SLOs, scalar targets) are *capability* gates:
        # each attaches to the single sweep point whose reference measurement
        # is best for that metric — the document asserts the target is
        # reachable at the system's best operating point within the sweep
        # envelope. Broadcasting them to every point would contradict the
        # document's own references (latency SLOs only hold at low load; the
        # throughput target only at high ISL).
        attach = _capability_attach_points(scenario, _scenario_level_gates(scenario))

        params: List[Any] = []
        for idx, point in enumerate(scenario.sweep):
            tier_kwargs: dict = {}
            target_priorities: dict = {}

            # The point's own reference measurements gate it (must): they are
            # the document's statement of what the reference stack achieves at
            # exactly this (ISL, OSL, concurrency).
            for key, attr in _REFERENCE_KEY_TO_ATTR.items():
                value = (point.reference or {}).get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                tier_kwargs[attr] = float(value)
                target_priorities[attr] = PRIORITY_MUST

            # A scenario-level gate attached here overrides the point's
            # reference value for the same metric (the target is contractual;
            # the reference is the incumbent's measurement).
            for attr, (value, priority) in attach.get(idx, {}).items():
                tier_kwargs[attr] = value
                target_priorities[attr] = priority

            targets = (
                {"target": PerformanceTarget(tolerance=0.0, **tier_kwargs)}
                if tier_kwargs
                else {}
            )
            params.append(
                BenchmarkTaskParams(
                    isl=point.isl,
                    osl=point.osl,
                    max_concurrency=point.concurrency,
                    num_prompts=_num_prompts_for(point.concurrency),
                    task_type="text",
                    targets=targets,
                    priority=_aggregate_priority(list(target_priorities.values())),
                    target_priorities=target_priorities or None,
                    goodput=goodput_constraints,
                )
            )
        return params

    def smoke_test_benchmark_config(self, config: Any, device: Any) -> Any:
        from reference_config.benchmarking.benchmark_config import (
            BenchmarkConfig,
            BenchmarkTask,
        )

        for task in config.tasks:
            points = task.param_map.get(device) or []
            if points:
                return BenchmarkConfig(
                    model_id=config.model_id,
                    tasks=[
                        BenchmarkTask(
                            param_map={device: [points[0]]},
                            workflow_venv_type=task.workflow_venv_type,
                        )
                    ],
                )
        return config

    # --- agentic traces (delegated) ---
    def agentic_traces_config(self, model_spec: Any) -> Optional[Any]:
        return self._delegate.agentic_traces_config(model_spec)

    def resolve_agentic_run_specs(
        self,
        config: Any,
        *,
        trace_sources: Any = None,
        git_ref_override: Optional[str] = None,
    ) -> Any:
        return self._delegate.resolve_agentic_run_specs(
            config, trace_sources=trace_sources, git_ref_override=git_ref_override
        )

    def agentic_traces_min_profile_seconds(self) -> int:
        return self._delegate.agentic_traces_min_profile_seconds()

    # --- measured reference data (delegated) ---
    def performance_targets_path(self):
        return self._delegate.performance_targets_path()

    def accuracy_targets_path(self):
        return self._delegate.accuracy_targets_path()

    # --- report metadata (delegated) ---
    def extra_spec_metadata_fields(self):
        return self._delegate.extra_spec_metadata_fields()


def _aggregate_priority(priorities: List[str]) -> Optional[str]:
    """A sweep point is ``must`` if any of its targets is must, else ``should``."""
    if not priorities:
        return None
    return PRIORITY_MUST if PRIORITY_MUST in priorities else PRIORITY_SHOULD


def _scenario_level_gates(scenario: Scenario) -> dict:
    """Scenario-level gates: ``{PerformanceTarget attr: (value, priority, lower_is_better)}``."""
    gates: dict = {}
    slo = scenario.slo
    if slo is not None:
        for attr, value in (
            ("ttft_ms", slo.ttft_ms),
            ("tpot_ms", slo.tpot_ms),
            ("e2el_ms", slo.e2el_ms),
        ):
            if value is not None:
                gates[attr] = (value, PRIORITY_MUST, True)
    for st in scenario.scalar_targets:
        attr = _SCALAR_METRIC_TO_ATTR.get(st.metric)
        if attr is None:
            logger.warning(
                "Ignoring unsupported scalar target metric %r in scenario %r.",
                st.metric,
                scenario.id,
            )
            continue
        gates[attr] = (st.target, st.priority, False)
    return gates


def _capability_attach_points(scenario: Scenario, gates: dict) -> dict:
    """Map each scenario-level gate to its capability point: ``{sweep index: {attr: (value, priority)}}``.

    The capability point for a metric is the sweep point whose reference
    measurement is best for it (min for latency SLOs, max for throughput /
    goodput percentages). A metric with no reference data anywhere attaches
    at the least-loaded point, the most charitable operating point.
    """
    attach: dict = {}
    if not gates:
        return attach
    ref_key = {attr: key for key, attr in _REFERENCE_KEY_TO_ATTR.items()}
    fallback = min(
        range(len(scenario.sweep)),
        key=lambda i: (
            scenario.sweep[i].concurrency,
            scenario.sweep[i].isl,
            scenario.sweep[i].osl,
        ),
    )
    for attr, (value, priority, lower_is_better) in gates.items():
        key = ref_key.get(attr)
        best_idx = None
        best_val = None
        if key is not None:
            for i, point in enumerate(scenario.sweep):
                ref = (point.reference or {}).get(key)
                if isinstance(ref, bool) or not isinstance(ref, (int, float)):
                    continue
                if best_val is None or (
                    ref < best_val if lower_is_better else ref > best_val
                ):
                    best_idx, best_val = i, ref
        attach.setdefault(best_idx if best_idx is not None else fallback, {})[attr] = (
            value,
            priority,
        )
    return attach


def _goodput_constraints(scenario: Scenario) -> Optional[str]:
    """``vllm bench serve --goodput`` constraint string from the scenario's SLOs.

    vLLM's keys are ttft/tpot/e2el in milliseconds — exactly the document's
    SLO metrics. Returns None when the scenario declares no SLOs, in which
    case goodput cannot be measured.
    """
    slo = scenario.slo
    if slo is None:
        return None
    parts = []
    for key, value in (
        ("ttft", slo.ttft_ms),
        ("tpot", slo.tpot_ms),
        ("e2el", slo.e2el_ms),
    ):
        if value is not None:
            parts.append(f"{key}:{value:g}")
    return " ".join(parts) or None


def _scenario_targets_goodput(scenario: Scenario) -> bool:
    """True if the document expresses any goodput expectation for the scenario."""
    if any(st.metric == "request_goodput" for st in scenario.scalar_targets):
        return True
    return any(
        isinstance((p.reference or {}).get("goodputPct"), (int, float))
        and not isinstance((p.reference or {}).get("goodputPct"), bool)
        for p in scenario.sweep
    )


__all__ = [
    "RequirementsModelSpecProvider",
    "RequirementsTargetPack",
    "unknown_eval_names",
]
