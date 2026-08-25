# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Requirements-driven adapters over the Tenstorrent validation content.

A Blaze customer-requirements document (parsed by
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

# Scenario scalar-target metric -> (PerformanceTarget attribute, lower_is_better).
# Only these aggregate metrics are graded; others are ignored with a warning.
_SCALAR_METRIC_TO_ATTR = {
    "system_throughput": "tput",
    "request_goodput": "goodput",
}

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
    def eval_config(self, model_name: str) -> Optional[Any]:
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
        peak_concurrency = max(p.concurrency for p in scenario.sweep)

        params: List[Any] = []
        for point in scenario.sweep:
            tier_kwargs: dict = {}
            priorities: List[str] = []

            slo = scenario.slo
            if slo is not None:
                # Per-request SLOs apply to every sweep point (default must).
                if slo.ttft_ms is not None:
                    tier_kwargs["ttft_ms"] = slo.ttft_ms
                    priorities.append(PRIORITY_MUST)
                if slo.tpot_ms is not None:
                    tier_kwargs["tpot_ms"] = slo.tpot_ms
                    priorities.append(PRIORITY_MUST)
                if slo.e2el_ms is not None:
                    tier_kwargs["e2el_ms"] = slo.e2el_ms
                    priorities.append(PRIORITY_MUST)

            # Aggregate scalar targets only make sense at the peak-throughput
            # operating point, so attach them to the max-concurrency point.
            if point.concurrency == peak_concurrency:
                for st in scenario.scalar_targets:
                    attr = _SCALAR_METRIC_TO_ATTR.get(st.metric)
                    if attr is None:
                        logger.warning(
                            "Ignoring unsupported scalar target metric %r in "
                            "scenario %r.",
                            st.metric,
                            scenario.id,
                        )
                        continue
                    tier_kwargs[attr] = st.target
                    priorities.append(st.priority)

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
                    priority=_aggregate_priority(priorities),
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


__all__ = ["RequirementsModelSpecProvider", "RequirementsTargetPack"]
