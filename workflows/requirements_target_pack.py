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
  catalog, re-gating it with the document's reference score/tolerance and
  measuring it under the document's ``genKwargs``) and the benchmark config
  (the document's sweep points + scalar targets / SLOs).
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from llm_module.goodput import AIPERF_GOODPUT_KEYS, GoodputSlo, render_goodput
from workflow_module.model_catalog import ModelSpecProvider
from workflow_module.requirements_schema import (
    PRIORITY_MUST,
    PRIORITY_SHOULD,
    AccuracyEval,
    RequirementsDoc,
    Scenario,
    Slo,
)
from workflow_module.target_pack import TargetPack

logger = logging.getLogger(__name__)

# Human-facing accuracy-eval names (from the document) -> catalog task names,
# most preferred first; matched case-insensitively after stripping whitespace.
# A tuple, not a single name, because the same benchmark is configured under
# several task names per model family (:meth:`_find_task_template` prefers
# whichever the document's model configures). Tau^3 spellings are listed out
# because ``_normalize_eval_name`` deliberately doesn't fold "^" or "benchmark".
_EVAL_NAME_TO_TASK = {
    "gpqa-diamond": ("gpqa_diamond_cot_zeroshot", "r1_gpqa_diamond"),
    "gpqa diamond": ("gpqa_diamond_cot_zeroshot", "r1_gpqa_diamond"),
    "swe-bench verified": ("swe_bench_verified",),
    "swe-bench-verified": ("swe_bench_verified",),
    "terminal-bench 2.0": ("terminal_bench_2",),
    "terminal-bench 2": ("terminal_bench_2",),
    "terminal-bench 2.1": ("terminal_bench_2_1",),
    "tau^3-banking benchmark": ("tau3_bench_banking",),
    "tau^3-banking": ("tau3_bench_banking",),
    "tau3-banking benchmark": ("tau3_bench_banking",),
    "tau3-banking": ("tau3_bench_banking",),
    "tau3-bench banking": ("tau3_bench_banking",),
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


_HARNESS_MODEL_ENV_KEYS = ("TAU2_USER_MODEL", "TAU2_NL_ASSERTIONS_MODEL")

# Sampling parameters that can be re-pointed inside a borrowed agentic config.
#
# The token budget is deliberately not among them. Its agentic spelling is
# ``max_tokens``, and it is one half of a coupled constraint -- an agent sends
# roughly ``max_input_tokens + max_output_tokens`` per request and the sum must
# stay under the server's max_context -- so moving one number in isolation
# turns a working budget into 400s mid-trial. An agent's budget is the agent's,
# not a per-answer sampling choice.
_AGENTIC_SAMPLING_KEYS = ("temperature", "top_p", "top_k", "reasoning_effort")

# Agent kwargs whose value is a JSON *string* rather than a mapping, and which
# therefore has to be parsed to be re-pointed. tau3 is the case: its adapter's
# ``build_llm_args()`` sets only temperature, so every other parameter reaches
# LiteLLM through this argument -- and it must stay a string, because the
# adapter shlex-quotes the value onto the container command line, where a dict
# raises TypeError.
_JSON_STRING_ARG_SUFFIX = "_json"

# LiteLLM selects its OpenAI-compatible provider on this prefix; the remainder
# is the model id sent to the server (see llm_module/drivers/agentic.py).
_LITELLM_OPENAI_PREFIX = "openai/"


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
        candidates = _EVAL_NAME_TO_TASK.get(_normalize_eval_name(ae.name))
        if candidates is None:
            available = sorted(set(_EVAL_NAME_TO_TASK))
            raise ValueError(
                f"Requirements accuracy eval {ae.name!r} has no known catalog "
                f"task mapping. Known eval names: {available}."
            )
        template, task_name = self._find_task_template(candidates)
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
            **self._gen_kwargs_overrides(template, task_name, ae),
            **self._harness_overrides(template, task_name, ae),
        )

    def _gen_kwargs_overrides(
        self, template: Any, task_name: str, ae: AccuracyEval
    ) -> Mapping[str, Any]:
        """The gen_kwargs a borrowed template runs with: the document's, then streaming.

        The document's ``genKwargs`` wins, per parameter and only per
        parameter, over the borrowed template's own tuned sampling (token
        budget, stop list, do_sample) so a document stating a temperature
        doesn't discard a budget it never chose.

        Then streaming, so a gateway in front of the deployed endpoint (e.g.
        Cloudflare's ~100s window) can't time out a long chain-of-thought
        answer before it starts responding. ``stream`` is deliberately not a
        document parameter: which side of a gateway the run sits on is the
        runner's knowledge, not the customer's contract.

        Harbor-backed tasks (agentic benchmarks) never read gen_kwargs -- their
        requests come from the agent via LiteLLM, not lm-eval; sampling is
        re-pointed in the agent's own config by
        :meth:`_repoint_agent_sampling`.
        """
        if getattr(template, "agentic_eval_config", None) is not None:
            return {}
        stated = ae.gen_kwargs.stated() if ae.gen_kwargs else {}
        original = dict(getattr(template, "gen_kwargs", None) or {})
        gen_kwargs = dict(original)
        if stated:
            logger.info(
                "Task %s: measuring under the document's genKwargs %r; the "
                "borrowed template's other parameters are unchanged.",
                task_name,
                stated,
            )
            gen_kwargs.update(stated)
        if str(gen_kwargs.get("stream", "")).lower() != "true":
            logger.info(
                "Task %s: streaming its generations, so a gateway cannot time "
                "out the request before the answer is complete.",
                task_name,
            )
            gen_kwargs["stream"] = "true"
        if gen_kwargs == original:
            return {}
        return {"gen_kwargs": gen_kwargs}

    def _harness_overrides(
        self, template: Any, task_name: str, ae: AccuracyEval
    ) -> Mapping[str, Any]:
        """Re-point a borrowed harness config at the document's own deployment.

        The template is borrowed from whichever catalog model happens to define
        the task (see :meth:`_find_task_template`), so anything in it that
        describes *that* model's deployment has to be replaced or the run
        silently validates against someone else's setup. Two such things:

        ``n_concurrent_trials`` describes the donor's deployment, and which
        donor gets borrowed is decided by catalog iteration order, so
        inheriting it would make the trial count arbitrary.
        ``deployment.maxConcurrencyPerInstance`` is the document's own
        statement of what the instance under test serves concurrently.
        Deliberately unclamped: the document is authoritative, and a trial
        count the host cannot afford is a property of the document rather than
        something to silently correct.

        The harness env keys in :data:`_HARNESS_MODEL_ENV_KEYS` name a model,
        not a credential -- tau3 runs the model under test as its own simulated
        user and NL-assertion judge. Every catalog entry therefore points them
        at its own repo, so a borrowed config would send that traffic to the
        donor, on an endpoint that is not serving it.

        Sampling is the third. The agent's own sampling is written into
        ``agent_kwargs`` (``temperature``, ``top_p``, and a ``top_k`` that has
        to ride in an ``extra_body`` to survive LiteLLM's ``drop_params``), and
        those values are the donor's -- ``top_p`` alone is 0.95 for some
        catalog models and 1.0 for others. The document's ``genKwargs`` replace
        them where the agent has the knob; see :func:`_repoint_sampling`.
        """
        cfg = getattr(template, "agentic_eval_config", None)
        if cfg is None:
            return {}
        changes: Dict[str, Any] = {}
        self._repoint_agent_sampling(cfg, task_name, ae, changes)

        concurrency = self._doc.deployment.max_concurrency_per_instance
        if concurrency and cfg.n_concurrent_trials != concurrency:
            logger.info(
                "Task %s: overriding borrowed n_concurrent_trials %s -> %s from "
                "the requirements document's deployment.maxConcurrencyPerInstance.",
                task_name,
                cfg.n_concurrent_trials,
                concurrency,
            )
            changes["n_concurrent_trials"] = concurrency

        for env_field in ("environment_env", "verifier_env"):
            env = getattr(cfg, env_field, None) or {}
            repointed = dict(env)
            for key in _HARNESS_MODEL_ENV_KEYS:
                donor = env.get(key)
                if not donor:
                    continue
                # Keep the provider prefix the catalog wrote (LiteLLM selects
                # its OpenAI provider on "openai/", see
                # llm_module/drivers/agentic.py) and swap only the model. A
                # value in some other shape is left alone and flagged, rather
                # than guessed at.
                if not donor.startswith(_LITELLM_OPENAI_PREFIX):
                    logger.warning(
                        "Task %s: borrowed %s[%s] is %r, which does not look "
                        "like an %r model reference; leaving it alone. It may "
                        "still name the model it was borrowed from.",
                        task_name,
                        env_field,
                        key,
                        donor,
                        _LITELLM_OPENAI_PREFIX,
                    )
                    continue
                ours = f"{_LITELLM_OPENAI_PREFIX}{self._doc.model.name}"
                if donor == ours:
                    continue
                logger.info(
                    "Task %s: re-pointing borrowed %s[%s] from %r to %r (the "
                    "requirements document's model).",
                    task_name,
                    env_field,
                    key,
                    donor,
                    ours,
                )
                repointed[key] = ours
            if repointed != env:
                changes[env_field] = repointed

        if not changes:
            return {}
        return {"agentic_eval_config": replace(cfg, **changes)}

    def _repoint_agent_sampling(
        self, cfg: Any, task_name: str, ae: AccuracyEval, changes: Dict[str, Any]
    ) -> None:
        """Apply the document's sampling to a borrowed agent, where it has the knob.

        Adds nothing the agent does not already read, so a parameter it has no
        knob for is reported rather than invented -- tau3 takes no ``top_k``
        anywhere, and no agent takes the document's token budget under a name
        this would recognise. Saying so matters more than it looks: the eval
        still runs and still produces a score, and without the warning that
        score reads as measured under parameters that never reached the server.
        """
        stated = ae.gen_kwargs.stated() if ae.gen_kwargs else {}
        if not stated:
            return
        sampling = {k: v for k, v in stated.items() if k in _AGENTIC_SAMPLING_KEYS}

        found: set = set()
        if sampling:
            agent_kwargs = _repoint_sampling(cfg.agent_kwargs or {}, sampling, found)
            if agent_kwargs != (cfg.agent_kwargs or {}):
                logger.info(
                    "Task %s: measuring under the document's genKwargs %r, "
                    "replacing the borrowed agent's own sampling.",
                    task_name,
                    {k: v for k, v in sampling.items() if k in found},
                )
                changes["agent_kwargs"] = agent_kwargs

        unhonoured = sorted(set(stated) - found)
        if unhonoured:
            logger.warning(
                "Task %s: the document states %r, but this agent exposes no "
                "such knob -- its config carries no place to put them, and "
                "inventing one would guess at the agent's own schema. The run "
                "samples the way the agent does, so its score is not measured "
                "under those parameters.",
                task_name,
                {k: stated[k] for k in unhonoured},
            )

    def _find_task_template(
        self, candidates: Sequence[str]
    ) -> Tuple[Optional[Any], str]:
        """A runnable EvalTask for the benchmark, and the task name it is.

        ``candidates`` are the task spellings that satisfy the document's eval,
        most preferred first. The document model's *own* entry wins for any of
        them, ahead of the preferred spelling borrowed from another model,
        because an EvalTask carries that model's sampling configuration --
        gen_kwargs, timeouts, the chat/completions class -- and those do not
        transfer. Borrowing GPQA from a reasoning model, for instance, brings
        its ``reasoning_effort`` along, which another server rejects, and its
        two-hour request timeout, which turns the rejection into a stall.

        Returns the preferred name with a ``None`` template when the catalog
        defines none of them, so the caller can try synthesizing that task.
        """
        from reference_config.evals.eval_config import EVAL_CONFIGS

        preferred = EVAL_CONFIGS.get(self._doc.model.name)
        if preferred is not None:
            for name in candidates:
                for task in preferred.tasks:
                    if task.task_name == name:
                        return task, name
        for name in candidates:
            for cfg in EVAL_CONFIGS.values():
                for task in cfg.tasks:
                    if task.task_name == name:
                        if name != candidates[0]:
                            logger.info(
                                "No catalog model defines %r; borrowing %r for "
                                "the same benchmark.",
                                candidates[0],
                                name,
                            )
                        return task, name
        return None, candidates[0]

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
        stated = ae.gen_kwargs.stated() if ae.gen_kwargs else {}
        logger.warning(
            "No catalog template for task %r; synthesizing with neutral "
            "defaults (max_length=%s from model.contextLength, streaming on, "
            "sampling: %s).",
            task_name,
            model_kwargs.get("max_length"),
            stated or "the server's own defaults",
        )
        return EvalTask(
            task_name=task_name,
            # Streaming on by default for synthesized tasks (long generations
            # against an OpenAI-compatible server). There is no template to
            # inherit sampling from, so the document's parameters are all there
            # is: whatever it does not state stays the server's own default.
            gen_kwargs={"stream": "True", **stated},
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
            params.extend(self._scenario_params(scenario, device, model_spec))

        task = BenchmarkTask(
            param_map={device: params},
            workflow_venv_type=select_vllm_benchmark_venv(model_spec),
        )
        return BenchmarkConfig(model_id=model_spec.model_id, tasks=[task])

    def _scenario_params(
        self, scenario: Scenario, device: Any, model_spec: Any
    ) -> List[Any]:
        from reference_config.benchmarking.benchmark_config import (
            SUPER_CLUSTER_MIN_NUM_PROMPTS_BATCH_MULTIPLE,
            get_num_prompts,
        )
        from workflows.utils_report import BenchmarkTaskParams, PerformanceTarget
        from workflows.workflow_types import DeviceTypes

        if not scenario.sweep:
            return []

        min_num_prompts = 0
        if device == DeviceTypes.SUPER_CLUSTER:
            model_max_concurrency = getattr(
                getattr(model_spec, "device_model_spec", None),
                "max_concurrency",
                None,
            )
            if model_max_concurrency:
                min_num_prompts = (
                    SUPER_CLUSTER_MIN_NUM_PROMPTS_BATCH_MULTIPLE * model_max_concurrency
                )
        # With per-row overrides in play, "the scenario has no SLOs" is no
        # longer the right predicate: a scenario can declare none itself and
        # still have every row supply its own.
        if _scenario_targets_goodput(scenario) and not any(
            _point_goodput_slo(p, scenario.slo) for p in scenario.sweep
        ):
            logger.warning(
                "Scenario %r declares goodput expectations but no sweep point "
                "declares its own SLOs, so goodput is not measured and those "
                "targets grade as NA.%s Bars belong on the rows: one set "
                "cannot hold across a sweep, since a bar that is satisfiable "
                "at one (ISL, OSL) is unreachable at another.",
                scenario.id,
                (
                    " The scenario-level slo is used as a capability gate on "
                    "the sweep's best point, not as goodput bars."
                    if scenario.slo is not None
                    else ""
                ),
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
                    num_prompts=get_num_prompts(
                        point.isl,
                        point.osl,
                        point.concurrency,
                        min_num_prompts=(
                            min_num_prompts if point.concurrency > 1 else 0
                        ),
                    ),
                    task_type="text",
                    targets=targets,
                    priority=_aggregate_priority(list(target_priorities.values())),
                    target_priorities=target_priorities or None,
                    goodput=_point_goodput_slo(point, scenario.slo),
                )
            )
        _warn_on_duplicate_shapes(scenario)
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

    # --- agentic traces (document sweep over a catalog/template base) ---
    def agentic_traces_config(self, model_spec: Any) -> Optional[Any]:
        # The run *shape* (scenario, dataset, InferenceX pin) still comes from
        # the catalog, or from the Kimi K2.7-Code template for an off-catalog
        # (synthesized) spec rather than refusing to run. What the document
        # contributes is the sweep: which concurrencies to replay it at.
        from reference_config.agentic_traces.agentic_traces_config import (
            get_agentic_traces_config_or_template,
            replace_agentic_runs,
        )

        base = self._delegate.agentic_traces_config(
            model_spec
        ) or get_agentic_traces_config_or_template(model_spec)
        if base is None:
            return None
        return replace_agentic_runs(
            base,
            self._agentic_concurrencies(),
            goodput=self._agentic_goodput_by_concurrency(),
            expected_sweep=self._agentic_expected_sweep(),
        )

    def _agentic_expected_sweep(self) -> List[Dict[str, Any]]:
        """The document's expected sweep points, verbatim and in order.

        ``AgenticSweepPoint.reference`` is the point as the document stated
        it, so what a run measures can be graded against its own operating
        point in the report. First workload wins on a shared concurrency,
        matching the dedupe in ``_agentic_concurrencies``.
        """
        points: Dict[int, Dict[str, Any]] = {}
        for workload in self._doc.agentic_workloads:
            for point in workload.sweep:
                if point.concurrency > 0:
                    points.setdefault(point.concurrency, dict(point.reference))
        return [points[c] for c in sorted(points)]

    def _agentic_concurrencies(self) -> List[int]:
        """Concurrencies the document's agentic workloads ask for, in order.

        Deduplicated because two workloads may share an operating point, and
        replaying the same concurrency twice would just burn an hour per
        duplicate.
        """
        seen: Dict[int, None] = {}
        for workload in self._doc.agentic_workloads:
            for point in workload.sweep:
                if point.concurrency > 0:
                    seen.setdefault(point.concurrency, None)
        return sorted(seen)

    def _agentic_goodput_by_concurrency(self) -> Dict[int, str]:
        """AIPerf ``--goodput`` constraints per operating point.

        Each row supplies its own bars only -- an agentic row with no ``slo``
        does not inherit the workload's, since an agentic document targets
        goodput at specific operating points rather than service-wide (see
        ``test_agentic_rows_do_not_inherit_the_workload_slo``). A concurrency
        with no row SLOs is omitted, leaving that run without bars rather than
        a misleading number. First workload wins on a shared concurrency,
        matching the dedupe in ``_agentic_concurrencies`` and
        ``_agentic_expected_sweep``.
        """
        for workload in self._doc.agentic_workloads:
            if workload.slo is not None and not any(
                p.slo is not None for p in workload.sweep
            ):
                logger.warning(
                    "Agentic workload %r declares workload-level SLOs (%r) but "
                    "no sweep row declares its own; agentic goodput is graded "
                    "per operating point, so these are ignored and goodput "
                    "will not be measured. Move the SLOs onto the rows that "
                    "carry a goodput target.",
                    workload.id,
                    workload.slo,
                )
        by_concurrency: Dict[int, str] = {}
        owner: Dict[int, str] = {}
        for workload in self._doc.agentic_workloads:
            for point in workload.sweep:
                if point.concurrency <= 0:
                    continue
                constraints = _aiperf_slo_constraints(
                    point.effective_slo(workload.slo) if point.slo else None
                )
                if not constraints:
                    continue
                existing = by_concurrency.get(point.concurrency)
                if existing is None:
                    by_concurrency[point.concurrency] = constraints
                    owner[point.concurrency] = workload.id
                elif existing != constraints:
                    # Not reconciled: an SLO set is a contract, not a lattice.
                    # First wins; warn that the other's bars are dropped.
                    logger.warning(
                        "Agentic workloads %r and %r both define concurrency %d "
                        "with different SLOs (%r vs %r); the sweep replays it "
                        "once, so %r's bars are used and %r's are dropped.",
                        owner[point.concurrency],
                        workload.id,
                        point.concurrency,
                        existing,
                        constraints,
                        owner[point.concurrency],
                        workload.id,
                    )
        return by_concurrency

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


def _repoint_sampling(node: Any, params: Mapping[str, Any], found: set) -> Any:
    """``node`` with ``params`` replaced wherever the config already carries them.

    Never adds a key. Each agent nests its sampling somewhere different --
    mini-swe-agent under ``config.model.model_kwargs``, terminus-2 with
    ``temperature`` at the top level but ``top_p`` inside ``llm_kwargs``, and
    ``top_k`` inside an ``extra_body`` because LiteLLM's ``drop_params`` throws
    away anything the OpenAI provider does not recognise. Encoding those
    layouts here would mean carrying each agent's config schema and re-checking
    it at every agent release.

    Replacing only what is already present needs none of that: the borrowed
    config demonstrates where its agent reads a parameter from, so the value is
    changed exactly there. It also cannot produce the duplicate that terminus-2
    breaks on -- a ``reasoning_effort`` in both the constructor args and
    ``llm_kwargs`` reaches LiteLLM twice and raises TypeError -- because a key
    that is absent stays absent.

    ``found`` collects the parameters that landed somewhere, so the caller can
    report the ones this agent has no knob for.
    """
    if isinstance(node, Mapping):
        out = dict(node)
        for key, value in node.items():
            if key in params and not isinstance(value, (Mapping, list)):
                out[key] = params[key]
                found.add(key)
            elif (
                isinstance(key, str)
                and key.endswith(_JSON_STRING_ARG_SUFFIX)
                and isinstance(value, str)
            ):
                out[key] = _repoint_sampling_json(value, params, found)
            else:
                out[key] = _repoint_sampling(value, params, found)
        return out
    if isinstance(node, list):
        return [_repoint_sampling(item, params, found) for item in node]
    return node


def _repoint_sampling_json(raw: str, params: Mapping[str, Any], found: set) -> str:
    """Re-point sampling inside a JSON-string agent argument, keeping it a string."""
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Not re-pointing agent argument %r: it is not valid JSON, so the "
            "sampling it carries is left as the donor wrote it.",
            raw,
        )
        return raw
    if not isinstance(parsed, dict):
        return raw
    repointed = _repoint_sampling(parsed, params, found)
    if repointed == parsed:
        return raw
    return json.dumps(repointed)


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


def _point_goodput_slo(point: Any, default: Optional[Slo]) -> Optional[GoodputSlo]:
    """Goodput bars in force at one sweep point -- none unless the row says so.

    A scenario-level ``slo`` is not broadcast across the sweep (a bar that
    holds at ISL 128 can be unreachable at ISL 1024; see the capability-gate
    comment in ``_scenario_params`` for the same field read as a target), and
    a target (aggregate) is not a bar (per-request) either. A row declaring
    only some fields still inherits the rest from the scenario.
    """
    if point.slo is None:
        return None
    return _goodput_slo(point.effective_slo(default))


def _goodput_slo(slo: Optional[Slo]) -> Optional[GoodputSlo]:
    """The document's SLOs as the tool-neutral bars the drivers carry.

    Benchmark points are handed this value, not a rendered string: which CLI
    vocabulary to use is the driver's business (``llm_module.goodput``), not
    the document adapter's.
    """
    if slo is None:
        return None
    return GoodputSlo(ttft_ms=slo.ttft_ms, tpot_ms=slo.tpot_ms, e2el_ms=slo.e2el_ms)


def _aiperf_slo_constraints(slo: Optional[Slo]) -> Optional[str]:
    """AIPerf ``--goodput`` string for a set of SLOs.

    The agentic path renders here rather than carrying the typed bars, because
    ``AgenticTracesRunSpec.goodput`` is a string the InferenceX client is
    handed verbatim -- there is no driver in between to do the rendering.
    """
    return render_goodput(_goodput_slo(slo), AIPERF_GOODPUT_KEYS)


def _warn_on_duplicate_shapes(scenario: Scenario) -> None:
    """Warn when two sweep points share an (ISL, OSL, concurrency) shape.

    Downstream the per-point values are keyed by that shape
    (``llm_module/benchmark_configs.py``), so a duplicate is last-wins and one
    point's targets and goodput bars are silently dropped. Cheap to detect
    here, where the document is still in hand and the ids can be named.
    """
    seen: Dict[Any, int] = {}
    for idx, point in enumerate(scenario.sweep):
        shape = (point.isl, point.osl, point.concurrency)
        if shape in seen:
            logger.warning(
                "Scenario %r declares sweep points %d and %d with the same "
                "(isl=%d, osl=%d, concurrency=%d); downstream configs are keyed "
                "by that shape, so only the last one's targets and goodput SLOs "
                "are used.",
                scenario.id,
                seen[shape],
                idx,
                *shape,
            )
        seen[shape] = idx


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
