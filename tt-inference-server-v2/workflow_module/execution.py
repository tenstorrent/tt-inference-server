# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Core workflow execution scaffolding.

A workflow  is a small decorator pipeline around the test_module
dispatcher:

    prepare -> run_tasks -> format_results -> acceptance -> metadata -> report
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, List, Optional, Sequence, Tuple

from report_module import (
    GenerateResult,
    ReportGenerator,
    ReportSchema,
    acceptance_criteria_check,
    build_acceptance_export,
)
from test_module.task_types import MediaTaskType

if TYPE_CHECKING:
    from test_module.context import MediaContext

from .blocks_sink import BlockAccumulator, get_default_accumulator

logger = logging.getLogger(__name__)


_SPEC_METADATA_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("model_id", "model_id"),
    ("model_repo", "hf_model_repo"),
    ("inference_engine", "inference_engine"),
    ("tt_metal_commit", "tt_metal_commit"),
    ("vllm_commit", "vllm_commit"),
)


@dataclass(frozen=True)
class TaskOutcome:
    """Result of a single ``run_media_task`` invocation inside a workflow."""

    task_type: str
    exit_code: int
    elapsed_seconds: float
    block_kind: Optional[str]

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0


@dataclass(frozen=True)
class WorkflowResult:
    workflow_name: str
    return_code: int
    task_outcomes: List[TaskOutcome] = field(default_factory=list)
    markdown_path: Optional[Path] = None
    json_path: Optional[Path] = None
    error: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        return self.return_code == 0


@dataclass(frozen=True)
class PrefixCacheOptions:
    """Prefix-caching benchmark knobs forwarded to ``BenchmarksWorkflow``.

    Threaded through
    ``OrchestratorMetadata`` so the CLI entry point in ``run.py`` stays
    decoupled from ``llm_module``.
    """

    preset: str = "full"
    scenarios: Optional[str] = None
    arrival_pattern: Optional[str] = None
    request_rate: Optional[float] = None
    scenarios_json: Optional[str] = None
    trace_path: Optional[str] = None
    # AIPerf --goodput SLO string (space-separated KEY:VALUE pairs).
    # Overrides the manifest preset/scenario goodput when set.
    goodput: Optional[str] = None
    auth_token: str = ""
    # Worker /metrics endpoints scraped by AIPerf (--server-metrics) for
    # the prefix-cache counters, independent of the load target. Repeatable
    # for multi-worker deployments. Empty keeps AIPerf's auto-derived
    # /metrics from --url (the frontend), which lacks the counters.
    metrics_urls: Tuple[str, ...] = ()
    venv_python: Optional[str] = None


@dataclass(frozen=True)
class SpecDecodeOptions:
    """Speculative-decoding benchmark knobs forwarded to ``BenchmarksWorkflow``.

    Threaded through ``OrchestratorMetadata`` so the CLI entry point in
    ``run.py`` stays decoupled from ``llm_module``.
    """

    preset: str = "full"
    warmup_requests: int = 4
    auth_token: str = ""
    venv_python: Optional[str] = None


@dataclass(frozen=True)
class ServingBenchOptions:
    """Serving-bench suite selection forwarded to ``ServingBenchWorkflow``.

    ``suites`` is the comma-separated suite list from ``--serving-bench-suites``;
    ``None`` runs every suite under ``test_module/serving_bench``.
    """

    suites: Optional[str] = None


@dataclass(frozen=True)
class LLMBenchOptions:
    """LLM performance-benchmark knobs forwarded to ``BenchmarksWorkflow``.

    ``tools`` value selecting the perf-tool driver
    (``vllm`` / ``aiperf`` / ``genai`` / ``guidellm``).
    ``auth_token`` is the bearer token (minted JWT) sent to the server.
    ``venv_python`` pins the interpreter whose ``bin/`` holds the perf-tool
    binary; set for the ``release`` path, where ``run.py`` runs in the
    V2_RUN_SCRIPT venv rather than the tool venv (a standalone benchmarks run
    is already inside the tool venv via run_llm_bench.py, so it stays ``None``).
    Threaded through ``OrchestratorMetadata`` so ``run.py`` stays decoupled
    from ``llm_module``.
    """

    tools: str = "vllm"
    auth_token: str = ""
    venv_python: Optional[str] = None
    # AIPerf --goodput SLO string (space-separated KEY:VALUE pairs). Only the
    # aiperf driver consumes it; other tools ignore it.
    goodput: Optional[str] = None


@dataclass(frozen=True)
class LLMEvalOptions:
    """Standard-eval knobs forwarded to ``EvalsWorkflow`` for LLM models."""

    auth_token: str = ""


@dataclass(frozen=True)
class OrchestratorMetadata:
    """Top-level metadata the per-task runners can't see themselves.

    Populated by the CLI entry point (``run.py``) and threaded into the
    schema by :meth:`WorkflowExecution.inject_metadata`.
    """

    server_mode: Optional[str] = None
    run_command: Optional[str] = None
    runtime_model_spec_json: Optional[str] = None
    prefix_cache: Optional[PrefixCacheOptions] = None
    spec_decode: Optional[SpecDecodeOptions] = None
    serving_bench: Optional[ServingBenchOptions] = None
    llm_bench: Optional[LLMBenchOptions] = None
    llm_eval: Optional[LLMEvalOptions] = None


class WorkflowExecution(ABC):
    """Template-method base for every workflow.

    Subclasses set ``name`` and ``task_types``. The default ``run()``
    invokes each task in order, formats the accumulated Blocks into a
    ``ReportSchema``, evaluates acceptance criteria, injects orchestrator
    metadata, and generates the report."""

    name: ClassVar[str] = ""
    task_types: ClassVar[Sequence[MediaTaskType]] = ()

    def __init__(
        self,
        ctx: MediaContext,
        *,
        accumulator: Optional[BlockAccumulator] = None,
        orchestrator_metadata: Optional[OrchestratorMetadata] = None,
    ) -> None:
        if not self.name:
            raise ValueError(
                f"{type(self).__name__} must set a non-empty class-level `name`."
            )
        self.ctx = ctx
        self.accumulator = accumulator or get_default_accumulator()
        self.orchestrator_metadata = orchestrator_metadata or OrchestratorMetadata()
        self.logger = logging.getLogger(f"workflow.{self.name}")

    def run(self) -> WorkflowResult:
        self.logger.info(
            "=== Workflow start: %s | model=%s device=%s port=%d output=%s ===",
            self.name,
            self.ctx.model_spec.model_name,
            self.ctx.device.name,
            self.ctx.service_port,
            self.ctx.output_path,
        )
        try:
            self.prepare()
            task_outcomes = self.run_tasks()
        except Exception as e:
            self.logger.exception(
                "Workflow %s aborted during task phase: %s", self.name, e
            )
            return WorkflowResult(
                workflow_name=self.name,
                return_code=1,
                task_outcomes=[],
                error=str(e),
            )

        schema = self.format_results()
        if schema is None:
            self.logger.error("No blocks accumulated — cannot generate report.")
            return WorkflowResult(
                workflow_name=self.name,
                return_code=1,
                task_outcomes=task_outcomes,
                error="no_blocks",
            )

        try:
            accepted, _blockers = self.apply_acceptance_criteria(schema)
            self.inject_metadata(schema)
            gen = self.generate_report(schema)
        except Exception as e:
            self.logger.exception(
                "Workflow %s aborted during report phase: %s", self.name, e
            )
            return WorkflowResult(
                workflow_name=self.name,
                return_code=1,
                task_outcomes=task_outcomes,
                error=str(e),
            )

        failed_tasks = [outcome for outcome in task_outcomes if not outcome.succeeded]
        return_code = 0 if accepted and not failed_tasks else 1
        if failed_tasks:
            self.logger.error(
                "Workflow %s had %d failed task(s)", self.name, len(failed_tasks)
            )
        self.logger.info("=== Workflow done: %s (rc=%d) ===", self.name, return_code)
        return WorkflowResult(
            workflow_name=self.name,
            return_code=return_code,
            task_outcomes=task_outcomes,
            markdown_path=gen.markdown_path,
            json_path=gen.json_path,
        )

    def prepare(self) -> None:
        # v2 runs against an external inference server, so there's no venv
        # / library install step to do here. Subclasses that need
        # environment prep (warmup pings, dataset downloads, etc.) override
        # this hook.
        self.logger.debug("prepare(): no-op")

    def run_tasks(self) -> List[TaskOutcome]:
        return [self._dispatch_task(t) for t in self.task_types]

    def _dispatch_task(self, task_type: MediaTaskType) -> TaskOutcome:
        from test_module.dispatch import run_media_task

        self.logger.info("→ task=%s", task_type.value)
        started = time.time()
        try:
            exit_code, block = run_media_task(self.ctx, task_type)
        except Exception as e:
            elapsed = time.time() - started
            self.logger.exception(
                "❌ task=%s raised after %.1fs: %s", task_type.value, elapsed, e
            )
            return TaskOutcome(task_type.value, 1, elapsed, None)

        elapsed = time.time() - started
        block_kind = block.kind if block is not None else None
        if exit_code != 0:
            self.logger.error(
                "❌ task=%s rc=%d block=%s (%.1fs)",
                task_type.value,
                exit_code,
                block_kind,
                elapsed,
            )
        elif block is None:
            # Runner intentionally produced no block (e.g. spec_tests found
            # no matching suites for this model+device)
            self.logger.info("⏭  task=%s no-op rc=0 (%.1fs)", task_type.value, elapsed)
        else:
            self.logger.info(
                "✅ task=%s block=%s (%.1fs)", task_type.value, block_kind, elapsed
            )
        return TaskOutcome(task_type.value, exit_code, elapsed, block_kind)

    def format_results(self) -> Optional[ReportSchema]:
        if not self.accumulator.blocks:
            return None
        return self.accumulator.build_schema()

    def apply_acceptance_criteria(self, schema: ReportSchema) -> Tuple[bool, list]:
        accepted, blockers, categories = acceptance_criteria_check(
            schema, known_issues=self._known_issues()
        )
        schema.metadata.update(
            build_acceptance_export(
                accepted, blockers, categories, self._model_status()
            )
        )
        self.logger.info(
            "Acceptance: %s (%d blocker(s))",
            "PASS" if accepted else "FAIL",
            len(blockers),
        )
        return accepted, blockers

    def _known_issues(self) -> Optional[list]:
        """model_spec known_issues (waivers) for this device, or None.

        Guarded with getattr so a spec that predates the field, or a differently
        shaped ctx, degrades to "no waivers" rather than raising.
        """
        spec = getattr(self.ctx, "model_spec", None)
        device_spec = getattr(spec, "device_model_spec", None)
        return getattr(device_spec, "known_issues", None) or None

    def _load_runtime_model_spec(self) -> Optional[dict]:
        """Return the ``runtime_model_spec`` sub-dict from the spec JSON, if any."""
        path = self.orchestrator_metadata.runtime_model_spec_json
        if not path:
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        spec = data.get("runtime_model_spec") if isinstance(data, dict) else None
        return spec if isinstance(spec, dict) else None

    def _model_status(self) -> Optional[str]:
        spec = self._load_runtime_model_spec()
        return spec.get("status") if spec else None

    def inject_metadata(self, schema: ReportSchema) -> None:
        meta = schema.metadata
        meta["workflow"] = self.name
        m = self.orchestrator_metadata
        if m.server_mode is not None:
            meta["server_mode"] = m.server_mode
        if m.run_command is not None:
            meta["run_command"] = m.run_command
        if m.runtime_model_spec_json is not None:
            meta["runtime_model_spec_json"] = m.runtime_model_spec_json
        self._inject_model_spec_metadata(meta)

    def _inject_model_spec_metadata(self, meta: dict) -> None:
        """Populate identity/provenance fields from the runtime model spec.

        Single source of truth for both media and LLM reports: every field is
        written whenever the spec is available (``None`` when the spec omits
        it, e.g. ``tt_metal_commit`` for a media image) so the report metadata
        schema is stable across workflows. ``model_impl`` comes from the
        nested ``impl.impl_name`` (the hyphenated display name, e.g.
        ``tt-transformers``); the rest map verbatim per
        :data:`_SPEC_METADATA_FIELDS`.
        """
        spec = self._load_runtime_model_spec()
        if not spec:
            return
        for meta_key, spec_key in _SPEC_METADATA_FIELDS:
            meta[meta_key] = spec.get(spec_key)
        impl = spec.get("impl")
        meta["model_impl"] = impl.get("impl_name") if isinstance(impl, dict) else None

    def generate_report(self, schema: ReportSchema) -> GenerateResult:
        report_dir = Path(self.ctx.output_path).parent
        result = ReportGenerator().generate(schema, report_dir)
        self.logger.info("Wrote markdown: %s", result.markdown_path)
        self.logger.info("Wrote json:     %s", result.json_path)
        return result


__all__ = [
    "ServingBenchOptions",
    "LLMBenchOptions",
    "LLMEvalOptions",
    "OrchestratorMetadata",
    "PrefixCacheOptions",
    "SpecDecodeOptions",
    "TaskOutcome",
    "WorkflowExecution",
    "WorkflowResult",
]
