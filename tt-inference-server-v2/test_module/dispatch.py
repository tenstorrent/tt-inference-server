# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Single dispatch entry point for the v2 test module.

Maps ``(model_type.name, task_type)`` to the appropriate ``run_<media>_<task>``
function and invokes it with uniform logging + process-style exit codes. Each
runner returns a :class:`report_module.schema.Block`, which is forwarded to
``workflow_module.accept_blocks`` so a sweep-level accumulator can assemble a
single base schema later.

In addition to per-model EVALUATION / BENCHMARK runners, this module also
exposes a ``SPEC_TESTS`` task type that loads matching test cases from
``test_module/test_suites/*.json`` via ``TestFilter``, instantiates each
test class, runs it, and forwards the resulting Blocks to the accumulator.
"""

from __future__ import annotations

import importlib
import logging
from copy import deepcopy
from typing import Callable, List, Optional, Tuple

from report_module.schema import Block
from workflow_module import accept_blocks
from workflows.workflow_types import ModelType

from ._test_common import (
    SkipTest,
    TestConfig,
    TestStatus,
    block_id,
    sweep_envelope,
)
from ._test_common.exceptions import TestOutcomeSignal
from .benchmark_tests import (
    run_audio_benchmark,
    run_cnn_benchmark,
    run_embedding_benchmark,
    run_image_benchmark,
    run_tts_benchmark,
    run_video_benchmark,
)
from .context import MediaContext
from .eval_tests import (
    run_audio_eval,
    run_cnn_eval,
    run_embedding_eval,
    run_image_eval,
    run_tts_eval,
    run_video_eval,
)
from .task_types import MediaTaskType

logger = logging.getLogger(__name__)

MediaRunner = Callable[[MediaContext], Block]

EVAL_DISPATCH: dict[str, MediaRunner] = {
    "CNN": run_cnn_eval,
    "IMAGE": run_image_eval,
    "AUDIO": run_audio_eval,
    "EMBEDDING": run_embedding_eval,
    "TEXT_TO_SPEECH": run_tts_eval,
    "VIDEO": run_video_eval,
}

BENCHMARK_DISPATCH: dict[str, MediaRunner] = {
    "CNN": run_cnn_benchmark,
    "IMAGE": run_image_benchmark,
    "AUDIO": run_audio_benchmark,
    "EMBEDDING": run_embedding_benchmark,
    "TEXT_TO_SPEECH": run_tts_benchmark,
    "VIDEO": run_video_benchmark,
}


_NO_MEDIA_RUNNER: frozenset[ModelType] = frozenset({ModelType.VLM, ModelType.LLM})

_registered_media_types = set(EVAL_DISPATCH) | set(BENCHMARK_DISPATCH)
assert not any(mt.name in _registered_media_types for mt in _NO_MEDIA_RUNNER), (
    "A ModelType in _NO_MEDIA_RUNNER also has a registered media runner"
)


_MEDIA_TASK_BLOCK_KIND = {
    MediaTaskType.EVALUATION: "evals",
    MediaTaskType.BENCHMARK: "benchmarks",
}


def _media_outcome_block(
    ctx: MediaContext,
    task_type: MediaTaskType,
    status: TestStatus,
    reason: Optional[str] = None,
    exc: Optional[Exception] = None,
) -> Block:
    """Build an evals/benchmarks Block for a non-pass, blockless outcome.

    Lets a media runner that was skipped (``SkipTest``), was not gradable
    (``NotApplicable``), or crashed produce a *visible* Block carrying an
    explicit :class:`TestStatus` — instead of vanishing behind a bare
    ``(1, None)`` return value.
    """
    kind = _MEDIA_TASK_BLOCK_KIND.get(task_type, "spec_tests")
    data: dict = {
        "success": False,
        "status": status.value,
        "test_name": ctx.model_spec.model_name,
    }
    if reason:
        data["reason"] = reason
    if exc is not None:
        data["error"] = {"type": type(exc).__name__, "message": str(exc)}
    return Block(
        kind=kind,
        id=block_id(ctx) or None,
        title=kind.replace("_", " ").title(),
        data=data,
    )


def run_media_task(
    ctx: MediaContext, task_type: MediaTaskType
) -> Tuple[int, Optional[Block]]:
    """Dispatch ``ctx`` to the correct media runner.

    Returns:
        ``(exit_code, block)`` where ``exit_code`` is ``0`` on success and
        ``1`` on any failure, and ``block`` is the runner's emitted Block on
        success (``None`` on failure). A model_type with no registered runner
        yields a *visible* Block instead of a silent failure: a non-blocking
        SKIP (``(0, block)``) if the type is runnerless by design
        (:data:`_NO_MEDIA_RUNNER`), otherwise a blocking ERROR (``(1, block)``).
        The block is also handed to ``workflow_module.accept_blocks`` so a
        sweep-level accumulator can pick it up alongside the return value.

        For ``SPEC_TESTS``, the returned ``block`` is the *last* Block
        produced (a representative); all Blocks from every test case are
        already in the accumulator by the time this returns.
    """
    if task_type == MediaTaskType.SPEC_TESTS:
        return run_spec_tests(ctx)

    model_type_name = ctx.model_spec.model_type.name
    logger.info(
        "Running %s for model_type=%s, model=%s, device=%s",
        task_type.value,
        model_type_name,
        ctx.model_spec.model_name,
        ctx.device.name,
    )

    dispatch = (
        EVAL_DISPATCH if task_type == MediaTaskType.EVALUATION else BENCHMARK_DISPATCH
    )
    runner = dispatch.get(model_type_name)
    if runner is None:
        if ctx.model_spec.model_type in _NO_MEDIA_RUNNER:
            reason = (
                f"{task_type.value} has no media runner for "
                f"model_type={model_type_name!r} (unsupported by design)"
            )
            logger.warning("⏭  %s -> skip: %s", task_type.value, reason)
            block = _media_outcome_block(
                ctx, task_type, TestStatus.SKIP, reason=reason
            )
            accept_blocks([block], envelope=sweep_envelope(ctx))
            return 0, block

        # Unexpected: a model type that should have a runner doesn't. Surface a
        # blocking, visible ERROR rather than a silent (1, None) failure.
        error = RuntimeError(
            f"no {task_type.value} runner registered for "
            f"model_type={model_type_name!r}; known types: {sorted(dispatch)}"
        )
        logger.error("❌ %s -> error: %s", task_type.value, error)
        block = _media_outcome_block(ctx, task_type, TestStatus.ERROR, exc=error)
        accept_blocks([block], envelope=sweep_envelope(ctx))
        return 1, block

    try:
        block = runner(ctx)
    except TestOutcomeSignal as e:
        # Intentional non-error outcome (skip / not-applicable): emit a
        # visible, non-blocking Block and keep the task green.
        status = TestStatus.SKIP if isinstance(e, SkipTest) else TestStatus.NA
        logger.info("⏭  %s -> %s: %s", task_type.value, status.value, e.reason)
        block = _media_outcome_block(ctx, task_type, status, reason=e.reason)
        accept_blocks([block], envelope=sweep_envelope(ctx))
        return 0, block
    except Exception as e:
        logger.exception("%s runner raised: %s", task_type.value, e)
        block = _media_outcome_block(ctx, task_type, TestStatus.ERROR, exc=e)
        accept_blocks([block], envelope=sweep_envelope(ctx))
        return 1, block

    accept_blocks([block], envelope=sweep_envelope(ctx))
    return 0, block


def _resolve_spec_test_suites(ctx: MediaContext) -> List[dict]:
    """Return matching expanded suites for ``ctx.model_spec.model_name`` + device."""
    from .test_categorization_system import TestFilter

    return (
        TestFilter()
        .filter_by_model(ctx.model_spec.model_name)
        .filter_by_device(ctx.device.name.lower())
        .get_tests()
    )


def _maybe_cap_num_prompts(case: dict, cap: Optional[int]) -> dict:
    """Clamp ``targets.request.num_prompts`` to ``cap`` if set.

    Returns a new test case dict so we don't mutate the suite definition
    in TestFilter's cache.
    """
    if cap is None:
        return case
    targets = case.get("targets") or {}
    request = targets.get("request") if isinstance(targets, dict) else None
    if not isinstance(request, dict) or "num_prompts" not in request:
        return case
    if request["num_prompts"] <= cap:
        return case
    new_case = deepcopy(case)
    new_case["targets"]["request"]["num_prompts"] = cap
    logger.info(
        "Capping %s targets.request.num_prompts: %s -> %s",
        case["name"],
        request["num_prompts"],
        cap,
    )
    return new_case


def _error_block(case: dict, ctx: MediaContext, exc: Exception) -> Block:
    """Build a spec_tests Block for a test that raised before producing one.

    Without this, a class that fails to import/instantiate/run would bump the
    failure counter but emit no Block, making the crash invisible in the
    report while still failing the workflow.
    """
    return Block(
        kind="spec_tests",
        id=block_id(ctx) or None,
        title=str(case.get("name") or "spec_test").replace("_", " ").title(),
        task_type=None,
        targets=dict(case.get("targets") or {}),
        data={
            "success": False,
            "status": TestStatus.ERROR.value,
            "attempts": 0,
            "test_name": str(case.get("name") or ""),
            "description": str(case.get("description") or ""),
            "error": {"type": type(exc).__name__, "message": str(exc)},
        },
    )


def _block_status(block: Block) -> TestStatus:
    """Resolve a Block's :class:`TestStatus`, falling back to legacy fields."""
    data = block.data if isinstance(block.data, dict) else {}
    resolved = TestStatus.from_value(data.get("status"))
    if resolved is not None:
        return resolved
    return TestStatus.from_legacy(
        data.get("success"), skipped=bool(data.get("skipped"))
    )


def _instantiate_spec_test(case: dict, ctx: MediaContext):
    """Import + construct a spec test class from a (filtered) test case dict.

    BaseTest accepts ``(config, targets, description="", ctx=None)`` but a
    handful of test classes (e.g. ImageGenerationEvalsTest) override
    ``__init__`` with just ``(config, targets)`` — so we try the rich form
    first and fall back to the minimal one, patching ``description`` on
    afterward so the summary block can still surface it.
    """
    config = TestConfig(case.get("test_config") or {})
    targets = case.get("targets") or {}
    description = case.get("description") or ""
    module = importlib.import_module(case["module"])
    cls = getattr(module, case["name"])
    try:
        return cls(config, targets, description=description, ctx=ctx)
    except TypeError:
        instance = cls(config, targets)
        instance.description = description
        return instance


def run_spec_tests(ctx: MediaContext) -> Tuple[int, Optional[Block]]:
    """Run all spec test cases that match ``ctx`` (model + device).

    Each test case produces one Block via ``BaseTest.run_tests()``; all are
    handed to the accumulator so the eventual ReportSchema includes one
    section per test. Returns ``(exit_code, last_block)`` where exit_code
    is non-zero if any test class raised or any Block did not explicitly
    report ``success=True`` (missing key, non-dict data, or any non-True
    value all count as failures).

    An empty filter result (no suite in ``test_module/test_suites/*.json``
    matches this model+device pair) is treated as a clean no-op: a warning
    is logged and ``(0, None)`` is returned, so a model that has not yet
    been wired into the spec-test config does not fail the whole workflow.
    """
    logger.info(
        "Running spec_tests for model=%s, device=%s",
        ctx.model_spec.model_name,
        ctx.device.name,
    )
    suites = _resolve_spec_test_suites(ctx)
    if not suites:
        logger.warning(
            "No spec test suites match model=%r device=%r — skipping spec_tests.",
            ctx.model_spec.model_name,
            ctx.device.name.lower(),
        )
        return 0, None

    cap = ctx.spec_tests_num_prompts_cap
    blocks: List[Block] = []
    failures = 0
    for suite in suites:
        suite_id = suite.get("id", "?")
        cases = suite.get("test_cases", [])
        logger.info("[%s] %d test case(s)", suite_id, len(cases))
        for case in cases:
            if "name" not in case or "module" not in case:
                logger.warning(
                    "  - skip malformed case (missing name/module): %r", case
                )
                continue
            if not case.get("enabled", True):
                logger.info("  - skip disabled %s", case["name"])
                continue
            class_name = case["name"]
            module_name = case["module"]
            logger.info("  -> %s (%s)", class_name, module_name)
            patched = _maybe_cap_num_prompts(case, cap)
            try:
                test = _instantiate_spec_test(patched, ctx)
                block = test.run_tests()
            except Exception as e:
                logger.exception("  ❌ %s raised: %s", class_name, e)
                blocks.append(_error_block(patched, ctx, e))
                failures += 1
                continue
            blocks.append(block)
            status = _block_status(block)
            if status.is_blocking:
                logger.error(
                    "  ❌ %s -> %s (data=%r)",
                    class_name,
                    status.value,
                    block.data,
                )
                failures += 1
            elif status in (TestStatus.SKIP, TestStatus.NA):
                logger.info("  ⏭  %s -> %s", class_name, status.value)

    if not blocks:
        return 1, None

    accept_blocks(blocks, envelope=sweep_envelope(ctx))
    exit_code = 0 if failures == 0 else 1
    logger.info(
        "spec_tests done: %d block(s), %d failure(s) -> exit=%d",
        len(blocks),
        failures,
        exit_code,
    )
    return exit_code, blocks[-1]


__all__ = [
    "MediaTaskType",
    "EVAL_DISPATCH",
    "BENCHMARK_DISPATCH",
    "run_media_task",
    "run_spec_tests",
]
