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
from enum import Enum
from typing import Callable, List, Optional, Tuple

from report_module.schema import Block
from workflow_module import accept_blocks

from ._test_common import TestConfig, sweep_envelope
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

logger = logging.getLogger(__name__)

MediaRunner = Callable[[MediaContext], Block]


class MediaTaskType(Enum):
    EVALUATION = "evaluation"
    BENCHMARK = "benchmark"
    SPEC_TESTS = "spec_tests"


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


def run_media_task(
    ctx: MediaContext, task_type: MediaTaskType
) -> Tuple[int, Optional[Block]]:
    """Dispatch ``ctx`` to the correct media runner.

    Returns:
        ``(exit_code, block)`` where ``exit_code`` is ``0`` on success and
        ``1`` on any failure, and ``block`` is the runner's emitted Block on
        success (``None`` on failure or when the model_type has no runner).
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
        logger.error(
            "No %s runner registered for model_type=%r. Known types: %s",
            task_type.value,
            model_type_name,
            sorted(dispatch),
        )
        return 1, None

    try:
        block = runner(ctx)
    except Exception as e:
        logger.exception("%s runner raised: %s", task_type.value, e)
        return 1, None

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


def _instantiate_spec_test(case: dict, ctx: MediaContext):
    """Import + construct a spec test class from a (filtered) test case dict.

    BaseTest accepts ``(config, targets, description="", ctx=None)`` but a
    handful of test classes (e.g. ImageGenerationEvalsTest) override
    ``__init__`` with just ``(config, targets)`` — so we try the rich form
    first and fall back to the minimal one rather than introspecting.
    """
    config = TestConfig(case.get("test_config") or {})
    targets = case.get("targets") or {}
    module = importlib.import_module(case["module"])
    cls = getattr(module, case["name"])
    try:
        return cls(config, targets, ctx=ctx)
    except TypeError:
        return cls(config, targets)


def run_spec_tests(ctx: MediaContext) -> Tuple[int, Optional[Block]]:
    """Run all spec test cases that match ``ctx`` (model + device).

    Each test case produces one Block via ``BaseTest.run_tests()``; all are
    handed to the accumulator so the eventual ReportSchema includes one
    section per test. Returns ``(exit_code, last_block)`` where exit_code
    is non-zero if any test class raised or any Block did not explicitly
    report ``success=True`` (missing key, non-dict data, or any non-True
    value all count as failures).
    """
    logger.info(
        "Running spec_tests for model=%s, device=%s",
        ctx.model_spec.model_name,
        ctx.device.name,
    )
    suites = _resolve_spec_test_suites(ctx)
    if not suites:
        logger.error(
            "No spec test suites match model=%r device=%r — nothing to do.",
            ctx.model_spec.model_name,
            ctx.device.name.lower(),
        )
        return 1, None

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
                logger.exception("  ✘ %s raised: %s", class_name, e)
                failures += 1
                continue
            blocks.append(block)
            if (
                not isinstance(block.data, dict)
                or block.data.get("success") is not True
            ):
                logger.error(
                    "  ✘ %s did not report success=True (data=%r)",
                    class_name,
                    block.data,
                )
                failures += 1

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
