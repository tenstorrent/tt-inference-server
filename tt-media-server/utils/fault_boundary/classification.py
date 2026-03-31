# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

from .fault_types import (
    COMPONENT_INFERENCE,
    COMPONENT_TORCH,
    COMPONENT_TTNN,
    ClassificationResult,
    FaultOrigin,
    FaultReport,
)

TTNN_PATH_MARKER = "site-packages" + os.sep + "ttnn"
TORCH_PATH_MARKER = "site-packages" + os.sep + "torch"
TT_METAL_COMMIT_ENV = "TT_METAL_COMMIT_SHA_OR_TAG"
IMAGE_TAG_ENV_KEYS: Tuple[str, ...] = ("IMAGE_TAG", "DOCKER_IMAGE_TAG")


def _normalize_path(path: str) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(path)))


def _normalize_roots(our_source_roots: Tuple[str, ...]) -> Tuple[str, ...]:
    return tuple(_normalize_path(r) for r in our_source_roots if r)


def _frame_paths_from_traceback(tb) -> list[str]:
    """Collect frame paths from outermost to innermost (``tb`` chain order)."""
    paths: list[str] = []
    while tb is not None:
        filename = tb.tb_frame.f_code.co_filename
        paths.append(_normalize_path(filename))
        tb = tb.tb_next
    return paths


def _module_suggests_external(module: str) -> Optional[Tuple[str, str]]:
    if not module:
        return None
    if module == "ttnn" or module.startswith("ttnn."):
        return COMPONENT_TTNN, module
    if module.startswith("torch") or module.startswith("triton"):
        return COMPONENT_TORCH, module
    return None


def _is_under_root(path: str, our_roots: Tuple[str, ...]) -> bool:
    return any(path.startswith(root + os.sep) or path == root for root in our_roots)


def _classify_from_stack_innermost_first(
    paths: list[str], our_roots: Tuple[str, ...]
) -> Optional[ClassificationResult]:
    """Prefer the innermost relevant frame to reduce UNKNOWN.

    *paths* is outermost→innermost as produced by CPython's ``tb`` chain; we walk
    innermost→outermost. Per frame: ttnn → torch (site-packages) → inference roots.
    Skip frames that match none (e.g. stdlib) and use the next outer frame.
    """
    for path in reversed(paths):
        if TTNN_PATH_MARKER in path:
            return ClassificationResult(FaultOrigin.EXTERNAL, COMPONENT_TTNN)
        if TORCH_PATH_MARKER in path:
            return ClassificationResult(FaultOrigin.EXTERNAL, COMPONENT_TORCH)
        if our_roots and _is_under_root(path, our_roots):
            return ClassificationResult(FaultOrigin.INTERNAL, COMPONENT_INFERENCE)
    return None


MAX_CAUSE_CHAIN_DEPTH = 16


def classify_exception(
    exc: BaseException,
    *,
    our_source_roots: Tuple[str, ...] = (),
    _cause_depth: int = 0,
) -> ClassificationResult:
    """Classify *exc* as internal, external (ttnn/torch), or unknown.

    Uses exception type module first, then traceback paths: **innermost frame
    wins** among ttnn, ``site-packages/torch``, and *our_source_roots*, so mixed
    stacks (inference calling into ttnn) usually resolve to EXTERNAL when the
    raise site is in ttnn.

    If *exc* has no traceback but ``__cause__`` does, classification follows the
    cause chain.

    *our_source_roots* should be absolute normalized prefixes of this service (e.g.
    the ``tt-media-server`` tree). An empty tuple means no path is treated as
    inference code (stack-only checks will classify ttnn paths as EXTERNAL).
    """
    module = getattr(type(exc), "__module__", "") or ""
    external = _module_suggests_external(module)
    if external is not None:
        comp, _mod = external
        return ClassificationResult(FaultOrigin.EXTERNAL, comp)

    roots = _normalize_roots(our_source_roots)
    tb = getattr(exc, "__traceback__", None)
    paths = _frame_paths_from_traceback(tb) if tb is not None else []

    if paths:
        from_stack = _classify_from_stack_innermost_first(paths, roots)
        if from_stack is not None:
            return from_stack
        return ClassificationResult(
            FaultOrigin.UNKNOWN,
            COMPONENT_INFERENCE,
            reason="no ttnn, torch, or inference root in traceback",
        )

    cause = getattr(exc, "__cause__", None)
    if cause is not None and cause is not exc and _cause_depth < MAX_CAUSE_CHAIN_DEPTH:
        from_cause = classify_exception(
            cause,
            our_source_roots=our_source_roots,
            _cause_depth=_cause_depth + 1,
        )
        if from_cause.origin != FaultOrigin.UNKNOWN or from_cause.reason is not None:
            return from_cause

    return ClassificationResult(
        FaultOrigin.UNKNOWN,
        COMPONENT_INFERENCE,
        reason="no traceback available",
    )


def _image_tag_from_env() -> Optional[str]:
    for key in IMAGE_TAG_ENV_KEYS:
        value = os.environ.get(key)
        if value:
            return value
    return None


def fault_report_from_exception(
    exc: BaseException,
    operation: str,
    *,
    classification: Optional[ClassificationResult] = None,
    our_source_roots: Tuple[str, ...] = (),
    request_id: Optional[str] = None,
    boundary_component: Optional[str] = None,
) -> FaultReport:
    """Build a :class:`FaultReport` with env metadata and optional classification."""
    result = classification or classify_exception(
        exc, our_source_roots=our_source_roots
    )
    component = result.component
    if result.origin == FaultOrigin.UNKNOWN and boundary_component:
        component = boundary_component
    return FaultReport(
        origin=result.origin,
        component=component,
        operation=operation,
        exc_type=type(exc).__name__,
        message=str(exc),
        request_id=request_id,
        tt_metal_commit=os.environ.get(TT_METAL_COMMIT_ENV),
        image_tag=_image_tag_from_env(),
        classification_reason=result.reason,
    )


def format_fault_log_line(report: FaultReport) -> str:
    """Single grep-friendly line for scrapers (Loki, CI)."""
    parts = [
        f"FAULT_ORIGIN={report.origin.value}",
        f"COMPONENT={report.component}",
        f"OP={report.operation}",
        f"EXC_TYPE={report.exc_type}",
        f"MSG={report.message!r}",
    ]
    if report.request_id:
        parts.append(f"REQUEST_ID={report.request_id}")
    if report.tt_metal_commit:
        parts.append(f"TT_METAL_COMMIT={report.tt_metal_commit}")
    if report.image_tag:
        parts.append(f"IMAGE_TAG={report.image_tag}")
    if report.classification_reason:
        parts.append(f"REASON={report.classification_reason!r}")
    return " ".join(parts)


def log_fault_report(
    report: FaultReport,
    logger: Optional[logging.Logger] = None,
    level: int = logging.ERROR,
) -> None:
    log = logger or logging.getLogger(__name__)
    log.log(level, format_fault_log_line(report))
