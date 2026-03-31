# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional, Tuple, TypeVar

from .classification import fault_report_from_exception, log_fault_report
from .fault_types import BoundaryError

T = TypeVar("T")


@contextmanager
def external_call_boundary(
    component: str,
    operation: str,
    *,
    our_source_roots: Tuple[str, ...],
    request_id: Optional[str] = None,
    logger: Optional[Any] = None,
) -> Iterator[None]:
    """Run a block; on exception, classify, log, and re-raise :class:`BoundaryError`."""
    try:
        yield
    except Exception as exc:
        report = fault_report_from_exception(
            exc,
            operation,
            our_source_roots=our_source_roots,
            request_id=request_id,
            boundary_component=component,
        )
        log_fault_report(report, logger=logger)
        raise BoundaryError(report) from exc


def wrap_external_call(
    component: str,
    operation: str,
    fn: Callable[..., T],
    /,
    *args: Any,
    our_source_roots: Tuple[str, ...],
    request_id: Optional[str] = None,
    logger: Optional[Any] = None,
    **kwargs: Any,
) -> T:
    """Call ``fn``; on failure, classify, log, and raise :class:`BoundaryError`."""
    with external_call_boundary(
        component,
        operation,
        our_source_roots=our_source_roots,
        request_id=request_id,
        logger=logger,
    ):
        return fn(*args, **kwargs)
