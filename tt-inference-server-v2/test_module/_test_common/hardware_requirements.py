# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Hardware-readiness tiers for media tests.

Two tiers cover every existing test kind in the repo:

- ``FULL_BOARD``: throughput tests (benchmarks, ``*LoadTest`` cases, stability
  runs). Need every declared worker to be ready, otherwise concurrency is
  bottlenecked and target times become meaningless.
- ``ANY_CHIP``: correctness/contract tests (evals, ``*ParamTest`` cases,
  integration / unit checks). A single ready worker is enough; the server
  will queue requests across whatever workers are alive.

The enum is consumed at two choke points:

1. Workflow dispatch path
   (:mod:`tt_inference_server_v2.test_module.context.require_health`) — each
   ``run_*_benchmark`` / ``run_*_eval`` callable passes the appropriate tier
   (or, for benchmarks, relies on ``FULL_BOARD`` being the default).
2. Spec-tests pipeline
   (:class:`tt_inference_server_v2.test_module._test_common.base_test.BaseTest`)
   — each spec test class declares ``HARDWARE_REQUIREMENT`` and the base
   class's ``run_tests()`` self-gates before executing.
"""

from __future__ import annotations

from enum import Enum


class HardwareRequirement(Enum):
    """Minimum chip-readiness a test needs to produce a meaningful result."""

    FULL_BOARD = "full_board"
    ANY_CHIP = "any_chip"


__all__ = ["HardwareRequirement"]
