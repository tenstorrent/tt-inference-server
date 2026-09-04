# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Pure loss-trajectory checking for the training workflow.

The forge server exposes per-step LoRA training loss via
``GET /v1/jobs/{id}/metrics`` as a list of records shaped like::

    {"global_step": 5, "metric_name": "train_loss", "value": 5.71, ...}

This module compares that trajectory against a checked-in expectation and
produces ``spec_tests``-shaped report records (consumed by
``report_module.generator``) plus an overall pass/fail verdict.

It is deliberately free of any HTTP / server / torch dependency so the whole
grading logic is unit-testable on a laptop. The expectation is intentionally
LOOSE ("did it converge?"), mirroring tt-blacksmith's own golden checks
(``rtol=0.5, atol=0.1``): the forge runner trains non-deterministically
(``seed=0`` but ``deterministic=False``), so exact per-step equality is not a
usable gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

TRAIN_LOSS = "train_loss"
VAL_LOSS = "val_loss"

# Local string constants so this module carries no report_module import; the
# values match report_module.status.TestStatus.{PASS,FAIL}.value.
TestStatus_PASS = "pass"
TestStatus_FAIL = "fail"


@dataclass(frozen=True)
class LossExpectation:
    """One expected metric value at a given optimizer step."""

    global_step: int
    metric_name: str
    value: float


@dataclass(frozen=True)
class LossCheckConfig:
    """Parsed expectation for a single (model, device, dataset) training run."""

    rtol: float = 0.5
    atol: float = 0.1
    require_decreasing: bool = True
    final_train_loss_max: Optional[float] = None
    expectations: List[LossExpectation] = field(default_factory=list)
    # Opaque request overrides forwarded verbatim into the TrainingRequest body
    # (batch_size, max_steps, steps_freq, lora_r, ...). Kept here so the run's
    # hyperparameters and its expected losses live in one file and version
    # together.
    request: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CheckResult:
    passed: bool
    records: List[Dict[str, Any]]
    summary: str


def parse_config(raw: Mapping[str, Any]) -> LossCheckConfig:
    """Build a :class:`LossCheckConfig` from the raw YAML mapping."""
    if not isinstance(raw, Mapping):
        raise TypeError("training expectation config must be a mapping")

    tolerances = raw.get("tolerances") or {}
    checks = raw.get("checks") or {}

    expectations: List[LossExpectation] = []
    for entry in raw.get("expected_losses") or []:
        if "global_step" not in entry or "value" not in entry:
            raise ValueError(
                "each expected_losses entry needs 'global_step' and 'value'"
            )
        expectations.append(
            LossExpectation(
                global_step=int(entry["global_step"]),
                metric_name=str(entry.get("metric_name", TRAIN_LOSS)),
                value=float(entry["value"]),
            )
        )

    final_max = checks.get("final_train_loss_max")
    return LossCheckConfig(
        rtol=float(tolerances.get("rtol", 0.5)),
        atol=float(tolerances.get("atol", 0.1)),
        require_decreasing=bool(checks.get("require_decreasing", True)),
        final_train_loss_max=None if final_max is None else float(final_max),
        expectations=expectations,
        request=dict(raw.get("request") or {}),
    )


def index_metrics(metrics: Sequence[Mapping[str, Any]]) -> Dict[Tuple[int, str], float]:
    """Flatten the server metrics list into ``{(step, metric_name): value}``.

    Later records for the same (step, metric) win, so a re-reported value
    overrides an earlier one.
    """
    indexed: Dict[Tuple[int, str], float] = {}
    for record in metrics or []:
        if "global_step" not in record or "metric_name" not in record:
            continue
        try:
            step = int(record["global_step"])
            value = float(record["value"])
        except (TypeError, ValueError, KeyError):
            continue
        indexed[(step, str(record["metric_name"]))] = value
    return indexed


def _record(
    *,
    model: str,
    device: str,
    test_name: str,
    status: str,
    description: str,
) -> Dict[str, Any]:
    """Build one ``spec_tests``-shaped report record."""
    return {
        "kind": "spec_tests",
        "model": model,
        "device": device,
        "test_name": test_name,
        "status": status,
        "attempts": 1,
        "elapsed_seconds": 0.0,
        "description": description,
    }


def _train_losses_in_step_order(
    indexed: Mapping[Tuple[int, str], float],
) -> List[Tuple[int, float]]:
    steps = sorted(step for (step, metric) in indexed if metric == TRAIN_LOSS)
    return [(step, indexed[(step, TRAIN_LOSS)]) for step in steps]


def evaluate(
    metrics: Sequence[Mapping[str, Any]],
    config: LossCheckConfig,
    *,
    model: str,
    device: str,
) -> CheckResult:
    """Grade ``metrics`` against ``config`` and return records + verdict."""
    indexed = index_metrics(metrics)
    records: List[Dict[str, Any]] = []

    # 1) No metrics at all is an unambiguous failure (job never produced loss).
    if not indexed:
        records.append(
            _record(
                model=model,
                device=device,
                test_name="training_produced_metrics",
                status=TestStatus_FAIL,
                description="server returned no training metrics",
            )
        )
        return CheckResult(
            passed=False, records=records, summary="no training metrics returned"
        )

    # 2) Per-checkpoint tolerance comparison against the expectation.
    for expectation in config.expectations:
        key = (expectation.global_step, expectation.metric_name)
        observed = indexed.get(key)
        name = f"{expectation.metric_name}@step{expectation.global_step}"
        if observed is None:
            records.append(
                _record(
                    model=model,
                    device=device,
                    test_name=name,
                    status=TestStatus_FAIL,
                    description=(
                        f"missing {expectation.metric_name} at step "
                        f"{expectation.global_step} (expected ~{expectation.value})"
                    ),
                )
            )
            continue
        close = math.isclose(
            observed, expectation.value, rel_tol=config.rtol, abs_tol=config.atol
        )
        records.append(
            _record(
                model=model,
                device=device,
                test_name=name,
                status=TestStatus_PASS if close else TestStatus_FAIL,
                description=(
                    f"observed={observed:.4f} expected={expectation.value:.4f} "
                    f"rtol={config.rtol} atol={config.atol}"
                ),
            )
        )

    train_curve = _train_losses_in_step_order(indexed)

    # 3) Roughly-decreasing check: final training loss should not exceed the
    # first recorded training loss (LoRA fine-tuning must make progress).
    if config.require_decreasing and len(train_curve) >= 2:
        first_step, first_loss = train_curve[0]
        last_step, last_loss = train_curve[-1]
        decreasing = last_loss <= first_loss + config.atol
        records.append(
            _record(
                model=model,
                device=device,
                test_name="train_loss_decreasing",
                status=TestStatus_PASS if decreasing else TestStatus_FAIL,
                description=(
                    f"first(step{first_step})={first_loss:.4f} "
                    f"last(step{last_step})={last_loss:.4f}"
                ),
            )
        )

    # 4) Absolute final-loss threshold, if configured.
    if config.final_train_loss_max is not None and train_curve:
        last_step, last_loss = train_curve[-1]
        within = last_loss <= config.final_train_loss_max
        records.append(
            _record(
                model=model,
                device=device,
                test_name="final_train_loss_threshold",
                status=TestStatus_PASS if within else TestStatus_FAIL,
                description=(
                    f"final(step{last_step})={last_loss:.4f} "
                    f"max={config.final_train_loss_max:.4f}"
                ),
            )
        )

    passed = all(r["status"] == TestStatus_PASS for r in records)
    n_pass = sum(1 for r in records if r["status"] == TestStatus_PASS)
    summary = f"{n_pass}/{len(records)} training checks passed"
    return CheckResult(passed=passed, records=records, summary=summary)
