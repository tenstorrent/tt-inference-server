# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for the CNN eval boundary that translates VisionEvalsTest status codes.

VisionEvalsTest emits its own accuracy status codes (0=undefined, 2=pass,
3=fail). Acceptance grading treats any unrecognized int as PASS, so the raw
``0`` (undefined / missing accuracy data) used to grade as a silent pass. These
tests pin the translation to ``ReportCheckTypes`` and the acceptance contract.
"""

from __future__ import annotations

from report_module.acceptance_criteria import (
    CATEGORY_EVALS,
    STATUS_NA,
    STATUS_PASS,
    acceptance_criteria_check,
)
from report_module.schema import Block, ReportSchema

from test_module._test_common import ReportCheckTypes
from test_module.eval_tests.cnn_eval_tests import _VISION_STATUS_TO_CHECK


def _eval_category(accuracy_check):
    block = Block(kind="evals", title="E", data={"accuracy_check": accuracy_check})
    schema = ReportSchema(metadata={"report_id": "r"}, sections=[block])
    accepted, blockers, categories = acceptance_criteria_check(schema)
    category = {c.name: c for c in categories}[CATEGORY_EVALS]
    return accepted, blockers, category


def test_vision_status_mapping_aligns_with_report_check_types():
    assert _VISION_STATUS_TO_CHECK == {
        0: ReportCheckTypes.NA,
        2: ReportCheckTypes.PASS,
        3: ReportCheckTypes.FAIL,
    }


def test_raw_undefined_vision_code_would_grade_as_pass():
    # Documents the hazard the mapping guards against: the raw undefined code
    # (0) bypasses acceptance's FAIL/NA cases and falls through to PASS.
    _, _, category = _eval_category(0)
    assert category.status == STATUS_PASS


def test_mapped_undefined_vision_code_grades_as_na():
    accepted, blockers, category = _eval_category(_VISION_STATUS_TO_CHECK[0])
    assert accepted is True and blockers == {}
    assert category.status == STATUS_NA
