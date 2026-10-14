# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""EvalTask.doc_ids: a configured lm-eval --samples subset that limit modes
slice instead of combining with --limit (lm-eval rejects both)."""

import json
from types import SimpleNamespace

import pytest

from llm_module.eval_command import (
    SMOKE_TEST_EVAL_LIMIT,
    _resolve_eval_samples,
    _slice_doc_ids_for_limit,
)
from reference_config.evals.eval_config import (
    LIVECODEBENCH_V6_DOC_IDS,
    EvalTask,
)


def _rc(**kw):
    base = {"eval_samples": None, "limit_samples_mode": None}
    base.update(kw)
    return SimpleNamespace(**base)


def _task(**kw):
    return EvalTask(task_name="livecodebench", doc_ids=[5, 6, 7, 8], **kw)


def test_no_doc_ids_and_no_cli_mapping_gives_no_samples():
    assert _resolve_eval_samples(EvalTask(task_name="x"), _rc()) is None


def test_doc_ids_are_passed_as_samples_without_cli_mapping():
    out = _resolve_eval_samples(_task(), _rc())
    assert json.loads(out) == {"livecodebench": [5, 6, 7, 8]}


def test_cli_mapping_for_the_task_wins_over_doc_ids():
    rc = _rc(eval_samples=json.dumps({"livecodebench": [1, 2]}))
    assert json.loads(_resolve_eval_samples(_task(), rc)) == {"livecodebench": [1, 2]}


def test_cli_mapping_for_another_task_falls_back_to_doc_ids():
    rc = _rc(eval_samples=json.dumps({"mmlu_pro": [1]}))
    assert json.loads(_resolve_eval_samples(_task(), rc)) == {
        "livecodebench": [5, 6, 7, 8]
    }


def test_smoke_test_keeps_the_first_three_doc_ids():
    task = EvalTask(task_name="livecodebench", doc_ids=list(range(10)))
    assert _slice_doc_ids_for_limit(task, _rc(limit_samples_mode="smoke-test")) == list(
        range(SMOKE_TEST_EVAL_LIMIT)
    )


def test_fraction_limit_keeps_that_share_of_doc_ids():
    from workflows.workflow_types import EvalLimitMode

    task = EvalTask(
        task_name="livecodebench",
        doc_ids=list(range(100, 110)),
        limit_samples_map={EvalLimitMode.CI_NIGHTLY: 0.5},
    )
    assert _slice_doc_ids_for_limit(task, _rc(limit_samples_mode="ci-nightly")) == list(
        range(100, 105)
    )


def test_int_limit_keeps_the_first_n_doc_ids():
    from workflows.workflow_types import EvalLimitMode

    task = EvalTask(
        task_name="livecodebench",
        doc_ids=[9, 8, 7],
        limit_samples_map={EvalLimitMode.CI_NIGHTLY: 2},
    )
    assert _slice_doc_ids_for_limit(task, _rc(limit_samples_mode="ci-nightly")) == [
        9,
        8,
    ]


@pytest.mark.parametrize("bad", [[], [-1], [1.5], [True], "0,1"])
def test_invalid_doc_ids_are_rejected(bad):
    with pytest.raises(ValueError):
        EvalTask(task_name="x", doc_ids=bad).validate_data()


def test_livecodebench_v6_window_is_131_unique_ids_in_the_sixth_release():
    ids = LIVECODEBENCH_V6_DOC_IDS
    assert len(ids) == 131 and len(set(ids)) == 131
    assert min(ids) >= 880 and max(ids) == 1054 and ids == sorted(ids)


def test_scoring_uses_the_doc_ids_count_not_lm_evals_dataset_size():
    from test_module.llm_tests.llm_eval_tests import _apply_samples_subset_count

    task = EvalTask(task_name="livecodebench", doc_ids=list(range(131)))
    counts = _apply_samples_subset_count(task, _rc(), {"livecodebench": 1055})
    assert counts == {"livecodebench": 131}
    # no subset -> counts untouched
    plain = EvalTask(task_name="mmlu_pro")
    assert _apply_samples_subset_count(plain, _rc(), {"mmlu_pro": 849}) == {
        "mmlu_pro": 849
    }
