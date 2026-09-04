# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Opt-in SWE-bench dataset-revision + content-digest reproducibility pin.

The pin composes with the #5056 ``run_with_progress`` harness: it is a
pre-agent step that, only when a ``dataset_revision`` is requested, loads that
exact immutable revision, materializes the selected rows into a local snapshot
both the agent and the grading harness read, and fails closed before any agent
starts when an ``expected_digest`` no longer matches. When no revision is
requested the run path is byte-for-byte unchanged.

The ``datasets`` dependency is only pulled in when pinning is used, so these
tests patch the single ``_load_hf_dataset`` seam instead of installing it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from llm_module.agentic import swebench


def _base_config(tmp_path: Path, **overrides) -> swebench.SWEbenchRunConfig:
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    config = swebench.SWEbenchRunConfig(
        task_name="swe",
        dataset_name="SWE-bench/SWE-bench_Verified",
        dataset_split="test",
        sweagent_subset="verified",
        agent_backend="mini-swe-agent",
        model_name="m",
        api_base="http://localhost:8000/v1",
        output_dir=output_dir,
        sweagent_config="config/default.yaml",
        mini_config="mini.yaml",
        mini_model_class="cls",
        mini_environment_class="env",
        n_concurrent_trials=1,
        max_workers=1,
        n_tasks=None,
        temperature=0.0,
        top_p=1.0,
        max_input_tokens=1,
        max_output_tokens=None,
        completion_kwargs={},
        swebench_timeout_sec=None,
        shuffle=False,
        random_delay_multiplier=0.0,
        score_existing_predictions=False,
    )
    return dataclasses.replace(config, **overrides)


def _digest(rows: list[dict]) -> str:
    return hashlib.sha256(swebench._canonical_selected_rows(rows)).hexdigest()


def test_pin_match_snapshots_and_binds_both_consumers(tmp_path):
    selected = {"instance_id": "django__django-11299", "problem_statement": "exact"}
    other = {"instance_id": "other", "problem_statement": "ignored"}
    full_rows = [other, selected]
    ordered = [selected]
    digest = _digest(ordered)

    cfg = _base_config(
        tmp_path,
        dataset_revision="a" * 40,
        expected_digest=digest,
        instance_ids=["django__django-11299"],
    )

    # First load = the pinned revision; second = the reload of the local
    # snapshot for the self-verification round-trip.
    with patch.object(
        swebench, "_load_hf_dataset", side_effect=[full_rows, [selected]]
    ) as load:
        dataset_source, actual = swebench._prepare_pinned_swebench_dataset(cfg)

    assert actual == digest
    # The pinned revision was requested by exact revision, not a mutable alias.
    assert load.call_args_list[0].args == ("SWE-bench/SWE-bench_Verified",)
    assert load.call_args_list[0].kwargs["revision"] == "a" * 40
    # Both consumers point at the same local snapshot directory.
    assert load.call_args_list[1].args == (dataset_source,)
    assert dataset_source == str(cfg.output_dir / "pinned_swebench_dataset")

    receipt = json.loads(
        (cfg.output_dir / "pinned_swebench_dataset_receipt.json").read_text()
    )
    assert receipt["content_sha256"] == digest
    assert receipt["expected_digest"] == digest
    assert receipt["dataset_revision"] == "a" * 40
    assert receipt["ordered_instance_ids"] == ["django__django-11299"]

    mini = swebench.build_mini_sweagent_command(
        cfg, cfg.output_dir / "mini.yaml", cfg.output_dir / "mini",
        dataset_source=dataset_source,
    )
    harness = swebench.build_swebench_harness_command(
        cfg, cfg.output_dir / "predictions.jsonl", "run-id",
        dataset_source=dataset_source,
    )
    assert mini[mini.index("--subset") + 1] == dataset_source
    assert harness[harness.index("--dataset_name") + 1] == dataset_source


def test_pin_mismatch_fails_closed_before_agent(tmp_path):
    selected = {"instance_id": "django__django-11299", "problem_statement": "x"}
    cfg = _base_config(
        tmp_path,
        dataset_revision="a" * 40,
        expected_digest="0" * 64,  # deliberately wrong
        instance_ids=["django__django-11299"],
    )

    with patch.object(
        swebench, "_load_hf_dataset", return_value=[selected]
    ) as load, pytest.raises(RuntimeError, match="content digest mismatch"):
        swebench._prepare_pinned_swebench_dataset(cfg)

    # Failure happens on the pinned load, before the snapshot reload round-trip.
    assert load.call_count == 1
    # No snapshot was materialized.
    assert not (cfg.output_dir / "pinned_swebench_dataset").exists()


def test_pin_off_by_default_leaves_run_path_unchanged(tmp_path):
    cfg = _base_config(tmp_path)  # no dataset_revision
    assert cfg.dataset_revision is None
    assert cfg.expected_digest is None

    with patch.object(swebench, "_load_hf_dataset") as load:
        dataset_source, actual = swebench._prepare_pinned_swebench_dataset(cfg)

    load.assert_not_called()
    assert (dataset_source, actual) == (None, None)

    # Builders keep resolving the mutable Hub alias exactly as before.
    mini = swebench.build_mini_sweagent_command(
        cfg, cfg.output_dir / "mini.yaml", cfg.output_dir / "mini",
        dataset_source=dataset_source,
    )
    harness = swebench.build_swebench_harness_command(
        cfg, cfg.output_dir / "predictions.jsonl", "run-id",
        dataset_source=dataset_source,
    )
    assert mini[mini.index("--subset") + 1] == cfg.sweagent_subset
    assert harness[harness.index("--dataset_name") + 1] == cfg.dataset_name
