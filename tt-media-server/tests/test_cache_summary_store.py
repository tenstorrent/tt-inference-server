# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import numpy as np
import pytest
from utils import cache_summary_store as store

SUMMARY = [
    {"name": "high_noise", "executed_steps": 15, "cached_steps": [[3, 5], [3, 5]]},
    {"name": "low_noise", "executed_steps": 25, "cached_steps": [[1, 2], [1, 2]]},
]


@pytest.fixture(autouse=True)
def _folder(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "_CACHE_SUMMARY_DIR", tmp_path / "summaries")


def test_a_jobs_summary_round_trips_and_is_deleted_with_it():
    store.write_cache_summary("4b53f494-edc2-4436-9104-b19a109e204e", SUMMARY)
    assert store.read_cache_summary("4b53f494-edc2-4436-9104-b19a109e204e") == SUMMARY
    store.delete_cache_summary("4b53f494-edc2-4436-9104-b19a109e204e")
    assert store.read_cache_summary("4b53f494-edc2-4436-9104-b19a109e204e") is None


def test_an_uncached_run_keeps_an_empty_summary_and_a_job_without_one_has_none():
    store.write_cache_summary("job_1", [])
    assert store.read_cache_summary("job_1") == []
    assert store.read_cache_summary("job_2") is None
    store.delete_cache_summary("job_2")


def test_numpy_values_are_written_as_plain_numbers():
    store.write_cache_summary(
        "job_1",
        [
            {
                "name": "high_noise",
                "executed_steps": np.int64(15),
                "cached_steps": np.array([[3, 5]]),
            }
        ],
    )
    assert store.read_cache_summary("job_1") == [
        {"name": "high_noise", "executed_steps": 15, "cached_steps": [[3, 5]]}
    ]


@pytest.mark.parametrize("task_id", ["", "../etc/passwd", "a/b", "a.b"])
def test_an_id_that_is_not_a_job_id_names_no_file(task_id, tmp_path):
    store.write_cache_summary(task_id, SUMMARY)
    assert store.read_cache_summary(task_id) is None
    assert not (tmp_path / "summaries").exists()
