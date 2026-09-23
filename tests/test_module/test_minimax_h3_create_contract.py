# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""URL hygiene of the MiniMax-H3 create-contract suite: the target is a validated http(s)
origin plus route constants, and only a well-formed job id is ever placed in a URL."""

from __future__ import annotations

import asyncio

import pytest

from test_module.load_param_tests import minimax_h3_create_contract_test as C


@pytest.mark.parametrize(
    ("given", "root"),
    [
        ("http://127.0.0.1:8000", "http://127.0.0.1:8000"),
        ("http://127.0.0.1:8000/", "http://127.0.0.1:8000"),
        ("https://h3.example.com/api/", "https://h3.example.com/api"),
        (" http://h3.example.com?x=1#f ", "http://h3.example.com"),
    ],
)
def test_service_root_keeps_origin_and_prefix_only(given, root):
    assert C._service_root(given) == root


@pytest.mark.parametrize(
    "bad", ["file:///etc/passwd", "127.0.0.1:8000", "ftp://x", "", "/v1"]
)
def test_service_root_refuses_anything_but_http(bad):
    with pytest.raises(ValueError):
        C._service_root(bad)


@pytest.mark.parametrize(
    "task_id",
    ["../../admin", "abc def", "", "x" * 129, "%0d%0aX: 1", "not-a-uuid", None],
)
def test_cancel_refuses_a_job_id_that_is_not_a_uuid(task_id):
    assert C._job_uuid(task_id) is None
    result = asyncio.run(  # returns before any request is made
        C._cancel_created_job(
            base_url="http://127.0.0.1:9",
            api_key="k",
            task_id=task_id,
            request_timeout=1.0,
        )
    )
    assert result is None


def test_job_uuid_is_the_canonical_form():
    assert (
        C._job_uuid("6F9619FF-8B86-D011-B42D-00C04FC964FF")
        == "6f9619ff-8b86-d011-b42d-00c04fc964ff"
    )
