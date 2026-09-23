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


class _NoNetwork:
    def post(self, *args, **kwargs):  # pragma: no cover - reaching this is the failure
        raise AssertionError(f"unexpected request {args} {kwargs}")


@pytest.mark.parametrize(
    "task_id", ["../../admin", "abc def", "", "x" * 129, "%0d%0aX: 1"]
)
def test_cancel_refuses_a_job_id_it_would_not_put_in_a_url(task_id):
    result = asyncio.run(
        C._cancel_created_job(
            _NoNetwork(), base_url="http://127.0.0.1:8000", api_key="k", task_id=task_id
        )
    )
    assert result is None
