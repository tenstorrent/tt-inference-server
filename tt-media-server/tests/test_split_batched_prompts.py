# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from domain.completion_request import CompletionRequest
from open_ai_api.llm import _split_batched_prompts


class TestSplitBatchedPrompts:
    """Per #3533 Problem 4: fan-out must regenerate _task_id for each sub-request,
    otherwise the scheduler's per-task result_queues collide and only one
    sub-request's await unblocks."""

    def test_string_list_yields_unique_task_ids(self):
        req = CompletionRequest(model="m", prompt=["hello", "world"], max_tokens=5)
        subs = _split_batched_prompts(req)
        assert len(subs) == 2
        assert [s.prompt for s in subs] == ["hello", "world"]
        task_ids = {s._task_id for s in subs}
        assert len(task_ids) == 2
        assert req._task_id not in task_ids

    def test_token_list_list_yields_unique_task_ids(self):
        req = CompletionRequest(
            model="m",
            prompt=[[1, 2, 3], [4, 5, 6], [7, 8]],
            max_tokens=5,
        )
        subs = _split_batched_prompts(req)
        assert len(subs) == 3
        assert [s.prompt for s in subs] == [[1, 2, 3], [4, 5, 6], [7, 8]]
        task_ids = {s._task_id for s in subs}
        assert len(task_ids) == 3
        assert req._task_id not in task_ids

    def test_single_string_prompt_passes_through_unchanged(self):
        req = CompletionRequest(model="m", prompt="single", max_tokens=5)
        subs = _split_batched_prompts(req)
        assert len(subs) == 1
        assert subs[0] is req
        assert subs[0]._task_id == req._task_id

    def test_flat_token_list_passes_through_unchanged(self):
        # Flat list of ints is a single tokenized prompt, not a batch.
        req = CompletionRequest(model="m", prompt=[1, 2, 3], max_tokens=5)
        subs = _split_batched_prompts(req)
        assert len(subs) == 1
        assert subs[0] is req
        assert subs[0]._task_id == req._task_id

    def test_empty_list_passes_through_unchanged(self):
        req = CompletionRequest(model="m", prompt=[], max_tokens=5)
        subs = _split_batched_prompts(req)
        assert len(subs) == 1
        assert subs[0] is req
