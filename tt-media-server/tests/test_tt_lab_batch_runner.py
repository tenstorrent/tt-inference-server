# SPDX-License-Identifier: Apache-2.0
"""TTLabRunner's Gemma batch mode (`tt-lab gemma --serve-batch`) against a fake
worker that speaks the same tagged pipe protocol (tests/fixtures). Needs the
Gemma 4 tokenizer (TT_LAB_TEST_TOKENIZER, default the local 31B checkpoint)."""
import asyncio
import os
from pathlib import Path

import pytest

TOKENIZER = os.environ.get("TT_LAB_TEST_TOKENIZER", "/home/ttuser/workspace/gemma31-reference")
pytestmark = pytest.mark.skipif(not Path(TOKENIZER, "tokenizer.json").exists(), reason="Gemma tokenizer not present")

SLOTS = 4
os.environ.update(
    MODEL_RUNNER="tt-lab-gemma", MODEL_WEIGHTS_PATH=TOKENIZER, SERVED_MODEL_NAME="google/gemma-4-31B-it",
    MAX_MODEL_LENGTH="4096", TT_LAB_CONTEXT="4096", MAX_NUM_SEQS=str(SLOTS), TT_GEMMA31_SLOTS=str(SLOTS),
    TT_LAB_BINARY=str(Path(__file__).with_name("fixtures") / "fake_tt_lab_batch_worker.py"),
    TT_LAB_GGUF="unused", TT_LAB_TTQ="unused", REQUEST_PROCESSING_TIMEOUT_SECONDS="30",
)
os.environ.pop("TT_LAB_DEVICE", None)

from domain.completion_request import CompletionRequest  # noqa: E402
from tt_model_runners.tt_lab_runner import TTLabRunner  # noqa: E402


def request(last, limit, stream=True, stop=None):
    return CompletionRequest(prompt=[2, 100, last], max_tokens=limit, temperature=0, stream=stream, stop=stop or [])


async def collect(runner, req):
    chunks = [c async for c in await runner._run_async([req])]
    return "".join(c["data"].text for c in chunks), chunks[-1]["data"]


@pytest.fixture(scope="module")
def runner():
    r = TTLabRunner("0")
    asyncio.run(r.warmup())
    yield r
    r.close_device()


def test_more_streams_than_slots(runner):
    async def main():
        reqs = [request(10 + i, 5 + i) for i in range(2 * SLOTS)]
        return await asyncio.gather(*(collect(runner, r) for r in reqs))

    for i, (text, final) in enumerate(asyncio.run(main())):
        assert final.finish_reason == "length"
        assert final.completion_tokens == 5 + i
        assert text == runner.tokenizer.decode([1000 + (10 + i + k) % 1000 for k in range(5 + i)],
                                               skip_special_tokens=True)
    assert runner.health_check()


def test_non_streaming_and_end_token(runner):
    async def main():
        plain = await runner._run_async([request(20, 4, stream=False)])
        ended = await collect(runner, request(7, 50))
        return plain, ended

    plain, (text, final) = asyncio.run(main())
    assert plain[0]["data"].completion_tokens == 4 and plain[0]["data"].finish_reason == "length"
    assert final.finish_reason == "stop" and final.completion_tokens == 4  # three tokens, then <end_of_turn>


def test_cancel_frees_the_slot(runner):
    async def main():
        async def partial():
            async for chunk in await runner._run_async([request(30, 4000)]):
                await asyncio.sleep(0.01)

        tasks = [asyncio.create_task(partial()) for _ in range(SLOTS)]
        await asyncio.sleep(0.5)
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        # Every slot must come back: SLOTS more requests complete.
        return await asyncio.wait_for(asyncio.gather(*(collect(runner, request(40 + i, 3)) for i in range(SLOTS))), 20)

    results = asyncio.run(main())
    assert all(final.completion_tokens == 3 for _, final in results)
    assert runner.health_check()


def test_stop_string_ends_early(runner):
    async def main():
        full, _ = await collect(runner, request(50, 12))
        stop = full[len(full) // 2:][:2]
        text, final = await collect(runner, request(50, 12, stop=[stop]))
        return full, stop, text, final

    full, stop, text, final = asyncio.run(main())
    assert final.finish_reason == "stop"
    assert text == full[:full.find(stop)]
    assert runner.health_check()
