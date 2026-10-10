# SPDX-License-Identifier: Apache-2.0
"""TTLabRunner's Gemma batch mode (`tt-lab gemma --serve-batch`) against a fake
worker that speaks the same tagged pipe protocol (tests/fixtures). No device is
opened. Needs the Gemma 4 tokenizer.json (TT_LAB_TEST_TOKENIZER, default the
local 31B checkpoint); it is loaded with the `tokenizers` library because this
suite's conftest replaces `transformers` with a mock."""
import asyncio
import os
from pathlib import Path

import pytest

TOKENIZER = os.environ.get("TT_LAB_TEST_TOKENIZER", "/home/ttuser/workspace/gemma31-reference")
tokenizers = pytest.importorskip("tokenizers")
pytestmark = pytest.mark.skipif(not Path(TOKENIZER, "tokenizer.json").exists(), reason="Gemma tokenizer not present")

SLOTS = 4
CONTEXT = 4096
MODEL = "google/gemma-4-31B-it"
FAKE = Path(__file__).with_name("fixtures") / "fake_tt_lab_batch_worker.py"

from domain.completion_request import CompletionRequest  # noqa: E402


class Tokenizer:
    """The slice of the Hugging Face tokenizer API the runner uses."""

    def __init__(self, path):
        self.inner = tokenizers.Tokenizer.from_file(str(Path(path, "tokenizer.json")))
        self.vocab_size = self.inner.get_vocab_size()

    @classmethod
    def from_pretrained(cls, path, **_):
        return cls(path)

    def encode(self, text, add_special_tokens=True):
        return self.inner.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids, skip_special_tokens=False):
        return self.inner.decode(list(ids), skip_special_tokens=skip_special_tokens)


@pytest.fixture(scope="module")
def configured():
    """Point the (already constructed) settings and validation at a Gemma batch deployment."""
    from config.settings import get_settings
    import domain.tt_lab_validation as validation
    import tt_model_runners.tt_lab_runner as runner_module

    mp = pytest.MonkeyPatch()
    settings = get_settings()
    mp.setattr(settings, "model_runner", "tt-lab-gemma")
    mp.setattr(settings, "model_weights_path", TOKENIZER)
    mp.setattr(settings, "request_processing_timeout_seconds", 30)
    mp.setattr(settings.vllm, "max_num_seqs", SLOTS)
    mp.setattr(validation, "CONTEXT_LENGTH", CONTEXT)
    mp.setattr(validation, "SERVED_MODEL_NAME", MODEL)
    mp.setattr(validation, "ACCEPTED_MODEL_NAMES", (None, MODEL, MODEL.rsplit("/", 1)[-1]))
    mp.setattr(validation, "IGNORE_EOS_SUPPORTED", True)
    mp.setattr(runner_module, "CONTEXT_LENGTH", CONTEXT)
    mp.setattr(runner_module, "AutoTokenizer", Tokenizer)
    for name, value in dict(TT_LAB_CONTEXT=str(CONTEXT), TT_GEMMA31_SLOTS=str(SLOTS), TT_LAB_BINARY=str(FAKE),
                            TT_LAB_GGUF="unused", TT_LAB_TTQ="unused", SERVED_MODEL_NAME=MODEL).items():
        mp.setenv(name, value)
    mp.delenv("TT_LAB_DEVICE", raising=False)
    yield runner_module.TTLabRunner
    mp.undo()


def start(runner_class, **env):
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        r = runner_class("0")
        asyncio.run(r.warmup())
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return r


@pytest.fixture(scope="module")
def runner(configured):
    # 2 ms per decode step, so a 400-token request is still decoding when a later one arrives.
    r = start(configured, FAKE_STEP_SECONDS="0.002")
    yield r
    r.close_device()


def request(last, limit, stream=True, stop=None, ignore_eos=False):
    return CompletionRequest(prompt=[2, 100, last], max_tokens=limit, temperature=0, stream=stream,
                             stop=stop or [], ignore_eos=ignore_eos)


def expected_tokens(last, count):
    return [1000 + (last + k) % 1000 for k in range(count)]


async def collect(runner, req):
    chunks = [c async for c in await runner._run_async([req])]
    return "".join(c["data"].text for c in chunks), chunks[-1]["data"]


def test_handshake_and_mode(runner):
    assert runner.batched and runner.slots == SLOTS
    assert runner.health_check()


def test_more_streams_than_slots(runner):
    async def main():
        reqs = [request(100 + i, 5 + i) for i in range(2 * SLOTS)]
        return await asyncio.gather(*(collect(runner, r) for r in reqs))

    for i, (text, final) in enumerate(asyncio.run(main())):
        assert final.finish_reason == "length"
        assert final.completion_tokens == 5 + i
        assert final.prompt_tokens == 3
        assert text == runner.tokenizer.decode(expected_tokens(100 + i, 5 + i), skip_special_tokens=True)
    assert runner.health_check()


def test_short_request_overtakes_long_one(runner):
    """Continuous batching: a request admitted while another decodes finishes first."""

    async def main():
        order = []

        async def run(req, name):
            await collect(runner, req)
            order.append(name)

        long = asyncio.create_task(run(request(60, 400), "long"))
        await asyncio.sleep(0.05)
        await run(request(61, 3), "short")
        await long
        return order

    assert asyncio.run(main()) == ["short", "long"]


def test_non_streaming_and_end_token(runner):
    async def main():
        plain = await runner._run_async([request(20, 4, stream=False)])
        ended = await collect(runner, request(7, 50))
        return plain, ended

    plain, (text, final) = asyncio.run(main())
    assert plain[0]["data"].completion_tokens == 4 and plain[0]["data"].finish_reason == "length"
    assert plain[0]["data"].text == runner.tokenizer.decode(expected_tokens(20, 4), skip_special_tokens=True)
    assert final.finish_reason == "stop" and final.completion_tokens == 4  # three tokens, then <end_of_turn>


def test_ignore_eos_generates_the_limit(runner):
    text, final = asyncio.run(collect(runner, request(7, 10, ignore_eos=True)))
    assert final.finish_reason == "length" and final.completion_tokens == 10


def test_cancel_frees_the_slot(runner):
    async def main():
        async def partial():
            async for _ in await runner._run_async([request(30, 4000)]):
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


def test_cancel_while_queued_for_a_slot(runner):
    """A request cancelled before it got a slot never reaches the worker."""

    async def main():
        holders = [asyncio.create_task(collect(runner, request(70 + i, 300))) for i in range(SLOTS)]
        await asyncio.sleep(0.05)
        waiting = asyncio.create_task(collect(runner, request(80, 3)))
        await asyncio.sleep(0.05)
        waiting.cancel()
        await asyncio.gather(waiting, return_exceptions=True)
        finals = await asyncio.gather(*holders)
        return finals, await collect(runner, request(81, 2))

    finals, (_, last) = asyncio.run(main())
    assert all(final.completion_tokens == 300 for _, final in finals)
    assert last.completion_tokens == 2 and runner.health_check()


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


def test_requests_after_a_stop_string_cancel(runner):
    """A stop string cancels the request in the worker; its slot is reused."""

    async def main():
        full, _ = await collect(runner, request(90, 20))
        stop = full[2:4]
        stopped = await asyncio.gather(*(collect(runner, request(90, 20, stop=[stop])) for _ in range(2 * SLOTS)))
        return stopped, await collect(runner, request(91, 5))

    stopped, (_, final) = asyncio.run(main())
    assert all(f.finish_reason == "stop" for _, f in stopped)
    assert final.completion_tokens == 5 and runner.health_check()


def test_worker_crash_fails_every_stream(configured):
    """The worker dies mid-decode: every open stream fails, health turns false."""

    class Runner(configured):
        exited = False

        def _child_exited(self):  # production exits the Python worker here
            Runner.exited = True

    r = start(Runner, FAKE_STEP_SECONDS="0.01")
    try:
        async def main():
            return await asyncio.gather(collect(r, request(13, 50)), collect(r, request(14, 50)),
                                        return_exceptions=True)

        results = asyncio.run(main())
        assert all(isinstance(x, RuntimeError) for x in results), results
        assert not r.health_check()
        assert Runner.exited
    finally:
        r.close_device()


def test_ignore_eos_rejected_without_batch_worker(monkeypatch):
    import domain.tt_lab_validation as validation

    monkeypatch.setattr(validation, "IGNORE_EOS_SUPPORTED", False)
    with pytest.raises(ValueError, match="ignore_eos"):
        validation.validate_request(request(5, 4, ignore_eos=True), Tokenizer(TOKENIZER))
