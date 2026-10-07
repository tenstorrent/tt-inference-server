# SPDX-License-Identifier: Apache-2.0
"""Persistent tt-lab silicon backend for the existing TT media-server LLM pipeline.

The subprocess runs all transformer layers on Blackhole silicon: GPT-OSS-20B on
one chip, GPT-OSS-120B or Gemma 4 31B across four. Python only encodes/decodes
tokens and transports requests. There is no CPU inference fallback.

Gemma 4 31B (MODEL_RUNNER=tt-lab-gemma) runs `tt-lab gemma --native-device`:
- TT_GEMMA31_SLOTS unset or 1: `--serve`, one sequence at a time, the same pipe
  protocol as GPT-OSS.
- TT_GEMMA31_SLOTS=N (2..16): `--serve-batch`. Each request owns one of N cache
  slots, is prefilled alone in it, and decodes together with the other active
  requests through tt-lab's exact batch-decode path. Requests are tagged, so
  several stream at once (the dynamic-batch device worker) and a cancelled
  stream frees its slot.
"""
import asyncio
import atexit
import os
import select
import struct
import subprocess
import time
import sys
import signal
import threading
from pathlib import Path

from domain.tt_lab_validation import CONTEXT_LENGTH, validate_request

from domain.completion_response import CompletionOutput, CompletionResult
from transformers import AutoTokenizer
from tt_model_runners.base_device_runner import BaseDeviceRunner

# --serve-batch replies: (id, value) pairs; value is a token or one of these.
STOPPED, LENGTH, READY, CANCELLED = -1, -2, -3, -4


class TextStream:
    """Turn generated token IDs into streamed text, holding back a possible
    stop-string prefix and an incomplete UTF-8 character."""

    def __init__(self, tokenizer, request):
        self.tokenizer = tokenizer
        self.prompt = request.prompt
        self.harmony = isinstance(request.prompt, str) and "<|start|>assistant" in request.prompt
        self.stops = [request.stop] if isinstance(request.stop, str) else (request.stop or [])
        self.generated = []
        self.previous = ""
        self.stopped = False

    def visible_text(self):
        if not self.harmony:
            return self.tokenizer.decode(self.generated, skip_special_tokens=True)
        raw = self.tokenizer.decode(self.generated, skip_special_tokens=False)
        # GPT-OSS emits analysis and then a final channel. Do not mix private
        # reasoning/channel delimiters into the API assistant content.
        marker = "<|channel|>final<|message|>"
        if marker in raw:
            return raw.rsplit(marker, 1)[1].split("<|return|>", 1)[0].split("<|end|>", 1)[0]
        if raw.startswith("final<|message|>"):
            return raw[len("final<|message|>"):].split("<|return|>", 1)[0]
        # Some prompts preselect the final channel and receive bare text.
        if self.prompt.endswith("final<|message|>"):
            return self.tokenizer.decode(self.generated, skip_special_tokens=True)
        return ""

    def push(self, token):
        """Add one token; return the newly visible text ("" if none)."""
        self.generated.append(token)
        text = self.visible_text()
        if self.stopped:
            return ""
        cut = min((text.find(s) for s in self.stops if s and s in text), default=-1)
        if cut >= 0:
            text = text[:cut]
            self.stopped = True
        hold = max((len(s) - 1 for s in self.stops if s), default=0)
        stable = text if self.stopped else text[:max(len(self.previous), len(text) - hold)]
        if stable.endswith("�") and not self.stopped:
            return ""
        if stable.startswith(self.previous) and len(stable) > len(self.previous):
            out = stable[len(self.previous):]
            self.previous = stable
            return out
        return ""

    def flush(self):
        """Text still held back once generation has ended."""
        if self.stopped:
            return ""
        text = self.visible_text()
        if text.startswith(self.previous) and len(text) > len(self.previous):
            out = text[len(self.previous):]
            self.previous = text
            return out
        return ""


def _chunk(text):
    return CompletionOutput(type="streaming_chunk", data=CompletionResult(text=text))


def _final(finish, prompt_tokens, completion_tokens, text=""):
    return CompletionOutput(type="final_result", data=CompletionResult(
        text=text, finish_reason=finish, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens))


class TTLabRunner(BaseDeviceRunner):
    restart_on_unhealthy = True

    def __init__(self, device_id):
        super().__init__(device_id)
        # A multiprocessing fork inherits uvicorn's handler; restore process
        # termination semantics so systemd and the scheduler can stop this worker.
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        if str(device_id).strip("() ") != "0":
            # One worker owns the whole card set: device 0 alone, or 0-3 at 32 tiles
            # across four chips. The native worker enumerates them itself.
            raise ValueError("tt-lab runs a single worker; configure DEVICE_IDS=(0)")
        # The native worker kills itself on a request longer than its own capacity,
        # so refuse to start with a context the API layer would over-accept.
        native_context = int(os.environ.get("TT_LAB_CONTEXT", CONTEXT_LENGTH))
        if native_context != CONTEXT_LENGTH:
            raise ValueError(
                f"TT_LAB_CONTEXT={native_context} disagrees with MAX_MODEL_LENGTH={CONTEXT_LENGTH}"
            )
        self.gemma = self.settings.model_runner == "tt-lab-gemma"
        self.vocab = 262144 if self.gemma else 201088
        self.slots = int(os.environ.get("TT_GEMMA31_SLOTS", "1")) if self.gemma else 1
        if self.gemma and "TT_LAB_DEVICE" in os.environ:
            raise ValueError("Gemma 4 31B owns all four cards; unset TT_LAB_DEVICE")
        if not 1 <= self.slots <= 16:
            raise ValueError("TT_GEMMA31_SLOTS must be 1..16")
        if self.slots > 1 and self.settings.vllm.max_num_seqs != self.slots:
            raise ValueError(f"MAX_NUM_SEQS must equal TT_GEMMA31_SLOTS={self.slots}")
        self.batched = self.slots > 1
        self.process = None
        self._closing = False
        self.tokenizer = AutoTokenizer.from_pretrained(self.settings.model_weights_path)
        self.timeout = self.settings.request_processing_timeout_seconds
        # --serve-batch state: per-request sinks, the reader thread, the slot limit.
        self._sinks = {}
        self._sinks_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._next_id = 0
        self._reader = None
        self._reader_error = None
        self._slot_limit = None
        atexit.register(self.close_device)

    def _read_words(self, count, deadline):
        """Read `count` int32 words; `deadline` None blocks (the batch reader thread)."""
        data = bytearray()
        while len(data) < 4 * count:
            remaining = None if deadline is None else deadline - time.monotonic()
            if (remaining is not None and remaining <= 0) or \
                    not select.select([self.process.stdout], [], [], remaining)[0]:
                self.close_device()
                raise TimeoutError("tt-lab silicon worker timed out")
            chunk = os.read(self.process.stdout.fileno(), 4 * count - len(data))
            if not chunk:
                raise RuntimeError(f"tt-lab silicon worker exited: {self.process.poll()}; inspect its stderr log")
            data.extend(chunk)
        return struct.unpack(f"<{count}i", data)

    def _read_token(self, deadline):
        return self._read_words(1, deadline)[0]

    def _write(self, packet):
        # FileIO writes may be partial for a large prompt.
        view = memoryview(packet)
        with self._write_lock:
            while view:
                written = os.write(self.process.stdin.fileno(), view)
                view = view[written:]

    async def warmup(self):
        binary = os.environ["TT_LAB_BINARY"]
        model = os.environ["TT_LAB_GGUF"]
        sidecar = os.environ["TT_LAB_TTQ"]
        if self.gemma:
            command = [binary, "gemma", "-m", model, "--ttq", sidecar, "--native-device",
                       "--serve-batch" if self.batched else "--serve", "--context", str(CONTEXT_LENGTH)]
        else:
            command = [binary, "serve", "-m", model, "--ttq", sidecar, "--device"]
            # 32 tiles mean four chips unless TT_LAB_SINGLE_CHIP32 is set; 120B needs four.
            if os.environ.get("TT_LAB_TILES"):
                command += ["--tiles", os.environ["TT_LAB_TILES"]]
        self.process = subprocess.Popen(
            [sys.executable, str(Path(__file__).with_name("tt_lab_child.py")), str(os.getpid())] + command,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0,
        )
        self._closing = False
        process = self.process
        def watch_child():
            process.wait()
            if not self._closing:
                # Let the scheduler recover even when the child dies while idle.
                # Forked workers inherit uvicorn's SIGTERM handler, which only
                # sets a server flag. Exit directly so multiprocessing observes
                # the death and the scheduler can replace this worker.
                os._exit(1)
        threading.Thread(target=watch_child, daemon=True).start()
        deadline = time.monotonic() + self.timeout
        if self.batched:
            if self._read_words(2, deadline) != (READY, self.slots):
                raise RuntimeError("Invalid tt-lab batch worker readiness handshake")
            self._reader = threading.Thread(target=self._read_replies, daemon=True)
            self._reader.start()
        elif self._read_token(deadline) != READY:
            raise RuntimeError("Invalid tt-lab worker readiness handshake")
        # Exercise actual device inference before publishing model readiness.
        from domain.completion_request import CompletionRequest
        self.run([CompletionRequest(prompt="Hello", max_tokens=1, temperature=0)])
        served = os.environ.get("SERVED_MODEL_NAME", "gpt-oss")
        mode = f", {self.slots} sequences batched" if self.batched else ""
        self.logger.info(f"{served} ready on Blackhole{mode}; persistent tt-lab PID={self.process.pid}")
        return True

    # ---- one sequence at a time (GPT-OSS, and Gemma with one slot) ----

    def _generate(self, request):
        if not self.health_check():
            raise RuntimeError("tt-lab silicon process is not running")
        tokens, limit = validate_request(request, self.tokenizer)
        self._write(struct.pack("<II", len(tokens), limit) + struct.pack(f"<{len(tokens)}i", *tokens))
        deadline = time.monotonic() + self.timeout
        stream = TextStream(self.tokenizer, request)
        finish = "length"
        while True:
            token = self._read_token(deadline)
            if token in (STOPPED, LENGTH):
                finish = "stop" if token == STOPPED or stream.stopped else "length"
                break
            if token < 0 or token >= self.vocab:
                self.close_device()
                raise RuntimeError("Invalid token from silicon worker")
            text = stream.push(token)
            if text:
                yield _chunk(text)
        text = stream.flush()
        if text:
            yield _chunk(text)
        yield _final(finish, len(tokens), len(stream.generated))

    # ---- several sequences (Gemma --serve-batch) ----

    def _read_replies(self):
        """Reader thread: route (id, value) pairs to their requests."""
        try:
            while True:
                rid, value = self._read_words(2, None)
                if value >= self.vocab or value < CANCELLED or value == READY:
                    raise RuntimeError(f"Invalid reply from silicon worker: {value}")
                with self._sinks_lock:
                    sink = self._sinks.get(rid)
                    if value < 0:
                        self._sinks.pop(rid, None)
                if sink is not None:
                    sink(value)
        except Exception as exc:
            self._reader_error = exc
            with self._sinks_lock:
                sinks, self._sinks = list(self._sinks.values()), {}
            for sink in sinks:
                sink(exc)

    def _submit(self, tokens, limit, sink):
        """Register `sink` for this request's replies and send it to the worker."""
        if not self.health_check():
            raise RuntimeError("tt-lab silicon process is not running")
        with self._sinks_lock:
            self._next_id = (self._next_id + 1) & 0x7FFFFFFF
            rid = self._next_id
            self._sinks[rid] = sink
        self._write(struct.pack(f"<4i{len(tokens)}i", 1, rid, len(tokens), limit, *tokens))
        return rid

    def _cancel(self, rid):
        if self.health_check():
            self._write(struct.pack("<2i", 2, rid))

    async def _generate_batched(self, request):
        """Stream one request through the batch worker. The slot is held from
        submit until the worker reports this request finished or cancelled."""
        tokens, limit = validate_request(request, self.tokenizer)
        loop = asyncio.get_running_loop()
        if self._slot_limit is None:
            self._slot_limit = asyncio.Semaphore(self.slots)
        slots = self._slot_limit
        await slots.acquire()
        replies = asyncio.Queue()

        def sink(value):
            # Reader thread: a terminal value also frees the slot.
            if isinstance(value, Exception) or value < 0:
                loop.call_soon_threadsafe(slots.release)
            loop.call_soon_threadsafe(replies.put_nowait, value)

        try:
            rid = self._submit(tokens, limit, sink)
        except BaseException:
            slots.release()
            raise
        stream = TextStream(self.tokenizer, request)
        done = False
        finish = "length"
        try:
            while True:
                value = await asyncio.wait_for(replies.get(), timeout=self.timeout)
                if isinstance(value, Exception):
                    done = True
                    raise RuntimeError(f"tt-lab silicon worker failed: {value}")
                if value < 0:
                    done = True
                    finish = "stop" if value == STOPPED or stream.stopped else "length"
                    break
                text = stream.push(value)
                if text:
                    yield _chunk(text)
                if stream.stopped:
                    # A stop string ended the text: free the slot now.
                    done = True
                    finish = "stop"
                    self._cancel(rid)
                    break
            text = stream.flush()
            if text:
                yield _chunk(text)
            yield _final(finish, len(tokens), len(stream.generated))
        finally:
            if not done:
                # Client went away (task cancelled or stream closed): stop decoding it.
                self._cancel(rid)

    async def _complete_batched(self, request):
        chunks = [c async for c in self._generate_batched(request)]
        return [_final(chunks[-1]["data"].finish_reason, chunks[-1]["data"].prompt_tokens,
                       chunks[-1]["data"].completion_tokens, "".join(c["data"].text for c in chunks))]

    def _run_blocking_batched(self, request):
        """Synchronous request (warmup) through the batch worker."""
        import queue
        replies = queue.Queue()
        tokens, limit = validate_request(request, self.tokenizer)
        self._submit(tokens, limit, replies.put)
        stream = TextStream(self.tokenizer, request)
        deadline = time.monotonic() + self.timeout
        while True:
            value = replies.get(timeout=max(0.0, deadline - time.monotonic()))
            if isinstance(value, Exception):
                raise RuntimeError(f"tt-lab silicon worker failed: {value}")
            if value < 0:
                break
            stream.push(value)
        text = stream.previous + stream.flush()
        return _final("stop" if value == STOPPED else "length", len(tokens), len(stream.generated), text)

    # ---- device runner interface ----

    def run(self, requests):
        results = []
        for request in requests:
            if self.batched:
                results.append(self._run_blocking_batched(request))
                continue
            chunks = list(self._generate(request))
            results.append(_final(chunks[-1]["data"].finish_reason, chunks[-1]["data"].prompt_tokens,
                                  chunks[-1]["data"].completion_tokens,
                                  "".join(c["data"].text for c in chunks)))
        return results

    async def _run_async(self, requests):
        if self.batched:
            # The dynamic-batch worker: a stream for streaming requests, a list otherwise.
            if getattr(requests[0], "stream", False):
                return self._generate_batched(requests[0])
            return await self._complete_batched(requests[0])

        async def stream():
            for chunk in self._generate(requests[0]):
                yield chunk
                await asyncio.sleep(0)
        return stream()

    def health_check(self, deep=False):
        alive = self.process is not None and self.process.poll() is None
        if self.batched and self._reader is not None:
            alive = alive and self._reader.is_alive() and self._reader_error is None
        return alive

    def close_device(self):
        self._closing = True
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            for pipe in (self.process.stdin, self.process.stdout):
                if pipe is not None:
                    pipe.close()
            self.process = None
        return True
