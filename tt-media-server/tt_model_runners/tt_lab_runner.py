# SPDX-License-Identifier: Apache-2.0
"""Persistent tt-lab silicon backend for the existing TT media-server LLM pipeline.

The subprocess runs all transformer layers on one Blackhole chip. Python only
encodes/decodes tokens and transports requests. There is no CPU inference fallback.
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

from domain.tt_lab_validation import validate_request

from domain.completion_response import CompletionOutput, CompletionResult
from transformers import AutoTokenizer
from tt_model_runners.base_device_runner import BaseDeviceRunner


class TTLabRunner(BaseDeviceRunner):
    restart_on_unhealthy = True

    def __init__(self, device_id):
        super().__init__(device_id)
        # A multiprocessing fork inherits uvicorn's handler; restore process
        # termination semantics so systemd and the scheduler can stop this worker.
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        if str(device_id).strip("() ") != "0":
            raise ValueError("tt-lab currently selects physical device 0; configure DEVICE_IDS=(0)")
        self.process = None
        self._closing = False
        self.tokenizer = AutoTokenizer.from_pretrained(self.settings.model_weights_path)
        self.timeout = self.settings.request_processing_timeout_seconds
        atexit.register(self.close_device)

    def _read_token(self, deadline):
        data = bytearray()
        while len(data) < 4:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.process.stdout], [], [], remaining)[0]:
                self.close_device()
                raise TimeoutError("tt-lab silicon worker timed out")
            chunk = os.read(self.process.stdout.fileno(), 4 - len(data))
            if not chunk:
                raise RuntimeError(f"tt-lab silicon worker exited: {self.process.poll()}; inspect its stderr log")
            data.extend(chunk)
        return struct.unpack("<i", data)[0]

    async def warmup(self):
        binary = os.environ["TT_LAB_BINARY"]
        model = os.environ["TT_LAB_GGUF"]
        sidecar = os.environ["TT_LAB_TTQ"]
        self.process = subprocess.Popen(
            [sys.executable, str(Path(__file__).with_name("tt_lab_child.py")), str(os.getpid()), binary, "serve", "-m", model, "--ttq", sidecar, "--device"],
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
        if self._read_token(time.monotonic() + self.timeout) != -3:
            raise RuntimeError("Invalid tt-lab worker readiness handshake")
        # Exercise actual device inference before publishing model readiness.
        from domain.completion_request import CompletionRequest
        self.run([CompletionRequest(prompt="Hello", max_tokens=1, temperature=0)])
        self.logger.info(f"GPT-OSS-20B ready on Blackhole; persistent tt-lab PID={self.process.pid}")
        return True

    def _generate(self, request):
        if not self.health_check():
            raise RuntimeError("tt-lab silicon process is not running")
        tokens, limit = validate_request(request, self.tokenizer)
        packet = struct.pack("<II", len(tokens), limit) + struct.pack(f"<{len(tokens)}i", *tokens)
        # FileIO writes may be partial for a large prompt.
        view = memoryview(packet)
        while view:
            written = os.write(self.process.stdin.fileno(), view)
            view = view[written:]
        deadline = time.monotonic() + self.timeout
        generated = []
        harmony = isinstance(request.prompt, str) and "<|start|>assistant" in request.prompt

        def visible_text():
            if not harmony:
                return self.tokenizer.decode(generated, skip_special_tokens=True)
            raw = self.tokenizer.decode(generated, skip_special_tokens=False)
            # GPT-OSS emits analysis and then a final channel. Do not mix private
            # reasoning/channel delimiters into the API assistant content.
            marker = "<|channel|>final<|message|>"
            if marker in raw:
                return raw.rsplit(marker, 1)[1].split("<|return|>", 1)[0].split("<|end|>", 1)[0]
            if raw.startswith("final<|message|>"):
                return raw[len("final<|message|>"):].split("<|return|>", 1)[0]
            # Some prompts preselect the final channel and receive bare text.
            if request.prompt.endswith("final<|message|>"):
                return self.tokenizer.decode(generated, skip_special_tokens=True)
            return ""
        previous = ""
        stopped = False
        stops = [request.stop] if isinstance(request.stop, str) else (request.stop or [])
        finish = "length"
        while True:
            token = self._read_token(deadline)
            if token in (-1, -2):
                finish = "stop" if token == -1 or stopped else "length"
                break
            if token < 0 or token >= 201088:
                self.close_device()
                raise RuntimeError("Invalid token from silicon worker")
            generated.append(token)
            text = visible_text()
            if not stopped:
                cut = min((text.find(s) for s in stops if s and s in text), default=-1)
                if cut >= 0:
                    text = text[:cut]
                    stopped = True
                # Hold a possible stop-string prefix and incomplete UTF-8 character.
                hold = max((len(s)-1 for s in stops if s), default=0)
                stable = text if stopped else text[:max(len(previous), len(text)-hold)]
                if stable.endswith("\ufffd") and not stopped:
                    continue
                if stable.startswith(previous) and len(stable) > len(previous):
                    yield CompletionOutput(type="streaming_chunk", data=CompletionResult(text=stable[len(previous):]))
                    previous = stable
        if not stopped:
            text = visible_text()
            if text.startswith(previous) and len(text) > len(previous):
                yield CompletionOutput(type="streaming_chunk", data=CompletionResult(text=text[len(previous):]))
        yield CompletionOutput(type="final_result", data=CompletionResult(text="", finish_reason=finish, prompt_tokens=len(tokens), completion_tokens=len(generated)))

    def run(self, requests):
        results = []
        for request in requests:
            chunks = list(self._generate(request))
            results.append(CompletionOutput(type="final_result", data=CompletionResult(
                text="".join(c["data"].text for c in chunks),
                finish_reason=chunks[-1]["data"].finish_reason,
                prompt_tokens=chunks[-1]["data"].prompt_tokens,
                completion_tokens=chunks[-1]["data"].completion_tokens,
            )))
        return results

    async def _run_async(self, requests):
        async def stream():
            for chunk in self._generate(requests[0]):
                yield chunk
                await asyncio.sleep(0)
        return stream()

    def health_check(self, deep=False):
        return self.process is not None and self.process.poll() is None

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
