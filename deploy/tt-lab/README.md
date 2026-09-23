# GPT-OSS-20B / tt-lab / single Blackhole deployment

This deployment uses the existing `tt-media-server/main.py` FastAPI application,
LLM service, scheduler, device worker, and API routes in tt-inference-server.
`MODEL_RUNNER=tt-lab-gpt-oss` selects `TTLabRunner`, which starts a persistent
`tt-lab serve ... --device` subprocess. The subprocess exclusively owns the first
Blackhole card and runs every transformer layer on 32 Tensix tiles. Python
handles tokenization, chat formatting, and API transport; there is no CPU or
simulator inference fallback. No QSFP cables are needed for this single-chip path.

## Start and inspect

```sh
sudo systemctl start ttlab-inference
journalctl -u ttlab-inference -f
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"openai/gpt-oss-20b","messages":[{"role":"user","content":"What is 2+2?"}],"temperature":0,"max_tokens":256}'
```

The tokenizer is loaded locally with Hugging Face offline mode enabled.
The listener binds to `0.0.0.0:8000` for LAN access, with authentication disabled.
The current LAN endpoint is `http://192.168.2.231:8000/v1`. Set `HTTP_HOST` in
`server.env` to change the bind address.
Stop with `sudo systemctl stop ttlab-inference`. Do not run other runtimes or
management commands that open device 0 while it is exclusively owned.

## Supported scope

* GPT-OSS-20B, one chip, 32 worker tiles, one sequence at a time.
* Greedy decoding (`temperature: 0` or omitted); 4096 total prompt/output tokens.
* Chat and text completion routes; SSE streaming and sequential queued requests.
* GPT-OSS Harmony analysis is suppressed from assistant content; final-channel
  text is returned. The output-token budget includes analysis tokens, so a short
  budget can finish with empty content and `finish_reason: length`.
* Token usage includes generated analysis/channel tokens, not just visible text.
* Sampling, adapters, and multi-choice generation are not supported.

## Local paths and configuration

`server.env` contains paths and runtime settings. `start.sh` is the entry point.
The prefill configuration enables `TT_LAB_SINGLE_CHIP32`, `TT_LAB_BATCH_PREFILL`,
`TT_LAB_BATCH_DENSE`, and `TT_LAB_PREFILL_PAIR_ONLY`, with a 128-token prompt chunk.
These switches share dense and expert weight loads across prompt tokens; decode
remains serial. They do not increase request concurrency. Unset the switches
(setting them to `0` still enables them) to return to eight-tile serial execution.
The isolated Python environment is `../../.venv-ttlab` relative to this directory.
Model files live under `/home/ttuser/models/`; GGUF provenance/hash and build/test
logs are in `/home/ttuser/workspace/bringup-logs/`.

The host uses IOMMU passthrough. `TT_LAB_HUGEPAGE_DIR=/dev/hugepages/ttlab` selects
an explicit 2 MiB hugetlb allocation for the roughly 396 KiB logits DMA buffer,
which is still pinned using the KMD PIN_PAGES ioctl. `/etc/sysctl.d/90-ttlab-hugepages.conf`
reserves eight 2 MiB pages, and the service creates the private hugetlb directory.
The normal tt-lab allocation path is unchanged when this variable is absent.

Driver source is pinned to tt-kmd commit
`3d5abc9f8a916bacc761a42cdd194e1ce0b3045c` (exclusive-open support) and installed
through DKMS. No board firmware flash was performed.

Build tt-lab with:

```sh
cd /home/ttuser/workspace/tt-lab
uv run --python 3.12 --no-project python make.py --env CXX=g++-12 :build :test
```

SFPI 7.76.0 lives at `/home/ttuser/sfpi-7.76.0`.

## Native pipe protocol

`tt-lab serve -m MODEL --ttq SIDECAR --device` emits little-endian int32 `-3`
after loading weights and starting firmware. Each request is two little-endian
uint32 values (prompt token count, output token limit), followed by int32 token
IDs. Responses are int32 generated tokens terminated by `-1` (EOS) or `-2`
(length). Malformed input terminates the worker. Requests reuse model weights and
reset token positions to zero; attention only reads the active sequence's KV.
The pipe is private to the server worker; there is no separate HTTP proxy service.

## Verified on this machine (2026-09-18)

* Full GGUF SHA-256 matched the Hugging Face LFS object.
* All 24 layers on silicon: 201,088 logits matched the device proxy bit-for-bit
  for each of two consecutive token commands.
* Live `/v1/chat/completions`: arithmetic returned `4`, capital question returned
  `Paris`, repeated arithmetic returned `4` again, and SSE arithmetic returned `7`.
* Native worker holds `/dev/tenstorrent/0` (Blackhole p150b at `0000:c1:00.0`).
* Service `ttlab-inference` is running and enabled for boot.

Evidence: `/home/ttuser/workspace/bringup-logs/verification-summary.json` and
`live-api-evidence.json`. Run `verify_live.py` with `.venv-ttlab/bin/python` to
repeat the live API checks. `test_runner_protocol.py` is a separate adapter unit
test suite using token fixtures; it is not evidence of hardware inference.

## Request validation and recovery

Clients can use `http://192.168.2.231:8000/v1` with model
`openai/gpt-oss-20b`. Omit temperature or set it to `0`, and use `n: 1`.
This is a greedy-only backend: unsupported sampling settings receive HTTP 400
with an explanation. Validation runs before queue submission and before SSE
headers, so invalid requests do not consume the worker's error budget. Empty
prompts, invalid token IDs, unknown models, and requests exceeding the 4096-token
context also receive HTTP 400. Both chat and text completions use this check.

Requests run sequentially on the chip, with up to eight waiting in the queue.
Excess load receives HTTP 429; retry with backoff. A disconnected stream is
finished internally to leave the native token pipe synchronized for the next
request. Completed or rejected synchronous requests do not accumulate unused
cancellation messages or per-request result queues.

The native child has a Linux parent-death signal so it cannot keep owning the
chip after its Python parent dies. If the native child exits unexpectedly, its
Python worker exits too. The scheduler checks workers every five seconds,
including during startup. On worker failure this single-device deployment exits
for systemd recovery (`Restart=on-failure`, ten-second backoff,
`KillMode=control-group`). Recovery recreates **all** IPC queues: reusing a queue
after killing its reader can retain a locked multiprocessing semaphore.
Readiness is published only after another actual silicon inference warmup.
Pending non-streaming requests are sent HTTP 503 before restart where possible;
a streaming or interrupted connection may instead close. Clients should retry
transient 429/503 or connection failures with backoff.

Run regression checks separately because the upstream scheduler test module
installs process-wide configuration mocks:

```sh
.venv-ttlab/bin/python -m pytest deploy/tt-lab/test_runner_protocol.py -q
.venv-ttlab/bin/python -m pytest tt-media-server/tests/test_scheduler.py -q
.venv-ttlab/bin/python -m pytest tt-media-server/tests/test_base_service.py tt-media-server/tests/test_chat_api.py -q
.venv-ttlab/bin/python deploy/tt-lab/verify_live.py
.venv-ttlab/bin/python deploy/tt-lab/verify_resilience.py
```

`verify_resilience.py --faults` additionally kills the native worker and Python
worker, and tests a crash during generation. Run it during a maintenance window;
it deliberately interrupts serving. Evidence is saved to
`/home/ttuser/workspace/bringup-logs/resilience-evidence.json`.
