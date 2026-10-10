# Gemma 4 31B text inference on four Blackhole cards

This deployment serves `google/gemma-4-31B-it` through the tt-lab runner
(`MODEL_RUNNER=tt-lab-gemma`, `tt-media-server/tt_model_runners/tt_lab_runner.py`)
with a persistent `tt-lab gemma --native-device` child process. All 60 layers
run on four P150 cards (64 Tensix workers). The server does tokenization, chat
templating and the OpenAI-compatible API (`/v1/chat/completions`,
`/v1/completions`, SSE streaming). Decoding is greedy only.

Two modes, chosen by the env file:

| | `server.env` | `server-batch.env` |
|---|---|---|
| native worker | `gemma --serve` | `gemma --serve-batch` |
| sequences | one at a time; the others queue | `TT_GEMMA31_SLOTS` (8) at once, continuous batching |
| prefill | 128-token D64 groups | 32-token groups, one request at a time, between decode steps |
| decode | single sequence | active sequences together through tt-lab's exact batch-decode path (`forward_batch`) |
| server worker | serial `device_worker` | dynamic-batch `device_worker` (`USE_DYNAMIC_BATCHER=true`) |
| cancel | the stream is drained to keep the pipe in sync | a disconnect or stop string sends `cancel`; the slot is freed |
| prefix caching | `TT_GEMMA_PREFIX_CACHE=1`, against the previous request | `TT_GEMMA_PREFIX_CACHE=1`, per slot; a request goes to the free slot with the longest shared prefix |
| `ignore_eos` | rejected (HTTP 400) | with `TT_LAB_IGNORE_EOS=1` and a tt-lab build from `ea8a211` on |

The native worker comes from tt-lab branch `work/gemma31-qb2-serve` (the
`--serve-batch` front end on top of `work/gemma31-qb2`). The env files carry its
accepted kernel flags; see that branch's `docs/gemma31-qb2-*.md`.

## Settings

- `TT_LAB_BINARY`, `TT_LAB_GGUF`, `TT_LAB_TTQ`: the native worker and model files.
- `MAX_MODEL_LENGTH` = `TT_LAB_CONTEXT`: the native context and the request limit
  (prompt + `max_tokens`). The runner refuses to start if they differ.
- `SERVED_MODEL_NAME`: the only accepted `model` field.
- `TT_GEMMA31_SLOTS` (2..16) selects `--serve-batch`; it must equal `MAX_NUM_SEQS`.
- `TT_LAB_DEVICE` must be unset: the worker owns all four cards. No other tt-lab or
  tt-metal workload may use any card while it runs.
- `DEVICE_IDS=(0)`: one server worker drives the whole card set.

## Start

```sh
bash deploy/tt-lab-gemma31/start.sh                                         # one sequence
bash deploy/tt-lab-gemma31/start.sh deploy/tt-lab-gemma31/server-batch.env  # eight
curl http://127.0.0.1:8002/tt-liveness
curl http://127.0.0.1:8002/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"google/gemma-4-31B-it","messages":[{"role":"user","content":"What is 17 times 23?"}],"max_tokens":32,"temperature":0}'
```

Weight loading and firmware start take a few minutes before readiness.

## `--serve-batch` pipe protocol

Little-endian int32 words. Host to worker: submit `1, id, n, limit, tokens[n]`;
submit with flags `3, id, n, limit, flags, tokens[n]` (flag 1: ignore end tokens);
cancel `2, id`. Worker to host: `(id, value)` pairs, a token (>= 0), `-1` stopped
at an end token, `-2` reached the limit, `-4` cancelled (also for an unknown id),
and `(-3, slots)` once ready. At most `slots` requests may be outstanding; the
runner holds a slot per request from submit to its final reply.

## Tests (no device)

```sh
PY=/home/ttuser/workspace/tt-inference-server/.venv-ttlab/bin/python
(cd tt-media-server && IS_GALAXY=false $PY -m pytest -q tests/test_tt_lab_batch_runner.py)
PYTHONPATH=/home/ttuser/workspace/gemma-reference/python \
  $PY -m pytest -q deploy/tt-lab-gemma31/test_adapter.py deploy/tt-lab/test_runner_protocol.py
```

`IS_GALAXY=false` matters: with the default (`true`) importing the settings runs
`tt-smi -glx_list_tray_to_device` for device discovery.

`test_tt_lab_batch_runner.py` runs the runner against
`tests/fixtures/fake_tt_lab_batch_worker.py`, which speaks the protocol above:
more streams than slots, continuous admission, non-streamed requests, end tokens,
`ignore_eos`, cancellation (streaming and while waiting for a slot), stop strings,
and a worker crash. Device validation: `DEVICE_TEST_PLAN.md`; earlier single-sequence
results: `VALIDATION.md`.
