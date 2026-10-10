# Gemma 4 31B four-card API validation — September 26, 2026

Server: this branch (`work/gemma31-qb2-serving`), `deploy/tt-lab-gemma31`, port
8002, `MAX_MODEL_LENGTH=32768`. Native worker: tt-lab `work/gemma31-qb2`
(`cc7d9a8`), matrix-only firmware with the flags in `server.env`.

- Adapter tests: 3 new 31B cases (served model, 32K limit, four-card guard)
  and the 4 existing 26B cases pass.
- `/tt-liveness`: model ready, one worker, no errors or restarts.
- Chat completion "What is 17 times 23? Answer briefly." -> `391`
  (26 prompt tokens, `finish_reason=stop`, 2.4 s end to end).
- Streaming completion ("Write a Python function that reverses a string.")
  delivers incremental chunks.
- A request naming `google/gemma-4-26B-A4B-it` is rejected with
  "This worker serves only google/gemma-4-31B-it".

Long-context behaviour (to 131,072 positions) is validated on the native
worker directly; see tt-lab `docs/gemma31-qb2-long-context.md`.

# Batch serving validation — October 10, 2026

Server: `server-batch.env` (8 slots, `MAX_MODEL_LENGTH=32768`, chunked prefill 2048, prefix
cache). tt-lab `work/gemma31-qb2-serve` `a4c97e8` (`work/gemma31-qb2` `989bd0a`, round 3: D64
prefill per request, the exact batch-decode path with its round-3 changes, prefill resumes and
chunks on the 256-token command grid), SHA-256 `5caddd77`. Logs:
`/home/ttuser/workspace/bringup-logs-gemma31/serving-r3/` (`final/` for this binary).

- **Native worker** (`gemma31_serve_batch_check.py`, `--serve-batch` against `--serve`):
  token-identical at 4K (7 prompts, 25–3,000 tokens, chunk 512, prefix cache, plus a follow-up
  turn that reuses 1,536 cached tokens) and at 32K (30,000 / 16,000 / 4,500, chunk 2,048), each
  with a mid-stream cancel, the next request identical, and ignore_eos.
- **Greedy identity through the API** (`verify_cli_identity.py`, CLI reference of 10 prompts up to
  28K tokens): 40 of 40 requests (sequential plain and streamed, then concurrency 8) match the
  CLI's text and completion-token counts.
- **Multi-turn chat** (4,028 → 4,125-token turns): text identical with and without the prefix
  cache. These turns do not hit the cache: past 1,088 positions a slot resumes only at a 256-token
  point within one 64-position window block before its cached end (tt-lab R3.4).
- **Standard benchmark** (`run.py --workflow benchmarks --tools vllm`,
  `benchmark-requirements.json`, the previous build `4a32ede`; the round-3 prefix fix changes only
  resumes): 0 failed requests at concurrency 1/8/16.

| ISL / OSL | Concurrency | Output tok/s | Mean TTFT | Mean TPOT |
| --- | ---: | ---: | ---: | ---: |
| 1024 / 128 | 1 | 29.1 | 1.19 s | 25.2 ms |
| 1024 / 128 | 8 | 70.8 | 5.31 s | 71.8 ms |
| 1024 / 128 | 16 | 70.9 | 18.7 s | 71.8 ms |
| 128 / 512 | 1 | 41.1 | 0.21 s | 24.0 ms |
| 128 / 512 | 8 | 204.4 | 0.91 s | 37.4 ms |
| 128 / 512 | 16 | 204.8 | 19.2 s | 37.3 ms |

  Concurrency 16 with 8 slots queues half the requests.
- **Found during this validation and fixed in tt-lab** (round 3, R3.4): prefix caching almost
  never hit past 1,088 positions, and chunks ending off the 256-token prefill grid changed a
  token of a 1,500-token prompt.
- Stopping: kill the `start.py` processes by PID (the `nohup` PID is not the server's); do not use
  `pkill -f`/`pgrep -f` with a pattern your own shell command contains.
