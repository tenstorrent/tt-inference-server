# agentic_bench

Long-running, multi-turn, **agentic-shape** soak benchmark for
OpenAI-compatible endpoints — designed to exercise a production LLM serving
stack (e.g. DeepSeek-R1-0528 on cpp_server) the way real Cursor / Claude
Code / OpenHands traffic would, for hours to days.

## Vision

`vllm bench serve` runs one fixed rate, one fixed ISL/OSL, one burst — great
for pinning down a single operating point, not for a realistic multi-day
soak. Real agentic traffic has:

- **Multi-turn conversations** where each turn replays the growing chat
  history.
- **Large shared system prompts per session** (tool schemas, coding rules,
  project context) reused turn after turn, so prefix caching matters.
- **Stochastic arrivals** — users/agents start sessions at random times.
- **Variable** prompt and output sizes across turns.

GuideLLM already has the building blocks (native `turns` / `prefix_buckets`
multi-turn synthetic data, Poisson profile, per-request TTFT/ITL records).
This tool is a thin wrapper that wires sensible agentic-shape defaults into
one `guidellm benchmark` invocation and lets you tune concurrency via
Little's Law instead of magic rate numbers.

## Quickstart

```bash
# one-time: install the bundled GuideLLM into your current env
pip install cpp_server/guidellm

# start cpp_server (or any OpenAI-compatible endpoint) on :8000, then:
./cpp_server/benchmarks/agentic_bench/agentic_bench.sh
```

That runs an 8-hour Poisson soak against `http://localhost:8000` with the
`DeepSeek-R1-0528` tokenizer and defaults tuned for HW sized for ~64
concurrent users, and writes `./runs/<ts>/benchmarks.json` (GuideLLM's
native report) when it finishes.

Example overrides:

```bash
./agentic_bench.sh --duration 3600 --turns 10                           # 1 h run, 10 turns/conv
./agentic_bench.sh --target-concurrency 128 --avg-turn-sec 30           # target 128 inflight, 30s/turn
./agentic_bench.sh --rate 2.0                                           # explicit rate, skip Little's Law
./agentic_bench.sh --target http://other-host:9000 --model some/Other    # different endpoint & tokenizer
./agentic_bench.sh --dry-run                                             # print the guidellm invocation
./agentic_bench.sh -- --random-seed 7                                    # forward extra flags to guidellm
```

Help: `./agentic_bench.sh --help`.

## Why these defaults

- **Poisson, not constant rate.** Real user/agent arrivals are stochastic.
  Constant rate is an artifact that hides burst behavior and makes tail
  latencies look better than they really are.
- **Rate derived from target concurrency via Little's Law.**
  `rate = target_concurrency / avg_turn_latency`. By default
  `64 / 40s = 1.6 req/s`. Watch your Grafana active-sessions metric and
  either set `--rate` explicitly or tune `--avg-turn-sec` to match your
  observed latency.
- **Pool of 8 shared prefixes, not 1 and not per-user.** One prefix →
  unrealistic 100% prefix-cache hit (server looks artificially fast).
  Per-user → 0% cross-user reuse (unrealistically slow). A small pool
  approximates a deployment with a handful of distinct projects/personas
  and yields realistic cache-hit rates. Prefix generation is serial and
  eager — see `--help` for the upper bound on `prefix_count x prefix_tokens`.
- **6 turns per conversation.** Most Claude-Code-style sessions are a
  handful of turns before the user changes task. Override with `--turns`.
- **`prompt_tokens=2000, stdev=1500`, clamped to [100, 30000].** Heavy-input-biased,
  matches a Cursor-like flow where each user turn carries growing context.
  GuideLLM uses *truncated normal*, not lognormal.
- **`output_tokens=800, stdev=600`, clamped to [64, 4096].** Most replies
  are short; occasional long code blocks are clamped.
- **No rolling chunks.** One `guidellm benchmark` invocation writes one
  `benchmarks.json` at the end. SIGINT (Ctrl-C) kills it with no partial
  report — acceptable.
- **No /metrics or tt-smi scraping inside this tool.** Capture those
  with the shared [cpp_server/monitoring/](../../monitoring/) stack
  (Prometheus + Grafana).

## CLI reference

See `./agentic_bench.sh --help` for the full list. Highlights:

```
--target URL                  OpenAI base URL              [http://localhost:8000]
--model NAME                  Model id                     [deepseek-ai/DeepSeek-R1-0528]
--duration SEC                Total run seconds            [28800]   # 8 h

Concurrency (Little's Law: target_concurrency = rate x avg_turn_latency):
  --rate RPS                  Poisson mean req/s (explicit)
  --target-concurrency N      Target in-flight turns       [64]
  --avg-turn-sec S            Expected per-turn latency    [40]

Agentic-shape synthetic data:
  --turns N                   Turns per conversation       [6]
  --prompt-tokens N           Mean user-prompt tokens/turn [2000]
  --output-tokens N           Mean assistant-output tokens [800]
  --prefix-tokens N           Shared system prompt size    [4000]
  --prefix-count N            Pool size                    [8]

Auth:
  --api-key KEY               also honors $OPENAI_API_KEY

--dry-run                     Print the guidellm command and exit
```

Everything after `--` is forwarded verbatim to `guidellm benchmark`, so any
advanced GuideLLM flag is reachable without re-plumbing. E.g. to switch
to closed-pool concurrency:

```bash
./agentic_bench.sh --rate 64 -- --profile concurrent
```

## Outputs

```
runs/<ts>/
├── config.txt              # full flag snapshot for reproducibility (api key masked)
└── benchmarks.json         # GuideLLM's native report written at end-of-run
```

Inspect `benchmarks.json` with `jq` or
`from guidellm.benchmark import GenerativeBenchmarksReport` if you want to
post-process.

## Data collection and debugging

Correlate against server-side signals:

- **`response_id`.** Each request record in `benchmarks.json` carries the
  OpenAI response `id` (field `response_id`). If cpp_server writes that id
  into its log lines, a `jq '.benchmarks[].requests.errored[].response_id'`
  list is immediately grep-able in `server.log`.
- **Prometheus / Grafana.** Run the shared
  [cpp_server/monitoring/](../../monitoring/) stack. Correlate on wall-clock
  UTC timestamps against the per-request timings in `benchmarks.json`.
- **Hang.** `py-spy dump --pid <server_pid>` for Python frames;
  `gdb -p <pid> -batch -ex "thread apply all bt"` for C++ frames.
- **Core dumps.** `ulimit -c unlimited` before starting cpp_server.

## Known limitations (explicit trade-offs)

- **SIGINT loses the current run.** GuideLLM writes `benchmarks.json` only
  at end-of-run. Acceptable by design — shorten `--duration` if you need
  finer checkpointing.
- **`turns` is a fixed integer, not a distribution** (GuideLLM limitation).
  Workaround: run multiple passes at different `--turns` values.
- **Prompt/output sizes are *truncated normal*, not lognormal.** Fine
  approximation in practice; the tails are controlled by the `_min`/`_max`
  clamps.
- **`ignore_eos: true` is sent by default** (vLLM-specific; GuideLLM adds
  it to force exact output lengths). If cpp_server rejects this field,
  pass `--extras '{"ignore_eos": false}'` and accept realistic-variable
  output lengths.
- **Closed-pool, sweep, and stress modes are passthrough only** — this
  tool is focused on one use case; advanced profiles still work via
  `-- --profile concurrent` etc. but the defaults are tuned for the
  Poisson-soak case.

## Files

- [agentic_bench.sh](./agentic_bench.sh) — CLI entrypoint; single `exec
  guidellm benchmark ...`.
