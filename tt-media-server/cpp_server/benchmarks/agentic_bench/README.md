# agentic_bench

Long-running, multi-turn, **agentic-shape** soak benchmark for
OpenAI-compatible endpoints — designed to exercise a production LLM serving
stack (e.g. DeepSeek-R1-0528 on cpp_server) the way real Cursor / Claude
Code / OpenHands traffic would, for hours to days, and produce one
self-contained HTML report.

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

GuideLLM v0.6.0 already has the building blocks (native `turns` /
`prefix_buckets` multi-turn synthetic data, Poisson profile, per-request
TTFT/ITL records). What it lacks for our use case is **interrupt-safe long
runs** and a **single aggregated report across many sub-runs**. This tool
adds exactly that and nothing more.

## Quickstart

```bash
# one-time: install the bundled GuideLLM into your current env
pip install cpp_server/guidellm

# start cpp_server (or any OpenAI-compatible endpoint) on :8000, then:
./cpp_server/benchmarks/agentic_bench/agentic_bench.sh
```

That runs an 8-hour Poisson soak at 0.15 req/s against
`http://localhost:8000` with the `DeepSeek-R1-0528` tokenizer and defaults
tuned for HW sized for 64 concurrent users, and writes
`./runs/<ts>/report.html` at the end (or at Ctrl-C).

Example overrides:

```bash
./agentic_bench.sh --rate 0.3 --duration 3600 --turns 10                # 1h at 0.3 rps, 10 turns/conv
./agentic_bench.sh --target http://other-host:9000 --model some/Other    # different endpoint & tokenizer
./agentic_bench.sh --dry-run                                             # print the guidellm invocation
./agentic_bench.sh -- --random-seed 7                                    # forward extra flags to guidellm
```

Help: `./agentic_bench.sh --help`.

## Why these defaults

- **Poisson, not constant rate.** Real user/agent arrivals are stochastic.
  Constant rate is an artifact that hides burst behavior and makes tail
  latencies look better than they really are.
- **Pool of 50 shared prefixes, not 1 and not per-user.** One prefix →
  unrealistic 100% prefix-cache hit (server looks artificially fast). One
  prefix per user → 0% cross-user reuse (unrealistically slow). A pool of
  ~50 distinct 8k-token prefixes approximates a deployment with ~50
  distinct projects/repos/personas and yields the ~80-95% cache-hit rate
  seen in production.
- **6 turns per conversation.** Most Claude-Code-style sessions are a
  handful of turns before the user changes task. Override with `--turns`.
- **`prompt_tokens=2000, stdev=1500`, clamped to [100, 30000].** Heavy-input-biased,
  matches a Cursor-like flow where each user turn carries growing context.
  GuideLLM uses *truncated normal*, not lognormal — see Limitations.
- **`output_tokens=800, stdev=600`, clamped to [64, 4096].** Most replies
  are short; occasional long code blocks are clamped.
- **Rolling 15-min chunks.** GuideLLM writes its JSON only at end-of-benchmark
  and has no SIGINT handler. Chunking means Ctrl-C loses **at most the
  in-flight requests of the current chunk** — all prior chunks survive.
  15 min is a balance between data safety and the 5% warmup overhead
  paid per chunk. Use `--chunk 300` for tighter safety.
- **Single self-contained `report.html`.** Copy-pasteable into a bug
  ticket. The full merged JSON is embedded as
  `<script type="application/json" id="data">…</script>` inside the
  same file, so CSV/JSON consumers lose nothing.
- **No `/metrics` or `tt-smi` scraping inside this tool.** Capture those
  with the shared [cpp_server/monitoring/](../../monitoring/) stack
  (Prometheus scrapes the server's `/metrics` plus per-binary CPU /
  memory / threads via process-exporter, Grafana auto-provisions both
  dashboards). Correlate on wall clock with the benchmark's
  `report.html` timeline.

## CLI reference

```
Target & duration
  --target URL              OpenAI-compatible base URL       [http://localhost:8000]
  --model NAME              Model id for the backend         [deepseek-ai/DeepSeek-R1-0528]
  --processor NAME          HF tokenizer id                  [<same as --model>]
  --rate RPS                Poisson mean req/s               [0.15]
  --duration SEC            Total run seconds (0 = forever)  [28800]   # 8 h
  --chunk SEC               Sub-run chunk seconds            [900]     # 15 min

Agentic-shape synthetic data (→ guidellm --data key=value,...)
  --turns N                 Turns per conversation           [6]
  --prompt-tokens N         Mean user-prompt tokens/turn     [2000]
  --prompt-tokens-stdev N   Stdev                            [1500]
  --prompt-tokens-min N     Clamp min                        [100]
  --prompt-tokens-max N     Clamp max                        [30000]
  --output-tokens N         Mean assistant-output tokens     [800]
  --output-tokens-stdev N   Stdev                            [600]
  --output-tokens-min N     Clamp min                        [64]
  --output-tokens-max N     Clamp max                        [4096]
  --prefix-tokens N         Size of each shared system prompt[8000]
  --prefix-count N          Pool size of distinct prefixes   [50]

Output & reliability
  --out DIR                 Run directory                    [./runs/<ts>]
  --max-errors N            Stop chunk after N errors        [500]
  --warmup FRAC             Per-chunk warmup fraction        [0.05]
  --extras 'JSON'           Extra guidellm --extras          [""]
  --dry-run                 Print the guidellm command and exit
  --help                    Show help and exit
```

Everything after `--` is forwarded verbatim to `guidellm benchmark`, so any
advanced GuideLLM flag is reachable without re-plumbing. E.g. to switch
to closed-pool for a single run:

```bash
./agentic_bench.sh --rate 64 -- --profile concurrent
```

## Outputs

```
runs/<ts>/
├── config.txt              # full flag snapshot for reproducibility
├── chunk_0000/
│   └── benchmarks.json     # one GuideLLM report per chunk
├── chunk_0001/...
└── report.html             # merged, single self-contained file (this is the deliverable)
```

You can re-run the merger at any time (even while the soak is still in
progress) to peek:

```bash
python3 scripts/merge_report.py runs/<ts>
```

## Data collection and debugging

The benchmark captures only client-side data; to debug weird server
behavior you need to correlate against server-side signals.

- **`response_id`.** Every request row in `report.html` carries the OpenAI
  response `id`. If cpp_server writes that id into its log lines, grepping
  `server.log` for a single anomalous row gives you the server-side trace.
  If it does not, adding that field is a small cpp_server change worth
  making for operability.
- **Server log.** Start cpp_server under
  `stdbuf -oL tee runs/<ts>/server.log` so its stdout/stderr land next to
  the run artifacts. Use whatever verbose log level your build supports.
- **Prometheus / Grafana.** Run these separately (they are not part of
  this tool). Correlate on wall-clock UTC timestamps; the timeline chart in
  `report.html` uses the same clock.
- **Hang.** `py-spy dump --pid <server_pid>` for Python frames;
  `gdb -p <pid> -batch -ex "thread apply all bt"` for C++ frames. Save
  under `runs/<ts>/dumps/`.
- **Regression band.** When the timeline shows a spike in a specific
  wall-clock window, `perf record -F 99 -p <server_pid> -g -- sleep 60`
  during the next occurrence.
- **Core dumps.** `ulimit -c unlimited` before starting cpp_server.

## Known limitations (explicit trade-offs)

- **No byte-level event log.** We rely on per-request records plus external
  Prometheus/Grafana for post-mortem. If you need a single append-only
  timeline of every token event, that is a much bigger custom tool.
- **Current-chunk loss on SIGINT.** All prior chunks are preserved; the
  chunk running at SIGINT time loses its in-flight requests. Use
  `--chunk 300` for tighter safety at a slight warmup-overhead cost.
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
  `-- --profile concurrent` etc. but the defaults and report are tuned for
  the Poisson-soak case.

## Files

- [agentic_bench.sh](./agentic_bench.sh) — CLI entrypoint, rolling loop,
  SIGINT trap.
- [scripts/merge_report.py](./scripts/merge_report.py) — aggregator; uses
  `guidellm.benchmark.GenerativeBenchmarksReport`; emits
  `report.html`.
