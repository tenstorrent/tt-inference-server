# Monitoring Stack

Prometheus + Grafana Docker Compose stack for the TT Media Server.

This stack covers **both** the C++ server and the (transitional) Python
server. Same Prometheus, same Grafana, two dashboards; you select which
server to scrape at start time via `SERVER_SERVICE`. Lives at the
`tt-media-server/` top level, alongside [`telemetry/`](../telemetry)
(which is the Python instrumentation that emits `/metrics`) and
[`cpp_server/`](../cpp_server) (which contains the C++ instrumentation).

## Quick Start

The inference server must be on the shared `tt_net` Docker network so Prometheus can reach it by container name:

```bash
# One-time: create the network and attach the inference container
docker network create tt_net 2>/dev/null || true
docker network connect tt_net <your-inference-container-name>

# From the tt-media-server/ directory, start Prometheus + Grafana pointing
# at the inference container. Pick SERVER_SERVICE=cpp or python. If
# PrefillGateway is running, set GATEWAY_TARGET to its --metrics-port endpoint.
SERVER_TARGET=<your-inference-container-name>:8000 \
SERVER_SERVICE=cpp \
GATEWAY_TARGET=<your-inference-container-name>:9091 \
  docker compose -f monitoring/docker-compose.yml up -d
```

If you're already inside `tt-media-server/monitoring/`, pass the file as
`./docker-compose.yml` (the leading `./` is required — a bare
`docker-compose.yml` is not resolved as a path):

```bash
SERVER_TARGET=<your-inference-container-name>:8000 \
SERVER_SERVICE=cpp \
GATEWAY_TARGET=<your-inference-container-name>:9091 \
  docker compose -f ./docker-compose.yml up -d
```

`SERVER_TARGET` defaults to `tt-inference-server:8000` if omitted.
`GATEWAY_TARGET` defaults to `prefill-gateway:9091` and should point at the
PrefillGateway `--metrics-port` endpoint when the gateway is enabled.
`SERVER_SERVICE` defaults to `python` (kept for backwards compatibility
with the original setup).

### Disaggregated prefill / decode

When you run the C++ server split into a prefill node and a decode node:

```bash
# Prefill server on :8001, decode server on :8000 (defaults shown in the prompt)
TT_LOG_LEVEL=debug LLM_MODE=prefill ./build/tt_media_server_cpp -p 8001
MAX_TOKENS_TO_PREFILL_ON_DECODE=0 TT_LOG_LEVEL=debug LLM_MODE=decode ./build/tt_media_server_cpp
```

point Prometheus at **both** servers — they expose identical metric names and
are told apart only by a `role` label that the scrape config attaches:

```bash
PREFILL_TARGET=<server-container-name>:8001 \
DECODE_TARGET=<server-container-name>:8000 \
  docker compose -f monitoring/docker-compose.yml up -d
```

`PREFILL_TARGET` defaults to `tt-prefill-server:8001` and `DECODE_TARGET` to
`tt-decode-server:8000`. Targets that aren't running are simply marked *down*
by Prometheus and emit no series, so these jobs are harmless to leave
configured in single-server mode (and vice versa — the `tt_media_server`
regular job is harmless when only prefill/decode are up).

Two dedicated dashboards are provisioned for this setup:

| Dashboard (Grafana title)                     | uid                          | filters on   | focus |
|-----------------------------------------------|------------------------------|--------------|-------|
| TT Inference Server — Prefill (disaggregated) | `tt-inference-server-prefill`| `role="prefill"` | prompt-token throughput, prefill (E2E) latency, queue depth, per-slot ISL |
| TT Inference Server — Decode (disaggregated)  | `tt-inference-server-decode` | `role="decode"`  | TSU / TPOT / ITL, decoding users, output throughput, per-slot decode perf |

> **Note (current iteration):** only the dashboards and scrape labels are
> disaggregation-aware. The metrics emitted by `metrics.cpp` are still the
> shared regular-mode set, so a handful of panels are mode-irrelevant by
> construction (e.g. TPOT on the prefill dashboard, were it shown). Tailoring
> the *exposed* metrics per mode is the next iteration.

### TTS server

A TTS deployment (`MODEL_SERVICE=tts`) is scraped by the plain
`tt_media_server` job — point `SERVER_TARGET` at it like any other single
server, and open the **TT Media Server — TTS (decode + vocoder)** dashboard
(uid `tt-media-server-tts`):

```bash
SERVER_TARGET=<tts-container-name>:8000 SERVER_SERVICE=cpp \
  docker compose -f monitoring/docker-compose.yml up -d
```

Speech generation is two stages, and the dashboard measures them separately so
a slowdown can be pinned to one of them.

**Stage 1 — codec-token decode.** `tt_tts_codec_tokens_total` is the cumulative
count of acoustic / codec tokens emitted by the TTS decoder, labelled by
`worker_id`, `device` (the worker's `DEVICE_IDS` group), `model_name` and
`voice_source`. Throughput is
`rate(tt_tts_codec_tokens_total[$__rate_interval])`, i.e. the autoregressive
decode capacity that has to stay ahead of playback.

**Stage 2 — vocoder / waveform reconstruction.**
`tt_tts_audio_frames_total` is the cumulative count of PCM frames (samples per
channel) the vocoder reconstructed from those tokens, labelled by `worker_id`,
`device`, `model_name` and `batch`. Both units come from that one counter, so
nothing can drift apart:

| Quantity | Query |
| --- | --- |
| samples/s | `rate(tt_tts_audio_frames_total[$__rate_interval])` |
| audio seconds/s (RTF) | `rate(tt_tts_audio_frames_total[$__rate_interval]) / scalar(max(tt_tts_audio_sample_rate_hz))` |

The second is the real-time factor: **below 1.0 the vocoder cannot keep up with
playback** even when the decoder can. `tt_tts_vocoder_chunks_total` (same
labels) pairs with the frame counter to give mean frames per chunk, which tells
apart "fewer chunks" from "shorter chunks" when audio throughput drops.

**Separating the two.** Tokens per second of audio is a constant of the codec,
so `rate(codec_tokens) / (rate(audio_frames) / sample_rate)` is flat while both
stages keep pace — it rises when the vocoder falls behind a healthy decoder and
falls when the decoder is starving. The two staleness clocks say the same thing
at a glance: `tt_tts_last_vocode_age_seconds` rising while
`tt_worker_last_output_age_seconds` stays flat is a vocoder stall; both rising
together points upstream at token generation.

Per-replica granularity comes from the scrape `instance` label. All of these are
published by the TTS worker process into its worker-metrics shared-memory slot
(`MetricsLayout::TTS_RUNNER`, see
`cpp_server/include/runtime/worker/tts_metrics_layout.hpp`) and rendered on the
main process's `/metrics` by `TtsWorkerMetricsRenderer`.

#### Label caveats

There is deliberately **no `voice` or `language` label**: the TTS API accepts
only `text`, a free-form `description` and an optional voice WAV, so neither
dimension exists to label by. `voice_source`
(`default` / `description` / `voice_sample`) is the bounded stand-in until
those fields are added to the request.

`batch` (`1`, `2`, `3-4`, `5-8`, `9-16`, `17+`) is **derived, not reported**.
The engine does not expose the batch its vocoder formed, so the runner counts
the distinct streams whose chunks came out of one `drainAudioOutputs()` sweep —
the engine vocodes a batch and pushes one chunk per stream in it, so a sweep
observes one batch's worth. Under load a single batch can straddle two sweeps,
which shows up as two smaller buckets rather than one larger one, so read the
bucket as "at least this many streams were reconstructed together". Buckets
rather than a raw count keep the label at 6 values instead of the 128 that
`PM_MAX_USERS` would admit. Replace this with the real batch size if the engine
starts reporting it.

### Docker Scrape Targets

Prometheus runs in Docker, so `localhost` inside Prometheus refers to the
Prometheus container, not the host or dev container running the server. Make
sure the server or gateway container is attached to `tt_net`, then use that
container name in `SERVER_TARGET` and `GATEWAY_TARGET`:

```bash
docker network create tt_net 2>/dev/null || true
docker network connect tt_net <server-container-name> 2>/dev/null || true

SERVER_TARGET=<server-container-name>:8001 \
SERVER_SERVICE=cpp \
GATEWAY_TARGET=<server-container-name>:9091 \
  docker compose -f monitoring/docker-compose.yml up -d
```

Verify Prometheus can see all targets:

```bash
docker exec tt_prometheus wget -qO- http://localhost:9090/api/v1/targets
```

The cpp dashboard is the default Grafana home. To make the python
dashboard the default home instead, set
`GF_HOME_DASHBOARD=/etc/grafana/provisioning/dashboards/tt_media_server_python.json`.

Open Grafana at **http://localhost:3000** (admin / admin). The dashboard loads
automatically. PrefillGateway panels are available in the `TT Prefill Gateway`
dashboard.

## Directory layout

```
monitoring/
├── docker-compose.yml                        # Prometheus + Grafana + process-exporter services
├── prometheus.yml                            # scrape config (server, gateway + process metrics)
├── prometheus/rules/prefill_gateway.yml      # PrefillGateway alert rules
├── process-exporter.yml                      # which host processes to expose CPU/memory/threads for
└── grafana/
    ├── provisioning/
    │   ├── datasources/prometheus.yml        # auto-registers Prometheus datasource
    │   └── dashboards/default.yml            # tells Grafana where to load dashboards from
    └── dashboards/
        ├── tt_media_server_cpp.json          # C++ server dashboard, regular mode (latency, throughput, queue)
        ├── tt_media_server_cpp_prefill.json  # C++ disaggregated prefill node (role="prefill")
        ├── tt_media_server_cpp_decode.json   # C++ disaggregated decode node (role="decode")
        ├── tt_media_server_cpp_tts.json      # C++ TTS server (MODEL_SERVICE=tts): codec-token + vocoder throughput
        ├── tt_media_server_python.json       # Python server dashboard (legacy, sunsetting)
        └── tt_prefill_gateway.json           # PrefillGateway routing, latency, registration-age dashboard
```

## Ports

| Service          | Port |
|------------------|------|
| Grafana          | 3000 |
| Prometheus       | 9090 |
| PrefillGateway   | 9091 by default (`--metrics-port`) |
| process-exporter | internal only (9256 on `monitoring` net) |

## Process metrics (CPU / memory / threads per binary)

`process-exporter` runs with `pid: host` and a read-only `/proc` mount so it
sees every process on the host without any change on the server side. It
groups processes by binary so you get per-server CPU/memory/threads even
when both the C++ and Python servers run on the same host.

For the C++ server, main and worker are the same binary
(`tt_media_server_cpp`); workers are distinguished by a `--worker N` argv.
[process-exporter.yml](./process-exporter.yml) splits them into two named
groups (via cmdline regex, worker rule first since first-match-wins). The
Python server (uvicorn) gets its own group:

| `groupname`                      | matches                                                  |
|----------------------------------|----------------------------------------------------------|
| `tt_media_server_cpp_worker`     | `...tt_media_server_cpp --worker N` processes            |
| `tt_media_server_cpp_main`       | main C++ server (comm `tt_media_server`, truncated)      |
| `prefill_gateway`                | PrefillGateway binary                                    |
| `tt_media_server_python`         | uvicorn / `python tt-media-server/main.py` processes     |

## PrefillGateway Alerts

Prometheus loads gateway rules from
[`prometheus/rules/prefill_gateway.yml`](./prometheus/rules/prefill_gateway.yml).
The initial rules cover stale prefill registrations, low prefix-match rate, high
prefill latency, and observed request timeouts. They appear in Prometheus and
Grafana as long as the gateway scrape target is reachable.

These are rendered by the "Infrastructure" row at the bottom of the
relevant Grafana dashboard: CPU %, memory RSS (MB), threads, open fds,
process count, and page-fault rate per binary.

Example PromQL if you want ad-hoc queries:

```promql
sum by (groupname) (rate(namedprocess_namegroup_cpu_seconds_total[1m])) * 100
sum by (groupname) (namedprocess_namegroup_memory_bytes{memtype="resident"}) / 1024 / 1024
sum by (groupname) (namedprocess_namegroup_num_threads)
```

After editing [process-exporter.yml](./process-exporter.yml) (e.g. a
renamed binary or a new worker kind), reload:

```bash
docker compose -f monitoring/docker-compose.yml restart process-exporter
```

## Stopping

```bash
docker compose -f monitoring/docker-compose.yml down
```

Add `-v` to also delete stored metrics and Grafana state.
