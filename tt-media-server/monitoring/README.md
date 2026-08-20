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

## Video generation metrics

The Python dashboard has a **Video Generation** row driven by the
`tt_media_server_video_*` family (emitted by `telemetry/telemetry_client.py`,
recorded in `model_services/video_service.py`). The generic request metrics
answer *how long*; these answer *how much video, how fast*:

| Metric | What it tells you |
|--------|-------------------|
| `video_generation_total{request_type,status}` | throughput and outcome, split t2v / i2v; `status` is `success` / `failure` / `cancelled` |
| `video_generation_duration_seconds{resolution,status}` | end-to-end latency; filter `status="success"` for latency, `"failure"` for time-to-failure |
| `video_frames_generated_total`, `video_content_seconds_total` | fleet output rate (frames/sec, video-seconds/sec) |
| `video_denoise_steps_total` | denoise steps **executed** (success only) — the steps/sec denominator |
| `video_frames_per_second`, `video_pixels_per_second` | per-generation throughput; pixels/sec makes 480p and 720p comparable |
| `video_step_duration_seconds` | mean wall-clock seconds per denoise step |
| `video_realtime_factor` | wall seconds per second of playable video (1.0 = realtime) |
| `video_output_size_bytes`, `video_output_frames` | what the client actually got back |
| `video_requested_inference_steps`, `video_conditioning_images` | what clients are asking for |
| `video_generations_in_progress` | live concurrency, split by request type |
| `video_last_generation_timestamp` | freshness — alert on `time() - <this>` to catch a pipeline that stopped producing |
| `video_encode_*` | ffmpeg mp4 encode cost |

Three things to know when reading these:

* **`cancelled` is not `failure`.** `POST /generations/{id}/cancel` is a normal
  client action, so it gets its own `status` value and is excluded from both
  sides of the Success Rate panel. Executed steps, frames, and throughput are
  recorded for successful generations only — a request that timed out or was
  cancelled after 6s of a 300s budget did not run its steps that fast.

* **Frame count and resolution come from probing the produced mp4** (PyAV, in
  `utils.video_manager.probe_video`). On a multihost `sp_runner` deployment the
  server only receives a file path from its MPI peer, so this is the only source
  of shape truth. If the probe fails, the resolution label is `unknown` and the
  shape-derived series are skipped rather than recorded as zero.

* **`video_encode_*` is absent on multihost `sp_runner` deployments.** The mp4 is
  encoded inside the external runner peer, which serves no `/metrics`. It *is*
  populated for in-process runners and the CPU postprocessing workers — including
  a small `resolution="64x64"` series from the postprocessing warmup task.

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
