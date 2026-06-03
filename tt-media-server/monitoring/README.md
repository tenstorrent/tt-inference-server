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
        ├── tt_media_server_cpp.json          # C++ server dashboard (latency, throughput, queue)
        ├── tt_media_server_python.json       # Python server dashboard (legacy, sunsetting)
        └── tt_prefill_gateway.json           # PrefillGateway routing, latency, heartbeat dashboard
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
The initial rules cover stale prefill heartbeats, low prefix-match rate, high
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
