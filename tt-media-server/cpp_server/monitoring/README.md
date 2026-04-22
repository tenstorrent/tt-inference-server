# Monitoring Stack

Prometheus + Grafana Docker Compose stack for the TT inference server.

## Quick start

The inference server must be on the shared `tt_net` Docker network so Prometheus can reach it by container name:

```bash
# One-time: create the network and attach the inference container
docker network create tt_net
docker network connect tt_net <your-inference-container-name>

# Start Prometheus and Grafana, pointing at the inference container
SERVER_TARGET=$(hostname):8000 docker compose -f monitoring/docker-compose.yml up -d
```

`SERVER_TARGET` defaults to `tt-inference-server:8000` if omitted. Using `$(hostname)` is convenient when the inference server container shares the host's name.

Open Grafana at **http://localhost:3000** (admin / admin). The dashboard loads automatically.

## Directory layout

```
monitoring/
├── docker-compose.yml                        # Prometheus + Grafana + process-exporter services
├── prometheus.yml                            # scrape config (server /metrics + process metrics)
├── process-exporter.yml                      # which host processes to expose CPU/memory/threads for
└── grafana/
    ├── provisioning/
    │   ├── datasources/prometheus.yml        # auto-registers Prometheus datasource
    │   └── dashboards/default.yml            # tells Grafana where to load dashboards from
    └── dashboards/
        └── tt_media_server.json              # pre-built dashboard (latency, throughput, queue)
```

## Ports

| Service          | Port |
|------------------|------|
| Grafana          | 3000 |
| Prometheus       | 9090 |
| process-exporter | internal only (9256 on `monitoring` net) |

## Process metrics (CPU / memory / threads per binary)

`process-exporter` runs with `pid: host` and a read-only `/proc` mount so it
sees every process on the host without any change on the server side.

Main and worker are the same binary (`tt_media_server_cpp`); workers are
distinguished by a `--worker N` argv. [process-exporter.yml](./process-exporter.yml)
splits them into two named groups (via cmdline regex, worker rule first
since first-match-wins):

| `groupname`                      | matches                                          |
|----------------------------------|--------------------------------------------------|
| `tt_media_server_cpp_worker`     | `...tt_media_server_cpp --worker N` processes    |
| `tt_media_server_cpp_main`       | main server (comm `tt_media_server`, truncated)  |

These are rendered by the "Infrastructure" row at the bottom of the existing
Grafana dashboard: CPU %, memory RSS (MB), threads, open fds, process count,
and page-fault rate per binary.

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
