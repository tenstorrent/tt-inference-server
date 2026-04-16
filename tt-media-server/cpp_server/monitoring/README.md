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
├── docker-compose.yml                        # Prometheus + Grafana services
├── prometheus.yml                            # scrape config (target container + /metrics path)
└── grafana/
    ├── provisioning/
    │   ├── datasources/prometheus.yml        # auto-registers Prometheus datasource
    │   └── dashboards/default.yml            # tells Grafana where to load dashboards from
    └── dashboards/
        └── tt_media_server.json              # pre-built dashboard (latency, throughput, queue)
```

## Ports

| Service    | Port |
|------------|------|
| Grafana    | 3000 |
| Prometheus | 9090 |

## Stopping

```bash
docker compose -f monitoring/docker-compose.yml down
```

Add `-v` to also delete stored metrics and Grafana state.
