# Deployment

Reference for the cloud team deploying the C++ inference server. The
server runs as a Docker container. This document describes the image,
the HTTP endpoints used for liveness/readiness probes and Prometheus
scraping, and the environment variables to set on the container.

## Contents

- [Image](#image)
- [HTTP endpoints](#http-endpoints)
  - [Readiness probe: /health](#readiness-probe-health)
  - [Liveness probe: /tt-liveness](#liveness-probe-tt-liveness)
  - [Build info: /info](#build-info-info)
  - [Prometheus metrics: /metrics](#prometheus-metrics-metrics)
- [Environment variables](#environment-variables)
  - [Required](#required)
  - [Authentication](#authentication)
  - [Logging](#logging)
  - [Capacity](#capacity)
  - [Generation](#generation)
  - [Timeouts (ms)](#timeouts-ms)
  - [Shared memory and IPC](#shared-memory-and-ipc)

## Image

```
ghcr.io/tenstorrent/tt-shield/tt-media-inference-server-blaze:<tag>
```

Built from [`tt-media-server/Dockerfile.blaze`](../Dockerfile.blaze) by the
[`on-dispatch-build-media-server`](https://github.com/tenstorrent/tt-shield/actions/workflows/on-dispatch-build-media-server.yml)
GitHub Actions workflow. Trigger the workflow manually to produce a new
tag.

## HTTP endpoints

All four endpoints below are unauthenticated.

> Despite their names, `/health` is the **readiness** probe (gate traffic
> on it) and `/tt-liveness` is the **liveness** probe (restart on it).
> The names are kept for backwards compatibility with the Python server.

| Endpoint           | Role              | 200 means                                                                                | Non-200                          |
| ------------------ | ----------------- | ---------------------------------------------------------------------------------------- | -------------------------------- |
| `GET /health`      | Readiness probe   | Workers are alive **and** ready, socket is connected (disaggregated prefill and decode). | `503` when any of those is false. |
| `GET /tt-liveness` | Liveness probe    | Process can respond.                                                                     | `500` only on unrecoverable failure (service unconfigured, internal exception). |
| `GET /info`        | Build info        | Always 200 with build identity.                                                          | —                                |
| `GET /metrics`     | Prometheus scrape | Always 200 with Prometheus text-format metrics.                                          | —                                |

### Readiness probe: /health

Returns 200 only when **all** of the following hold:

- At least one worker is alive (process running).
- At least one worker is ready (model loaded and warmed up).
- The inter-server socket is connected — or there is no socket
  (collocated prefill and decode in one pod, `LLM_MODE=regular`).

Otherwise returns 503 with `status: "unhealthy"` and an `error` string.

Response fields:

| Field           | Type    | When present                                                   | Meaning                                                                                                                             |
| --------------- | ------- | -------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `status`        | string  | always                                                         | `"healthy"` (200) or `"unhealthy"` (503).                                                                                           |
| `timestamp`     | int     | always                                                         | Server's wall-clock time in Unix seconds when the response was built.                                                               |
| `error`         | string  | 503 only                                                       | One of `"no workers are alive"`, `"no workers are ready"`, `"socket not connected"`.                                                |
| `socket_status` | string  | disaggregated prefill and decode (`LLM_MODE=prefill`/`decode`) | One of `"disabled"`, `"stopped"`, `"server:waiting"`, `"client:connecting"`, `"server:connected"`, `"client:connected"`.            |

**Healthy (200), collocated prefill and decode:**

```json
{
  "status": "healthy",
  "timestamp": 1747094400
}
```

**Healthy (200), disaggregated prefill and decode — this pod is the decode server and the prefill peer is connected:**

```json
{
  "status": "healthy",
  "timestamp": 1747094400,
  "socket_status": "server:connected"
}
```

**During warmup (503) — container is up, model isn't loaded yet. Expected for the first few minutes after start; the readiness probe should fail so traffic is not routed to the container yet:**

```json
{
  "status": "unhealthy",
  "timestamp": 1747094400,
  "error": "no workers are ready"
}
```

**Worker crashed (503) — traffic will stop being routed to this container; page on this in production:**

```json
{
  "status": "unhealthy",
  "timestamp": 1747094400,
  "error": "no workers are alive"
}
```

**Disaggregated peer disconnected (503) — the decode container can see the prefill container is gone:**

```json
{
  "status": "unhealthy",
  "timestamp": 1747094400,
  "socket_status": "server:waiting",
  "error": "socket not connected"
}
```

### Liveness probe: /tt-liveness

Returns 200 whenever the process can answer. Only returns 500 when the
service container isn't configured, or `getSystemStatus()` throws — i.e.
the process is wedged. Wire this to your liveness probe; a non-200
means the container should be restarted.

Response fields:

| Field            | Type    | When present                 | Meaning                                                                                 |
| ---------------- | ------- | ---------------------------- | --------------------------------------------------------------------------------------- |
| `status`         | string  | always                       | Always `"alive"` (even on 500).                                                         |
| `model_ready`    | bool    | always                       | `true` once any worker has loaded the model. Informational — does **not** gate the 200. |
| `queue_size`    | int     | on 200                       | Current number of in-flight + queued requests.                                          |
| `max_queue_size` | int     | on 200                       | `MAX_QUEUE_SIZE` — capacity before 429s start.                                          |
| `socket_status`  | string  | on 200 in disaggregated mode | Same values as `/health`.                                                               |
| `workers`        | array   | on 200                       | One entry per worker process (see below).                                               |
| `error`          | string  | 500 only                     | Reason the liveness check itself failed.                                                |

Each `workers[]` entry:

| Field       | Type   | Meaning                                                                                |
| ----------- | ------ | -------------------------------------------------------------------------------------- |
| `worker_id` | string | Stable per-worker identifier (index into `DEVICE_IDS`).                                |
| `is_alive`  | bool   | The worker process exists and is responsive to the server's worker supervisor.         |
| `is_ready`  | bool   | The worker has finished warmup and is accepting work.                                  |
| `pid`       | int    | Worker process OS PID. Useful for correlating logs and `ps`/`htop` output.             |

**Warmed up, idle (200):**

```json
{
  "status": "alive",
  "model_ready": true,
  "queue_size": 0,
  "max_queue_size": 1000,
  "workers": [
    { "worker_id": "0", "is_alive": true, "is_ready": true, "pid": 142 }
  ]
}
```

**Warming up — container is alive, model is loading. Liveness still 200 (don't restart yet); readiness `/health` is 503 during this window, so traffic isn't routed here:**

```json
{
  "status": "alive",
  "model_ready": false,
  "queue_size": 0,
  "max_queue_size": 1000,
  "workers": [
    { "worker_id": "0", "is_alive": true, "is_ready": false, "pid": 142 }
  ]
}
```

**Worker crashed, server is alive (200) — the server's worker supervisor will restart the worker. Liveness stays 200, so the container is **not** restarted; readiness `/health` will be 503 while `is_alive: false`, so traffic is shed:**

```json
{
  "status": "alive",
  "model_ready": false,
  "queue_size": 0,
  "max_queue_size": 1000,
  "workers": [
    { "worker_id": "0", "is_alive": false, "is_ready": false, "pid": 142 }
  ]
}
```

**Service wedged (500) — the container will be restarted via the liveness probe:**

```json
{
  "status": "alive",
  "model_ready": false,
  "error": "Liveness check failed: <exception message>"
}
```

### Build info: /info

Build identity baked into the image. First thing to grab when filing a
bug or correlating an incident with a release.

Response fields:

| Field                          | Meaning                                                                |
| ------------------------------ | ---------------------------------------------------------------------- |
| `tt_inference_server.version`  | Semver of this server build (from `VERSION` file).                     |
| `tt_inference_server.commit`   | Git commit of the `tt-inference-server` repo at build time.            |
| `tt_blaze.commit`              | Git commit of the `tt-blaze` submodule used at build time.             |
| `tt_metal.commit`              | Git commit of the `tt-metal` submodule (inside tt-blaze) at build time. |

**Example:**

```json
{
  "tt_inference_server": {
    "version": "0.5.0",
    "commit": "56741604fa1e1d2cb8a..."
  },
  "tt_blaze": {
    "commit": "8df8a38675123db7b56..."
  },
  "tt_metal": {
    "commit": "002502a73221309123..."
  }
}
```

### Prometheus metrics: /metrics

Prometheus text-format (version 0.0.4) exposition. Returned
content-type is `text/plain; version=0.0.4; charset=utf-8`. The body
combines server-side metrics with worker metrics collected from the
worker processes.

Always returns 200. Scrape it from Prometheus on the container's HTTP
port at `/metrics`.

A ready-to-import Grafana dashboard for these metrics is in this repo at
[`../monitoring/grafana/dashboards/tt_media_server_cpp.json`](../monitoring/grafana/dashboards/tt_media_server_cpp.json).

**Example response (truncated):**

```
# HELP tt_num_requests_in_flight Requests being processed (queued + prefilling + decoding)
# TYPE tt_num_requests_in_flight gauge
tt_num_requests_in_flight 7

# HELP tt_num_decoding_requests Requests currently in the decode phase
# TYPE tt_num_decoding_requests gauge
tt_num_decoding_requests 5

# HELP tt_num_active_sessions Active sessions tracked by the server
# TYPE tt_num_active_sessions gauge
tt_num_active_sessions 38

# HELP tt_max_queue_size Configured request queue capacity
# TYPE tt_max_queue_size gauge
tt_max_queue_size 1000

# HELP tt_prompt_tokens_total Cumulative prompt tokens processed since process start
# TYPE tt_prompt_tokens_total counter
tt_prompt_tokens_total 184523

# HELP tt_generation_tokens_total Cumulative generation tokens produced since process start
# TYPE tt_generation_tokens_total counter
tt_generation_tokens_total 92341

# HELP tt_time_to_first_token_seconds Latency from request arrival to first generated token
# TYPE tt_time_to_first_token_seconds summary
tt_time_to_first_token_seconds{quantile="0.5"} 0.231
tt_time_to_first_token_seconds{quantile="0.9"} 0.482
tt_time_to_first_token_seconds{quantile="0.95"} 0.612
tt_time_to_first_token_seconds{quantile="0.99"} 1.034
tt_time_to_first_token_seconds_sum 1284.5
tt_time_to_first_token_seconds_count 5421

# HELP tt_http_requests_total HTTP requests served, by method and status code
# TYPE tt_http_requests_total counter
tt_http_requests_total{method="POST",status_code="200"} 5420
tt_http_requests_total{method="POST",status_code="429"} 12
tt_http_requests_total{method="GET",status_code="200"} 8341

# HELP tt_worker_heartbeat_age_seconds Seconds since the worker last produced a heartbeat (frozen if growing)
# TYPE tt_worker_heartbeat_age_seconds gauge
tt_worker_heartbeat_age_seconds{worker_id="0"} 0.04
```

## Environment variables

Set these on the container. Most values are cached on first read, so
changing them after the container starts has no effect — restart to
apply.

### Required

These two are required for the C++ server to start correctly inside the
image.

| Variable      | Value   | Description                                                                                                                    |
| ------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `SERVER_MODE` | `cpp`   | Selects the C++ server in the image's entrypoint. Without it (or any value other than `cpp`) the image runs the Python server. |
| `CACHE_ROOT`  | `/tmp/` | Path the image's entrypoint adjusts permissions on so the non-root container user can write to mounted volumes.                |

### Authentication

| Variable         | Default           | Description                                                                                                                                                                                              |
| ---------------- | ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `OPENAI_API_KEY` | `your-secret-key` | Bearer token required on protected endpoints. The default is public — always override in production with a secret injected from your secrets manager. `/health`, `/tt-liveness`, `/info`, `/docs`, `/metrics` are open. |

### Logging

| Variable       | Default | Description                                                            |
| -------------- | ------- | ---------------------------------------------------------------------- |
| `TT_LOG_LEVEL` | `info`  | One of `trace`, `debug`, `info`, `warn`, `error`, `critical`, `off`.   |
| `TT_LOG_FILE`  | (unset) | If set, additionally write logs to this file.                          |

### Capacity

| Variable              | Default | Description                                                |
| --------------------- | ------- | ---------------------------------------------------------- |
| `MAX_QUEUE_SIZE`      | `1000`  | Queue capacity before the server returns 429.              |
| `MAX_IN_FLIGHT_COUNT` | `32`    | Max simultaneous in-flight requests.                       |
| `MAX_CONTEXT_LENGTH`  | `65536` | Max total tokens (prompt + completion) per request.        |
| `MAX_SESSIONS_COUNT`  | `128`   | Max concurrent sessions tracked.                           |
| `PM_MAX_USERS`        | `128`   | Max concurrent users the pipeline manager accepts.         |

### Generation

| Variable                 | Default | Description                                             |
| ------------------------ | ------- | ------------------------------------------------------- |
| `USE_FAST_MODE`          | `false` | Enable fast-mode generation path.                       |
| `USE_DEEPSEEK_MD_FORMAT` | `false` | Apply DeepSeek-specific markdown formatting to outputs. |

### Timeouts (ms)

The defaults below are smoke-test values. Bringing the runner up on real
hardware takes a long time, so production deployments typically set
`PM_CONNECT_TIMEOUT_MS` to several hours and `WARMUP_TIMEOUT_MS` to
roughly an hour.

| Variable                 | Default | Description                                                                                              |
| ------------------------ | ------- | -------------------------------------------------------------------------------------------------------- |
| `WARMUP_TIMEOUT_MS`      | `10000` | Max wait for the first token during runner warmup.                                                       |
| `OUTPUT_HANG_TIMEOUT_MS` | `60000` | Max gap with no model output (while a request is in flight) before the worker self-terminates.           |
| `PM_CONNECT_TIMEOUT_MS`  | `30000` | Pipeline manager connect timeout. Must be large enough to ride out runner startup.                       |

### Shared memory and IPC

The server uses `/dev/shm` for the descriptors and queues that wire the
main process to its worker processes and to the Tenstorrent device. The
container needs access to the host's shared-memory namespace to reach
the hardware, which means `/dev/shm` is shared with every other
container on the same host. The default names below are hard-coded, so
two deployments using the defaults on the same host will collide on the
same segments, and a crashed container can leave stale segments behind
that block the next one from starting.

**Set unique values per deployment.** Prefix or suffix every variable in
this section with an identifier that is unique to your deployment (e.g.
the release name plus an install timestamp). Pair it with an automated
cleanup of `/dev/shm/<prefix>_*` on container start and shutdown to
recover after crashes.

| Variable                         | Default              | Description                                                          |
| -------------------------------- | -------------------- | -------------------------------------------------------------------- |
| `BLAZE_SOCKET_DESCRIPTOR_PREFIX` | `deepseek`           | Prefix for the Blaze runner's shared-memory socket descriptors.      |
| `TT_TASK_QUEUE`                  | `tt_tasks`           | Main → worker task queue name.                                       |
| `TT_RESULT_QUEUE`                | `tt_results`         | Worker → main result queue name.                                     |
| `TT_CANCEL_QUEUE`                | `tt_cancels`         | Cancel-request queue name.                                           |
| `TT_WARMUP_SIGNALS_QUEUE`        | `tt_warmup_signals`  | Worker warmup-signal queue name.                                     |
| `TT_MEMORY_REQUEST_QUEUE`        | `tt_mem_requests`    | Memory allocation request queue name.                                |
| `TT_MEMORY_RESULT_QUEUE`         | `tt_mem_results`     | Memory allocation result queue name.                                 |
| `TT_WORKER_METRICS_SHM`          | `/tt_worker_metrics` | POSIX shared-memory segment for the worker metrics transport.        |
| `RESULT_QUEUE_CAPACITY`          | `65536`              | Capacity (messages) of the result queue.                             |
| `CANCEL_QUEUE_CAPACITY`          | `1024`               | Capacity (messages) of the cancel queue.                             |
| `MEMORY_QUEUE_CAPACITY`          | `128`                | Capacity (messages) of each memory queue.                            |
