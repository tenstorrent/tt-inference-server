# Video service benchmark

`video_benchmark.py` measures the video service over HTTP. It follows the shape of the
Python benchmark (`test_module/benchmark_tests/video_benchmark_tests.py`: submit, poll
until `completed`, download, `inference_steps_per_second = steps / elapsed`), and also
reads the `tt_video_*` series from `/metrics` before and after each phase, so the time
is split into queue wait, generation and server overhead.

| Phase | Needs | What it measures |
|---|---|---|
| `sequential` | real runner | N videos one after another (single user) |
| `burst` | real runner | K videos at once: queue wait, completion times, videos per minute |
| `control` | mock runner | latency of the cheap endpoints as the store grows (also `GET /jobs` size) |
| `throughput` | mock runner | jobs per second through the whole pipeline |

The same script runs against the Python server (same API); only the `/metrics` part is
specific to the C++ server and is skipped when the series are missing.

```bash
# server overhead, no device (starts and stops its own mock server)
./tests/video/benchmark/run_mock_benchmark.sh /tmp/bench-mock

# real device: start the server first (see tests/video/real_device/README.md), then
python3 tests/video/benchmark/video_benchmark.py --phase sequential --calls 3 --steps 40 --out seq40.json
python3 tests/video/benchmark/video_benchmark.py --phase burst --burst 3 --steps 4 --out burst.json
```

Telemetry added to the C++ server (`GET /metrics`):

| Series | Meaning |
|---|---|
| `tt_video_queue_wait_seconds` | submit until a worker takes the task |
| `tt_video_generation_seconds` | the worker's own generation time |
| `tt_video_e2e_seconds{outcome}` | submit until the result reaches the server |
| `tt_video_server_overhead_seconds{phase}` | `request_write`, `shm_enqueue`, `response_handling` |
| `tt_video_tasks_total{outcome}` | finished tasks, `ok` / `error` |
| `tt_video_jobs{status}` | jobs in the store by status |
| `tt_video_job_cleanup_total{kind}` | `evicted` (expired) and `stuck` (`stale_job`) |
| `tt_video_worker_restarts_total{result}` | `ok`, `failed`, `gave_up` |

Not covered: per-stage times inside the worker (denoise, VAE), which the Python
`VideoStageRecorder` records in the worker process. They do not reach the C++ server.

## Where the Python server spends a `POST`

`py_post_profiler.py` starts the Python server with timing wrappers around the
`POST /v1/videos/generations` path (endpoint helpers, job creation, FastAPI dependency
resolution, body validation, each ASGI middleware layer) without changing the repository.
`post_latency.py` sends N identical `POST` requests and prints the client latency; it runs
unchanged against either server.
