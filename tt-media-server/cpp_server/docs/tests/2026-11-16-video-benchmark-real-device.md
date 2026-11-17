# Video service benchmark and telemetry (T3K, Wan2.2)

Date: 2026-11-16. This is a benchmark, not a functional validation: it measures how long
things take and what the server itself costs. Functional results are in the other reports
in this folder.

## Summary

- **The server is not the bottleneck.** The server-side cost around one video is about
  0.3 ms (request file write, shm enqueue, response handling), against 79 to 725 seconds of
  device time. The HTTP control plane answers in 0.2 to 0.6 ms (p50).
- **Generation time is linear in `num_inference_steps`:** about 17.95 s per step plus about
  7 s fixed (4 steps: 78.8 s, 20 steps: 365.6 s, 40 steps: 725.0 s).
- **One worker means one video at a time.** A burst of 3 videos finished at 79, 158 and
  237 s: 0.76 videos per minute, the same as sequential. Queueing neither helps nor hurts.
- **`GET /v1/videos/jobs` grows linearly with the store** (about 350 bytes and 0.04 ms per
  job): 95 ms and 810 KB at 2300 jobs. There is no pagination.
- **One device hang was seen** during the first burst attempt (see "Incident"). It was not
  reproduced in the repeat run. The cause is not known.

## Setup

| | |
|---|---|
| Machine | T3K (8 Wormhole chips), one video worker on mesh 2x4 |
| Server | `tt_media_server_cpp -p 8000`, `MODEL_SERVICE=video`, runner `tt-wan2.2` (Wan2.2-T2V-A14B-Diffusers), populated `tt_dit` weight cache, warmup finished 101 s after start (both starts, from the log timestamps) |
| Client | `tests/video/benchmark/video_benchmark.py` (submit, poll every 1 s, download), same shape as `test_module/benchmark_tests/video_benchmark_tests.py` |
| Request | default resolution, `num_inference_steps` 4, 20 or 40 (40 is what the Python benchmark uses for Wan2.2) |
| Mock runs | `tests/video/benchmark/run_mock_benchmark.sh`: mock runner with zero delay, no device, to isolate the server cost |
| Chips | reset (`tt-smi -r`) before the first start and again after the hang |

Server-side numbers come from the new `tt_video_*` series on `/metrics`, read before and
after each phase (deltas). Means are exact (`_sum / _count`). **Quantiles read from the
histograms are interpolated inside wide buckets and are not meaningful for durations of
tens of seconds, so only means are reported for them.** Quantiles in the tables below come
from client-side timings.

## Telemetry added

The C++ server had no timing measurements for video. Added to `GET /metrics`:

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

Code: `ServerMetrics` (`metrics.hpp/.cpp`), hooks in `VideoService`, and
`VideoJobStore::countByStatus()` (unit test added, 17/17 `VideoJobStoreTest` pass).

Not covered: the per-stage times inside the worker (denoise loop, VAE decode) that the
Python `VideoStageRecorder` records. They are measured in the worker process and do not
reach the C++ server, so the benchmark cannot split the 79 s into denoise and VAE. Sentry
was not checked.

## Results on the device

### Sequential, one video at a time

| Steps | Videos | Client elapsed (s) | Server generation, mean (s) | Python-style steps/s | Download | mp4 size |
|---|---|---|---|---|---|---|
| 4 | 3 | 79.14, 79.14, 79.14 | 78.76 | 0.0505 | 1.2 to 1.8 ms | 1.5 to 1.6 MB |
| 20 | 2 | 366.67, 365.67 | 365.65 | 0.0546 | 1.3 to 1.7 ms | 1.4 to 1.5 MB |
| 40 | 2 | 725.31, 726.33 | 724.96 | 0.0551 | 0.9 to 1.1 ms | 1.0 to 1.3 MB |

- Client elapsed is about 0.4 to 1 s above the server generation time. That gap is the 1 s
  poll interval of the client, not the server: the server's own submit-to-result time
  (`tt_video_e2e_seconds`) is only 0.5 ms above the generation time.
- Run-to-run variation is tiny: 79.137 to 79.144 s over three videos at 4 steps.
- Linear fit from the three points: **17.95 s per step** (36 steps: 725.0 - 78.8; 16 steps:
  365.6 - 78.8) and **about 7 s fixed**. The Python server's stage metrics later split this into about 3.4 s at the start of the denoise loop, 3.2 s of VAE decode and 0.2 s of mp4 encoding (see [the comparison](2026-11-16-video-benchmark-cpp-vs-python.md)). The fixed
  part is therefore under 10% even at 4 steps. Python-style `steps / elapsed` rises with the
  step count only because of that fixed part and tends to 0.0557.
- `POST` (create job) took 0.7 to 0.9 ms on the real server.

### Burst, 3 videos submitted at once (4 steps)

| | |
|---|---|
| Submit latency | 1.0, 1.3, 1.4 ms (all three accepted with 202) |
| Completion times | 79.2 s, 158.3 s, 237.5 s |
| Gap between completions | 79.16 s, 79.14 s (equal to one video) |
| Wall time / throughput | 237.5 s / 0.758 videos per minute |
| Queue wait, server mean | 78.8 s (the three waited about 0, 79 and 158 s) |
| Generation, server mean | 78.9 s |
| End to end, server mean | 157.7 s |

Throughput equals 60 / 79.14 = 0.758: a single worker serves tasks strictly one after the
other, so a queue adds waiting but costs no device time. Latency of the last video in a
burst of K is about K times a single video.

### Server overhead per task (means, all device runs)

| Phase | Mean | What it covers |
|---|---|---|
| `request_write` | 0.07 to 0.11 ms | writing `request-<id>.json` to the payload folder |
| `shm_enqueue` | 0.02 ms | pushing the task into the shared-memory queue |
| `response_handling` | 0.16 to 0.17 ms | reading the response json, finishing the job, cleanup |
| **Total** | **about 0.3 ms** | against 78.8 s or more of generation (under 0.0005%) |

## Results with the mock runner (server cost without a device)

The mock worker takes about 40 ms per job (it writes a file), so it, not the server, limits
throughput here.

| Measurement | Value |
|---|---|
| 300 jobs, 8 client threads, full submit/poll/download | 24.5 jobs per second, no rejections |
| `POST` create job (store with about 2000 jobs) | p50 0.56 ms, p95 0.73 ms, p99 0.89 ms |
| `GET` job status | p50 0.25 ms, p95 0.31 ms, p99 0.34 ms |
| `GET` unknown id (404) | p50 0.22 ms, p95 0.28 ms |
| `GET /tt-liveness` | p50 0.22 ms, p95 0.29 ms |
| `GET /metrics` | p50 0.37 ms, p95 0.44 ms |
| Server overhead means | request write 0.13 ms, shm enqueue 0.01 ms, response handling 0.12 ms |

### `GET /v1/videos/jobs` against the size of the store

| Jobs in store | Response size | Latency p50 | p95 |
|---|---|---|---|
| 400 | 142 KB | 16.5 ms | 16.9 ms |
| 800 | 282 KB | 32.3 ms | 33.6 ms |
| 1300 | 458 KB | 52.7 ms | 54.3 ms |
| 2300 | 810 KB | 94.9 ms | 97.5 ms |

Linear: about 352 bytes and 0.041 ms per job. `GET` of one job stays at 0.24 to 0.30 ms
at every size. The default limit is `MAX_JOBS=10000`, which would be roughly 3.5 MB and
400 ms per call (extrapolated, not measured). The list is built while the job store mutex
is held (from reading the code, the effect on other requests was not measured), and the
web console calls it every 2 s. This is the measured basis for adding pagination.

## Incident: device hang during the first burst attempt

During the first burst run (3 videos, 4 steps, submitted 1 s after a 725 s video finished),
the first burst video entered the VAE decode at 20:00:05 and never finished. Evidence:

- the worker used 100% of a core for 12 minutes (a normal VAE decode takes about 3 s);
- a backtrace of the worker (`gdb`) showed it waiting in
  `tt::tt_metal::distributed::FDMeshCommandQueue::read_completion_queue()`, that is, for
  the device to return the result;
- the job stayed `in_progress`, the other two stayed `queued`, no error was logged, and the
  stale-job cleanup (3 h) had not fired yet.

The server was stopped (SIGTERM during a job), the chips were reset with `tt-smi -r`, and
the burst was repeated on a fresh server: all three videos succeeded (the numbers above).

What is not known: whether the hang is caused by the job that ran just before it, by the
sequence of step counts (4, 20, 40, then 4 again), by the very short gap, or by the
device itself. It happened once and was not reproduced. A targeted repeat (a 40-step video,
then a 4-step video, no reset) would tell. What the server does about it: nothing until the
3 h stale-job limit, and even then the worker is not interrupted, so a hung worker blocks
the queue until the process is restarted.

## Not done


- **One resolution and one worker**, text-to-video only. Other resolutions and multi-worker
  scaling were not measured.
- **Per-stage times (denoise, VAE) are not available** (see "Telemetry added").
- 2 to 3 videos per configuration: enough to show the scaling, not to give confident
  percentiles.

## Reproducing

```bash
# server overhead without a device
./tests/video/benchmark/run_mock_benchmark.sh /tmp/bench-mock
# on the T3K, with the real server running
python3 tests/video/benchmark/video_benchmark.py --phase sequential --calls 3 --steps 40
python3 tests/video/benchmark/video_benchmark.py --phase burst --burst 3 --steps 4
```
See `tests/video/benchmark/README.md`. Raw output: `2026-11-16-video-benchmark-real-device.log`.
