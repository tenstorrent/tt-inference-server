# Video benchmark: C++ server against the Python server (T3K, Wan2.2)

Date: 2026-11-16. Same benchmark, same machine, same model and requests, run once against
each server. A benchmark, not a functional validation. The C++ run is described in
detail in [the C++ benchmark report](2026-11-16-video-benchmark-real-device.md).

## Summary

- **Device time is the same.** For the same request both servers take the same time to
  generate (within 0.2 to 0.6 s on 79 to 725 s), and every mp4 has exactly the same size
  on both servers, so the same work was done.
- **The C++ control plane is faster:** `POST` 0.4 to 0.7 ms against 3.5 to 6 ms (measured
  step by step on the Python server: about 1 ms is a cross-process start event, 0.5 ms is
  thread-pool hops for dependencies, and the image-to-video checks the C++ server lacks
  cost only 0.04 ms), download
  1 to 4 ms against 48 to 65 ms. The download difference comes from a design change in the
  C++ service (see "Download"). Both are negligible next to generation time.
- **Throughput under a queue is the same:** 0.758 against 0.759 videos per minute (one
  worker, videos run one after another).
- **The two servers measure different things.** Python measures what happens inside the
  worker (denoise, VAE decode, encoding, frames, output size); the C++ server measures the
  queue and the server (queue wait, overhead, job states, cleanup, restarts). Neither covers
  everything; the gaps are listed under "Telemetry: what each server shows".
- **Python's per-request video metrics include the queue wait.** In the burst run they
  report a 4-step video as 158 s of "generation" and a step as 39.5 s, when the device
  spent 79 s and about 18.8 s.
- **The C++ server is ready about 30 s sooner** (101 s against 131 s from start to the end
  of warmup). About half of it is before the model starts loading (Python imports its
  libraries in the main process first); the other half is in the warmup itself and was not
  explained.
- **No hang in the Python run,** although it repeated the exact sequence after which the
  C++ run hung once (a 40-step video, then a burst 1 s later). This supports, but does not
  prove, that the hang seen in the C++ run is not caused by the server.

## Setup

| | C++ server | Python server |
|---|---|---|
| Start | `tt_media_server_cpp -p 8000` | `uvicorn main:app --lifespan on --port 8000` |
| Environment | `MODEL_SERVICE=video`, `MODEL=Wan2.2-T2V-A14B-Diffusers`, `MODEL_RUNNER=tt-wan2.2`, `DEVICE=t3k`, `DEVICE_IDS=(0,1,2,3)`, the same populated `tt_dit` weight cache | same |
| Chips | reset (`tt-smi -r`) before the start | reset before the start |
| Warmup finished after start (log timestamps) | 101 s (two starts, both 101 s) | 131 s (one start) |
| Model code | the same worker code (`tt_model_runners`) | same |

Client: `tests/video/benchmark/video_benchmark.py` (submit, poll every 1 s, download), the
same command lines for both servers: 3 videos at 4 steps, 2 at 20 steps, 2 at 40 steps
(sequential), then 3 videos at 4 steps submitted at once (burst). Output resolution
832x480, 81 frames (from the Python metrics). Prompts are the same on both runs.

## Results

### Generation time

| Steps | C++ client elapsed (s) | Python client elapsed (s) | C++ worker time, mean (s) | Python `generation_duration_seconds`, mean (s) |
|---|---|---|---|---|
| 4 | 79.14, 79.14, 79.14 | 79.35, 79.28, 80.28 | 78.76 | 79.05 |
| 20 | 366.67, 365.67 | 367.24, 366.23 | 365.65 | 366.32 |
| 40 | 725.31, 726.33 | 725.59, 724.40 | 724.96 | 724.62 |

Client elapsed is quantized by the 1 s poll interval, so differences under 1 s are within
noise. mp4 sizes per prompt were identical on both servers (for example 1,514,666 bytes for
the first 4-step prompt).

### Where the time goes (Python stage metrics, means)

| Steps | Denoise loop | VAE decode | mp4 encode | Rest | Per step (denoise / steps) |
|---|---|---|---|---|---|
| 4 | 75.09 s | 3.20 s | 0.21 s | about 0.5 s | 18.77 s |
| 20 | 362.34 s | 3.20 s | 0.21 s | about 0.6 s | 18.12 s |
| 40 | 720.65 s | 3.19 s | 0.21 s | about 0.6 s | 18.02 s |

A linear fit of the denoise loop gives 17.93 s per step plus about 3.4 s that does not
depend on the step count. So the "about 7 s fixed" seen in the C++ report is roughly
3.4 s at the start of the denoise loop, 3.2 s of VAE decode, 0.2 s of encoding and a few
hundred milliseconds of the rest. The C++ server cannot show this split today (see below).

### Burst, 3 videos at once (4 steps)

| | C++ | Python |
|---|---|---|
| Submit latency | 1.0 to 1.4 ms | 8.4 to 9.6 ms |
| Completion times | 79.2, 158.3, 237.5 s | 79.3, 158.9, 237.2 s |
| Throughput | 0.758 videos per minute | 0.759 videos per minute |

### Startup

From the log timestamps (a readiness poll every 10 s gave 105 s and 134 s earlier, with up
to 10 s error).

| From process start to | C++ | Python |
|---|---|---|
| first log line | 0 s | 11 s |
| device opened (cluster started) | 6 s | 19 s |
| first weight load from the cache | 11 s | 26 s |
| warmup finished | 101 s | 131 s |

About 15 s of the gap is before the weights start loading (Python imports FastAPI, torch
and ttnn in the main process before the worker starts). The other about 15 s is in the warmup
(90 s against 105 s from the first weight load), which runs the same model code; it was not
investigated.

### Control plane

| | C++ | Python |
|---|---|---|
| `POST` create job, idle server | 0.7 to 0.9 ms | 3.5 to 6.0 ms |
| Download of a 1.0 to 2.3 MB mp4 | 0.9 to 3.6 ms | 48 to 65 ms |

### Why `POST` is slower on the Python server (measured)

Both servers were sent the same 300 `POST` requests one after another (`post_latency.py`,
4-step text-to-video body). The Python server ran with timing wrappers around every step of
the request path (`py_post_profiler.py`); the C++ server ran with the mock runner, since a
`POST` does not reach the runner.

| | C++ | Python |
|---|---|---|
| Client latency, p50 / mean | 0.41 / 0.43 ms | 4.21 / 4.33 ms |

Where the Python server spends its 4.33 ms (means; times are inclusive per row, the
indented rows are parts of the row above):

| Step | Time | Has a counterpart in the C++ server? |
|---|---|---|
| HTTP handling outside the measured middleware (uvicorn, error middleware) | 0.31 ms | yes (Drogon) |
| `DeprecatedPathMiddleware` (a `BaseHTTPMiddleware` for old paths) | 0.32 ms | no |
| Dependency resolution and body validation | 0.71 ms | partly |
| &nbsp;&nbsp;three sync dependencies (API key, service resolver, t2v-on-i2v check), each run in a thread pool | 0.49 ms (3 x 0.16 ms) | the key check only, inline |
| &nbsp;&nbsp;Pydantic body validation | 0.09 ms | yes (JSON parse and field checks) |
| Endpoint body | 1.34 ms | |
| &nbsp;&nbsp;start event: `multiprocessing.Manager().Event()`, a round trip to the manager process | 0.97 ms | no (the C++ server uses a shared-memory queue) |
| &nbsp;&nbsp;`JobManager.create_job` (job record, asyncio task) | 0.29 ms | yes (job store insert, request file, shm push) |
| &nbsp;&nbsp;all image-to-video and media checks together (6 helpers) | 0.04 ms | no |
| &nbsp;&nbsp;`model_dump` of the request | 0.01 ms | no |
| Rest inside the router: route match, reading the body, building and sending the response, and the first part of the new job task that runs before the response is sent | about 1.6 ms | partly |

What this shows:

- **The missing image-to-video and media features explain almost nothing:** 0.04 ms of
  4.3 ms. Removing everything the C++ server does not have at all (those checks,
  `model_dump`, the deprecated-path middleware) would still leave the Python server at
  about 4 ms, ten times the C++ server.
- **The largest single cost is the cross-process start event** (about 1 ms per job), then
  the thread-pool hops for the three synchronous dependencies (about 0.5 ms). Both come
  from how the server is built, not from what it checks.
- **About 1.6 ms is not split further.** It includes framework work and the start of the job
  task, which the event loop runs before the response is sent, so the two cannot be
  separated with this method.

Limits: the timing wrappers add about a microsecond each. Whether the Prometheus
instrumentation middleware is inside or outside the measured part was not determined; its
cost is in either the 0.31 ms row or the rest. One run of 300 requests per server.

#### Download

The Python server remuxes the mp4 to faststart (moves the `moov` box before the data) into
a temporary copy on every download request, without checking whether the file already is
faststart. Its own metric `tt_media_server_download_result_duration_seconds` shows 45 ms per
download on the server side. The C++ service checks the file once, when the video is
finished, and remuxes it only if needed; the download then sends the file as it is. In
these runs the files were already faststart (the worker encodes with `+faststart`), so the
C++ server did not remux anything. This is a deliberate difference of the C++ service.

## Telemetry: what each server shows

Measured on the `/metrics` output of both servers during the runs above.

| Question | C++ server | Python server |
|---|---|---|
| How long did the device work on a video? | `tt_video_generation_seconds` (worker time) | `tt_media_server_model_inference_duration_seconds` (no per-video labels; counts the warmup too) |
| How long did a video wait in the queue? | `tt_video_queue_wait_seconds` | not measured |
| Submit to result | `tt_video_e2e_seconds{outcome}` | `tt_media_server_video_generation_duration_seconds` (named "generation", includes the queue wait) |
| Denoise / VAE split | not available | `..._video_denoise_duration_seconds`, `..._video_vae_decode_duration_seconds` |
| mp4 encoding | not available | `..._video_encode_duration_seconds`, `..._encode_frames_per_second` |
| Frames, pixels, content seconds, output bytes | not available | `..._frames_generated_total`, `..._vae_pixels_total`, `..._content_seconds_total`, `..._output_bytes_total`, `..._output_size_bytes`, `..._output_frames` |
| Speed per video | not available | `..._frames_per_second`, `..._pixels_per_second`, `..._realtime_factor`, `..._step_duration_seconds` (all derived from the queue-inclusive duration) |
| What was requested | not available | `..._requested_inference_steps`, `..._requested_duration_seconds`, `..._requested_aspect_ratio_total` |
| Server overhead around the device | `tt_video_server_overhead_seconds{phase}` | not measured |
| Jobs by status | `tt_video_jobs{status}` | `tt_media_server_jobs_in_progress` only |
| Success / failure counts | `tt_video_tasks_total{outcome}` | `..._video_generation_total{status}` |
| Cleanup (expired, stale jobs) | `tt_video_job_cleanup_total{kind}` | not measured |
| Worker restarts | `tt_video_worker_restarts_total{result}` | not measured |
| Download time | not measured | `tt_media_server_download_result_duration_seconds` |
| Model warmup | not measured | `tt_media_server_device_warmup_duration_seconds` |
| HTTP requests | `tt_http_requests_total{method,status}` | HTTP request count, duration and size |

### Problems seen in the Python metrics

- **Queue time is counted as generation time.** `..._video_generation_duration_seconds`
  runs from submit to result. In the burst its mean was 158.2 s for 4-step videos whose
  device time was 79 s. Every metric derived from it is skewed the same way: step duration
  39.5 s (true about 18.8 s), realtime factor 31.2 (true about 15.6), frames per second
  0.63 (true about 1.03). Without a queue (the sequential runs) the values are correct.
- **`step_duration_seconds` is the whole duration divided by the steps,** so it includes the
  VAE decode and the fixed costs: 19.8 s at 4 steps against 18.8 s from the denoise loop.
- **Warmup is counted** in `tt_media_server_video_encode_total` and
  `tt_media_server_model_inference_*` (one 64x64 encode and one inference before the first
  request).

## What the C++ server needs for the same picture

1. **The stage times from the worker.** The worker in the C++ server runs the same model
   code, so `VideoStageRecorder` most likely records denoise and VAE times there too, but in
   the worker process, where nothing exports them (not verified). The cheapest path is for
   the worker to put the stage times and the output facts (frames, resolution, bytes,
   encode time) into its response file, and for the C++ server to record them.
2. **The same series names** (`tt_media_server_video_*`) and labels for those values, so a
   dashboard built for the Python server works on both. The queue-inclusive Python
   duration should not be copied as is: keep generation and queue wait separate, as the C++
   server already does.
3. **Download and warmup durations,** which Python has and C++ does not.

## Not done

- One run per server, 2 to 3 videos per configuration; enough to compare, not for
  confident percentiles.
- Startup: two samples for the C++ server, one for the Python server.
- Sentry was not checked on either server.
- Only text-to-video at the default resolution, one worker.

Raw output: `2026-11-16-video-benchmark-cpp-vs-python.log`.
