# Video service: extended checks on a real device and on the mock runner

Date: 2026-11-14. Raw outputs sit next to this file as
`2026-11-14-video-real-device-extended-checks-*.log` (not committed).

This report covers the checks that earlier reports listed as not done: request fields, the video
itself, resumed downloads, behaviour under load, start/stop/restart, queue limits, the background
cleaner, a comparison with the Python server, and a second defect found in the worker exit path
(fixed). Earlier reports: [cancel](2026-11-14-video-cancel-real-server.md),
[cancel race](2026-11-14-video-cancel-race-real-server.md),
[queue, download, sync](2026-11-14-video-real-server-queue-download-sync.md),
[curl examples and worker crash](2026-11-14-video-curl-examples-and-worker-crash-real-device.md).

## Environment

- T3K (4 cards, 8 chips), Wan2.2-T2V-A14B-Diffusers, runner `tt-wan2.2`, `DEVICE_IDS="(0,1,2,3)"`,
  tt_dit weight cache, `HF_HUB_OFFLINE=1`. Tracing is disabled in the Python runner
  (`"traced": False`; with the default the output is noise).
- `cpp_server` branch `cpp-video-service`, including the two fixes of `VideoDeviceRunner`
  (section 8). Jobs use 4 steps unless stated; one 4-step job takes about 78 to 81 s.
- Python server for the comparison: `uvicorn main:app` of `tt-media-server`, same machine and
  configuration, started when the C++ server was stopped (only one server can own the chips).
- Mock tests need only a build of the server.

## Summary

| Area | Where | Result |
|---|---|---|
| C++ unit tests (`Video*`, `CancelQueue*`) | build tree, `ctest -R "Video\|CancelQueue"` | 55/55 |
| Mock: contract cases | `tests/video/mock/run_contract_tests.sh` | 36/36 |
| Mock: server cases (5 phases) | `tests/video/mock/run_server_tests.sh` | 58/58 and the race log check |
| Mock: worker crash and warmup failure | `tests/video/mock/worker_crash_cases.py` | 15/15 |
| Mock: lifecycle, shutdown and load (new) | `tests/video/mock/lifecycle_cases.py` | 53/53 |
| Device: fields, content, Range, load, 50 steps (new) | `real_device/jobs/fields_content_load.py` | 35 PASS, 1 FAIL: the failure was a wrong expectation, corrected in the script (3.6) |
| Device: dropped sync request, 30 queued jobs (new) | `real_device/jobs/disconnect_and_queue_depth.py` | 9/9 |
| Device: start, stop, restart, purge, unwritable folder (new) | `real_device/lifecycle/restart_cases.py` | 27/27 |
| Device: warmup answers and queue limits (new) | `real_device/lifecycle/warmup_and_limits.py` | 8/8 |
| Device: probes sent to both servers (new) | `real_device/jobs/api_parity.py` | 72 probes each, differences in section 6 |

New defect found and fixed: a worker whose warmup failed died with a segmentation fault on exit
(section 8). Open findings are in section 9.

## 1. Mock runner: lifecycle, shutdown and load

`python3 tests/video/mock/lifecycle_cases.py` starts its own server for each case (private ports,
shm names and output folder), so it can run next to a server that owns the chips.

| Case | What it checks | Result |
|---|---|---|
| cleaner | with `JOB_CLEANUP_INTERVAL_SECONDS=1` and `JOB_RETENTION_SECONDS=3` a finished job is evicted without new requests; the list is empty; the log has the cleanup line | PASS |
| stale | with `JOB_MAX_STUCK_TIME_SECONDS=2` a slow job becomes `failed` / `stale_job`; its late result does not revive it and the late mp4 is removed | PASS |
| sigterm | SIGTERM during a job: exit code 0 in 0.8 s, no worker process, no ipc folder, no shm queues | PASS |
| disconnect | the client drops a synchronous request: the task runs to the end, the mp4 stays on disk (the same as for a finished sync request, a documented limitation), no job record, the server keeps serving | PASS |
| sync2 | two synchronous requests at once: both 200, the second waits behind the first (3.0 s and 5.8 s with a 3 s task) | PASS |
| restart | after `kill -9` of the worker, `/tt-liveness` reports `model_ready=true` in 11 of 11 samples while the new worker warms up (6 s), and `POST` answers 202 throughout | observed (finding 9.2) |
| preclaim | the worker dies after taking a task and before `CLAIMED`: the job stays `queued` 8 s later although `JOB_MAX_STUCK_TIME_SECONDS=3`; a new job completes | observed (finding 9.3) |
| stress | 400 concurrent POST plus GET, cancel and DELETE (64 threads): all accepted in 0.3 s, only 200/404/409 returned, every job reaches a terminal status (266 completed, 108 cancelled, none failed) | PASS |
| bigtable | 10000 jobs created in 4.4 s; the list returns all of them in 0.44 s; a single GET while four threads list in a loop: median 1 ms, max 450 ms | PASS |
| downloads | 64 parallel full downloads and 64 parallel `bytes=N-` downloads of one mp4: identical bytes, correct tail | PASS |
| noffmpeg | no ffmpeg on `PATH`: the job completes and the log says the remux failed and the original is kept | PASS |

## 2. Device: validation and the request fields

`real_device/jobs/fields_content_load.py` (step 1). No device time is used: a blocker job runs
first and the echo jobs are cancelled while queued.

| Check | Result |
|---|---|
| a request with only `prompt` is echoed with `num_inference_steps: 20` | PASS |
| all optional fields (`negative_prompt`, `num_inference_steps`, `seed`, `aspect_ratio`, `height`, `width`, `fps`, `duration`, `num_frames`) are accepted and echoed unchanged | PASS |
| steps 3 and 51: 422; steps 4 and 50: accepted | PASS |
| 422 for: empty prompt, blank prompt, no prompt, `height` 0, negative `fps`, prompt not a string, `seed` not a number, broken JSON | PASS (8 of 8) |
| a prompt of about 20000 characters with Unicode (č ć ž š đ, Cyrillic, Japanese, emoji) is accepted, generates, and is echoed unchanged | PASS |

The Wan2.2 runner reads no shape field: `height`, `width`, `num_frames`, `fps`, `duration` and
`aspect_ratio` are accepted and ignored, and the video is always 832x480, 81 frames. So no value of
those fields makes the runner fail.

## 3. Device: the video, Range, load, memory

### 3.1 The mp4

`ffprobe`: h264, 832x480, 16 fps, 81 frames. The box order is `ftyp`, `moov`, `free`, `mdat`
(faststart) and the log says `mp4 already faststart`, so no remux ran. Frames 0, 40 and 80 of a
4-step video were extracted and looked at: a coherent scene with a figure and snow, no noise (the
quality is low with 4 steps, as expected).

### 3.2 Range and resumed downloads

| Check | Result |
|---|---|
| `bytes=0-99`: 206, 100 bytes, `Content-Range: bytes 0-99/<size>` | PASS |
| `Accept-Ranges: bytes`, `Content-Length` equals the file size | PASS |
| a download cut at 40% and resumed with `bytes=N-` gives the same file (143610 + 215415 = 359025 bytes) | PASS |
| `bytes=-50`: 206; `bytes=<size>-`: 416 | PASS |
| `bytes=5-2` (last before first): 200 with the whole file | allowed, see 3.6 |

### 3.3 Responsiveness while the device generates

During one job, a loop of `GET` of the job and `GET /v1/videos/jobs` plus 48 parallel downloads of an
older video (16 threads): `GET` of a job median 1 ms, max 3 ms (179 samples); `GET /jobs` median 1 ms,
max 5 ms; all 48 downloads byte-identical; the job completed normally. HTTP threads are not blocked
by a generation.

### 3.4 Deep queue and a dropped synchronous request

`real_device/jobs/disconnect_and_queue_depth.py`:

- The client drops a `?async=false` request after 20 s. The generation runs to the end (the worker
  logged two `wrote` lines for the dropped request and the job queued behind it), the job behind it
  completed after 138 s, no job record exists for the dropped request, and both mp4 files stay in the
  output folder (the documented sync limitation).
- 30 jobs queued: all accepted in under a second, one `in_progress` and 29 `queued`, `queue_size` 30.
  The 29 cancels took under 0.1 s and all became `cancelled` at once, the running job completed, the
  queue drained to 0.

### 3.5 Step count and memory

- 4 steps: about 78 s. 20 steps: 364 s (from the worker log). 50 steps: 904 s, completed. About 18 s
  per step plus a fixed part.
- Resident memory after about six jobs including the 50-step one: server process 25 MB to 32 MB,
  worker 1537 MB to 1571 MB. No growth worth noting.

### 3.6 One wrong expectation

The first run of the script reported 35 PASS and 1 FAIL: it expected 416 for `Range: bytes=5-2`, and the
server answers 200 with the full file. An invalid byte range may be ignored (RFC 7233), so the
expectation was wrong; the script now accepts 200 or 416. The same 200 was seen again in the
Python-vs-C++ probes. The corrected check was not run as part of a second complete run.

## 4. Device: start, stop, restart

`real_device/lifecycle/restart_cases.py` starts and stops the server itself.

| Check | Result |
|---|---|
| start from the weight cache is ready after 102 s | PASS |
| purge on start with planted files: the three top-level `*.mp4` files are removed (`old1.mp4`, `old2.mp4`, `x_silent.mp4`); `audio.wav`, `notes.txt`, a subfolder with an mp4 and a symlink named `*.mp4` are kept; the log says `removed 3 orphan mp4` | PASS |
| the log names the tt_dit weight cache folder | PASS |
| `DELETE` of a finished job removes its mp4; the job is then 404 | PASS |
| SIGTERM of the idle server: exit code 0 in 0.9 s, no process left, ipc folder and shm queues removed, log says `graceful` | PASS |
| the next start after an idle SIGTERM works without `tt-smi -r` (102 s) | PASS |
| `TT_VIDEO_PURGE_ON_START=0` keeps an existing mp4 | PASS |
| seed 7 after a restart: the mp4 is byte-identical (4316534 bytes, same SHA-256 prefix) | PASS (record) |
| SIGTERM with a job in step 1 or 2 of 4: the server exits with code 0 after 0.8 s, no process left, ipc folder and shm queues removed | PASS |
| the start right after that SIGTERM | **fails**: ETH heartbeat timeout, the chips need `tt-smi -r` (finding 9.1) |
| an output folder that is not writable: the server starts | PASS |
| a job with such an output folder | `failed` / `processing_error`, message `Failed to export video: [Errno 32] Broken pipe` (finding 9.4); the server keeps accepting jobs and stays alive | PASS |

On seed repeatability: an earlier run on the same day, before the configuration was final, gave different
bytes after a restart (PSNR 30.9 dB). With the final configuration (tracing off in the runner) two runs in
different server processes produced identical files.

## 5. Device: warmup answers and limits

`real_device/lifecycle/warmup_and_limits.py` (`MAX_QUEUE_SIZE=3`, `MAX_JOBS=5`):

| Check | Result |
|---|---|
| during the warmup (102 s, 33 samples): `/tt-liveness` is 503 with `model_ready=false`, `POST` is 503 with `Retry-After: 30`, `POST` without a token is 401 | PASS |
| three active jobs are accepted, the fourth and fifth get 429 | PASS |
| with the table full, `POST` gets 503 (finished jobs count; 2 accepted, then 503 three times) and the table holds at most 5 records | PASS |

## 6. Python server and C++ server, same probes

`real_device/jobs/api_parity.py` sent 72 probes to each server on the same machine (auth, validation,
unknown ids, a job through its life, download headers, Range, synchronous mode). Identical answers on
the rest; the differences are:

| Probe | Python server | C++ server |
|---|---|---|
| error body of 401/422/404 | `{"detail": ...}` | `{"detail": ..., "error": {...}}` (a superset) |
| `model` in the job | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | `tt-wan2.2` (falls back to the runner name when `SERVED_MODEL_NAME` and `MODEL_WEIGHTS_PATH` are not set) |
| `download` of a queued or running job | 404 | 409 |
| `ETag` and `Last-Modified` on `download` | present | absent |
| `HEAD` on `download` | 405 | 200 |
| `POST ?async=false` | ignored: 202 and a job (async/sync is only the server setting) | 200 with the mp4, `Content-Type: video/mp4`, `X-Generation-Time` |
| `POST ?async=banana` | 202 | 422 |
| synchronous request leaves a job record | (not applicable) | no |

Equal on both: 401 for a missing and for a wrong token on all six routes; 422 for all ten invalid
bodies; 404 for unknown, malformed and numeric ids on all four id routes; the `202` body fields; `409`
for `DELETE` of an active job; cancel of a queued job gives `cancelled`, a second cancel 404; `DELETE`
body `{"id", "object": "video", "deleted": true}`; `Accept-Ranges: bytes`, `206` for a range, `416` past
the end; the completed job keys.

### Worker crash on the Python server

`kill -9` of the device worker while a job runs and a second job waits: the job stayed `in_progress`
and the second `queued` for the whole 8 minute observation. `/tt-liveness` went to 503 after about 30 s.
The service restarted the worker five times (the chips needed a reset, as for the C++ server), logged
`has died too many times (5), restart did not help` every 30 s, and kept running. Compare the C++
server: the held job becomes `failed` after 1 s and, past the restart limit, the process exits with
code 1.

## 7. Not tested

- Several workers (a T3K mesh uses all chips for one worker), multihost launches, LTX-2.3 and other
  runners.
- A disk that is full (only a folder that is not writable was tried), a video of hundreds of MB, ffmpeg
  missing with a real non-faststart video (the mock check used a file that is not a valid mp4).
- `DELETE` of a job whose result is outside the output folder (needs a runner that returns such a path).
- A stuck-job sweep against a real device hang (the sweep was tested with the slow mock).
- The Python server's own test suite, its `stale_job` cleanup, and its sync mode through the environment
  setting.
- Not part of this run: whether the Python server's first start failed because of the C++ server stopped
  before it. It failed with an ETH heartbeat timeout right after the previous C++ server (the one of the
  unwritable-folder phase) had exited, and started normally after `tt-smi -r`. The cause was not
  investigated.

## 8. Defect found and fixed: segmentation fault on exit after a failed warmup

The earlier fix (release the GIL when warmup fails) let a worker with a failed warmup finally exit, and
exposed a second defect: it exited with SIGSEGV ("killed by signal 11 -- core dumped") instead of code 1.
Reproduced without a device (a wrong `TT_PYTHON_PATH` fails the warmup at the import) and caught with
`gdb`: `VideoDeviceRunner::stop()` called `attr()` on `python->runner` or `python->loop` while they were
null handles (the warmup failed before they were created). `is_none()` is false for a null handle, so the
guard did not stop it. The fix checks the handle first (`!python->runner || ...is_none()`, and
`python->loop && ...`). After it the worker exits with code 1 and the log shows no signal. The service
behaviour is unchanged (it restarts the worker up to the limit and then exits), but the exit is now clean
and no core dump is produced for each failed warmup.

The regression test is case C of `tests/video/mock/worker_crash_cases.py`, which now also checks that the
log has no `killed by signal`.

## 9. Open findings

1. **SIGTERM while a job runs leaves the chips unusable until `tt-smi -r`.** The server closes the mesh
   device from the shutdown path while the generation is still in flight, the worker exits within a
   second, the log says `Server shutdown complete (graceful)`, and the next start fails with an ETH
   heartbeat timeout. A SIGTERM of an idle server is clean. A drain (finish or bound the running job, then
   close) or at least a different log line would help; this needs a decision on how long a 15 minute job
   may delay a shutdown. Not changed.
2. **`/tt-liveness` stays `model_ready=true` while a restarted worker warms up** (mock: 11 of 11 samples),
   so `POST` is accepted (202) for jobs that wait for a worker that cannot take them yet. Not changed.
3. **A task taken by a worker that dies before `CLAIMED` stays `queued` for good.** The stuck-job sweep
   only fails running jobs, so it does not catch it, even with a 3 s limit after 8 s. Mock only; a client
   would have to cancel and resubmit. Not changed.
4. **A job fails after the whole generation when the output folder cannot be written**, so the device time
   is lost, and the message is `Broken pipe` rather than a permission error. **Fixed on 2026-11-16**, see
   [the follow-up report](2026-11-16-video-output-dir-check-real-device.md).
5. **Synchronous mp4 files stay on disk until the next start** (known), also for a dropped request.
6. **Orphan `/tmp/tt-video-ipc-<pid>` folders** of servers that were killed (not stopped) stay in `/tmp`;
   the next start does not remove folders of other pids. They are empty, 4 KB each.
7. The C++ server sends no `ETag` or `Last-Modified`, so `If-Range` cannot be used for a resumed download;
   `Range: bytes=N-` works.
8. Differences from the Python server in section 6 (`model` value, 409 against 404, `?async`, `HEAD`).

## Commands

From `tt-media-server/cpp_server`, with the server and the device ready (the server log and output folder
are optional):

```bash
# mock, no device
python3 tests/video/mock/lifecycle_cases.py
python3 tests/video/mock/worker_crash_cases.py
./tests/video/mock/run_server_tests.sh
./tests/video/mock/run_contract_tests.sh

# real device, server already running
python3 tests/video/real_device/jobs/fields_content_load.py --server-log <log> --output-dir <TT_VIDEO_OUTPUT_DIR>
python3 tests/video/real_device/jobs/disconnect_and_queue_depth.py --server-log <log> --output-dir <TT_VIDEO_OUTPUT_DIR>
python3 tests/video/real_device/jobs/api_parity.py --base-url http://127.0.0.1:8000 --api-key <token> > answers.txt

# real device, the test starts and stops the server itself (nothing else may use the chips)
python3 tests/video/real_device/lifecycle/restart_cases.py --tt-metal <tt-metal> --media-server <tt-media-server> --dit-cache <cache> --api-key <token>
python3 tests/video/real_device/lifecycle/warmup_and_limits.py --tt-metal <tt-metal> --media-server <tt-media-server> --dit-cache <cache> --api-key <token>
```

Run `api_parity.py` against each server and `diff` the two outputs after replacing the UUIDs.
