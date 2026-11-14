# Video service: job queue, download and sync mode on a real server

Date: 2026-11-14
Result: **45 of 45 checks passed.** Duration 397 s, 6 real generations in total on a T3K with Wan2.2.
Script: [tests/video/real_device/jobs/queue_download_sync.py](../../tests/video/real_device/jobs/queue_download_sync.py)
First report (cancel in the queue and while running, basic flow): [2026-11-14-video-cancel-real-server.md](2026-11-14-video-cancel-real-server.md)

## Environment

T3K, `tt_media_server_cpp -p 8000`, `MODEL_RUNNER=tt-wan2.2`, a populated `tt_dit` weight cache, `DEVICE_IDS="(0,1,2,3)"`. The server was not restarted between the two sets of tests. Jobs were sent with `num_inference_steps=4`, except the job that deliberately omits the field (E).

Note: the server was started with a binary that did not yet have the new cancel INFO log line, so it does not appear in the log of this run.

## Prompts and jobs

| Job | ID | Prompt | Seed | Steps | Outcome | Video file in `cpp_server/videos/` |
|---|---|---|---|---|---|---|
| A | `ded2c74a-c84c-4c21-a4ff-a049f274ff3b` | `a red ball rolling on grass` | 7 | 4 | `completed` | `f7e15192-81d0-4bdc-af2f-f43c27cbca27.mp4` (2 935 180 B) |
| B | `8c926344-a532-4661-a09c-4d9ff47c3077` | `job B (bice otkazan)` ("will be cancelled") | 11 | 4 | `cancelled` in the queue | none |
| C | `02eae4ff-30cc-4704-84ea-44eb6f9709e6` | `a blue cube spinning slowly` | 7 | 4 | `completed` | `61dbf402-be12-4346-8922-f6dba171e684.mp4` (635 229 B) |
| D | `3c2ac9c0-7c0e-48c0-8a53-412ce67b8c9e` | `job D (bice otkazan)` ("will be cancelled") | 12 | 4 | `cancelled` in the queue | none |
| E | `b270a970-69da-4d7d-8cb4-01f5d93169d1` | `default koraci` ("default steps") | not set | not set (default 20) | `cancelled` in the queue | none |
| F | `7cf916c8-74f4-4347-95e5-68bec4afc979` | `a blue cube spinning slowly` (same as C) | 7 | 4 | `completed` | `dcf90c9d-ca54-467a-bf07-66a45642da0c.mp4` (635 229 B) |
| sync 1 | none (sync creates no job) | `sync one` | 21 | 4 | 200 + mp4 | `0296967c-8a61-45b4-85fc-0fe0b75375e0.mp4` (1 205 085 B) |
| sync 2 | none | `sync two` | 22 | 4 | 200 + mp4 | `ad785610-70da-4af1-84e9-817ebf632b1b.mp4` (1 462 105 B) |

File-to-job pairs were matched by the file size reported by `download` (A, C, F) and by creation time. Sync files cannot be linked by id because sync creates no job, so they were matched by the size of the response. The prompts of B, D and E are deliberately meaningless: those jobs are never executed.

## 1. Auth and validation (no device needed)

| Check | Result |
|---|---|
| `POST` without a token | 401, PASS |
| `POST` with a wrong token | 401, PASS |
| `GET /v1/videos/jobs` without a token | 401, PASS |
| Empty prompt | 422 (`prompt must not be empty`), PASS |
| `num_inference_steps=3` (below 4) | 422, PASS |
| `num_inference_steps=51` (above 50) | 422, PASS |
| Invalid JSON | 422, PASS |
| Unknown job (GET) | 404, PASS |
| Cancel of an unknown job | 404, PASS |

## 2. Several jobs in the queue, and cancel

Order: A is generating, B, C, D and E wait. B, D and E are cancelled; C must stay and still run.

| Check | Result |
|---|---|
| B, C, D, E are all `queued` while A runs | PASS |
| Job E without `num_inference_steps`: the job record has `num_inference_steps = 20` | PASS |
| `DELETE` of a running job (A) | 409, PASS |
| `DELETE` of a queued job (C) | 409, PASS |
| Cancel B, D, E | each `cancelled` at once (HTTP 200), PASS |
| Markers in the payload directory | three markers (`cancel-9`, `cancel-11`, `cancel-12`), PASS |
| C after the other cancels | still `queued`, PASS |
| A and C | both `completed`, PASS |
| Server log | exactly 3 lines `skipping cancelled task ... (before claim)`, PASS |
| B, D, E | `cancelled` without an `error` field, PASS |
| Payload directory at the end | empty (markers and files removed), PASS |

C ran even though it sat between cancelled jobs in the queue: the worker skips only tasks with a marker and runs the others in order.

## 3. Download, Range, faststart, ffprobe (job A)

| Check | Result |
|---|---|
| `download` | 200, `Accept-Ranges: bytes`, `video/mp4`, `Content-Length` equal to the size, `attachment` in `Content-Disposition`, PASS |
| `Range: bytes=0-99` | 206, `Content-Range: bytes 0-99/2935180`, PASS |
| `Range: bytes=-10` | 206, the last 10 bytes, PASS |
| `Range` beyond the file | 416, `Content-Range: bytes */2935180`, PASS |
| `Range` from the middle to the end (seeking) | 206, 1 467 590 B, PASS |
| Faststart | `moov` at 36, `mdat` at 1187, PASS |
| `ffprobe` | H.264, 832x480, 81 frames, 16 fps, duration 5.063 s, PASS |

`ffprobe` confirms that the file is a valid video. It does not mean the picture shows what the prompt describes, because the content was not inspected by the test.

For every job the worker logs `mp4 already faststart`, so the remux was skipped.

## 4. Repeatability with the same seed

C and F have the same prompt and seed (7).

| Check | Result |
|---|---|
| C and F | both 635 229 B, same SHA-256 prefix `e12cc6158219ac95`, **byte-for-byte identical**, PASS |
| A (different prompt and seed) | differs from C (`abbfe000f8cfa96e`), PASS (sanity check that the comparison is not trivial) |

With the same seed and parameters, generation is deterministic **within one server session**.

A limitation found later the same day: between two server sessions (with a restart and a chip reset) the same request (seed 7) once did not produce a byte-identical file. The earlier `A.mp4` was 2 935 180 B, while the same request built later was 2 951 811 B (identical in three consecutive jobs). PSNR between them is 30.9 dB and frame 40 shows the same scene in both. So repeatability across restarts is not confirmed (seeds 1 and 4 matched by size between sessions, seed 7 did not). The same request built with the earlier C++ (worker index as the runner id) gave exactly the same file as the new one (SHA-256 prefix `03dbef99d2c4ee66`), so the difference is not caused by the runner-id change.

## 5. Sync mode (`?async=false`) and two simultaneous requests

| Check | Result |
|---|---|
| Sync 1 | 200, `video/mp4`, `ftyp`, 1 205 085 B, 79 s, PASS |
| Sync 2 (sent 1 s after the first) | 200, mp4, 1 462 105 B, 156 s, PASS |
| Sync 2 waits behind sync 1 | 156 s vs 79 s (one device), PASS |
| `X-Generation-Time` header | present on both (`78.74`), PASS |
| Sync creates no job record | job count 10 before, 10 after, PASS |
| Payload directory | empty at the end, PASS |
| `CRASHED` or `TT_FATAL` in the log | none, PASS |
| `/tt-liveness` afterwards | `model_ready: true`, PASS |

## 6. Observations and open questions

These are not test failures. They are things seen along the way that should be looked at.

1. **Sync mp4 files stay on disk.** After both sync requests, both mp4 files remained in `cpp_server/videos/`. Sync creates no job record, so the job cleaner does not see them, and `syncFileResponse` in `src/api/video_controller.cpp` only sends the file without deleting it. From the code it looks like every sync request leaves a file on disk until the next start with purge enabled (`TT_VIDEO_PURGE_ON_START`). This is the same gap as for a client that disconnects in the middle of a sync generation, but it is now visible in the normal case too. What the Python server does in the same situation was not checked.
2. **`X-Generation-Time` is identical for both sync requests (`78.74`).** The requests differ (different seed and prompt), and the time is measured in the worker around `processJsonTask`. Earlier measurements of the same kind of job varied by tenths of a second (78.2 s, 79.1 s, 79.1 s by wall clock). An identical value to two decimals may be a coincidence, but it is not explained. Reading the time from several consecutive sync requests would settle it.
3. **Numbers in the log are the internal `task_id`, not the UUID.** Lines such as `skipping cancelled task 9` refer to the task, not to the job UUID. Linking them requires knowing the submission order.
4. **The `videos/` directory now holds 7 mp4 files**, about 10.5 MB from both sets of tests. Files of jobs A, C and F are removed when the job is deleted or when the cleaner removes it after the retention period. Sync files are not removed (point 1).

## 7. What is not covered

- **The video content was not looked at by the test.** Format, dimensions, frame count and repeatability were checked, but not whether the picture shows the described scene. The content was reviewed manually afterwards.
- **All jobs use 4 steps.** Quality and timing at the default of 20 steps were not measured (job E with 20 steps was cancelled before it ran).
- **The cancel/CLAIMED race** is not part of this test. It has its own test and report: [2026-11-14-video-cancel-race-real-server.md](2026-11-14-video-cancel-race-real-server.md) (10/10).
- **A client disconnecting in the middle of a sync generation** was not tried.
- **Stress and worker crash** (`kill -9`) were not tried on a real device.

## Repeated after the C++ runner-id change

The same test was repeated in the final configuration: the C++ side passes the device group (`0,1,2,3`) as the runner id, the Python `runner_utils.py` is unchanged from HEAD, and tracing is disabled for Wan2.2 in the Python runner (`"traced": False`, a local workaround that is not part of these commits; with tracing enabled the output on a T3K is noise).

Result: **45 of 45 checks passed**, none skipped (the checks that read the server log were included). The seed-7 request, for which the earlier `A.mp4` was recorded, gives a different file in this session; see the repeatability note in section 4.

## How to repeat

The server must run with a real runner and be `model_ready` (the start command is in [2026-11-14-video-cancel-real-server.md](2026-11-14-video-cancel-real-server.md), section 2). From `tt-media-server/cpp_server`:

```bash
# everything (about 7 minutes, 6 generations)
python3 tests/video/real_device/jobs/queue_download_sync.py --server-log /path/to/server.log

# only auth and validation (seconds, no device work)
python3 tests/video/real_device/jobs/queue_download_sync.py --phase validation
```

Without `--server-log` and `--payload-dir` the checks that read the log and the payload directory are skipped (SKIP), not failed.
