# Video service: cancel test on a real server

Date: 2026-11-14
Result: **21 of 21 checks passed.** Run on a T3K with the real Wan2.2 model, not the mock runner.
Script: [tests/video/real_device/cancel/cancel_queue_and_running.py](../../tests/video/real_device/cancel/cancel_queue_and_running.py)

## Environment

| | |
|---|---|
| Machine | T3K, 8 Wormhole chips |
| Server | `tt_media_server_cpp -p 8000`, `MODEL_SERVICE=video`, runner `tt-wan2.2` |
| Weight cache | a populated `tt_dit` cache for mesh 2x4 (see `TT_DIT_CACHE_DIR` below) |
| Jobs | `num_inference_steps=4` (instead of the default 20, to keep the run short) |
| Test duration | 242 s |

## 1. How cancel works

### Two channels, because there are two processes

- The **service** (Drogon, `VideoJobStore`) changes the job status. This is what the client sees.
- The **worker** is a separate process that owns the device. Cancel reaches it through a **marker file** `cancel-<taskId>` next to `request-<taskId>.json`.

### Why a file and not a message

- The task queue (shared memory) is one-way and cannot remove an element from the middle, so the service cannot take back a task it has already sent.
- A marker is **state**, not an event: it stays on disk until someone removes it. So it also works when the cancel arrives long before the worker takes the task.
- It reuses a mechanism that already exists (request and response are files on disk anyway).

### Why the worker checks twice

```
worker takes the task from the queue
   |
   +-- check 1: marker?  -- yes --> skip the task (A)
   |
   +-- send CLAIMED to the service
   |
   +-- check 2: marker?  -- yes --> skip the task (B)
   |
   +-- run the model on the device    (C: a cancel that arrives later; the result is discarded)
```

| When the cancel arrives | What happens |
|---|---|
| **A.** while the task waits in the queue | check 1 catches it |
| **B.** after check 1, before CLAIMED | check 2 catches it |
| **C.** after check 2 | the model runs to the end, the job is `cancelling`, then `cancelled`, the result is discarded |

Without check 2, case B would turn into case C and the device would be spent on a job the client already cancelled.

### What this gives

- The device is not used for jobs cancelled in the queue.
- A queued job becomes `cancelled` at once, without waiting for the worker.
- No new IPC protocol.
- Cleanup in one place (`cleanupTask`).

### What gets worse or stays open

- Generation on the device cannot be interrupted: a running job stays `cancelling` until it finishes.
- A window of a few microseconds remains after check 2.
- It depends on the file system: service and worker must share it (reasoning from the code, not tested).
- Read errors are swallowed: if `exists()` fails the task is treated as not cancelled; if the marker cannot be written, only a WARN is logged.
- The marker is deliberately not removed in `deleteJob`; if the worker dies without a result it stays until `onStopped`.

### CLAIMED and status

The worker sends `CLAIMED` as an ordinary message on the result queue. The service's consumer thread waits in `blockingPop`, so `onTaskClaimed` runs `jobs.markInProgress(...)` immediately. There is no separate event channel.

The Python server marks a video job `in_progress` in a similar way: the device worker sets `start_event` when it really starts the task, and the service polls it every 0.5 s, so `in_progress` can lag by up to half a second there. Python has no `cancel_event` for video jobs (it exists only for training jobs); see "Difference from Python" in section 5.

## 2. Setup: what had to be fixed before the test could run

The server did not start at first. Causes:

1. **Chips in a bad state** (`TT_FATAL: Read unexpected run_mailbox value`). Fix: `tt-smi -r`. The reset had to be repeated after every shutdown in the middle of initialization. See [2026-11-14-device-reset-problem.md](2026-11-14-device-reset-problem.md).
2. **`DEVICE_IDS="(0,1,2,3)"`.** The default `(0)` gives the worker `TT_VISIBLE_DEVICES=0`, and the Wan mesh then fails with `Fabric is being used but Device 1 is not active`.
3. **`MODEL=Wan2.2-T2V-A14B-Diffusers`.** Without it the Python settings pick a galaxy configuration with a 1x1 mesh instead of 2x4.
4. **`TT_DIT_CACHE_DIR`** pointing at an existing weight cache. Otherwise the server uses an empty `~/.cache/tt_dit_cache` and builds the weights from PyTorch, which is slow (log: `Cache does not exist. Loading PyTorch state dict.`).

Command used to start the server:

```bash
cd tt-media-server/cpp_server
source <tt-metal>/python_env/bin/activate
MODEL_SERVICE=video \
MODEL=Wan2.2-T2V-A14B-Diffusers \
MODEL_RUNNER=tt-wan2.2 \
DEVICE=t3k \
DEVICE_IDS="(0,1,2,3)" \
TT_DIT_CACHE_DIR=<path to the tt_dit weight cache> \
TT_METAL_HOME=<tt-metal> \
TT_PYTHON_PATH=<path to tt-media-server> \
HF_HUB_OFFLINE=1 \
OPENAI_API_KEY=<token> \
./build/tt_media_server_cpp -p 8000
```

## 3. Tests and results

### Prompts and jobs

All jobs were sent with `num_inference_steps=4`. The other fields (resolution, frame count, duration) are defaults.

| Job | ID | Prompt | Seed | Outcome | Video file in `cpp_server/videos/` |
|---|---|---|---|---|---|
| A | `c2114026-eb2b-4354-a95b-de32c802cc53` | `a red ball rolling on grass` | 1 | `completed` | `f298518e-1f13-40b3-a21b-13fe966ea83e.mp4` (3 113 780 B) |
| B | `9e3a2c6f-026d-4555-9bdc-eb8c00b162a5` | `a blue cube` | 2 | `cancelled` in the queue | none (task skipped, model not run) |
| D | `860f12f7-4da9-4670-8c24-88f893c4c0e8` | `a green forest at dawn` | 3 | `cancelled` during generation | none (result discarded, mp4 deleted) |
| E | `a127a12d-089f-4ee7-bdc7-263c03ec26d8` | `a yellow bird flying` | 4 | `completed` | `ea7eb2a9-054e-425a-8989-27479bdd88f0.mp4` (748 228 B) |

Only A and E produced a video, so only those two can be inspected. The file name does not match the job id (the file gets its own UUID); the pairs were matched by the size reported by `download` and by creation time. Video quality (whether it shows a red ball on grass and a yellow bird in flight) is not checked by the test, which only checks the mp4 header and size.

### Test 1: cancel a job waiting in the queue (job B)

Job A is generating, B waits in the queue, then B is cancelled.

| Check | Result |
|---|---|
| B right after the cancel | `cancelled` (HTTP 200), PASS |
| Marker in the payload directory | `cancel-2` next to `request-2.json`, PASS |
| Cancelling the same job again | 404, PASS |
| Server log | `Worker 0 skipping cancelled task 2 (before claim)`, PASS |
| B later via GET | `cancelled`, no `error` field, PASS |
| Files at the end | payload directory empty, marker removed, PASS |

The task was skipped without touching the device, and cleanup is complete.

### Test 2: cancel a job that is generating (job D)

The cancel arrives 5 s after generation started.

| Check | Result |
|---|---|
| Right after the cancel | `cancelling` (HTTP 200), PASS |
| Duration | the model ran for another 74 s, then `cancelled`, PASS |
| `download` for D | 409 (not available), PASS |
| Output directory | no mp4 of the cancelled job, PASS |
| Marker | none left, PASS |

Generation cannot be interrupted from outside: the job stays `cancelling` to the end and the result is discarded.

### Test 3: basic flow and device health (jobs A and E)

| Job | Outcome | Duration |
|---|---|---|
| A (before the cancel) | `completed`, mp4 3.1 MB (`ftyp` header), PASS | about 78 s |
| E (after cancelling D) | `completed`, mp4 0.7 MB, PASS | about 79 s |

E works normally, so cancelling in the middle of generation did not break the device or the worker.

### All 21 checks

1. A submit 202
2. A in_progress
3. B is queued while A runs
4. cancel B gives `cancelled` at once
5. marker `cancel-<id>` exists
6. second cancel of B is 404
7. A completed
8. A download is an mp4
9. worker skipped B (before claim) in the log
10. B cancelled without an error field
11. marker removed (`cleanupTask`)
12. no leftover request/response files
13. D in_progress
14. cancel D gives `cancelling`
15. D finished
16. D ends as `cancelled`
17. D download is not available
18. no mp4 of the cancelled job left in the output directory
19. no leftover marker
20. E completed after the cancel
21. E download is an mp4

## 4. What the test does not cover

- **The mp4 files were not inspected.** Only the header and size were checked. Playback and whether the picture is noise must be checked by eye (files in `cpp_server/videos/`). E is smaller than A (0.7 vs 3.1 MB), probably because of the prompt and seed, but this was not verified.
- **The race (cancel after the task was taken from the queue, before CLAIMED)** is not covered here. It has its own test and report: [2026-11-14-video-cancel-race-real-server.md](2026-11-14-video-cancel-race-real-server.md).
- **Several jobs in the queue** were not tried in this test, only one queued job. Covered in [2026-11-14-video-real-server-queue-download-sync.md](2026-11-14-video-real-server-queue-download-sync.md) (three queued jobs cancelled, one between them still runs).
- **Timings are not representative**: 4 steps instead of the default 20.

## 5. Observations

- During the first run the service logged nothing when a job was cancelled (neither for a queued job nor for a running one), so the log did not show which case occurred. An INFO line was therefore added to `VideoService::cancelJob` (`src/services/video_service.cpp`). **Verified** by repeating the same test with the new binary (see "Repeated with the new log line" below).
  - The line is logged only when the cancel succeeds; for an unknown or already finished job (404) there is none (not tested, follows from the code).
  - The line carries the internal `task_id` (a number), not the public job UUID, like the other worker lines (`skipping cancelled task 2`).
- `download` of a job that is not finished returns **409**, while the Python server returns 404. A known difference.
- **Difference from Python in cancelling a running job** (from reading the Python code; the Python server was not tried on a device for this case). In Python, `cancel_job` marks a running video job `cancelling` and cancels its asyncio task, and the cancellation handler marks it `cancelled` straight away, so `cancelling` is almost instantaneous and the status says `cancelled` while the device is still generating. There is no cancel event or marker for video jobs. The service does put the task id into the scheduler's `cancel_queue` when the waiting coroutine is cancelled, but the worker used for Wan (`device_worker`, the default; the dynamic-batch worker that reads `cancel_queue` is used only when `use_dynamic_batcher` is on, for LLMs) does not read it (the code says "unused here"). So a video request that is already in the worker queue stays there and the worker runs it, although the job is shown as `cancelled`. This is read from the code. The log of one Python run on the T3K is consistent with it (a 78 s generation started right after the previous job finished, when the only request waiting was the job that had been cancelled in the queue), but it was not a controlled test. In the C++ service the job stays `cancelling` until the generation really ends and only then becomes `cancelled`, and a job cancelled while queued is skipped by the worker without touching the device. The C++ behaviour is more honest about the state of the device, and it saves device time for jobs cancelled in the queue; the price is that a client waits longer to see `cancelled` for a running job.
- Leftovers after runs that did not shut down cleanly: a stale `/tmp/tt-video-ipc-<pid>` directory from the dead process, and possibly an empty `~/.cache/tt_dit_cache`. They are harmless.

## Repeated with the new log line

The same test (same checks, same script) was repeated after rebuilding the server with the cancel INFO line. A `tt-smi -r` was needed before it (the `run_mailbox` error appeared again on the first start, see [2026-11-14-device-reset-problem.md](2026-11-14-device-reset-problem.md)). 

Result: **21 of 21 checks passed**, as the first time. Duration 243 s.

The new server log line, for both cases:

| Case | Log line | Logged |
|---|---|---|
| Cancel of a queued job (B, task 2) | `[VideoService] Cancel requested for task 2: job is now cancelled (queued, worker will skip it)` | PASS |
| Cancel of a running job (D, task 3) | `[VideoService] Cancel requested for task 3: job is now cancelling (already running, result will be discarded)` | PASS |

The order of the log lines for a queued job shows the whole path of the cancel:

```
13:57:55  [VideoService] Cancel requested for task 2: job is now cancelled (queued, worker will skip it)
13:59:13  [MediaIpcRunner] Worker 0 skipping cancelled task 2 (before claim)
```

The first line is written by the service at the moment of the cancel, the second by the worker only when it gets to the task (after A finished), so the gap between the request and the skip is about 78 s, the generation time of A.

Prompts and jobs of the repeated test:

| Job | ID | Prompt | Seed | Outcome |
|---|---|---|---|---|
| A | `b7def045-f7c3-4185-95dd-a0cfa6dd62ad` | `a red ball rolling on grass` | 1 | `completed` |
| B | `f8082c79-4328-428c-9424-fe8f5afa6886` | `a blue cube` | 2 | `cancelled` in the queue |
| D | `7d72de49-f75c-47ef-8970-028de5512ee5` | `a green forest at dawn` | 3 | `cancelled` during generation |
| E | `ec7e984a-8422-43c0-aca1-75459661379d` | `a yellow bird flying` | 4 | `completed`, 748 228 B |

Job E had exactly the same file size (748 228 B) as in the first run with the same prompt and seed. This is only a size comparison, not a byte comparison.

## Repeated after the C++ runner-id change

The test was repeated twice after changing `video_device_runner.cpp` (the C++ side now passes the device group as the runner id instead of the worker index): once on a server whose Python code used the original `runner_utils.py` from HEAD (in a copy of the Python tree), once in the final configuration with the real tree. Both times **21 of 21 checks passed**. The video sizes (A 3 113 780 B, E 748 228 B) are the same as in the first run. Without that change a mesh worker had its device group overwritten with the worker index, and Wan2.2 could not open the mesh on a T3K (`Requested mesh shape [2, 4] requires 8 devices, but only 2 devices are available`).

## How to repeat

The server must run with a real runner and be `model_ready`. From `tt-media-server/cpp_server`:

```bash
# cancel in the queue and while running, basic flow (about 4 minutes, 4 generations)
python3 tests/video/real_device/cancel/cancel_queue_and_running.py --server-log /path/to/server.log \
    --output-dir /path/to/TT_VIDEO_OUTPUT_DIR
```

Without `--server-log`, `--payload-dir` and `--output-dir` the checks that use them are skipped (SKIP), not failed. The race test is separate: [2026-11-14-video-cancel-race-real-server.md](2026-11-14-video-cancel-race-real-server.md).
