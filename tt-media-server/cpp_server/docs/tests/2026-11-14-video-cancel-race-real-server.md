# Video service: cancel/CLAIMED race on a real runner

Date: 2026-11-14
Result: **10 of 10 checks passed.** The test ran three times with the same outcome (task 4, task 6 and task 1, the last one after a later change, see below). This report describes the second run, from [tests/video/real_device/cancel/cancel_race.py](../../tests/video/real_device/cancel/cancel_race.py).
Related: [cancel in the queue and while running](2026-11-14-video-cancel-real-server.md), which explains how cancel works.

## What is tested

The service and the worker are two processes, so there is a window in which they disagree:

1. The worker takes the task from the queue and makes the **first check** for the cancel marker: it is not there.
2. The worker sends the service a `CLAIMED` message.
3. The worker makes the **second check** for the marker, and only then runs the model.

The service considers a job `queued` until it has processed `CLAIMED`. If a cancel arrives between steps 1 and 2, the service still sees `queued`, immediately marks the job `cancelled` and writes the marker. The first check has already passed and cannot catch it, so the second check must. If it did not, the model would run on the device for a job the client already received as cancelled (about 80 s of device time).

The test verifies that the second check really stops the task, that the model is not run, and that the device is immediately free for the next job.

## Setup: test hook

In production the window is microseconds wide and cannot be hit. The server therefore has to be started with the test hook **`TT_MEDIA_TEST_PRECLAIM_DELAY_MS=1500`**: after taking a task the worker waits 1.5 s before sending `CLAIMED`. During that time the service still sees the job as `queued`.

The rest of the command is the same as in the [cancel report](2026-11-14-video-cancel-real-server.md), section 2. Example:

```bash
cd tt-media-server/cpp_server
source <tt-metal>/python_env/bin/activate
MODEL_SERVICE=video MODEL=Wan2.2-T2V-A14B-Diffusers MODEL_RUNNER=tt-wan2.2 \
DEVICE=t3k DEVICE_IDS="(0,1,2,3)" \
TT_MEDIA_TEST_PRECLAIM_DELAY_MS=1500 \
TT_DIT_CACHE_DIR=<path to the tt_dit weight cache> \
TT_METAL_HOME=<tt-metal> \
TT_PYTHON_PATH=<path to tt-media-server> \
HF_HUB_OFFLINE=1 OPENAI_API_KEY=<token> \
./build/tt_media_server_cpp -p 8000
```

The hook is not for production: every job would wait 1.5 s. After the test the server should be restarted without that variable. Because the server was restarted for this test, a `tt-smi -r` was needed before starting (see the [device reset analysis](2026-11-14-device-reset-problem.md)).

## Jobs

| Job | ID | Prompt | Seed | Steps | Outcome |
|---|---|---|---|---|---|
| race | `997eaeea-9767-42b8-8d80-3e639c7a07be` (task 6) | `race` | 5 | 4 | `cancelled`, model not run |
| after race | `4b751719-f95b-4a3a-a199-7a65a2e807a4` | `after race` | 6 | 4 | `completed` |

The `race` job has no video: its task was skipped. The `after race` job only shows that the device is immediately free; its mp4 was not inspected.

## Timeline and result

| Time | Who | What |
|---|---|---|
| 0.0 s | client | sends job `race` |
| 0.0 s | worker | takes the task, first marker check: not there |
| 0.5 s | test | job is still `queued` (confirms the hook is active) |
| 0.5 s | client | cancel: job is `cancelled` at once, marker written |
| 1.5 s | worker | sends `CLAIMED`, the service does not change `cancelled` |
| 1.5 s | worker | second marker check: finds it, skips the task |
| 3.5 s | test | job is still `cancelled`, no `error` field |
| 3.5 s | test | sends `after race` |
| 83.7 s | test | `after race` is `completed` |

The same events from the server log:

```
14:46:53.816  [VideoService] Cancel requested for task 6: job is now cancelled (queued, worker will skip it)
14:46:54.814  [MediaIpcRunner] Worker 0 skipping cancelled task 6 (after claim)
```

| Check | Result |
|---|---|
| Submit | 202, PASS |
| 0.5 s after submit the job is `queued` (the worker is still waiting before `CLAIMED`) | PASS |
| Cancel | `cancelled` (HTTP 200), PASS |
| Job stays `cancelled` | PASS |
| After 3 s: `cancelled` without an `error` field | PASS |
| Worker log: `skipping cancelled task 6 (after claim)`, exactly one line | PASS |
| No `VideoIpcRunner` or `VideoDevice` lines for task 6 (the model was not run) | PASS |
| `after race` is `completed` | PASS |
| `after race` took 80 s, under 120 s (had the race job run, it would take about 160 s) | PASS |
| Payload directory empty | PASS |

The text `(after claim)` in the worker line shows that the second check, not the first, stopped the task. The first check logs `(before claim)`.

## What the test does not cover

- **The window after the second check.** A cancel that arrives when the worker has passed both checks and the model is about to start cannot be hit deliberately without another hook. In that case the behaviour is the same as cancelling a running job: the model runs to the end, the job is `cancelling`, then `cancelled`, and the result is discarded. The window is estimated at a few microseconds.
- **Order of operations in `cancelJob`.** The service first changes the status in the store and only then writes the marker. This may widen window C slightly, but the test does not measure it.
- **The check that the model did not run relies on the log.** If the worker logged differently, the check would pass because of missing lines. The duration of the next job is an additional confirmation.
- **One job per run.** There are no simultaneous cancels of several jobs in this window.

## Repeated after the C++ runner-id change

The test was repeated in the final configuration: the C++ side passes the device group (`0,1,2,3`) as the runner id, the Python `runner_utils.py` is unchanged from HEAD, `"traced": False` is a local change, and the server was started with `TT_MEDIA_TEST_PRECLAIM_DELAY_MS=1500` (with `tt-smi -r` before the start). Result: **10 of 10 checks passed**, the same as before the change.

```
17:20:47.454  [VideoService] Cancel requested for task 1: job is now cancelled (queued, worker will skip it)
17:20:48.452  [MediaIpcRunner] Worker 0 skipping cancelled task 1 (after claim)
```

The second marker check stopped it (`after claim`), the model was not run for that task, and the next job finished in the expected time. The test has now passed three times (task 4, task 6 and this one, task 1), each time 10/10.

## How to repeat

The server must run with a real runner, with the hook from "Setup", and be `model_ready`. From `tt-media-server/cpp_server`:

```bash
python3 tests/video/real_device/cancel/cancel_race.py --server-log /path/to/server.log
```

`--server-log` is required, because the evidence is in the worker log. If the server was not started with the hook, the test notices this at the first check (the job is not `queued` after 0.5 s), cancels the job and stops, without spending device time.
