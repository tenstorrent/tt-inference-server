# Curl examples and worker crash on a real device (T3K, Wan2.2)

Date: 2026-11-14. Raw output: `2026-11-14-video-curl-examples-and-worker-crash-real-device.log` (not committed).

## Environment

- T3K, Wan2.2-T2V-A14B-Diffusers, `MODEL_RUNNER=tt-wan2.2`, `DEVICE_IDS="(0,1,2,3)"`, tracing disabled in the Python runner (see the README of `tests/video/real_device`).
- `cpp_server` on branch `cpp-video-service`, server on port 8000, 4 inference steps unless stated.

## 1. The curl examples of `docs/VIDEO_SERVICE.md`

Every block of the section "Trying it with curl" was extracted from the markdown and run as written (only `num_inference_steps` 20 changed to 4, and the output path of `-o`), against the real server.

| Command | Result |
|---|---|
| `GET /tt-liveness` | `model_ready: true` |
| create (with `jq`, without `jq`, all optional fields) | `202`, job object with an id |
| job status, wait loop | `queued`, `in_progress`, `completed` |
| list jobs | `200` |
| download | `200`, mp4 h264 832x480, 81 frames |
| download with `Range: bytes=0-1023` | `206`, 1024 bytes |
| download with a range past the end of the file | `416` |
| cancel of a finished job | `404` (as documented) |
| delete | `200`, `deleted: true` |
| synchronous `?async=false` | `200`, mp4 832x480, 81 frames, 633 KB, after 13 min 25 s (it waited in the queue behind two jobs, then ran 20 steps) |

All passed. All optional fields (`height`, `width`, `num_frames`, `fps`, `seed`, `negative_prompt`) were accepted; the Wan2.2 runner ignores shape fields and always produces 832x480, 81 frames.

## 2. Input that the runner could reject

The Wan2.2 runner reads no shape field, and the API and the Python request model validate the same ranges, so there is no value of `height`, `width`, `num_frames`, `fps`, `duration` or `aspect_ratio` that produces a `failed` job. Tried on the device:

| Input | Result |
|---|---|
| `seed: -1` | `202`, `completed` |
| `seed: 9223372036854775807` and `-9223372036854775808` | `422` from the API (`LargestInt out of Int range`) |
| `negative_prompt` of 200 KB | `202`, `completed` |
| `aspect_ratio: "not-a-ratio"` (earlier run) | accepted, video generated |

A `failed` job on a device can therefore only come from a runtime failure, tested below.

## 3. Worker crash on the device

Steps: two jobs (the first running for 15 s, the second queued), `kill -9` of the worker process.

The rows marked FINDING are not failed test checks: they are behaviours that turned out to be wrong or limited, which is what this test was meant to find. PASS means the service behaved as intended.

| Check | Result |
|---|---|
| job 1 becomes `failed`, `processing_error`, "Worker process 0 exited while processing request" | PASS, after 1 s |
| the service starts a new worker | PASS ("Restarting crashed worker 0") |
| the new worker opens the device | **FINDING** (the chips need a reset): `ETH training timed out after 900000 ms, on eth core 23, 16` (device 3) |
| the failed worker process exits | **FINDING, fixed** (see section 4): the process stayed alive (state `Sl`) for more than 4 minutes after "Worker 0 exiting after fatal error" |
| `/tt-liveness` reflects it | **FINDING** (open): `model_ready: true`, `is_ready: true` for the new pid the whole time, `queue_size: 1` |
| the queued job is served | **FINDING**: it stayed `queued` (a consequence of the rows above) |

After that the server was stopped, `tt-smi -r` was run and the server restarted. A job then completed normally (832x480, 81 frames), so the chips and the server are fine after a reset.

### Cause of the hang on exit

`gdb` on the hung worker: the thread that stops the runner (`VideoDeviceRunner::stop`) waits in `gil_scoped_acquire`, and the main thread of the worker is blocked in the join of the shutdown thread while it still holds the Python GIL.

In `VideoDeviceRunner::warmup()` the interpreter is created with `py::initialize_interpreter()`, which leaves the GIL held by the calling thread. The success path releases it with `PyEval_SaveThread()`. The two failure paths (`return false` in the `catch` blocks) do not. When warmup fails, the GIL stays held, `stop()` cannot take it, the process never exits, and the service manager never sees the worker die, so it neither restarts it nor shuts the server down.

This only shows up when warmup fails in a worker that created the interpreter, which on a device happens after a crash that leaves the chips in a bad state. The mock runner does not use `VideoDeviceRunner`, so the mock crash tests do not cover it. The same pattern (`PyEval_SaveThread()` only on success) exists in `sdxl_runner.cpp`, `embedding_runner.cpp` and `qwen3_tts_runner.cpp` and was not checked.

## 4. The fix and the repeated test

`VideoDeviceRunner::warmup()` now releases the GIL (`PyEval_SaveThread()`) on the failure path as well, not only on success.

- **Reproduction without a device:** a wrong `TT_PYTHON_PATH` makes the import of the Python runner fail in warmup, the same path as the device failure. Before the fix the server was still running after 30 s (worker hung, state `Sl`); after the fix the server exits with code 1 after about 1 s.
- **Regression test:** `tests/video/mock/worker_crash_cases.py`, case C. It fails on the old code (the server is still running after 30 s) and passes with the fix. The whole file: 14/14.
- **Repeated on the T3K** (same steps as in section 3, fixed binary): job 1 becomes `failed`/`processing_error` after 1 s; the restarted workers cannot open the chips (`Timed out waiting for ETH heartbeat`), each one now exits after its failed warmup, the service restarts it, and after 3 restarts within the hour it logs `giving up` and the server exits with code 1, about 23 s after the kill. So the restart limit works on the device too. The server needs a supervisor, and the chips need `tt-smi -r` before the next start; after the reset a job completed normally.

A second defect was found later: the worker then exited with a segmentation fault (`stop()` used null Python handles).
It is fixed too; see [the extended checks](2026-11-14-video-real-device-extended-checks.md), section 8.

### Not tested

- A crash where the chips recover by themselves (here they did not).
- The same `PyEval_SaveThread()` pattern in `embedding_runner.cpp` and `qwen3_tts_runner.cpp` was not checked; `sdxl_runner.cpp` already releases the GIL after its `catch`.

## Observations

- After the crash the reset of the chips is required (see `2026-11-14-device-reset-problem.md`); a restart of the worker alone does not help.
- `model_ready` stays true while a restarted worker is still warming up, so clients get `202` for jobs that no worker can take yet.
