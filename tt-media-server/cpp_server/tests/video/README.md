# Video service tests

The tests are split into folders by what they need to run.

| Folder | What it needs | What is in it |
|---|---|---|
| `mock/` | only a build of the server, no device | a local HTTP stub, contract cases, server tests, worker crash and lifecycle tests that use the mock runner. Guide: `mock/README.md` |
| `real_device/` | a real server with a device and the model loaded | tests of cancel, the job queue, download and sync mode, grouped by topic. Guide: `real_device/README.md` |
| `python/` | Python with numpy and pytest, no device | checks of the `python/cpp_dit_adapters` package (fake `tt_dit` pipelines), and of the runner list and mesh table against the Python server. The C++ runner layer is tested by `build/dit_runner_test`. Run: `python -m pytest -q tests/video/python` |
| `benchmark/` | mock runner (server overhead) or a real server with a device | timings, not validation: sequential and burst video generation, control-plane latency, jobs per second. Guide: `benchmark/README.md` |
| (this folder) | | `run_device_tests.sh` (the contract checks against a real server with a device, uses `mock/contract_cases.py`) and `UNIT_TESTS.md` |

The unit tests (job store, request parser, JSON field order, modes) are C++
tests in `tests/unit/api/`, described in `UNIT_TESTS.md`. Reports of the runs on
a device are in `docs/tests/`.

## Contract test against a real server

From `tt-media-server/cpp_server`, when the server and the device are running:

```bash
./tests/video/run_device_tests.sh
```

This step waits for a job to become `completed` (`VIDEO_JOB_TIMEOUT`, default
1800 s) and requires `/download` to return an mp4.

## Tests on a real server and device (`real_device/`)

Unlike `mock/server_cases.py` (mock runner), these tests send requests to an
already running `tt_media_server_cpp` with a real runner (for example Wan2.2 on
a T3K). They are split into folders by topic, and each test has its own report in
`docs/tests/`. An overview of the folders, the parameters and how to add a new
test is in `real_device/README.md`.

| Folder | Test | What it checks | Report |
|---|---|---|---|
| `real_device/cancel/` | `cancel_queue_and_running.py` | cancel of a queued job (the worker skips it), cancel of a running job (`cancelling`, then `cancelled`), the device is fine afterwards | `2026-11-14-video-cancel-real-server.md` |
| `real_device/cancel/` | `cancel_race.py` | cancel after the task was taken from the queue and before `CLAIMED`: the second marker check must stop the task. The server needs `TT_MEDIA_TEST_PRECLAIM_DELAY_MS=1500` | `2026-11-14-video-cancel-race-real-server.md` |
| `real_device/jobs/` | `queue_download_sync.py` | auth and validation (phase `validation`, no generation), several jobs in the queue with cancel, `DELETE` 409, `download` with `Range`, faststart, `ffprobe`, the same seed gives the same file, two simultaneous sync requests | `2026-11-14-video-real-server-queue-download-sync.md` |
| `real_device/jobs/` | `fields_content_load.py` | request fields and limits (422), a Unicode prompt of 20000 characters, the mp4 (codec, size, faststart), `Range` and a resumed download, latency while the device generates, a 50-step job, memory | `2026-11-14-video-real-device-extended-checks.md` |
| `real_device/jobs/` | `disconnect_and_queue_depth.py` | a dropped synchronous request, 30 queued jobs cancelled at once | `2026-11-14-video-real-device-extended-checks.md` |
| `real_device/jobs/` | `api_parity.py` | the same 72 probes to the C++ and to the Python server; prints the answers, asserts nothing | `2026-11-14-video-real-device-extended-checks.md` |
| `real_device/lifecycle/` | `restart_cases.py` | starts and stops the server itself: purge on start, `PURGE_ON_START=0`, SIGTERM idle and in the middle of a job, seed repeatability across a restart, an unwritable output folder | `2026-11-14-video-real-device-extended-checks.md` |
| `real_device/lifecycle/` | `warmup_and_limits.py` | starts the server itself: `503` + `Retry-After` during the warmup, `429` at `MAX_QUEUE_SIZE`, `503` at `MAX_JOBS` | `2026-11-14-video-real-device-extended-checks.md` |

```bash
# from tt-media-server/cpp_server
python3 tests/video/real_device/cancel/cancel_queue_and_running.py --server-log /path/to/server.log
python3 tests/video/real_device/cancel/cancel_race.py --server-log /path/to/server.log
python3 tests/video/real_device/jobs/queue_download_sync.py --server-log /path/to/server.log
python3 tests/video/real_device/jobs/queue_download_sync.py --phase validation
python3 tests/video/real_device/jobs/fields_content_load.py --server-log /path/to/server.log
python3 tests/video/real_device/jobs/disconnect_and_queue_depth.py --server-log /path/to/server.log
# these two start and stop the server themselves (nothing else may use the chips);
# on a T3K run them with TT_VIDEO_TRACED=0 in the environment
TT_VIDEO_TRACED=0 python3 tests/video/real_device/lifecycle/restart_cases.py --tt-metal <tt-metal> --dit-cache <cache>
TT_VIDEO_TRACED=0 python3 tests/video/real_device/lifecycle/warmup_and_limits.py --tt-metal <tt-metal> --dit-cache <cache>
```
