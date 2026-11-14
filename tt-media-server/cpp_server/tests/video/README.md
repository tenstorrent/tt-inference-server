# Video service tests

The tests are split into folders by what they need to run.

| Folder | What it needs | What is in it |
|---|---|---|
| `mock/` | only a build of the server, no device | a local HTTP stub, contract cases and server tests that use the mock runner. Guide: `mock/README.md` |
| `real_device/` | a real server with a device and the model loaded | tests of cancel, the job queue, download and sync mode, grouped by topic. Guide: `real_device/README.md` |
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

```bash
# from tt-media-server/cpp_server
python3 tests/video/real_device/cancel/cancel_queue_and_running.py --server-log /path/to/server.log
python3 tests/video/real_device/cancel/cancel_race.py --server-log /path/to/server.log
python3 tests/video/real_device/jobs/queue_download_sync.py --server-log /path/to/server.log
python3 tests/video/real_device/jobs/queue_download_sync.py --phase validation
```
