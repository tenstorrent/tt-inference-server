# Mock tests of the video service (no device)

Tests that work without a device: contract cases, server tests and worker crash
tests that run against the server with the mock runner. The unit tests are C++ tests in
`tests/unit/api/`, and the tests on a real device are in
`tests/video/real_device/`.

## Files in this folder

| File | What it is |
|---|---|
| `contract_cases.py` | contract checks of the job API (async mode). Takes `--base-url`, so it runs against the mock-runner server and against a real server with a device |
| `server_cases.py` | checks against a real `tt_media_server_cpp` with the mock runner, by phase |
| `run_server_tests.sh` | starts the server with the mock runner once per phase (private shm names) and runs `contract_cases.py` and `server_cases.py` |
| `worker_crash_cases.py` | starts its own server with the mock runner, kills the worker with `SIGKILL` and checks the failed job, the restart, the restart limit and that a worker whose warmup fails exits |

`contract_cases.py` checks the job API (async mode; sync mode is covered by
`server_cases.py`): `POST /v1/videos/generations` returns 202 and a job id,
polling until the job is `completed`, the job list, `download`, `cancel`,
`DELETE`, an unknown job 404 and body validation. `tests/video/run_device_tests.sh`
uses the same checks against a real server with a device.

## Server tests (contract, 401, 409, cancel, 429)

`./tests/video/mock/run_server_tests.sh` starts a real `tt_media_server_cpp` with the
mock runner (once per phase). The `contract` phase runs `contract_cases.py` with
`--require-file`. The other phases run `server_cases.py`: 401, 409 on `DELETE` of
an active job, cancel of a queued and a running job, sync mode (mp4), 429 when
the queue is full, 503 at `MAX_JOBS` and during warmup, and the cancel/CLAIMED
race. No device is needed. The script needs a built `./build/tt_media_server_cpp`.

Phases of `server_cases.py`: `main`, `sync-error`, `limits`, `warmup`,
`cancel-race`.

## Worker crash tests

```bash
python3 tests/video/mock/worker_crash_cases.py
```

Starts its own server (ports 18150 and 18151, private shm names, a temporary
output folder) and kills the worker with `SIGKILL`: the job the worker had taken
becomes `failed` (`processing_error`), a new worker starts, the job waiting in the
queue is served, and with `TT_VIDEO_WORKER_MAX_RESTARTS=1` the second crash makes the
server exit with code 1. Case C starts the real device runner with a wrong `TT_PYTHON_PATH` (no chip is opened) and checks that the failed worker exits, so the server exits with code 1; it guards the fix of a GIL deadlock on that path. Needs a built `./build/tt_media_server_cpp`.

## Cancel in the queue (manually, no device)

The contract tests do not cover `cancel` of a job waiting in the queue (the
server tests above do). A manual test with a slowed-down mock runner
(`TT_VIDEO_MOCK_DELAY_MS`) and its own shm queue names, so that it does not
disturb another server on the same machine:

```bash
MODEL_SERVICE=video MODEL_RUNNER_TYPE=video_mock MODEL_RUNNER=tt-wan2.2 \
OPENAI_API_KEY=test TT_VIDEO_MOCK_DELAY_MS=4000 \
TT_MEDIA_TASK_QUEUE=cxl_tasks TT_MEDIA_RESULT_QUEUE=cxl_results \
TT_WARMUP_SIGNALS_QUEUE=cxl_warm TT_WORKER_METRICS_SHM=/cxl_metrics \
./build/tt_media_server_cpp -p 18099 &
# three POSTs, then cancel 2 and 3 (queued) and 1 (running) at once
```

Expected: the log shows `MockVideoRunner task 1` and twice `skipping cancelled
task`; all three jobs end as `cancelled`, without `error`.
