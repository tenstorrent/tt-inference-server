# Video service: worker killed after taking a task and before CLAIMED (real device)

Date: 2026-11-16. Raw output: `2026-11-16-video-preclaim-worker-death-real-device.log` (not committed).

Result: **the finding reproduces on the device.** A task taken by a worker that dies before it reports
`CLAIMED` stays `queued` for good; the stuck-job sweep does not catch it.

## Environment

T3K, Wan2.2-T2V-A14B-Diffusers, runner `tt-wan2.2`, `DEVICE_IDS="(0,1,2,3)"`, tt_dit weight cache,
branch `cpp-video-service`. Extra settings: `TT_MEDIA_TEST_PRECLAIM_DELAY_MS=30000` (the worker waits 30 s
between taking the task and sending `CLAIMED`), `JOB_MAX_STUCK_TIME_SECONDS=20`,
`JOB_CLEANUP_INTERVAL_SECONDS=2`. The mock version of this case is `preclaim` in
`tests/video/mock/lifecycle_cases.py` (4/4, same outcome).

## Steps and observations

1. Server ready after about 110 s.
2. One job (`num_inference_steps=4`, seed 7) was created; 6 s later the worker was killed with `kill -9`
   (inside the 30 s window, so it held the task and had not sent `CLAIMED`).
3. The service logged `Restarting crashed worker 0` and started a new worker within a second.
4. For the next 120 s (sampled every 5 s) the job stayed `queued` with no error, although the stuck limit was
   20 s. `/tt-liveness` stayed 200 throughout.
5. The server was stopped by SIGTERM at the end (log: `graceful`). The restarted worker was still loading
   and had not finished; the run does not show whether it would ever pick the task up.

## What this shows

- A job whose task was taken but never claimed is neither `failed` nor processed: the client waits without
  an answer. The sweep only looks at running jobs, so it does not see it.
- Compare a crash after `CLAIMED`: the job fails after 1 s (see the worker crash report).

## Not covered

- Whether a restarted worker that comes up healthy would re-read the lost task. After `kill -9` the chips
  need `tt-smi -r`, so a healthy restart cannot be reproduced on this device; the mock run shows that a new
  job completes but the lost one stays `queued`.
- The fix is not made. Options: let the sweep also fail `queued` jobs whose task was taken, or let the
  service re-queue or fail the job when it restarts the worker.

## Afterwards

The chips need `tt-smi -r` before the next start.
