# Test reports

Reports of the tests of the video service in `cpp_server`.

Rules:
- One report per test, named `YYYY-MM-DD-<topic>.md`.
- Raw outputs of the runs are kept locally (`*.log` is ignored by git) and are not committed.
- A report contains the environment, the exact command, the results per check, what is not covered, and observations.

## Reports

| Date | Report | Result |
|---|---|---|
| 2026-11-14 | [Cancel on a real server (T3K, Wan2.2)](2026-11-14-video-cancel-real-server.md) | 21/21 (repeated with the new log line and after the C++ runner-id change: 21/21) |
| 2026-11-14 | [Cancel/CLAIMED race on a real runner](2026-11-14-video-cancel-race-real-server.md) | 10/10 (repeated after the C++ runner-id change: 10/10) |
| 2026-11-14 | [Job queue, download/Range, repeatability, sync mode (T3K, Wan2.2)](2026-11-14-video-real-server-queue-download-sync.md) | 45/45 (repeated after the C++ change: 45/45) |
| 2026-11-14 | [Curl examples and worker crash on a real device (T3K, Wan2.2)](2026-11-14-video-curl-examples-and-worker-crash-real-device.md) | curl examples: all pass; worker crash: job fails correctly; hang on a failed warmup found and fixed, repeated on the device |
| 2026-11-14 | [Extended checks: fields, content, load, restart, limits, Python comparison (T3K and mock)](2026-11-14-video-real-device-extended-checks.md) | mock 53/53 + 15/15 + 58/58 + 36/36, unit 55/55; device 35/36 (wrong expectation) + 9/9 + 27/27 + 8/8; a segfault on worker exit fixed; 4 open findings |
| 2026-11-16 | [Worker killed before CLAIMED on a real device (T3K, Wan2.2)](2026-11-16-video-preclaim-worker-death-real-device.md) | finding reproduced: the job stays `queued` for 120 s, the sweep does not catch it |
| 2026-11-16 | [Output directory check on a real device (T3K, Wan2.2)](2026-11-16-video-output-dir-check-real-device.md) | all checks pass: 503 before any device time is spent, recovery without a restart |
| 2026-11-16 | [Video service benchmark and telemetry (T3K, Wan2.2)](2026-11-16-video-benchmark-real-device.md) | benchmark, not a validation: server cost about 0.3 ms per video against 79 to 725 s of device time; 17.95 s per step plus 7 s fixed; one worker gives 0.76 videos per minute; `GET /jobs` grows linearly (95 ms at 2300 jobs); one device hang seen once, not reproduced |
| 2026-11-16 | [Video benchmark: C++ server against the Python server (T3K, Wan2.2)](2026-11-16-video-benchmark-cpp-vs-python.md) | same device time on both (within 0.6 s); C++ control plane faster (`POST` 0.7 ms vs 3.5 to 6 ms, download 1 to 4 ms vs 48 to 65 ms); C++ ready about 30 s sooner (101 s vs 131 s); Python shows stage metrics (denoise, VAE) that C++ lacks, C++ shows queue/server metrics that Python lacks; Python's video durations include queue wait |

## Problem analyses

| Date | Document |
|---|---|
| 2026-11-14 | [The server cannot open the device without `tt-smi -r` (`run_mailbox` error)](2026-11-14-device-reset-problem.md) |
