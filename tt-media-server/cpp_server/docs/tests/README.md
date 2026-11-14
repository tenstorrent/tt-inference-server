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

## Problem analyses

| Date | Document |
|---|---|
| 2026-11-14 | [The server cannot open the device without `tt-smi -r` (`run_mailbox` error)](2026-11-14-device-reset-problem.md) |
