# Video service: refuse work when the output directory cannot be written (real device)

Date: 2026-11-16. Raw output: `2026-11-16-video-output-dir-check-real-device.log` (not committed).

Result: **all checks pass.** Fixes finding 4 of the
[extended checks report](2026-11-14-video-real-device-extended-checks.md): a job used to fail only after the
whole generation ("Broken pipe"), so the device time was lost.

## Change

- `tt::utils::checkDirWritable` (`include/utils/video_output_dir.hpp`): creates the directory if it is
  missing, writes one byte to a probe file and removes it; returns a short reason or an empty string.
- `VideoService` logs an error at start when the directory is not usable (the server still starts) and
  exposes `outputDirError()`. Only the real device runner checks; the mock runner writes no files.
- `VideoController::generate` calls it for every `POST /v1/videos/generations`, async and sync, and answers
  `503` (`service_unavailable`) with the reason before a task is created.
- Unit tests `VideoOutputDir.CheckDirWritable*` (missing directory is created and no probe is left, a path
  that is a file, a read-only directory); `video_generate_request_test` 60/60.

## Test on a T3K

Wan2.2-T2V-A14B-Diffusers, `num_inference_steps=4`, `TT_VIDEO_OUTPUT_DIR` set to a folder with mode 555
(chips reset before the start, server ready after about 105 s).

| Check | Result |
|---|---|
| the startup log has `Video output directory is not usable: cannot write in ...: Permission denied` | PASS |
| async `POST` on the read-only folder: `503`, message names the directory | PASS |
| `?async=false` `POST` on the read-only folder: `503` | PASS |
| no job record was created (`GET /v1/videos/jobs` is `[]`) and no probe file is left in the folder | PASS |
| after `chmod 755` (server not restarted) a `POST` gives `202` and the job completes (about 80 s), the mp4 is in the folder | PASS |
| `chmod 555` again while the server runs: the next `POST` is `503` again | PASS |
| the server stopped by SIGTERM when idle | PASS |

## Not covered

- A disk that fills up in the middle of a generation (the check happens before the run) and a folder whose
  permissions change during a run.
- Phase 4 of `tests/video/real_device/lifecycle/restart_cases.py` was rewritten for the new behaviour
  (503, then recovery after the fix) but that script was not run as a whole; the same checks were run by hand
  as listed above.
- The check runs on every request (create, write, delete of one file). Its cost was not measured; it is a
  few syscalls compared with a generation of 80 s or more.
