# MiniMax-H3 test system (tt-media-server)

Everything the quad deployments broke on in Aug/Sep 2026 -- slow warmup, fl2va rejected at admission,
"9 s produces no audio", ref2va OOMs, silently dropped `duration_seconds`, rung-switch crashes -- is
now a test somebody can re-run. Tests are layered so the cheap ones run everywhere and the expensive
ones run on a quad inside tmux.

| tier | what | needs | time | run |
|---|---|---|---|---|
| 0 | admission/policy contract, request mapping, side-file wire format, media-check heuristics | nothing (ffmpeg for the media tests) | 25 s | `pytest tests/test_minimax_h3_*.py tests/test_h3_*.py tests/test_sp_runner.py tests/test_video_runner.py` |
| 1 | deployment health: warmup envs, time-to-ready / first / warm budgets, failed-job recovery | a deployment + `h3ctl.sh` | 15 min | `bash tests/run_live_test.sh -m h3_tier1` |
| 2 | shapes: every duration 4-15 s and aspect ratio, rung orders, audio/video content, fl2va keyframes | same | 45 min per task | `bash tests/run_live_test.sh tests/test_minimax_h3_live_shapes.py -k T2va` |
| 3 | trace residency (6 rungs resident), ref2va reference-count matrix, second-request OOMs | same | hours | `bash tests/run_live_test.sh -m h3_tier3`, `... tests/test_minimax_h3_live.py -k Ref2va` |

`run_live_test.sh` wires `H3_LIVE_*` from `H3_DEPLOY_DIR` (default: the `h3-deploy` next to the
checkout or `~/h3-deploy`), refuses to start while another live run or a generation is in flight,
and leaves a log + JSON report under `$H3_DEPLOY_DIR/live-test/`. Every live module is opt-in
(`H3_LIVE_URL`), starts fresh workers per case when it can (`h3ctl.sh start <task>`), resets the
chips (`tt-smi -glx_reset_auto`) after a failed or hung job, and never leaves the mesh wedged.

## Modules

| file | tier | contents |
|---|---|---|
| `h3_live_common.py` | – | shared machinery: config, HTTP, `Deployment` (fresh/reset/poison), `Assets`, `Combo`, `Report`, DELETE / cancel / routing contract checks |
| `h3_live_sequences.py` | – | ordered request sequences on one process; `Spec` -> body, `expected_canvas`/`expected_rung`, per-output judging, worker-log facts (rung, capture, JIT) |
| `h3_media_checks.py` | – | content validity: audio loudness/clipping, garbage-frame detection, duration/canvas echo |
| `test_h3_media_checks.py`, `test_h3_live_sequences_pure.py` | 0 | the two helpers on synthetic clips / known numbers |
| `test_minimax_h3_policy.py`, `test_minimax_h3_video_request.py`, `test_minimax_h3_fl2va_request.py` | 0 | published policy constants, pydantic admission (H3 runner process) |
| `test_minimax_h3_admission_gaps.py` | 0 | the SAME admission in the SP API process -- xfail(strict) until the gate keys on `MODEL` too |
| `test_sp_runner.py`, `test_video_runner.py` (H3 parts) | 0 | side-file wire format carries `duration_seconds`/`aspect_ratio`/keyframes/references; worker request mapping |
| `test_minimax_h3_live_deploy.py` | 1 / 3 | `h3ctl.sh check`, warmup envs in `/proc/<pid>/environ`, budgets with cold-JIT detection, API 422 contract (xfail), failed job then success, all six rungs resident |
| `test_minimax_h3_live_shapes.py` | 2 / 3 | t2va ladder 4-15 s (+ second pass), aspect echo, audio-replay repro, switch-back, determinism; fl2va one-keyframe ladder/aspects, two keyframes per canvas; tier 3 full duration x aspect matrix per task |
| `test_minimax_h3_live.py` | 2 / 3 | fl2va keyframe combos, ref2va reference matrix (#5044 table), cancel, failed-job deletable, DELETE contract, routing 422s |

## Issue -> test

| issue (where seen) | test | state |
|---|---|---|
| slow warmup; fixed by `TT_METAL_SHM_TRACKING_DISABLED=1` + `TT_METAL_LOGS_PATH` (first request 173 s -> 42 s) | `live_deploy::TestDeploymentKnobs::test_worker_env_has_the_warmup_knobs`, `TestBudgets::test_first_and_warm_request` (budget 120 s warm cache / 600 s after a rebuild, detected via `BuildKernels` lines), `test_time_to_ready` | strict |
| fl2va first+last keyframes at a 1008-row canvas: rejected (`prompt tokens 2053 > 2048`) with a >= ~33-token prompt, or ADMITTED with a shorter prompt and the audio comes back as noise (text padded to 3072 rows overruns the 2048-row prompt arena into the audio rows) | `live_shapes::TestFl2vaShapes::test_first_and_last_keyframes[21:9/16:9/9:16]` | xfail(strict) until the cap/resize fix lands; `[4:3/1:1/3:4]` strict (two keyframes fit under the cap) |
| fl2va one keyframe must work at every duration / canvas | `test_one_keyframe_duration_ladder`, `test_one_keyframe_aspect_ratios` | strict |
| "9 s produces no audio" / audio full-scale noise on rung replays (quad1 2026-09-05) | `TestT2vaShapes::test_audio_survives_replay_of_a_length_seen_after_capture` (12 s, 13 s, 13 s), `test_duration_ladder_second_pass_replays` | xfail(strict); every completed output in every live test is also judged by `h3_media_checks` |
| garbage video frames (old metal tree) | `h3_media_checks.judge` on every output (entropy > 7.5 bits or > 0.14 B/px at q3) | strict |
| switching back to a smaller rung crashed in `ttnn.copy` | `test_switch_back_to_a_smaller_rung` (5 s, 9 s, 5 s) | strict |
| second rung bound in a process produced garbage (old tree) | `test_duration_ladder_4_to_15s`, `test_aspect_ratios_echo_the_canvas` (4-6 rungs per process, content judged) | strict |
| SP path dropped `duration_seconds`/`aspect_ratio` (9 s request -> 5 s video) | `test_sp_runner.py` side-file object tests, `test_video_runner.py` mapping tests; live: every `Spec` asserts served duration (17n+5 rounding) and canvas | strict |
| API accepts out-of-policy duration/aspect/unknown fields in the SP process (failed job instead of 422) | `test_minimax_h3_admission_gaps.py` | xfail(strict) |
| `num_inference_steps` is not a request lever for H3 but is accepted | `test_minimax_h3_video_request.py::test_minimax_rejects_unsupported_request_fields[num_inference_steps-50]` | failing today (pre-existing, unmarked) |
| a failed job must not poison the worker process | `live_deploy::TestRecovery::test_failed_job_does_not_poison_the_process` | strict |
| out-of-policy duration must be a 422 at the API, not a failed job | `live_deploy::TestApiContract::test_out_of_policy_duration_is_refused_at_admission` | xfail(strict) |
| identical trees / cache overlay / weights on all ranks (`h3ctl.sh check`) | `live_deploy::TestHygiene::test_check_passes_on_all_ranks` | strict |
| every duration x every aspect ratio in one process (t2va, fl2va one keyframe) | `live_shapes::TestFullMatrix::test_t2va`, `test_fl2va_one_keyframe` (tier 3, 72 requests each) | strict on status/echo/video; audio-noise outputs reported as xfail |
| trace region 150 MB vs six resident captures (upstream validates 450 MB) | `live_deploy::TestTraceResidency::test_all_rungs_bound_then_replayed` | strict on status (tier 3) |
| ref2va reference-count limits, second-request OOM (#5044 table) | `test_minimax_h3_live.py::TestRef2va` (`SECOND_REQUEST_OOM` xfail on img8 / mix_6i_3v) | as before |
| DELETE contract, cancel, routing 422 for endpoints the deployment does not serve | `test_minimax_h3_live.py` (both classes) | strict |
| t2va output determinism | `test_same_seed_is_byte_identical` | strict |

## Content heuristics (h3_media_checks)

Calibrated on 2026-09-05 quad1 outputs (14 clean clips over 6 canvases, 8 corrupted-audio clips,
2 corrupted-video clips): clean audio is mean -28..-53 dB / max -10..-35 dB, corrupted audio has
max 0.0 dB (clipping) and mean -0.3..-13 dB -> **fail when max > -1 dB or mean > -15 dB**; clean
frames are 0.068-0.115 JPEG bytes/pixel (q3, 672 px wide) at 5.9-7.55 bits Y-entropy (a detailed scene
alone reaches 7.5+, so entropy never decides on its own), garbage frames 0.29-0.46 B/px at 7.77-7.83
bits -> **fail when > 0.20 B/px, or > 7.7 bits together with > 0.12 B/px**, six frames per clip; the
container bit rate is a third, decode-free signal: clean <= 0.34 bits/pixel/frame, garbage 0.77-0.88 ->
**fail when > 0.6**.
Every judged output also has to echo the request: duration = `expected_frames(seconds)/24`
(rounded UP to 17n+5 frames) and the canvas from `expected_canvas(aspect)`.

## Reading a run

* `[deployment hh:mm:ss] ...` lines are worker restarts / chip resets.
* Each sequence request prints `tag: status in Ns rung=R [capture] [jit=N] | OK/BAD dur=... audio mean/max=... frames[...]`.
* The JSON report (`H3_LIVE_REPORT`) has one row per request with `rung`, `captured`,
  `compiled_kernels`, `sha256`, `audio_mean_db`/`audio_max_db`, `content_ok`, `content_reasons`.
* An **XPASS** on a strict xfail means a known bug got fixed: remove the mark in the same PR.
