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
| `test_minimax_h3_deploy_constants.py` | 0 | source-text tripwires on `dit_runners.py` deployment constants (trace region 150 vs 450 MB, shape resolution reads duration/aspect) |
| `test_minimax_h3_contract_gaps.py` | 0 | request/serving contract tripwires from the 2026-09-06 gap hunt (steps, prompt bytes, negative_prompt, blank prompt, seed bounds, rank-outcome exchange, upload route) -- all xfail(strict) |
| `test_sp_runner.py`, `test_video_runner.py` (H3 parts) | 0 | side-file wire format carries `duration_seconds`/`aspect_ratio`/keyframes/references; worker request mapping |
| `test_minimax_h3_live_deploy.py` | 1 / 3 | `h3ctl.sh check`, warmup envs in `/proc/<pid>/environ`, budgets with cold-JIT detection, API 422 contract (xfail), failed job then success, liveness under load, API restart, worker-log alarms, all six rungs resident, opt-in chaos (kill a rank) |
| `test_minimax_h3_live_shapes.py` | 2 / 3 | t2va ladder 4-15 s (+ second pass), aspect echo, audio-replay repro, switch-back, determinism (+ different seed), prompt/field variants, queue order, cancel-then-next, fl2va side-file leak; fl2va one-keyframe ladder/aspects, two keyframes per canvas x prompt length; tier 3: full duration x aspect matrix per task, soak, two-keyframe repeats, ref2va video reference, per-rung audio map |
| `test_minimax_h3_live.py` | 2 / 3 | fl2va keyframe combos, ref2va reference matrix (#5044 table), cancel, failed-job deletable, DELETE contract, routing 422s |

## Issue -> test

| issue (where seen) | test | state |
|---|---|---|
| slow warmup; fixed by `TT_METAL_SHM_TRACKING_DISABLED=1` + `TT_METAL_LOGS_PATH` (first request 173 s -> 42 s) | `live_deploy::TestDeploymentKnobs::test_worker_env_has_the_warmup_knobs`, `TestBudgets::test_first_and_warm_request` (budget 120 s warm cache / 600 s after a rebuild, detected via `BuildKernels` lines), `test_time_to_ready` | strict |
| fl2va first+last keyframes at a 1008-row canvas: rejected (`prompt tokens 2053 > 2048`) with a >= 33-token prompt, or ADMITTED with a shorter prompt and the audio comes back as noise (text padded to 3072 rows overruns the 2048-row prompt arena into the audio rows) | `live_shapes::TestFl2vaShapes::test_first_and_last_keyframes[<aspect>-<short|long>]` | verdict by symptom: cap refusal -> xfail A, admitted-but-noise-audio -> xfail B, clean -> pass (fix landed), anything else -> FAIL; 768/576-row canvases pass today |
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
| every duration x every aspect ratio in one process (t2va, fl2va one keyframe) | `live_shapes::TestFullMatrix::test_t2va`, `test_fl2va_one_keyframe` (tier 3, 72 requests each) | strict on status/echo/video and on bind/capture audio; noise audio on traced REPLAYS reported as xfail |
| trace region 150 MB vs six resident captures (upstream validates 450 MB) | `live_deploy::TestTraceResidency::test_all_rungs_bound_then_replayed` | strict on status (tier 3) |
| served denoise steps: API default 20 reaches the pipeline, policy/warmup say 50 ("not a request lever") | `test_minimax_h3_contract_gaps.py::test_default_steps_resolve_to_the_h3_policy_value`, `::test_explicit_steps_are_refused_for_h3`; live `live_deploy::TestServedContract::test_requests_run_at_the_policy_step_count` (worker-log `N steps` fact) | xfail(strict) |
| prompt silently cut to 2048 bytes / negative_prompt to 512 on the SHM wire | `contract_gaps::test_prompt_longer_than_the_wire_slot_is_refused_or_delivered_whole`, `::test_negative_prompt_longer_than_the_wire_slot_is_refused`; live `live_shapes::TestRequestFields::test_long_prompt_is_delivered_whole` (same video with/without text past byte 2048) | xfail(strict) |
| `negative_prompt` accepted but never reaches the H3 pipeline | `contract_gaps::test_negative_prompt_is_refused_as_not_a_lever`; live `TestRequestFields::test_negative_prompt_changes_the_output` | xfail(strict) |
| blank / whitespace prompt, out-of-int64 seed admitted | `contract_gaps::test_blank_prompt_is_refused`, `::test_out_of_int64_seed_is_refused` | xfail(strict) |
| rank outcomes not exchanged: rank 0 ships the mp4 while another rank errored (ref2va 2026-09-04) | `contract_gaps::test_inference_loop_exchanges_rank_outcomes` | xfail(strict) |
| `/i2v/upload` defaults diverge from the JSON route, no duration/aspect | `contract_gaps::test_upload_route_matches_the_json_route` | xfail(strict) |
| a dead rank takes 25-83 min to fail the job (only the 300 s op timeout and the 5000 s request timeout bound it) | `live_deploy::TestChaos` (opt-in) | xfail(strict) |
| cancel is API-side only: the mesh finishes the abandoned clip and leaves an orphan mp4 | `test_minimax_h3_live.py::*::test_cancel_then_delete` now waits for the worker to finish before the on-disk check | xfail by symptom |
| t2va deployment must refuse i2v and ref2va bodies (the silent-conditioning-drop incident) | `test_minimax_h3_live.py::TestT2va::test_routing_and_delete_negatives` | strict (tier 1) |
| default API key in production; `/v1/models` cannot tell the served task; `/tt-liveness` says device n150 on a quad; not-ready answered with 405 instead of 503 | `live_deploy::TestServedContract::{test_api_key_is_not_the_default, test_models_endpoint_names_the_served_task, test_liveness_fields_describe_this_deployment, test_not_ready_is_a_503}` | xfail(strict) |
| 202 / GET bodies echo the full base64 media; `/download` while running is a 404; cancel of a finished job is a 404 | `live_shapes::TestApiBehaviour::{test_submit_response_does_not_echo_media, test_download_while_running_is_not_a_404, test_cancel_of_a_finished_job_is_a_409}` | xfail(strict) |
| untagged BT.601 colour (players assume 709); `-shortest` drops the last frame at 4/6/15 s | `live_shapes::TestOutputContract::{test_colour_is_tagged, test_all_frames_are_delivered_at_4s}` | xfail(strict) |
| fl2va keyframes honoured (first frame ~ first keyframe, last ~ last; SSIM) | `live_shapes::TestFl2vaConditioning::test_keyframes_are_honoured` | strict |
| fl2va not deterministic for a fixed seed (PSNR 22.6 dB between identical requests) | `live_shapes::TestFl2vaConditioning::test_same_seed_is_byte_identical` | xfail(strict) |
| trace region: server opens the mesh with 150 MB, upstream validates the bucketed design at 450 MB | `test_minimax_h3_deploy_constants.py::test_trace_region_matches_what_upstream_validates` (tier 0, source-text tripwire) | xfail(strict) |
| ref2va reference-count limits, second-request OOM (#5044 table) | `test_minimax_h3_live.py::TestRef2va` (`SECOND_REQUEST_OOM` xfail on img8 / mix_6i_3v) | as before |
| DELETE contract, cancel, routing 422 for endpoints the deployment does not serve | `test_minimax_h3_live.py` (both classes) | strict |
| t2va output determinism | `test_same_seed_is_byte_identical` | strict |
| `/tt-liveness` must stay alive + model_ready while a generation runs (k8s canary killed the pod during a 52 s job) | `live_deploy::TestLivenessUnderLoad::test_liveness_stays_ready_during_a_generation` | strict |
| API restart while workers keep running (SHM rings are create-or-attach) | `live_deploy::TestApiRestart::test_api_restart_keeps_serving` | strict |
| unexpected critical / TT_FATAL / TIMEOUT / Permission denied lines in an otherwise clean run | `live_deploy::TestWorkerLog::test_no_unexpected_alarms_in_the_current_worker_log` | strict (server host) |
| a dead rank must fail the job within a bound, not hang (k8s: chips held by a surviving worker) | `live_deploy::TestChaos::test_killed_rank_fails_the_job_within_a_bound` (opt-in `H3_LIVE_CHAOS=1`, tier 3) | strict |
| disk / tmpfs headroom and leftover side-files on every rank | `h3ctl.sh check` (root < 20 GB, /dev/shm < 4 GB, > 5 `tt_img_*` = FAIL) via `TestHygiene` | strict |
| queued jobs complete in submission order, all valid; `queue_size` recorded | `live_shapes::TestQueue::test_three_queued_jobs_complete_in_order` | strict |
| cancelling a running job must not corrupt the next one | `live_shapes::TestQueue::test_cancel_running_job_then_next_request_is_clean` | strict |
| fl2va side-files (`/dev/shm/tt_img_*.json`) must be unlinked after each request | `live_shapes::TestSideFiles::test_fl2va_requests_leave_no_side_files` | strict (server host) |
| unicode prompt, `negative_prompt`, ~1000-token prompt | `live_shapes::TestRequestFields::test_prompt_variants` | strict |
| a prompt beyond the 2048-row prompt arena: today a failed job after the encoder ran, contract is 422 | `live_shapes::TestRequestFields::test_prompt_beyond_the_arena_cap_is_refused` | xfail by symptom -- observed 2026-09-06: the ~1000- and ~2500-token prompts produced byte-identical outputs, i.e. the text encoder silently truncates long prompts (see Notes) |
| a different seed must change the output | `test_same_seed_is_byte_identical` (third request, seed 8) | strict |
| 12 identical requests: outputs identical, wall time flat (no residual growth) | `live_shapes::TestSoak::test_identical_requests_stay_flat` (tier 3) | strict |
| two keyframes at a canvas that fits, three times (bind/capture/replay) | `live_shapes::TestFl2vaRepeats::test_two_keyframes_repeat_at_4x3` (tier 3) | strict |
| ref2va video reference with duration/aspect | `live_shapes::TestRef2vaReferences::test_video_reference_with_duration` (tier 3) | strict |
| 24 fps, audio and video streams within one frame of each other (truncated / missing soundtrack) | `h3_media_checks.judge` on every output | strict |
| ref2va honours `duration_seconds` / `aspect_ratio` | `live_shapes::TestRef2vaShapes::test_duration_and_aspect_echo` | strict |
| which rungs corrupt audio on the replay of their capture duration | `live_shapes::TestRungAudioMap::test_replay_of_capture_duration[44032/61440/86016/118784]` (tier 3) | strict on status/echo/video; replay audio noise -> xfail |
| old-suite fl2va/ref2va combo outputs judged for content (a 41 MB 5 s ref2va clip passed the stream check on 2026-09-04) | `_generate_repeatedly` now calls `h3_media_checks.judge` | strict |
| tt-metal python tree drift between ranks (the H3 fixes are .py-only) | `h3ctl.sh check` compares `models/tt_dit/**.py` too; `live_deploy::TestHygiene` runs it | strict |

## Content heuristics (h3_media_checks)

Calibrated on 2026-09-05 quad1 outputs (14 clean clips over 6 canvases, 8 corrupted-audio clips,
2 corrupted-video clips): clean audio is mean -28..-53 dB / max -10..-35 dB, corrupted audio has
max 0.0 dB (clipping) and mean -0.3..-13 dB -> **fail when max > -1 dB or mean > -15 dB**; clean
frames are 0.068-0.133 JPEG bytes/pixel (q3, 672 px wide) at 5.9-7.55 bits Y-entropy (a detailed scene
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
* An **XPASS** on a strict xfail means a known bug got fixed: remove the mark in the same PR. Tests that
  decide by symptom (two keyframes, full matrix, rung audio map) use imperative `pytest.xfail` and simply
  PASS once the bug is gone.
* `pytest tests/` (what CI runs) must keep skipping every live module: nothing may set `H3_LIVE_URL` at
  import time; `test_h3_live_sequences_pure.py` imports `h3_live_common` without it.
* Cold-JIT detection: >= 500 `BuildKernels | compiled` lines in the request's worker-log window
  (a new-canvas bind on a warm cache compiles ~140, a cold cache ~1800).

## Notes / observations

* **Long prompts are silently truncated**, not refused: on 2026-09-06 a ~1000-token and a ~2500-token
  prompt (the same sentence repeated) produced byte-identical t2va outputs. The API only rejects
  absurd lengths (100k chars); anything between the tokenizer limit and that passes and is cut. Decide
  whether that is the contract (then assert the truncation length) or a 422.
* **Intermittent load hang** (once in ~10 fl2va fresh starts that day): the vision-tower weight load hit
  `TIMEOUT: device timeout in fetch queue wait` after 300 s on two ranks (`LoadingError ... blocks.16.mlp.linear_fc2`).
  The harness marks the deployment poisoned and resets; the failure is recorded in the report.
* Per-module fixtures mean every live module starts with a chip reset (~110 s); run one module per
  invocation when time matters.
