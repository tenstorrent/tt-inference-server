# C++ runner layer for all dit_runners models, checked on a T3K (Wan2.2)

Date: 2026-11-17. Functional check of a code change on a T3K with Wan2.2 T2V, plus
device-free tests of every model. The change moves everything the Python server's
`tt_model_runners/dit_runners.py` does, except the `tt_dit` pipeline calls, into C++.

## Change

Before this change the video worker imported the Python server (`runner_fabric`,
`config.settings`, `dit_runners.py`, `utils.video_manager`, `multihost/`) from
`TT_PYTHON_PATH`. An intermediate step replaced that by one Python file for Wan2.2 only.
Now:

| | Before (Python server) | Now |
|---|---|---|
| Models | `MODEL_RUNNER` values of `dit_runners.py` | the same 23 (Wan2.2 T2V, I2V, AniSora, Distill, LoRA, Lightning, Prodia T2V/I2V; Mochi-1; LTX-2.3; MiniMax-H3 t2va, fl2va, ref2va, FastH3, LightX2V x2; SD3.5, FLUX.1 dev/schnell/Kontext, Motif, Qwen-Image x2) |
| Runner classes, registry | Python `TTDiTRunner` and subclasses | C++ `DiTModel` (abstract) with one subclass per family: `WanModel`, `MiniMaxModel`, `ImageModel`, ...; `DiTRunner` runs any of them |
| Mesh per `DEVICE`, fabric, trace regions, guidance, steps, resolution | Python `config/` and `dit_runners.py` | C++ tables and `deviceParams()` in `src/runtime/dit/` (same values) |
| Request parsing and validation (image prompts, LTX shape, MiniMax keyframes/references/duration/aspect ratio) | pydantic models and runners | C++ `DiTRequest`, `DiTModel::validate()` |
| Opening and closing the mesh | `BaseMetalDeviceRunner` | C++ through ttnn (`openMesh`) |
| mp4 and mp4 + AAC export | `utils/video_manager.py` (ffmpeg from Python) | C++ (`media_export.cpp`: ffmpeg through a pipe, WAV writer) |
| Multihost lockstep and follower | `multihost/` (pickle over `ttnn.distributed_context_*`) | C++ (`dit_multihost.cpp`; JSON messages over the same ttnn calls) |
| Calls into `tt_dit` | inside the runner classes | `python/cpp_dit_adapters/`: per family a function that builds the pipeline and one that calls it; no validation, export or configuration |
| Needed variables | `TT_METAL_HOME`, `TT_PYTHON_PATH` | `TT_METAL_HOME`; `TT_VIDEO_PYTHON_DIR` optional (adapters, default `cpp_server/python`) |
| Tracing | `"traced"` hard-coded in `dit_runners.py` | `TT_VIDEO_TRACED`, default per model as in Python (`True`; Kontext `False`); `0` turns it off |

Added to the C++ request: `image_prompts` and `references` (base64 only, URLs are
refused); the job record shows only `frame_pos` and the reference counts.

## Difference from the run before the change, on the device

Same machine, same chips (reset not needed), same environment and request set; the
"before" server is the binary built from HEAD's `video_device_runner.cpp` with
`TT_PYTHON_PATH` pointing at the Python server, the "after" server has
`TT_PYTHON_PATH` unset (`MODEL=Wan2.2-T2V-A14B-Diffusers`, `MODEL_RUNNER=tt-wan2.2`,
`DEVICE=t3k`, `DEVICE_IDS=(0,1,2,3)`, `TT_VIDEO_TRACED=0`, `HF_HUB_OFFLINE=1`).

| | Before (Python runner) | After (C++ runner) |
|---|---|---|
| Ready after start | 101.2 s | 101.2 s |
| `a red ball rolling on grass`, seed 7, 4 steps: generation | 78.4 s | 78.8 s |
| mp4 of that request | 2 919 606 B, md5 `5ce1308d...` | 2 919 606 B, md5 `5ce1308d...` (**byte-identical**) |
| The same request again | md5 `5ce1308d...` | md5 `5ce1308d...` |
| Same request with `negative_prompt: "blurry"` | 3 398 399 B, md5 `a7782163...` | 3 398 399 B, md5 `a7782163...` (**byte-identical**) |
| mp4 export time | inside the generation time | 0.3 s (logged separately) |
| Stop with SIGTERM | exit 0, 0.8 s | exit 0, 0.9 s, `Closed mesh device` logged, nothing left |

The mp4 being byte-identical means the C++ frame conversion and ffmpeg command produce
the same file as `VideoManager.export_to_mp4`.

What differs for a user:
- no `TT_PYTHON_PATH`, and no `tt-media-server` code is imported (Python imports only
  tt-metal's `ttnn` and `models.tt_dit`, plus numpy, torch, PIL);
- `TT_VIDEO_TRACED` replaces editing `dit_runners.py` for tracing;
- a request is validated by the C++ model; refusals come back as a failed job with the
  runner's message (`[VideoDevice] request refused: ...`);
- a body with a duplicate `frame_pos` or a `url` reference is refused with `422` at
  submit (checked on the server while the first job ran), and a queued job with
  `image_prompts` shows `[{"frame_pos": 0}]` instead of the base64 (checked, `200`;
  cancelled afterwards);
- the server logs `[DiT]` lines (pipeline created in 54.8 s, warmup run 42.5 s, per
  task inference and export times).

## Device-free tests

| Test | Result |
|---|---|
| `build/dit_runner_test` (new, `tests/unit/dit/dit_runner_test.cpp`): the registry serves the 23 runners; every model's mesh, device parameters, pipeline call, validation (Wan I2V, LTX shape, MiniMax shape/keyframes/references/LightX2V), image models; request round trip; mp4 export from float, grey and RGBA frames (faststart), WAV, mp4 with AAC and no temp files left; lockstep and follower with a fake transport | 29/29 |
| `build/video_generate_request_test` (two new cases: `image_prompts`, `references`) | 63/63 |
| `python -m pytest tests/video/python` (adapters with fake `tt_dit` pipelines; the C++ mesh table and runner list against the Python server's `ModelConfigs` and `dit_runner_log_map`) | 18/18 |
| `tests/video/mock/worker_crash_cases.py` | 15/15 |

## Not done

- Only `tt-wan2.2` ran on the device. The other models are tested with their pipeline
  arguments and adapters (fakes); their weights are not on this machine, so no other
  model has been built or run through its real pipeline.
- `traced=True` was not run on the T3K (noise there, as with the Python server).
- The multihost lockstep was tested with a fake transport only: the ttnn message calls
  (`distributed_context_send_bytes`, ...) and the JSON message format were not run on
  several hosts, and the follower rank was not started.
- Media URLs and the `i2v/upload` route are not supported (base64 only).
- ffmpeg is looked up on `PATH`; the Python server also fell back to the
  `imageio_ffmpeg` wheel.

## Check of this change by itself (2026-11-18)

The branch with only this change (a clean build, no other pending work): `dit_runner_test` 35/35,
`video_generate_request_test` 63/63, `python -m pytest tests/video/python` 18/18, and on the mock runner
`worker_crash_cases.py` 15/15, `lifecycle_cases.py` 53/53, `run_server_tests.sh` 42 + 2 + 5 + 4 + 5, contract
36/36. On the T3K (Wan2.2, `DEVICE_IDS="(0,1,2,3)"`, `TT_VIDEO_TRACED=0`, chips reset): ready after about 102 s;
`a red ball rolling on grass`, seed 7, 4 steps: 79 s, 2,919,606 B, md5 `5ce1308d...` (identical); the same with
`negative_prompt` 80 s, 3,398,399 B, md5 `a7782163...` (identical); inference 78.1 s and 78.7 s, export 0.3 s; SIGTERM
when idle exits 0 after 0.8 s.

The other real-device suites, on the same branch and the same T3K (a worker process, `TT_VIDEO_TRACED=0`):

| Suite | Result |
|---|---|
| `cancel_queue_and_running.py` | 21/21 |
| `queue_download_sync.py` (phase `validation`, then the whole script) | 9/9, 45/45 |
| `disconnect_and_queue_depth.py` | 9/9 |
| `fields_content_load.py` | 36/36 |
| `cancel_race.py` (server with `TT_MEDIA_TEST_PRECLAIM_DELAY_MS=1500`) | 10/10 |
| `warmup_and_limits.py` | 8/8 |
| `restart_cases.py` | 31/31 |

`restart_cases.py` first failed at phase 1 ("ETH heartbeat" while the worker warmed up): the suite before it,
`warmup_and_limits.py`, stops its server in the middle of a warmup, which leaves the chips in the known stale state
(`tt-smi -r` needed). After a reset it passed. `api_parity.py` was not run (it records answers and needs the Python
server for the comparison).

