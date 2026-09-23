# MiniMax-H3 in CI

How the MiniMax-H3 video model (text-to-video-and-audio; first/last-frame and
omni-reference to follow) is tested by tt-shield, what each tier does, what a red
job means, and how to run the same thing by hand. Single-host Blackhole Galaxy first.

## Where it is wired

| Piece | File |
|---|---|
| CI membership (nightly + weekly on `BH_GALAXY`) | `.github/workflows/models-ci-config.json` -> `MiniMaxAI/MiniMax-H3` |
| Model spec (dev catalog, what `run.py --dev-mode` resolves) | `workflows/model_specs/dev/video.yaml` |
| Media-server device config for `DEVICE=blackhole_galaxy` | `tt-media-server/config/constants.py` (`TT_MINIMAX_H3_T2VA`, `BLACKHOLE_GALAXY`) |
| Release gate registration | `reference_config/evals/eval_config.py`, `eval_targets/model_accuracy_reference.json`, `benchmark_targets/model_performance_reference.json` |
| Benchmarks / evals hooks | `test_module/benchmark_tests/video_benchmark_tests.py`, `test_module/eval_tests/video_eval_tests.py` |
| Spec tests | `test_module/server_tests_config.json` (`minimax_h3`), `test_module/test_suites/video.json` |
| The generation benchmark (vendored `quad-agent/h3-benchmark`) | `test_module/_test_common/minimax_h3_bench/`, `test_module/load_param_tests/minimax_h3_benchmark_test.py` |
| Inputs | `test_fixtures/datasets/minimax_h3/` (prompts + manifests; media staged, see its README) |
| Hardware-free tests | `tests/test_module/benchmark_tests/test_minimax_h3_bench_unit.py`, `test_minimax_h3_bench_mock.py`, `tests/test_module/fixtures/mock_tt_h3_server.py`, `tt-media-server/tests/test_minimax_h3_video_request.py` |

tt-shield needs no change: `BH_GALAXY` already maps to runner label `bh-galaxy`
(type `blackhole_galaxy`), MEDIA models go through `media-inference-server`, and
the nightly builds the media image from tt-metal `main` (the H3 pipeline,
`models/tt_dit/pipelines/minimax_h3`, is on main since 2026-08-17).

## What a run does

`run.py --model MiniMaxAI/MiniMax-H3 --workflow release --device blackhole_galaxy
--docker-server --dev-mode --engine media --ci-mode --override-docker-image <img>
--host-volume /mnt/MLPerf/tt-shield/persistent-volume`:

1. `setup_host` downloads `MiniMaxAI/MiniMax-H3` into the host volume
   (`weights/MiniMax-H3`, ~180 GB with `transformer_ref/`), checks 300 GB / 64 GB RAM.
2. The container starts with `MODEL=MiniMax-H3 DEVICE=blackhole_galaxy` and the spec
   env: `MODEL_RUNNER=tt-minimax-h3-t2va`, `MESH_DEVICE=(4, 8)`, `MAX_QUEUE_SIZE=2`,
   `MINIMAX_H3_MODEL_PATH=/home/container_app_user/cache_root/weights/MiniMax-H3` (the pipeline
   reads it when no weights directory is mounted; `--host-weights-dir` runs mount one instead),
   `TT_DIT_CACHE_DIR` on the persistent volume (~68 GB ttnn cache, 20-30 min on the
   very first start), `MINIMAX_H3_WARM_SHAPES=16:9@5,16:9@10,16:9@15` (each shape
   compiles 4-16 min at startup instead of inside the first request).
3. `release` = evals -> benchmarks -> spec tests, one accumulator, one report:
   * **evals**: `MiniMaxH3VideoQualityTest` (one 16:9/5 s clip, structural checks:
     decodable, right duration and ratio, not black/flat/frozen; CLIP off until a BH
     Galaxy `clip_valid_range` is measured -> `accuracy_check` NA).
   * **benchmarks**: two 16:9/5 s clips with the H3 request fields (no
     `num_inference_steps`), `ttft` = wall time per clip vs the `blackhole_galaxy`
     reference (72 s; the functional tier passes below 720 s).
   * **spec tests** (`minimax-h3-blackhole_galaxy`):
     `MiniMaxH3CreateContractTest` (auth, unknown/provider fields, aspect ratio,
     duration, explicit step count -> 422), `MiniMaxH3LifecycleDownloadTest` (one
     job: create, query, list, download, video + non-silent audio) and
     `MiniMaxH3BenchmarkTest` (below). `MiniMaxH3CancelLifecycleTest` is registered
     but disabled: cancelling an in-flight job took the multi-host deployment down
     (PR #5053); enable once a single-host cancel is verified harmless.

Any spec test with `success: false`, a failed generation in the benchmark, or a
crashed task makes `run.py` exit non-zero, which is the red job.

## MiniMaxH3BenchmarkTest

The h3-benchmark contract, in order:

1. **Health pre-gate** (read-only): `/health` 200, `/tt-liveness` ready with an
   empty queue, no foreign job in flight (waits `idle_wait_s`), `/metrics` canary not
   dead and the newest generation not a failure without a success in 30 min. A
   deployment that fails this is reported as "not fit to measure" and nothing is
   generated: a green `/health` never proves generation works.
2. **Probe**: openapi lists the task's route, the job and download routes; the
   deployment's gate does not refuse the task; the API key is accepted; every asset
   the plan needs is present and matches `sha256s-bundle.txt`.
3. **Smoke**: one 5 s clip (`SMOKE`), judged.
4. **Plan**: per case, 1 warmup + N measured runs, each with its own budget
   (`BH1X` table in `models.py`, one 2x extension while the status keeps moving),
   then the newest N ok clips are judged. Every clip must be an mp4 with a video and
   an audio stream, 24 fps, the canvas of its aspect ratio (16:9 -> 1344x768), a
   duration within 0.25 s of `17n+5` frames, and a soundtrack that is neither silent
   nor at the rails (> 5 % of samples at 0 dB is the corruption signature).
5. **Host state**: a job still generating when its budget runs out wedges the host
   (remaining cases skip); three device-trouble failures in a row (`TT_THROW`,
   `device timeout`, `hang`, ...) stop it; a synchronous build (POST answers 200 with
   the mp4) stops it; a job that never left `queued` is a strike, not a wedge. Every
   job this run submitted and did not see finish is cancelled at the end.
6. **Verdict** per case: `pass` (all runs ok, all clips clean), `xfail` (every
   failure is a documented capability limit, e.g. FL2VA-H's 27 MB keyframe against the
   10,000,000-char base64 cap), `fail`, or `skipped` (host stopped earlier). Timing
   against `target_times_s` is reported and, with `enforce_timing`, fails a case whose
   median exceeds 1.25x the target.

Plans (`test_suites/video.json`): nightly (`--ci-mode`) runs `T2VA-L` x3 and
`T2VA-M`/`T2VA-H` x1 (~15 min warm); weekly runs `T2VA-L/M/H` x3 (~30 min warm).
Rows land in `<output>/minimax_h3_bench/results.jsonl` and `results.csv`, clips under
`out/`, the narrative in `run.log`, `summary.json` next to them -- the same files
`quad-agent/h3-benchmark`'s `h3bench report` and `h3bench gaps` read.

## Running it by hand

```bash
# against any deployment (hosted or local), no run.py:
python -m test_module.load_param_tests.minimax_h3_benchmark_test \
    --base-url http://127.0.0.1:8000 --task t2va --cases T2VA-L,T2VA-M --runs 3 \
    --out /tmp/h3bench --assets /path/to/h3-assets      # API_KEY / TT_MINIMAX_API_KEY in the env

# the whole nightly on a Galaxy you own:
python3 run.py --model MiniMaxAI/MiniMax-H3 --workflow release --device blackhole_galaxy \
    --docker-server --dev-mode --engine media --ci-mode \
    --override-docker-image ghcr.io/tenstorrent/tt-shield/tt-media-inference-server:latest \
    --host-volume /path/with/300GB

# no hardware: unit tests + the mock end to end (needs ffmpeg + ffprobe)
pytest tests/test_module/benchmark_tests/test_minimax_h3_bench_unit.py \
       tests/test_module/benchmark_tests/test_minimax_h3_bench_mock.py
```

A one-off tt-shield dispatch (Actions -> "On dispatch"): `custom-model=MiniMaxAI/MiniMax-H3`,
`runner-label=bh-galaxy`, `device-type=blackhole_galaxy`, `workflow=benchmarks` first
(two clips, fastest signal), then `release`; tt-metal ref `main`, tt-inference-server ref
= the branch carrying this catalog entry. Expect the first run on a fresh runner to spend
the readiness window (2 x 3600 s) on the weight download and the ttnn cache.

## Known gaps and next steps

* **FL2VA / Ref2VA are not on main yet** (runners, `POST /generations/ref2va`,
  `DELETE`): they live on `sadesoye/add_h3_fl2va_ref2va`, which also needs a
  `policy.py` that is not on tt-metal main. `cases.json` already carries their 15
  cases; when they land, add `MiniMax-H3-FL2VA` / `MiniMax-H3-Ref2VA` specs
  (`MODEL_RUNNER` per task, same device block), `minimax_h3_fl2va` / `_ref2va` model
  configs and suites with `task: fl2va|ref2va`, stage the media pack
  (`test_fixtures/datasets/minimax_h3/README.md`), and take the contract tests from
  PR #5053. Weekly Ref2VA is ~8 h on one Galaxy and must be split in two jobs.
* **Budgets and targets are provisional** (`BH1X` row, `target_times_s`,
  `model_performance_reference.json`): re-derive them from the first green weekly's
  `results.jsonl` (3x the slowest observed run; target = 1.25x the median).
* **Cancel** is disabled in CI until verified on a single host.
* **Seed determinism** is not asserted (the hosted deployments were not deterministic).
* The H3 step-count contract (`tt-media-server/domain/video_generate_request.py`): an
  explicit `num_inference_steps` is refused with 422, and an omitted one is pinned to the
  50-step schedule (main previously ran the schema default of 20 steps when the field was
  omitted, so the "fixed 50" the rows record was not what ran). The multi-host rank workers
  rebuild the request from shared memory with that field present; `video_runner.py` drops
  it for H3 before the rebuild so the refusal stays at the API boundary.
