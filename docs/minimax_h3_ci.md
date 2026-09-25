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
`models/tt_dit/pipelines/minimax_h3`, is on main since 2026-08-13).

## What a run does

`run.py --model MiniMaxAI/MiniMax-H3 --workflow release --device blackhole_galaxy
--docker-server --dev-mode --engine media --ci-mode --override-docker-image <img>
--host-volume /mnt/MLPerf/tt-shield/persistent-volume`:

1. `setup_host` downloads the t2va partitions of `MiniMaxAI/MiniMax-H3` into the host
   volume (`weights/MiniMax-H3`, ~144 GB; `FL2VA/`, `Ref2VA/` and `transformer_ref/` are
   excluded, the full HF repo is 498 GB) and checks 300 GB disk / 64 GB RAM. This runs
   before the container starts; an interrupted download leaves `.incomplete` files, which
   setup_host detects and resumes on the next run.
2. The container starts with `MODEL=MiniMax-H3 DEVICE=blackhole_galaxy` and the spec
   env: `MODEL_RUNNER=tt-minimax-h3-t2va`, `MESH_DEVICE=(4, 8)`, `MAX_QUEUE_SIZE=2`,
   `MINIMAX_H3_MODEL_PATH=/home/container_app_user/cache_root/weights/MiniMax-H3` (the pipeline
   reads it when no weights directory is mounted; `--host-weights-dir` runs mount one instead),
   `TT_DIT_CACHE_DIR` on the persistent volume (~68 GB ttnn cache, 20-30 min on the
   very first start), `MINIMAX_H3_WARM_SHAPES=16:9@5` (one warm shape). The readiness
   window is 3600 s per boot attempt x 2 attempts and covers weight load, the cache build
   and that warm shape. The 10 s and 15 s shapes compile inside the first request of each,
   which the benchmark's warmup runs absorb within the `BH1X` budgets (T2VA-M 1500 s,
   T2VA-H 1800 s).
3. `release` = evals -> benchmarks -> spec tests, one accumulator, one report:
   * **evals**: `MiniMaxH3VideoQualityTest` (one 16:9/5 s clip at seed 0, 8 frames sampled
     from it; structural checks: decodable, right duration and ratio, not black/flat/frozen).
     Black/flat/frozen is enforced: the `accuracy` block keyed `"1"` (the number of clips) in
     `model_accuracy_reference.json` requires the generation to succeed with 0 invalid and
     0 frozen clips. Each sample's outcome, or its error, is logged. CLIP stays
     off (`enable_clip: false`) until a BH Galaxy `clip_valid_range` is measured.
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
   (`BH1X` table in `models.py`; effectively hard on this deployment, since the server
   exposes no progress and the extension-while-moving cannot trigger),
   then the newest N ok clips are judged. Every clip must be an mp4 with a video and
   an audio stream, 24 fps, the canvas of its aspect ratio (16:9 -> 1344x768), a
   duration within 0.25 s of `17n+5` frames, and a soundtrack that is neither silent
   nor at the rails (> 5 % of samples at 0 dB is the corruption signature).
5. **Host state**: a job still generating when its budget runs out wedges the host
   (remaining cases skip); three device-trouble failures in a row (`TT_THROW`,
   `device timeout`, `hang`, ...) stop it; a synchronous build (POST answers 200 with
   the mp4) stops it; a job that never left `queued` is a strike, not a wedge. Every
   job this run submitted and did not see finish is cancelled at the end. In CI the
   benchmark never resumes from a previous `results.jsonl`: every run generates fresh
   (resume is CLI-only, `--force` semantics). The whole benchmark has a cooperative
   deadline 600 s before its template `timeout` (13 800 s of 14 400, leaving room for cancel
   and teardown); a case not started by then is marked `skipped` and the test fails.
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

The weekly entry builds the image from tt-metal `stable`, not `main`, and runs `plan_full`
(`T2VA-L/M/H` x3). A weekly import failure in the runner is therefore a stable/main skew,
not a model regression.

## Running it by hand

```bash
# against any deployment (hosted or local), no run.py:
python -m test_module.load_param_tests.minimax_h3_benchmark_test \
    --base-url http://127.0.0.1:8000 --task t2va --cases T2VA-L,T2VA-M --runs 3 \
    --out /tmp/h3bench --assets /path/to/h3-assets      # API_KEY / MINIMAX_API_KEY / TT_MINIMAX_API_KEY
                                                        # in the env, the same for every H3 test

# the whole nightly on a Galaxy you own:
python3 run.py --model MiniMaxAI/MiniMax-H3 --workflow release --device blackhole_galaxy \
    --docker-server --dev-mode --engine media --ci-mode \
    --override-docker-image ghcr.io/tenstorrent/tt-shield/tt-media-inference-server:latest \
    --host-volume /path/with/300GB

# no hardware: unit tests + the mock end to end (needs ffmpeg + ffprobe; under
# GITHUB_ACTIONS a missing ffmpeg/ffprobe fails these loudly instead of skipping)
pytest tests/test_module/benchmark_tests/test_minimax_h3_bench_unit.py \
       tests/test_module/benchmark_tests/test_minimax_h3_bench_mock.py
```

A one-off tt-shield dispatch (Actions -> "On dispatch"): `model=MiniMaxAI/MiniMax-H3` (in
the dropdown once tt-shield #1154 merges; `custom-model=` on an older tt-shield),
`runner-label=bh-galaxy`, `device-type=blackhole_galaxy`, `workflow=benchmarks` first
(two clips, fastest signal), then `release`; tt-metal ref `main`, tt-inference-server ref
= the branch carrying this catalog entry. The weight download runs in setup_host before the
container starts and is bounded only by tt-shield's job timeout -- 360 min on the scheduled
nightly, and a timed-out job uploads no report -- so the FIRST run on an unstaged runner
should be this on-dispatch workflow (1080 min), which leaves the volume holding the weights
and the ttnn cache. The readiness window itself is 3600 s per boot attempt x 2 attempts and
covers weight load + cache build + the warm shape. If a download was interrupted, setup_host
detects the `.incomplete` files and resumes it.

## Per-task dispatch branches (FL2VA, Ref2VA)

t2va, fl2va and ref2va are separate deployments of the same weights (one `MODEL_RUNNER`
each), while the dev catalog holds one `MiniMaxAI/MiniMax-H3` BLACKHOLE_GALAXY spec. Two
dispatch-only branches point that spec at the other runners, so tt-shield runs every case
of `cases.json` without a shield change: `zni/h3-ci-fl2va` (6 FL2VA cases, one suite entry)
and `zni/h3-ci-ref2va` (12 REF2VA / SIZE cases in five entries of <= 14400 s, rising risk,
smoke only in the first). Do not merge them as is: main's spec serves t2va. This is
`zni/h3-ci-fl2va`; `zni/h3-ci-ref2va` is the other one. Both branches share, on top of this CI branch:

* `sadesoye/add_h3_fl2va_ref2va` (the FL2VA / Ref2VA runners, `/ref2va`, DELETE, the H3
  policy read from tt-metal) merged in, with the three tasks' BLACKHOLE_GALAXY `(4, 8)`
  configs in `tt-media-server/config/constants.py`;
* the media-limit commits from `zni/h3-media-limits`;
* task-aware plumbing, all keyed on the spec's `MODEL_RUNNER`: `setup_host` downloads the
  task's weight set (ref2va reads `transformer_ref/` instead of `transformer/`) and treats a
  volume missing the task's folders or index shards as incomplete; the benchmark takes its
  task, default plans, output directory and `skip_smoke` per entry; the contract,
  lifecycle, quality-eval and generic-benchmark requests follow the deployment's task
  (fl2va gets the text-only shape it serves, ref2va one reference image);
* the task's media, committed under `test_fixtures/datasets/minimax_h3/` with `git add -f`
  (FL2VA ~27 MB, Ref2VA ~101 MB; pinned by `sha256s-bundle.txt`).

tt-metal must be >= `816841ddc93` (#57097: the `minimax_h3/policy.py` the server imports and
the `create_pipeline` `dit_fsdp` / `trace_denoise` / `bucket_denoise` arguments). The image of
the t2va runs (`d9c2c92d05c`) and tt-metal stable `de546d3b` lack both. Dispatch with a pinned
sha, a fresh build (`docker-image` empty) and `run-full-evals=false` first:

```bash
gh workflow run on-dispatch.yml -R tenstorrent/tt-shield --ref main \
  -f model=MiniMaxAI/MiniMax-H3 -f runner-label=bh-galaxy -f device-type=blackhole_galaxy \
  -f workflow=release -f tt-metal-git-ref=<tt-metal sha >= 816841ddc93> \
  -f inference-server-git-ref=<40-hex sha of zni/h3-ci-fl2va> -f impl-of-model=default \
  -f run-full-evals=false -f create-issue-comment=false -f run-ai-summary=false
```

Expect: FL2VA ~30 min of generation (CI) / ~50 min (full) plus boot; Ref2VA ~5.5 h (CI) /
~11 h (full) plus the `transformer_ref/` download (66 GB on a volume t2va warmed, 144 GB cold).
The max-input cases (`FL2VA-H`, the `REF2VA-H` family) have out-of-memory history on 4x8 and
run last; an out-of-memory there leaks device DRAM until the ranks restart.

## Known gaps and next steps

* **FL2VA / Ref2VA are not on main yet** (runners, `POST /generations/ref2va`,
  `DELETE`): they live on `sadesoye/add_h3_fl2va_ref2va`, which also needs a
  `policy.py` that is not on tt-metal main. `cases.json` already carries their 18
  cases (6 fl2va + 12 ref2va); when they land, add `MiniMax-H3-FL2VA` / `MiniMax-H3-Ref2VA` specs
  (`MODEL_RUNNER` per task, same device block), `minimax_h3_fl2va` / `_ref2va` model
  configs and suites with `task: fl2va|ref2va`, stage the media pack
  (`test_fixtures/datasets/minimax_h3/README.md`), and take the contract tests from
  PR #5053. Weekly Ref2VA is ~8 h on one Galaxy and must be split in two jobs.
* **Budgets and targets are provisional** (`BH1X` row, `target_times_s`,
  `model_performance_reference.json`): re-derive them from the first green weekly's
  `results.jsonl` (3x the slowest observed run; target = 1.25x the median).
* **Cancel** is disabled in CI until verified on a single host.
* **Seed determinism** is not asserted (the hosted deployments were not deterministic).
* **Media goes as base64 only**. On main the 27 MB keyframe of `FL2VA-H`/`FL2VA-L2`/
  `FL2VA-M2` exceeds the 10,000,000-char image cap and those cases can only `xfail`; URL
  media would not help (downloads are capped at 7.5 MB and re-checked against the same
  cap). The per-task dispatch branches carry the media-limit commits (image 30 MiB, video
  50 MiB, audio 15 MiB, body 64 MiB, no base64 echo), so there they are real runs.
* **The shared volume is not everywhere**: `/mnt/MLPerf/tt-shield/persistent-volume` exists
  only on runners that have that mount; tt-shield falls back to `/localdev/persistent-volume`
  or a per-run directory, and the media pack has to be staged wherever the volume lands.
* **CLIP is off in CI** (`enable_clip: false`). By hand, pass `--skip-clip` or the ViT-B/32
  download happens before any generation.
* The H3 step-count contract (`tt-media-server/domain/video_generate_request.py`): an
  explicit `num_inference_steps` is refused with 422, and an omitted one is pinned to the
  50-step schedule (main previously ran the schema default of 20 steps when the field was
  omitted, so the "fixed 50" the rows record was not what ran). The multi-host rank workers
  rebuild the request from shared memory with that field present; `video_runner.py` drops
  it for H3 before the rebuild so the refusal stays at the API boundary.
