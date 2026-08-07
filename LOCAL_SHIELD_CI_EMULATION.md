# Emulating tt-shield `release` CI locally on the BH Galaxy

Goal: run exactly what tt-shield's `release` workflow runs (benchmarks + evals +
acceptance gate), on this box, without GitHub Actions — so we can catch failures
before spending a CI dispatch. Verified against tt-shield and tt-inference-server
source on 2026-08-06.

---

## 1. What tt-shield `release` actually runs

The whole thing is **one command**. There is no separate warmup / serve /
benchmark / eval / gate job — CI's hardware job (`workflow_run-tests-with-inference-server.yml`)
runs a single:

```
python3 run.py --model <M> --workflow release --device <D> \
    --impl <I> --docker-server --dev-mode [--ci-mode] [--override-docker-image <img>]
```

inside a container. Everything downstream (server bring-up, health/warmup,
benchmark sweep, eval sweep, acceptance verdict, report JSON/MD) is produced by
that one `run.py` call.

Job graph (on-dispatch, `workflow=release`):
1. `determine-server-type` → FORGE engine ⇒ `forge-media-inference-server`
2. `resolve-shas` (skipped if `docker-image` given)
3. `build-forge-media-inference-server` — builds the forge image; the
   `tt-forge-version-override` input is applied HERE (rewrites the forge
   requirements / resolves a tt-xla ref via `override_from_xla_ref.py`), not at
   run time.
4. `run-tests` on `runs-on: bh-galaxy` — the single `run.py --workflow release`.
5. `ai-run-summary` / issue-comment / Slack — reporting only (infra-only).

### `release` decomposes (inside run.py → v2 engine) into children:
`tt-inference-server-v2/workflow_module/workflows.py:375-470`
- LLM/VLM: `("evals", "benchmarks", "spec_tests")`
- **plus `agentic`** — appended automatically iff the model has ANY
  `EVALS_AGENTIC` task (`_has_agentic_tasks`).
All children accumulate into one `ReportSchema`; the acceptance gate runs once
over the combined schema.

---

## 2. The acceptance gate (what "green" means)

Live path: `tt-inference-server-v2/report_module/acceptance_criteria.py`
(NOT the v1 `workflows/acceptance_criteria.py`, which is dead code for this path —
imported only by `tests/`). `accepted = (len(blockers) == 0)`; process rc=0 iff
`accepted and not failed_tasks`.

- **Benchmarks** pass if ANY tier (functional/complete/target) passes. `NA` counts
  as passing. Tiers are computed from `theoretical` targets ×/÷
  `TIER_MULTIPLIERS = {functional:10, complete:2, target:1}`
  (`test_module/_test_common/target_check.py`). A **zero** target → NA → passes
  vacuously.
- **Evals** block if a row self-reports `success=False`, OR `accuracy_check`==FAIL,
  OR `accuracy_check` missing. `accuracy_check`==NA is harmless. Model-spec
  `known_issues` waivers (matching workflow_type + task_name) demote a blocker to
  a non-fatal waiver.
- **`status: EXPERIMENTAL` does NOT waive failures.** The v2 checker takes no
  model_status; status is metadata only. (The v1 `required_target_tiers` map that
  made EXPERIMENTAL informational is dead code here.)

Thresholds live at:
- benchmarks: `benchmarking/benchmark_targets/model_performance_reference.json`
  (`model → device(lowercased) → [ {isl,osl,max_concurrency,targets.theoretical} ]`)
- evals: inline in `evals/eval_config.py` (`EvalTaskScore`: `gpu_reference_score`,
  `published_score`, `tolerance=0.05`). `evals/eval_targets/model_accuracy_reference.json`
  is image/video-only, NOT LLMs.

---

## 3. Local emulation — the command (verified string forms)

`--impl` takes the **hyphenated** `impl_name` (`forge-vllm-plugin`), NOT the
underscore `impl_id`. `--device` takes the lowercased enum name
(`blackhole_galaxy`). Both are argparse `choices=`, so a wrong form fails fast.

### Mode A — CI-faithful (run.py builds/runs its own Docker server)
```bash
cd /data/ssalice/temp/tt-inference-server
export JWT_SECRET=... HF_TOKEN=hf_... AUTOMATIC_HOST_SETUP=1
export PERSISTENT_VOLUME_ROOT=/localdev/persistent-volume HOST_HF_HOME=/localdev/hf_home
python3 run.py \
  --model <Qwen3-32B | Devstral-2-123B-Instruct-2512> \
  --workflow release \
  --device blackhole_galaxy \
  --impl forge-vllm-plugin \
  --docker-server --dev-mode --ci-mode \
  --override-docker-image <prebuilt-forge-image>   # or omit to build
```
`--ci-mode` present ⇔ CI's default `run-full-evals=false` (shrinks eval sample
sizes). Drop it to reproduce a full-eval run.

### Mode B — against an already-running server (fastest iteration)
Start your forge server yourself (as we've been doing), then:
```bash
cd /data/ssalice/temp/tt-inference-server
export HF_TOKEN=hf_...                 # REQUIRED for --workflow release
export API_KEY=<same-key-server-uses>  # FORGE uses a LITERAL bearer, not JWT
python3 run.py --dev-mode --model <M> --impl forge-vllm-plugin \
  --device blackhole_galaxy --workflow release --service-port 8000
# add --server-url http://host:port if not localhost
```
No `--docker-server`/`--local-server` ⇒ run.py targets the running server.

### Mode C — children separately (recommended for bring-up)
`--workflow benchmarks` and `--workflow evals` are standalone; **`evals` never
pulls in the agentic child** (only `release` does). So to emulate release
evals+benchmarks WITHOUT the heavy agentic harness:
```bash
python3 run.py --dev-mode --model <M> --impl forge-vllm-plugin \
  --device blackhole_galaxy --workflow benchmarks --tools vllm --service-port 8000
python3 run.py --dev-mode --model <M> --impl forge-vllm-plugin \
  --device blackhole_galaxy --workflow evals --service-port 8000
```
Downsample: `--limit-samples-mode smoke-test` (first task, `--limit 3`) or
`--eval-samples '{"task":[ids]}'`.

### Under the hood each child shells out to:
- **evals**: `<venv>/bin/lm_eval --tasks <t> --model local-completions
  --model_args model=<hf_repo>,base_url=<host:port>/v1/completions,tokenizer_backend=huggingface,num_concurrent=N
  --gen_kwargs stream=False --num_fewshot .. --batch_size .. --log_samples [--limit N]`
  (lm-eval is a TT fork; bearer exported as `OPENAI_API_KEY`).
- **benchmarks**: `vllm bench serve --backend openai-chat --endpoint
  /v1/chat/completions --model <hf_repo> --dataset-name random --max-concurrency c
  --num-prompts n --random-input-len isl --random-output-len osl
  --percentile-metrics ttft,tpot,itl,e2el --save-result`. Online only.

### Reports land at
`workflow_logs/reports_output/<workflow>/report_<id>.{md,json}` (or
`$CACHE_ROOT/...`). Verdict keys in the JSON metadata: `acceptance_criteria`
(bool), `acceptance_blockers`, `acceptance_summary_markdown`. Run log:
`workflow_logs/run_logs/run_<id>.log`.

---

## 4. Prerequisites / gotchas that break a naive local repro
- `--dev-mode` MANDATORY: Qwen3-32B / Devstral live only in the dev catalog
  (`workflows/model_specs/dev/cnn.yaml`). Omitting it silently uses the prod
  catalog and the model won't resolve.
- `HF_TOKEN` required for `--workflow release` (release isn't a client-side
  workflow, so `handle_secrets` demands it even without launching a server).
- Auth: FORGE clients send a **literal** bearer (`VLLM_API_KEY`→`API_KEY`→
  `"your-secret-key"`), NOT a JWT. Mismatch = 401. (`JWT_SECRET` only matters for
  `--workflow server --docker-server`.)
- First run auto-provisions uv venvs (lm-eval fork, vllm tool, and the heavy
  EVALS_AGENTIC venv if the model has agentic tasks).
- `--device-id` from tt-smi / `find_pci_id.py`; omit on a dedicated single-device
  box.

---

## 5. Model-specific consequences (why local emulation matters here)

### Devstral-2-123B (mesh 4x8, ctx 1024, batch 16, gmu 0.15)
- Has 3 `EVALS_AGENTIC` tasks (swe_bench_verified, swe_bench_multilingual,
  terminal_bench_2), all `min_context_required = 192K`.
- Standard `evals` child = clean no-op (agentic tasks filtered out).
- **`release` appends the `agentic` child, and `AgenticWorkflow` does NOT honor
  `min_context_required`** — it will try to run SWE-bench/Terminal-Bench against a
  1024-token server (needs Docker + SWE-agent/harbor + huge contexts) and will
  almost certainly fail → can flip the release verdict to FAIL. There is no
  in-`release` flag to skip agentic. **Use Mode C (benchmarks + evals separately)
  for Devstral, or expect the agentic child to block `release`.**
- Benchmarks: only (128,128) survives the isl+osl≤1024 filter; placeholder-zero
  target → passes vacuously.

### Qwen3-32B (mesh 8x4, target ctx 4096, batch 32, gmu ~0.25-0.50)
- 3 tasks (r1_aime24, r1_math500, r1_gpqa_diamond), all `EVALS_COMMON` — **NO
  agentic child**. The agentic trap does NOT apply to Qwen.
- On branch `ssalice/devstral-123b-integration` these tasks have **NO
  `min_context_required` guard**, so at ctx 4096 they WILL run. r1_* are reasoning
  tasks that emit very long generations; at max_model_len 4096 they'll hit the
  context ceiling → truncation / HTTP-400 / low scores → possible eval blocker.
  (Confirm the guard state on whichever branch drives CI; earlier work added
  40960 guards — verify they're present. With a 40960 guard, all 3 SKIP at 4096
  and the eval child is a non-blocking no-op = "green but no accuracy measured".)
- Benchmarks: **Qwen3-32B has NO `blackhole_galaxy` entry** in
  model_performance_reference.json (only galaxy/t3k/p300x2). Confirm whether a
  missing device entry → NA/pass or a "missing reference" blocker; may need a
  placeholder `blackhole_galaxy` block added before CI.

---

## 6. Deferred (per user): the batch-32 / ctx-4096 Qwen config
- Qwen3-32B DP+TP chunked-prefill test: tt-xla commit `50b19a6c4`,
  `tests/integrations/vllm_plugin/generative/test_data_tensor_parallel_generation.py`.
  Knobs: `prefill_chunk_size:128`, `min_num_seqs:1`, `prefill_batch_threshold:16`
  (all `< max_num_seqs`), greedy (`temperature=0.0, top_p=1.0`), `cpu_sampling:False`,
  opt=1, `experimental_kv_cache_dtype:bfp_bf8`.
- `LOCAL_RESULTS.md` is STALE (2026-07-30, old branch
  `ssalice/devstral-qwen-wip-07-13-2026`). Every clean full-depth Qwen run in it
  is **batch 16 @ gmu 0.20** (run 11 cleanest: FSDP `shard_weights_on_batch_axis:True`,
  16/16 coherent, 4299s). Cheat sheet puts **batch 32 @ 4096 ≈ gmu 0.50**, vs a
  TP-only weight ceiling ~0.49 → batch 32 likely needs FSDP weight sharding.
  The user's "batch 32 @ 4096 ran locally" postdates this file — get the exact
  config from the newer `[notes]` commits on `ssalice/devstral-qwen-5893` (several
  are dated after 07-30) before dispatching CI.
