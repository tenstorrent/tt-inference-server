# Common Mistakes to Avoid (Gemma 4 / QB2 workflow)

A running list of recurring mistakes made while bringing up Gemma 4 on QB2,
with the correct action. Read before launching a build or release run.

## Serving / run.py flags
- **Forgetting `--docker-server`.** Without it the release workflow silently
  polls for a server it never started (or uses an ad-hoc API endpoint). Symptom:
  first report had `server_mode: API`, or run.py "polling for non-existent
  server". → Always pass `--docker-server` for a self-managed release run.
- **Forgetting `--tt-device p300x2`.** Auto-detection raises
  `ValueError: Unable to map tt-smi board counts...`. → Pass it explicitly.
- **Forgetting `--override-docker-image` for a local-only image.** If the image
  isn't pushed to GHCR, run.py tries to pull the derived tag and fails / polls.
  → Pass `--override-docker-image <local tag>` when validating locally. (The
  nightly builds+pushes its own image, so it doesn't need this.)
- **Tool-choice flags for agentic evals.** mini-swe-agent / SWE-Bench send
  `tool_choice: "auto"`, which 400s with *"auto tool choice requires
  --enable-auto-tool-choice and --tool-call-parser to be set"* unless the server
  has both. → Now baked into the spec (`tool_call_parser_name: gemma4` +
  `enable-auto-tool-choice: true` in `vllm_args`). Don't strip these.

## Auth
- **Minting a JWT by hand inside the container.** Caused `ModuleNotFoundError:
  No module named 'jwt'` and 401s. → Let `run.py` do it: it `load_dotenv()`s
  `.env` (`JWT_SECRET`/`VLLM_API_KEY`) and sets `OPENAI_API_KEY` from the JWT.
  Just make sure `.env` exists.
- **Starting a server with `--no-auth` when the client still sends a key** (or
  vice versa). Keep auth consistent between server and eval/benchmark client.

## Image build
- **Wrong Ubuntu version.** `build_single_docker.sh` defaults to `--ubuntu-version
  20.04`, whose cold base build hits `GLIBC_2.34 not found (libzstd.so.1)`. All
  working images are 22.04. → Always pass `--ubuntu-version 22.04`.
- **Editing the shared Dockerfile for a model-specific need.** The
  `vllm.tt-metal.src.dev.Dockerfile` is used by *every* model; a "reinstall vllm
  after the plugin" hack there is not a safe no-op and risks other models. →
  Fix upstream instead (the plugin clobber was fixed in tenstorrent/vllm #433;
  we just bumped the pinned commit).
- **Assuming only the vllm layer rebuilds.** The vllm image's builder stage runs
  `build_metal.sh`, so a pruned buildkit cache means a full tt-metal recompile.
  Budget ~45–90 min when the cache is cold.

## Monitoring
- **Grepping build logs for bare `ERROR`.** BuildKit echoes RUN commands that
  literally contain `echo "ERROR: ..."`, so the monitor false-fires instantly.
  → Wait on the build **process exiting** (or the image tag appearing), or grep
  strict markers (`failed to solve`, `did not complete successfully`,
  `build_docker.sh completed successfully`).

## Device hygiene
- **Not resetting a wedged board.** After a fabric/eth-core hang, runs fail with
  *"Timed out while waiting for active ethernet core... Try resetting the
  board."* → `tt-smi -r` (warm reset all PCI devices) before relaunching.
- **Reusing a stale eval output dir.** harbor/terminal-bench raise
  `FileExistsError: Job directory ... cannot be resumed with a different
  config.` → Remove the stale
  `workflow_logs/evals_output/.../agentic/<task>/` dir before rerunning.

## Git hygiene
- **Committing build/run artifacts.** `temp_docker_build_dir_*/` and
  `workflow_logs/` are not source. Don't stage them.
- **Committing local-only docs to the remote branch.** Benchmark/triage
  writeups (including this folder) are local notes; keep them out of the PR
  unless explicitly wanted.
