# Running evals and agentic evals

This guide covers running the `evals` and `agentic` workflows against an
already-running inference server. The examples use `GLM-5.3` on
`super_cluster`, but the same flow applies to any model that has eval tasks in
`evals/eval_config.py`.

Both workflows are client-side: `run.py` does not start a model server, it only
sends requests to the endpoint you pass with `--server-url`.

## Get the repository

The GLM configuration and the image bootstrap script live on a feature branch,
so clone and check that branch out:

```bash
git clone git@github.com:tenstorrent/tt-inference-server.git
cd tt-inference-server
git checkout ipastalTT/glm-5.3-c8
```

Submodules are not needed for evals or agentic evals, a plain clone is enough.

## Prerequisites

- An inference server that is already up and reachable at `--server-url`.
- An API key for that server, exported as `API_KEY`. This report assumes
  we will use an API_KEY but modifications can be made to test with 
  additional headers as well.
- Python 3.8 or newer to run `run.py`, with **PyYAML installed in that
  interpreter**. This is the one dependency `run.py` needs before it can
  bootstrap anything else, and on a clean machine it is usually missing:

  ```bash
  python3 -c "import yaml" || pip install pyyaml
  ```

- Docker with Compose (agentic only). See
  [Docker network configuration](#docker-network-configuration-required-for-agentic)
  below, this is a required host change for agentic runs at high concurrency.
  With c=8 this is not needed.
- Nothing else needs to be installed by hand. `run.py` creates the Python
  virtual environments it needs under `.workflow_venvs/` on first use
  (`.venv_evals_common` for evals, `.venv_evals_agentic` for agentic). The first
  run therefore spends several minutes building environments before any request
  is sent, which is expected and only happens once.
- `HF_TOKEN` exported is recommended. It is not required for the endpoint
  itself, but eval datasets and agentic task packages are fetched from Hugging
  Face and anonymous access gets rate limited. GPQA is gated so you will need it
  for this dataset

## Commands

Export your API key. Ask the model owner for one if you do not have it,
and keep it out of committed files and shared logs:

```bash
export API_KEY=sk-tt-...
```

Standard evals (lm-eval based, GPQA Diamond for Kimi,~10-15mins):

```bash
python run.py \
  --model GLM-5.3 \
  --workflow evals \
  --device super_cluster \
  --server-url https://<endpoint>:443 \
  --skip-system-sw-validation \
  --dev-mode
```
Benchmarks (vllm bench serve)

```bash
python run.py \
  --model GLM-5.3 \
  --workflow benchmarks \
  --device super_cluster \
  --server-url https://<endpoint>:443 \
  --skip-system-sw-validation \
  --dev-mode
```

For the optional real-text benchmark, see
[Added: custom LongBench benchmark](#added-custom-longbench-benchmark) below.

AgentX (agentic traces)

```bash
python run.py \
  --model GLM-5.3 \
  --workflow agentic_traces \
  --device super_cluster \
  --server-url https://<endpoint>:443 \
  --skip-system-sw-validation \
  --dev-mode
```

Agentic evals (Harbor based, Terminal Bench 2.1 for Kimi, ~2-2.5 hours):

```bash
python run.py \
  --model GLM-5.3 \
  --workflow agentic \
  --device super_cluster \
  --server-url https://console.tenstorrent.com:443 \
  --skip-system-sw-validation \
  --dev-mode
```

**Note that in case of a localhost run without API_KEY you should target the http endpoint, otherwise the workflow might not reach the server**

Both commands run as written, provided the prerequisites above are met. The
things that most often stop a first run are a `python` without PyYAML, a missing
`API_KEY`, and, for agentic, an unmodified Docker address pool configuration (in
case of concurrency>30) 
Long first-run delays are normal, that is venv creation, not a hang.

If you cancel an agentic run before it finishes you must delete the containers and the networks manually, since they occupy slots in network address pool indefinetely.

```bash
docker ps -q | xargs -r docker stop
docker network prune -f
```

### What the flags do

| Flag | Meaning |
| --- | --- |
| `--model` | Short model name, must be a key in the model specs (`GLM-5.3` maps to `zai-org/GLM-5.3`). You can also pass `zai-org/GLM-5.3` directly instead |
| `--workflow` | `evals` runs the lm-eval tasks; `agentic` runs the `EVALS_AGENTIC` tasks. They are separate runs, you cannot get both from one invocation. Unless you run release workflow |
| `--device` | Target hardware, `super_cluster` for the Blackhole Super-Cluster. Used for report labelling and spec selection. |
| `--server-url` | Base URL of the running OpenAI-compatible server. Mutually exclusive with `--local-server` and `--docker-server`. |
| `--skip-system-sw-validation` | Skips the tt-smi / tt-topology host checks, which are irrelevant when the hardware lives behind a remote endpoint. |
| `--dev-mode` | Selects the dev model catalog under `workflows/model_specs/dev/`. |

Useful extras: `--limit-samples-mode smoke-test` for a fast sanity run ( I don't recommend this at this stage but letting you know),
`--reset-venvs` if a virtual environment gets into a bad state.

`API_KEY` is picked up automatically and forwarded as `OPENAI_API_KEY` to
lm-eval and to the agentic harness, together with `OPENAI_BASE_URL` derived from
`--server-url`.

## Choosing which agentic benchmarks run

The `agentic` workflow runs whatever agentic tasks are enabled for the model in
[`evals/eval_config.py`](../evals/eval_config.py). For `GLM-5.3` Terminal Bench 
2.1 and Tau3 banking are present.

You can select which agentic eval to run as in the following commands:

For tau3
```bash
python run.py \
  --model GLM-5.3 \
  --workflow agentic \
  --agentic-benchmark tau3 \
  --device super_cluster \
  --server-url https://<endpoint>:443 \
  --skip-system-sw-validation \
  --dev-mode
```

And terminal bench 2.1

```bash
python run.py \
  --model GLM-5.3 \
  --workflow agentic \
  --agentic-benchmark tb2.1 \
  --device super_cluster \
  --server-url https://<endpoint>:443 \
  --skip-system-sw-validation \
  --dev-mode
```

Enabling them changes what the run needs in three ways:

- Both benchmarks pull their own images, so re-run the image pre-download after
  uncommenting.
- Tau3 creates two containers per trial rather than one, which roughly doubles
  the number of Docker networks and the required slots, so we run it at concurrency
  **4**.

## Pre-download the agentic Docker images

**This is mainly for downloading SWEBench so you can skip this part if you are running 
Terminal Bench 2.1 and tau3. However I leave these instructions here in case you encounter issues with docker images.**

Agentic trials pull their task images on demand, and a pull that is slow or rate
limited will fail the trial rather than just delay it. Pre-pull the images once
before the first agentic run.

Log in to Docker Hub first with a PAT. The Terminal Bench and SWE-bench images
come from Docker Hub, where anonymous pulls are rate limited well below what a
full image set needs:

```bash
docker login -u <docker-hub-username>
# paste a personal access token from https://app.docker.com/settings/personal-access-tokens
```

Then run the bootstrap script from the repository root:

```bash
./scripts/bootstrap_agentic_docker_images.sh --max-workers 8
```

That works as written. The script creates the `EVALS_AGENTIC` venv if it does
not exist yet, then pulls the images with eight concurrent workers instead of
the default four. Any option after the script name is forwarded to
`scripts/pull_agentic_docker_images.py`, and for `--max-workers` the value you
pass overrides the script's default. 


Two things to know before you start it:

- **It pulls a lot.** The script covers Terminal Bench 2.0 and 2.1 (for the most part they are the same images), Tau3-bench
  base images, and the complete SWE-bench Verified set for both the agent and
  the scoring harness. The SWE-bench portion alone is hundreds of gigabytes.
  Images already present locally are skipped unless you pass `--force`.
- **The wrapper is all-or-nothing.** It hardcodes
  `--benchmark terminal-bench-2 --benchmark tau3-bench --benchmark swe-bench`
  and appends your arguments after them. Because `--benchmark` collects values
  into a list instead of replacing them, passing
  `--benchmark terminal-bench-2` yourself adds a fourth entry rather than
  narrowing the selection, and all three families are still pulled.

If you left the Kimi config as-is, Terminal Bench is the only benchmark that
will run, so the Tau3 and SWE-bench images are dead weight. To pull only what
the model actually uses, skip the wrapper and call the Python script directly
with the agentic venv's interpreter:

```bash
.workflow_venvs/.venv_evals_agentic/bin/python \
  scripts/pull_agentic_docker_images.py \
  --benchmark terminal-bench-2 --max-workers 8
```

That interpreter only exists after the venv has been created, which normally
happens on your first agentic run. To create it up front without pulling
anything, run the same bootstrap step the wrapper performs:

```bash
python3 -c "
from workflows.bootstrap_uv import bootstrap_uv
from workflows.workflow_types import WorkflowVenvType
from workflows.workflow_venvs import VENV_CONFIGS
bootstrap_uv()
VENV_CONFIGS[WorkflowVenvType.EVALS_AGENTIC].setup(model_spec=None)
"
```

## Docker network configuration

**Again this part is not needed at concurrency 8**.

The agentic workflow runs Terminal Bench through Harbor, and Harbor starts one
Docker Compose project per trial. Each Compose project creates its own bridge
network. If a config runs with `n_concurrent_trials=64` for example, up to 64 networks
exist at the same time, plus whatever else is already running on the host.

Docker's built-in address pools only yield about 32 usable
networks by default, so a run at this concurrency exhausts them and trials start
failing with:

```text
could not find an available, non-overlapping IPv4 address pool among the
defaults to assign to the network
```

The fix is to widen the pools in `/etc/docker/daemon.json`. This is the config
currently in use on the run host:

```json
{
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    },
    "default-address-pools": [
        { "base": "172.20.0.0/16", "size": 24 },
        { "base": "172.21.0.0/16", "size": 24 }
    ]
}
```

Each `/16` base carved into `/24` subnets gives 256 networks, so the two pools
provide 512, which is comfortably above what a 64-trial run needs.

Two things to be careful about:

- Keep the existing keys in the file. The `runtimes` block above is unrelated to
  evals, it just already existed on that host, and overwriting the file with
  only `default-address-pools` would break it.
- Make sure `172.20.0.0/16` and `172.21.0.0/16` do not collide with subnets your
  host actually needs to reach, for example VPN or lab networks. Pick different
  bases if they do.
- Also it is recommended to run `docker prune network -f` before restarting.
Apply and verify:

```bash
sudo systemctl restart docker
docker network create pool-check && docker network inspect pool-check \
  --format '{{ (index .IPAM.Config 0).Subnet }}' && docker network rm pool-check
```

The printed subnet should be a `/24` inside one of the configured bases.
Restarting the daemon stops running containers, so do it between runs, not
during one.

## Where results land

Both workflows write under `workflow_logs/` in the repository root (or under
`$CACHE_ROOT/workflow_logs` if `CACHE_ROOT` is set)

## Troubleshooting

- **Run exits immediately with an unknown model error.** The model is only in
  the dev catalog, add `--dev-mode`.
- **401 or 403 from the endpoint.** `API_KEY` was not exported into the same
  shell that ran `run.py`. Note that prefixing the variable on the command line
  only applies to that single invocation.
- **Agentic trials fail in bursts with network errors.** Almost always address
  pool exhaustion, see the Docker section above. Check for leftover networks
  from a previous crashed run with `docker network ls` and clean them up with
  `docker network prune`.
- **Agentic run refuses to start or reports confusing pre-existing results.**
  Rename `workflow_logs` as described above.
- **Trials fail to pull images, or pulls return `toomanyrequests`.** Run
  `docker login` and pre-pull with
  [`scripts/bootstrap_agentic_docker_images.sh`](#pre-download-the-agentic-docker-images).
- **`ModuleNotFoundError: No module named 'yaml'`.** `run.py` was started with an
  interpreter that does not have PyYAML, see the prerequisites.

---

# Evaluation campaign additions and changes

> [!NOTE]
>
> This section summarizes campaign additions and benchmark fixes applied to
> [the base branch, `ipastalTT/glm-5.3-c8`](https://github.com/tenstorrent/tt-inference-server/tree/ipastalTT/glm-5.3-c8).
> They add model validation and repair benchmark failures.
> Client requests, test cases, sampling settings, concurrency, and scoring rules
> remain unchanged. Additional tests require explicit selection. The Tau3 fix
> pins a working benchmark revision so the existing test can run.

<a id="added-custom-longbench-benchmark"></a>
<a id="custom-longbench-benchmark"></a>

## 1. Custom LongBench benchmark

Run the GLM-5.3 performance sweep with real-text prompts from LongBench V1/V2.
This campaign addition is also present in the base branch at
[`b14327dc`](https://github.com/tenstorrent/tt-inference-server/tree/b14327dc3abb5531a8a111ebfb90a133f556e5ac).

- [`datasets/custom-longbench/`](../datasets/custom-longbench/README.md) supplies
  368 prompts across all 12 input lengths, from 128 to 255,872 tokens.
- [`llm_module/custom_longbench.py`](../llm_module/custom_longbench.py),
  `build_longbench_configs()`, selects prompts for each input length.
  It preserves the 28 sweep conditions, output lengths, concurrency, and request
  counts. Insufficient rows cause an error; the client does not duplicate prompts.
- [`test_module/llm_tests/llm_benchmark_tests.py`](../test_module/llm_tests/llm_benchmark_tests.py),
  `run_llm_bench()`, selects this path only with `--benchmark custom-longbench`.
  The default random-input benchmark and its acceptance targets stay unchanged.
  Random-input targets do not grade custom-input results.

From the repository root:

```bash
python run.py \
  --model GLM-5.3 \
  --workflow benchmarks \
  --benchmark custom-longbench \
  --dataset-path "$PWD/datasets/custom-longbench/custom-longbench.jsonl" \
  --device super_cluster \
  --server-url https://<endpoint>:443 \
  --skip-system-sw-validation \
  --dev-mode
```

Use a fresh `CACHE_ROOT` for each run. Compare matching sweep conditions with
prefix caching disabled on both Prefill and Decode for both datasets.
The client does not change server cache settings.
See [Custom LongBench](custom_longbench.md) for dataset preparation and reports.

<a id="glm-53-swe-bench"></a>

## 2. GLM-5.3 SWE-bench

Add an optional SWE-bench Verified evaluation using the existing mini-swe-agent
workflow.

- [`reference_config/evals/eval_config.py`](../reference_config/evals/eval_config.py)
  adds `swe_bench_verified` to GLM-5.3 with `requires_explicit_selection=True`.
  It reuses the GLM-5.2 SWE-bench recipe with concurrency set to 8.
- [`test_module/llm_tests/agentic_eval_tests.py`](../test_module/llm_tests/agentic_eval_tests.py),
  `_select_agentic_tasks()` and `_filter_agentic_tasks_by_benchmark()`, require
  `--agentic-benchmark swebench` or the full task name to select it.
  Unset or `all` still selects only Terminal-Bench and Banking for GLM-5.3.
- [`run.py`](../run.py) and [`run_workflows.py`](../run_workflows.py) update the CLI
  help to explain this selection rule.
  [Selection tests](../tests/test_module/llm_tests/test_glm53_swebench.py) cover
  the unchanged default campaign and the new explicit selection.

```bash
python run.py \
  --model GLM-5.3 \
  --workflow agentic \
  --agentic-benchmark swebench \
  --device super_cluster \
  --server-url https://<endpoint>:443 \
  --skip-system-sw-validation \
  --dev-mode
```

<a id="banking-evaluator-dependency"></a>

## 3. Tau3 Banking runtime pin

Harbor did not pin the Tau3 Banking runtime source. New image builds picked up
an upstream bug. We pin the runtime to the commit before that bug.
Client requests, test cases, model settings, and scoring rules are unchanged.

- tt-inference-server runs Tau3 Banking through Harbor. The
  [Tau3 implementation and evaluator](https://github.com/dcvijeticTT/harbor/blob/a7f80f9baf674909b98da952e102b37b0a846b0d/adapters/tau3-bench/README.md#overview)
  live in `sierra-research/tau2-bench`; the Python package is still named `tau2`.
- Harbor's [Dockerfile](https://github.com/dcvijeticTT/harbor/blob/a7f80f9baf674909b98da952e102b37b0a846b0d/adapters/tau3-bench/src/tau3_bench/task-template/environment/Dockerfile)
  cloned the latest default branch. [PR #523](https://github.com/sierra-research/tau2-bench/pull/523),
  merged September 10, 2026, made the evaluator import voice code requiring
  `websockets`. The installed `knowledge` extra omitted that dependency, so
  evaluator import failed even without a tt-inference-server code change.
- [`llm_module/agentic/banking_docker.py`](../llm_module/agentic/banking_docker.py)
  pins both the evaluator and user-simulator images to
  [`b351ed5`](https://github.com/sierra-research/tau2-bench/commit/b351ed5f9281d4bdfa5629262f54c8781da0d5be),
  the preceding commit. It checks evaluator import during each image build.
  The extra `websockets` installation is removed.
- [`llm_module/agentic/harbor.py`](../llm_module/agentic/harbor.py), `run()`,
  applies the pin automatically to Banking tasks in Tau3 Docker runs.
  Other tasks are unchanged. Remove any external Banking Docker wrapper.

## 4. AgentX 1M corpus

TT's [instructions for the full 1M corpus](https://github.com/tenstorrent/tt-inference-server/pull/5163#issuecomment-5693020667)
remove `_256k` from the dataset name. This fork exposes that choice without
editing the default configuration.

- Add `--agentic-traces-corpus 1m` to the GLM-5.3 AgentX command above.
- The default remains `semianalysis_cc_traces_weka_062126_256k`.
  Explicit `256k` selects the same run.
- `1m` selects `semianalysis_cc_traces_weka_062126`. Only the dataset and result
  label change. Lane 8, duration, seed, sampling requests, warmup, and scoring
  remain unchanged. Model context length and corpus selection are separate.
- [`run.py`](../run.py), [`run_workflows.py`](../run_workflows.py), and their
  runtime/engine forwarding pass the selection to
  [`run_agentic_traces()`](../test_module/llm_tests/agentic_traces_tests.py).
  It selects a copy of the run specification; the default registry is unchanged.
- Use a fresh `CACHE_ROOT` for each corpus. The existing native JSON records
  `public_dataset`, lane count, duration, seed, and the InferenceX revision.

## 5. PoC environment check

TT reloads the repository's `.env` after startup. An old `OPENAI_BASE_URL` can
send Banking simulator requests to a different server than `--server-url`.

[`scripts/check_poc_environment.py`](../scripts/check_poc_environment.py) checks
both the shell environment and `.env`. It rejects conflicting endpoints, output
paths, model/target overrides, Harbor settings, and AIPerf overrides. It does not
change either source or print credential values. Run it before each PoC client:

```bash
python scripts/check_poc_environment.py \
  --server-url "$SERVER_URL" --cache-root "$CACHE_ROOT"
```

The check writes `poc-environment.json` under `CACHE_ROOT`. Normal credential
settings, including `HF_TOKEN` and `API_KEY`, remain available to TT.
