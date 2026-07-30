# Running evals and agentic evals

This guide covers running the `evals` and `agentic` workflows against an
already-running inference server. The examples use `Kimi-K2.7-Code` on
`super_cluster`, but the same flow applies to any model that has eval tasks in
`evals/eval_config.py`.

Both workflows are client-side: `run.py` does not start a model server, it only
sends requests to the endpoint you pass with `--server-url`.

## Get the repository

The Kimi configuration and the image bootstrap script live on a feature branch,
so clone and check that branch out:

```bash
git clone git@github.com:tenstorrent/tt-inference-server.git
cd tt-inference-server
git checkout ipastalTT/dl-docker-images
```

Submodules are not needed for evals or agentic evals, a plain clone is enough.

## Prerequisites

- An inference server that is already up and reachable at `--server-url`.
- An API key for that server, exported as `API_KEY`.
- Python 3.8 or newer to run `run.py`, with **PyYAML installed in that
  interpreter**. This is the one dependency `run.py` needs before it can
  bootstrap anything else, and on a clean machine it is usually missing:

  ```bash
  python3 -c "import yaml" || pip install pyyaml
  ```

- Docker with Compose (agentic only). See
  [Docker network configuration](#docker-network-configuration-required-for-agentic)
  below, this is a required host change for agentic runs at high concurrency.
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

Export your console API key. Ask the model owner for one if you do not have it,
and keep it out of committed files and shared logs:

```bash
export API_KEY=sk-tt-...
```

Standard evals (lm-eval based, GPQA Diamond for Kimi,~10-15mins):

```bash
python run.py \
  --model Kimi-K2.7-Code \
  --workflow evals \
  --device super_cluster \
  --server-url https://console.tenstorrent.com:443 \
  --skip-system-sw-validation \
  --dev-mode
```

Agentic evals (Harbor based, Terminal Bench 2.1 for Kimi, ~2-2.5 hours):

```bash
python run.py \
  --model Kimi-K2.7-Code \
  --workflow agentic \
  --device super_cluster \
  --server-url https://console.tenstorrent.com:443 \
  --skip-system-sw-validation \
  --dev-mode
```
Both commands run as written, provided the prerequisites above are met. The
things that most often stop a first run are a `python` without PyYAML, a missing
`API_KEY`, and, for agentic, an unmodified Docker address pool configuration or
a `workflow_logs` directory left over from the previous run. Long first-run
delays are normal, that is venv creation, not a hang.

If you cancel an agentic run before it finishes you must delete the containers and the networks manually:

```bash
docker ps -q | xargs -r docker stop
docker network prune -f
```

### What the flags do

| Flag | Meaning |
| --- | --- |
| `--model` | Short model name, must be a key in the model specs (`Kimi-K2.7-Code` maps to `moonshotai/Kimi-K2.7-Code`). |
| `--workflow` | `evals` runs the lm-eval tasks; `agentic` runs the `EVALS_AGENTIC` tasks. They are separate runs, you cannot get both from one invocation. Unless you run release workflow |
| `--device` | Target hardware, `super_cluster` for the Blackhole Super-Cluster. Used for report labelling and spec selection. |
| `--server-url` | Base URL of the running OpenAI-compatible server. Mutually exclusive with `--local-server` and `--docker-server`. |
| `--skip-system-sw-validation` | Skips the tt-smi / tt-topology host checks, which are irrelevant when the hardware lives behind a remote endpoint. |
| `--dev-mode` | Selects the dev model catalog under `workflows/model_specs/dev/`. `Kimi-K2.7-Code` currently only exists there, so the run fails without it. |

Useful extras: `--limit-samples-mode smoke-test` for a fast sanity run ( I don't recommend this at this stage but letting you know),
`--reset-venvs` if a virtual environment gets into a bad state.

`API_KEY` is picked up automatically and forwarded as `OPENAI_API_KEY` to
lm-eval and to the agentic harness, together with `OPENAI_BASE_URL` derived from
`--server-url`.

## Choosing which agentic benchmarks run

The `agentic` workflow runs whatever agentic tasks are enabled for the model in
[`evals/eval_config.py`](../evals/eval_config.py). For `Kimi-K2.7-Code` only
Terminal Bench 2.1 is enabled today. Tau3 banking and SWE-bench Verified are
present but commented out, so an agentic run currently covers Terminal Bench
only.

To run the full agentic suite, uncomment those two blocks:
[`evals/eval_config.py` lines 509-613](../evals/eval_config.py#L509-L613), which
are the `tau3_bench_banking` and `swe_bench_verified` entries in the
`moonshotai/Kimi-K2.7-Code` config. Line numbers drift as the file changes, so
search for those two task names inside the Kimi `EvalConfig` if the range no
longer matches.

Enabling them changes what the run needs in three ways:

- Both benchmarks pull their own images, so re-run the image pre-download after
  uncommenting.
- Tau3 creates two containers per trial rather than one, which roughly doubles
  the number of Docker networks in use and makes the address pool change below
  more important, not less.
- The run gets substantially longer. SWE-bench Verified is 500 instances with a
  separate scoring pass after patch generation.

## Pre-download the agentic Docker images

**This is mainly for downloading SWEBench so you can skip this part if you are running 
Terminal Bench 2.1 to be honest.**

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

## Docker network configuration (required for agentic)

The agentic workflow runs Terminal Bench through Harbor, and Harbor starts one
Docker Compose project per trial. Each Compose project creates its own bridge
network. The Kimi config runs `n_concurrent_trials=64`, so up to 64 networks
exist at the same time, plus whatever else is already running on the host.

Docker's built-in address pools only yield about 32 usable (31+1 for the host bridge)
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

## Tip: rotate `workflow_logs` between agentic runs

After every agentic run, move the log directory aside before starting the next
one:

```bash
mv workflow_logs workflow_logs_run1
```

The agentic harness will not start cleanly on top of an existing populated
`workflow_logs` tree, and reusing it also mixes artifacts from different runs
into the same Harbor job directories, which makes the resulting report hard to
attribute. Renaming is the reliable way to avoid both problems, and it keeps the
previous run's results intact for comparison.

Use a descriptive suffix so runs stay identifiable later, for example
`workflow_logs_kimi_tb21_run1`. Only the plain `workflow_logs` name is
recreated by the next run, the renamed directories are ignored.

## Where results land

Both workflows write under `workflow_logs/` in the repository root (or under
`$CACHE_ROOT/workflow_logs` if `CACHE_ROOT` is set):

- `workflow_logs/run_logs/run_<run_id>.log` is the top-level driver log, this is
  the first place to look when a run dies early.
- `workflow_logs/reports_output/evals/` and `workflow_logs/reports_output/agentic/`
  hold the generated markdown reports, named
  `report_<model>_<timestamp>.md`, plus a `data/` folder with the matching JSON.
- Raw per-task artifacts sit in a per-run subdirectory such as
  `reports_output/agentic/Kimi-K2.7-Code_super_cluster_agentic/`, which contains
  the Harbor job tree with each trial's transcript and verifier output.

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

## Related documents

- [Agentic evaluation container lifecycle](agentic_container_lifecycle.md) for
  how Harbor, mini-swe-agent, and the SWE-bench harness create containers.
- [Workflows user guide](workflows_user_guide.md) for the full `run.py` CLI.
