<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC -->

# serving_bench — quickstart

Two self-contained shell benchmark suites that run against an **already-running,
OpenAI-compatible** inference server, driven through `run.py` like any other
workflow:

- **`agentic_bench`** — a [GuideLLM](https://github.com/vllm-project/guidellm)
  agentic-shape load soak against `/v1/chat/completions`.
- **`benchmark`** — the cpp serving-bench harness
  (`tt-media-server/cpp_server/benchmarks/run_benchmarks.sh`; needs a built
  cpp_server, so `agentic_bench` is the hardware-free one).

**No Tenstorrent hardware is required** when you target a running server with
`--server-url`: the `serving_bench` path runs no `tt-smi`/topology validation.

## 1. Setup (clone → ready) — one-time

```bash
git clone https://github.com/tenstorrent/tt-inference-server.git
cd tt-inference-server
git checkout ddjukic/copy-exabox-tests-from-tt-shield   # until merged to main

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
```

`requirements-dev.txt` is **enough to launch `run.py`**. On first run, `run.py`
bootstraps `uv` and auto-provisions the heavier venvs it needs (the v2 runner
venv, plus a per-suite venv that installs `guidellm`) — you do **not** pip-install
runtime deps yourself. These steps need internet access.

No secrets are needed for `serving_bench`: `HF_TOKEN` is only required with
`--docker-server`, and `JWT_SECRET` only for `--workflow server --docker-server`.
`run.py` writes a `.env` on first run.

## 2. Provide a server

`serving_bench` benchmarks whatever `--server-url` points at. Without TT hardware,
start any OpenAI-compatible server — e.g. a tiny ungated model on CPU:

```bash
pip install vllm
vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 8000
```

## 3. Run

```bash
MODEL=Qwen/Qwen2.5-0.5B-Instruct \
python run.py \
  --model meta-llama/Llama-3.1-8B-Instruct --device gpu \
  --workflow serving_bench --serving-bench-suites agentic_bench \
  --server-url http://127.0.0.1 --service-port 8000 \
  --limit-samples-mode smoke-test
```

- `--model` / `--device` must be a registered `MODEL_SPECS` combo; for
  `serving_bench` it is only **report metadata**, not the model actually
  benchmarked. `Llama-3.1-8B-Instruct` / `gpu` is a valid pair.
- The `MODEL` env var is the model your server actually serves: GuideLLM sends it
  in each request **and** downloads its HF tokenizer to count tokens — set it to
  match your server.
- Omit `--serving-bench-suites` to run both suites.

## limit-samples-mode presets

`--limit-samples-mode` selects a knob preset (see `presets.py`):

| mode         | DURATION | TARGET_CONCURRENCY | notes                  |
|--------------|----------|--------------------|------------------------|
| `smoke-test` | 30s      | 2                  | + small token shape    |
| `ci-commit`  | 120s     | 8                  |                        |
| `ci-nightly` | 3600s    | 32                 | full soak              |
| `ci-long` / unset | — | —                  | each suite's defaults.env |

Any knob you export yourself (e.g. `DURATION=60 ...`) overrides the preset:
precedence is **caller env > preset > suite defaults.env**.

## See the command without running anything

`agentic_bench.sh` has a `--dry-run` that prints the `guidellm` command and exits
— no server, no setup:

```bash
test_module/serving_bench/agentic_bench/agentic_bench.sh \
  --target http://localhost:8000 --model Qwen/Qwen2.5-0.5B-Instruct \
  --duration 30 --target-concurrency 2 --dry-run
```
