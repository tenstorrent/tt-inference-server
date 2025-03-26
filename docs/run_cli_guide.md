# `run.py` CLI tool


A command-line interface (CLI) tool to run various AI workflows such as model evaluation, benchmarking, and server deployment, either locally or in Docker.

---

## 📦 Requirements

The `run.py` and the workflow scripts bootstraps the various required python virtual environments as needed using `venv` and `uv` (https://github.com/astral-sh/uv). There are not requirements to set up beyond having:

- Python 3.8+ (default `python3` on Ubuntu 20.04)
- python3-venv: 
```bash
$ apt install python3-venv
```

---

## 🔧 CLI Options

### Required Arguments

| Option         | Description                                                                                  |
|----------------|----------------------------------------------------------------------------------------------|
| `--model`      | **(Required)** Name of the model to run. Available choices are defined in `MODEL_CONFIGS`. |
| `--workflow`   | **(Required)** Type of workflow to run. Valid choices include:<br>`evals`, `benchmarks`, `server`, `release`, `reports` |
| `--device`     | **(Required)** Target device for execution. Valid choices include: `n150`, `n300`, `t3k`, `galaxy` |

---

### Optional Arguments

| Option                   | Description                                                                                      |
|--------------------------|--------------------------------------------------------------------------------------------------|
| `--local-server`         | Run the inference server on **localhost**.                                                      |
| `--docker-server`        | Run the inference server inside a **Docker container**.                                          |
| `--service-port`         | Set a custom service port. Defaults to `8000` or value of `$SERVICE_PORT`.                      |
| `--disable-trace-capture`| Skip trace capture requests for faster execution if traces are already captured.                |
| `--dev-mode`             | Enable **developer mode** for mounting file edits into Docker containers at run time.                                     |

> ⚠️ `--local-server` and `--docker-server` are **mutually exclusive**. You cannot specify both.

---

## 🧪 Workflow Types

Supported values for `--workflow`:

- `evals`: Run evaluation workflows using `EVAL_CONFIGS`
- `benchmarks`: Run benchmark workflows using `BENCHMARK_CONFIGS`
- `release`: Run both evaluation and benchmarking (must exist in both configs)
- `server`: Start inference server only
- `reports`: Reserved for report generation
- `tests`: Test the bounaries of model implementations (⚠️ Not implemented yet)

---

## 🛡️ Environment Secrets

Required secrets will be prompted interactively if not already available in the environment:

- `HF_TOKEN`: HuggingFace API token
- `JWT_SECRET`: JWT secret used for authentication

These are stored in a `.env` file after first entry.

---

## 🚀 Example Usage

See user guide for examples in full context.

```bash
# start vLLM server running model
python3 run.py --model Llama-3.2-1B-Instruct --device n300 --workflow server --docker-server

# run release workflow
python3 run.py --model Llama-3.2-1B-Instruct --device n300 --workflow benchmarks 

# benchmarks workflow already captured traces, can skip that
python3 run.py --model Llama-3.2-1B-Instruct --device n300 --workflow evals --disable-trace-capture
```

or 

```bash
python3 run.py --model Llama-3.2-1B-Instruct --device n300 --workflow evals --docker-server
```


```bash
python3 run.py --model Llama-3.2-1B-Instruct --device n300 --workflow benchmarks --docker-server
```

```bash
python3 run.py --model Llama-3.2-1B-Instruct --device n300 --workflow reports
```
