# Model Readiness Workflows User Guide

The main entry point is the `run.py` command-line interface (CLI) tool to run different workflows.

`--workflow`:
- `evals`: Run evaluation tasks for given model defined in `EVAL_CONFIGS`, e.g. sends specific dataset of prompts to LLM, processes and scores output.
- `benchmarks`: Run benchmark tasks for given model defined in  `BENCHMARK_CONFIGS`, e.g. sends random data prompts to LLM, profiles output.
- `reports`: Generates reports for comparison with other hardward and validating model performance and accuracy.
- `release`: Runs evals, benchmarking,and reports workflows.
- `server`: Start inference server only [🚧 only currently implemented for `--docker-server`]

For example, the following command will start the vLLM server in a Docker container from the released Docker Image hosted on GHCR, and then run the client side benchmarks script against it:
```bash
python3 run.py --model Llama-3.2-1B-Instruct --device n150 --workflow benchmarks --docker-server
```

## Table of Contents

- [Requirements](#requirements)
  - [System Requirements](#system-requirements)
  - [Hugging Face Requirements](#hugging-face-requirements)
- [`run.py` CLI Options](#runpy-cli-options)
  - [Required Arguments](#required-arguments)
  - [Optional Arguments](#optional-arguments)
- [Serving LLMs with vLLM](#serving-llms-with-vllm)
  - [Server Workflow](#server-workflow)
    - [Docker Server](#docker-server)
    - [First Run](#first-run)
      - [Secrets](#secrets)
      - [Weights Download](#weights-download)
      - [Release Docker Images](#release-docker-images)
- [Release Workflow](#release-workflow)
- [Performance Benchmarks](#performance-benchmarks)
  - [Benchmarking Steps](#benchmarking-steps)
- [Accuracy Evaluations](#accuracy-evaluations)
- [Reports](#reports)
- [Logs](#logs)
- [Additional Documentation](#additional-documentation)

## Requirements

Using `run.py` the workflow scripts bootstraps the various required python virtual environments as needed using `venv` and `uv` (https://github.com/astral-sh/uv). With this design there are no typical python install steps such as `pip install`.

⚠️ NOTE: the first run setup of the Python virtual envs with `uv` and `venv` will take some time, up to 15 minutes in runs that install many required venvs and have low-bandwidth network speeds. This only happens once. If you have errors with a venv [file an issue](https://github.com/tenstorrent/tt-inference-server/issues), when applying a fix to the venv you should remove the specific venv to allow a clean installation.

### System Requirements

The system requirements for `run.py` and the Model Readiness Workflows are:

- Python 3.8+ (Python 3.8.10 is default `python3` on Ubuntu 20.04)
- python3-venv: likely already installed, if needed install via apt:
```bash
$ apt install python3-venv
```

### Hugging Face requirements

HF_TOKEN: for access to gated HF datasets (access your token from https://huggingface.co, go to `Settings` -> `Access Tokens`)

You will need to accept the terms for any specific gated datasets or model repositories required, e.g. https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct

## `run.py` CLI Options

### Required Arguments

| Option         | Description                                                                                  |
|----------------|----------------------------------------------------------------------------------------------|
| `--model`      | **(Required)** Name of the model to run. Available choices are defined in `MODEL_SPECS`. |
| `--workflow`   | **(Required)** Type of workflow to run. Valid choices:<br>`evals`, `benchmarks`, `release`, `reports`, `server` |
| `--device`     | **(Required)** Target device for execution. Valid choices:<br>`n150`, `n300`, `t3k`, `galaxy` |

---

### Optional Arguments

| Option                   | Description                                                                                      |
|--------------------------|--------------------------------------------------------------------------------------------------|
| `--docker-server`        | Run the inference server inside a **Docker container**.                                          |
| `--local-server`         | [🚧 not implemented yet] Run the inference server on **localhost**.                                                      |
| `--service-port`         | Set a custom service port. Defaults to `8000` or value of `$SERVICE_PORT`.                      |
| `--disable-trace-capture`| Skip trace capture requests for faster execution if traces are already captured.                |
| `--dev-mode`             | Enable **developer mode** for mounting file edits into Docker containers at run time.                                     |

---

## Serving LLMs with vLLM

Note: you can serve a model with vLLM or another Open AI API compatible inference server however you like to use the additional workflows that only send prompts to the inference server (`evals` and `benchmarks`) or process the output data (`reports`).

For example, if you run vLLM following the docs at https://github.com/tenstorrent/vllm/tree/dev/tt_metal during development, you can run the client side workflows mentioned (`evals`, `benchmarks`, `reports`, or all of them with `release`) against that already running inference server.

This section describes how to use `run.py` automation to also run the inference server (currently only vLLM for Tenstorrent hardware is supported).

**Options:**
1. `run.py` with `--docker-server`: automates using Docker run
2. Manual Docker run command: use pre-built or custom built Docker images with tt-metal and vLLM
3. custom: build tt-metal and vLLM from source

Each combination of {model_name} and {device} corresponds to a specific run. This is because this is what is running on Tenstorrent device, and loading the model weights to the device, starting the model, and compiling the kernel binaries for all different input sizes can take several minutes, e.g. ~5 minutes for 70B+ models.

### Server workflow

The `server` workflow runs the vLLM inference server for the model in detached process using either `--docker-server` or `--local-server` and exits.

Once the server is running, multiple runs of the client side workloads, i.e. `benchmarks` and `evals` can be run without tearing the model inference server down. This saves time during development.

#### Docker server

To run the inference server with docker use `--docker-server` flag:
```bash
# run inference server with docker
python3 run.py --model Llama-3.2-1B-Instruct --device n300 --workflow server --docker-server
```
Note: add `--dev-mode` flag if you are making changes to the tt-inference-server files and want those to be mounted into the Docker container and used.

```log
2025-03-26 01:29:38,051 - run_docker_server.py:113 - INFO: Created Docker container ID: 6b8c7038a44a
2025-03-26 01:29:38,051 - run_docker_server.py:114 - INFO: Access container logs via: docker logs -f 6b8c7038a44a
2025-03-26 01:29:38,051 - run_docker_server.py:115 - INFO: Docker logs are also streamed to log file: /home/<username>/tt-inference-server/workflow_logs/docker_server/vllm_2025-03-26_01-29-33_Llama-3.2-11B-Vision-Instruct_n300_server.log
2025-03-26 01:29:38,051 - run_docker_server.py:118 - INFO: Stop running container via: docker stop 6b8c7038a44a
```

You will get a log message with the container ID (`6b8c7038a44a` in the case above).

The stddout and stderr will also be streamed to a log file for this run: `/home/<username>/tt-inference-server/workflow_logs/docker_server/vllm_2025-03-26_01-29-33_Llama-3.2-11B-Vision-Instruct_n300_server.log` in the case above.

The running container can be viewed with:
```bash
docker ps -a
```

and stopped with:
```bash
# container ID was 
docker stop <container-id>
```
#### First run

##### Secrets

On first run you will be asked to enter:
- HF_TOKEN: for access to gated HF repositories (access your token from https://huggingface.co, go to `Settings` -> `Access Tokens`)
- JWT_SECRET: this is your JWT Token secret for any vLLM servers deployed (you can fill in any secret string you like)

These secrets will be stored in the `.env` file in the repo top-level and referenced there after where needed.

##### Weights download

Any model weights that are required will be default downloaded from Hugging Face using the token provided above. This happens via `workflows/setup_host.py` which created the default directory structure expected for the `--docker-server` automation.

##### Release Docker Images

Each model implementation is mapped to a pre-built release Docker Image that contains a pre-built tt-metal and vLLM source builds.
These Docker images are tested with the `release` workflow to ensure correctness for each model supported.

The Model Name -> Docker Image mapping is in the main repo README.md LLMs table: [README.md](../README.md#LLMs)

# Release workflow

For the same model device combination the `release` workflow runs in sequence:
1. `benchmarks` workflow
2. `evals` workflow
3. `tests` workflow [🚧 not implemented yet]
4. `reports` workflow

This is an added convenience so that the run on device can run both the benchmarks and evals before being shutdown and next run started. This workflow runs all workflows required to certify a model implementation on Tenstorrent hardware is working correctly and ready for release.

# Performance Benchmarks

The `benchmarks` workflow

```bash
python3 run.py --model Llama-3.2-1B-Instruct --device n300 --workflow benchmarks
```

### Benchmarking Steps

Here, we dissect the runtime logs (stdout and stderr + streamed line-by-line to `workflow_logs/run_logs`):

1. Set up workflow virtual environments:
```log
2025-03-26 19:26:52,173 - run.py:175 - INFO: model:            Llama-3.2-1B-Instruct
2025-03-26 19:26:52,173 - run.py:176 - INFO: workflow:         benchmarks
2025-03-26 19:26:52,173 - run.py:177 - INFO: device:           n300
2025-03-26 19:26:52,173 - run.py:178 - INFO: local-server:     False
2025-03-26 19:26:52,173 - run.py:179 - INFO: docker-server:    False
2025-03-26 19:26:52,173 - run.py:180 - INFO: workflow_args:    None
2025-03-26 19:26:52,173 - run.py:182 - INFO: tt-inference-server version: 0.0.4
2025-03-26 19:26:52,174 - run_workflows.py:47 - INFO: Python version: 3.8.10
2025-03-26 19:26:52,174 - workflow_venvs.py:227 - INFO: running setup_benchmarks_run_script() ...
2025-03-26 19:26:52,174 - utils.py:130 - INFO: Running command: /home/tstesco/projects/tt-inference-server/.workflow_venvs/.venv_benchmarks_run_script/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch numpy
Looking in indexes: https://download.pytorch.org/whl/cpu
Requirement already satisfied: torch in ./.workflow_venvs/.venv_benchmarks_run_script/lib/python3.10/site-packages (2.6.0+cpu)
Requirement already satisfied: numpy in ./.workflow_venvs/.venv_benchmarks_run_script/lib/python3.10/site-packages (2.1.2)
...
```
2. run workflow:
```log
2025-03-26 19:26:57,887 - run_workflows.py:120 - INFO: Starting workflow: benchmarks
2025-03-26 19:26:57,889 - utils.py:130 - INFO: Running command: /home/tstesco/projects/tt-inference-server/.workflow_venvs/.venv_benchmarks_run_script/bin/python /home/tstesco/projects/tt-inference-server/benchmarking/run_benchmarks.py --model Llama-3.2-1B-Instruct --device n300 --output-path /home/tstesco/projects/tt-inference-server/workflow_logs/benchmarks_output --service-port 8000
2025-03-26 19:27:00,930 - workflows.utils - INFO - Directory '/home/tstesco/projects/tt-inference-server/.workflow_venvs' is readable and writable.
2025-03-26 19:27:00,931 - run_benchmarks.py:125 - INFO: Running /home/tstesco/projects/tt-inference-server/benchmarking/run_benchmarks.py ...
2025-03-26 19:27:00,932 - run_benchmarks.py:131 - INFO: workflow_config=: WorkflowConfig(workflow_type=<WorkflowType.BENCHMARKS: 1>, workflow_run_script_venv_type=<WorkflowVenvType.BENCHMARKS_RUN_SCRIPT: 2>, run_script_path=PosixPath('/home/tstesco/projects/tt-inference-server/benchmarking/run_benchmarks.py'), name='benchmarks', workflow_log_dir=PosixPath('/home/tstesco/projects/tt-inference-server/workflow_logs/benchmarks_logs'), workflow_path=PosixPath('/home/tstesco/projects/tt-inference-server/benchmarking'))
2025-03-26 19:27:00,932 - run_benchmarks.py:132 - INFO: model_spec=: ModelSpec(device_configurations={<DeviceTypes.N150: 3>, <DeviceTypes.N300: 4>, <DeviceTypes.T3K: 5>}, tt_metal_commit='v0.56.0-rc47', vllm_commit='e2e0002ac7dc', hf_model_repo='meta-llama/Llama-3.2-1B-Instruct', model_name='Llama-3.2-1B-Instruct', model_id='id_tt-metal-Llama-3.2-1B-Instruct-v0.0.1', impl_id='tt-metal', version='0.0.1', param_count=1, min_disk_gb=4, min_ram_gb=5, repacked=0, docker_image='ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-20.04-amd64:0.0.4-v0.56.0-rc47-e2e0002ac7dc', max_concurrency_map={<DeviceTypes.N150: 3>: 32, <DeviceTypes.N300: 4>: 32, <DeviceTypes.T3K: 5>: 32}, max_context_map={<DeviceTypes.N150: 3>: 131072, <DeviceTypes.N300: 4>: 131072, <DeviceTypes.T3K: 5>: 131072}, status='supported')
2025-03-26 19:27:00,932 - run_benchmarks.py:133 - INFO: device=: n300
2025-03-26 19:27:00,932 - run_benchmarks.py:134 - INFO: service_port=: 8000
2025-03-26 19:27:00,932 - run_benchmarks.py:135 - INFO: output_path=: /home/tstesco/projects/tt-inference-server/workflow_logs/benchmarks_output
2025-03-26 19:27:00,932 - run_benchmarks.py:149 - INFO: OPENAI_API_KEY environment variable set using provided JWT secret.
```
3. wait for inference server to be ready:
```log
2025-03-26 19:27:00,932 - run_benchmarks.py:162 - INFO: Wait for the vLLM server to be ready ...
2025-03-26 19:27:00,933 - utils.prompt_client - WARNING - Health check failed: HTTPConnectionPool(host='127.0.0.1', port=8000): Max retries exceeded with url: /health (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 
...
2025-03-26 19:31:28,381 - utils.prompt_client - INFO - vLLM service is healthy. startup_time:= 0.012565135955810547 seconds
```
4. trace capture:
```log
2025-03-26 19:31:28,381 - utils.prompt_client - INFO - Capturing traces for input configurations...
2025-03-26 19:31:28,381 - utils.prompt_client - INFO - Capturing traces for input_seq_len=8000, output_seq_len=64
2025-03-26 19:31:28,381 - root - INFO - generate_prompts args=PromptConfig(input_seq_len=8000, max_prompt_length=8000, num_prompts=1, distribution='fixed', dataset='random', tokenizer_model='meta-llama/Llama-3.2-1B-Instruct', template=None, save_path=None, print_prompts=False, include_images=False, images_per_prompt=1, image_width=256, image_height=256, use_chat_api=False)
2025-03-26 19:31:28,381 - utils.prompt_generation - INFO - Generating random prompts...
2025-03-26 19:31:29,630 - root - INFO - No template applied to generated prompts.
2025-03-26 19:31:29,785 - utils.prompt_client - INFO - Starting text trace capture: input_seq_len=8000, output_seq_len=64
2025-03-26 19:31:29,785 - utils.prompt_client - INFO - calling: http://127.0.0.1:8000/v1/completions, response_idx=0
2025-03-26 19:31:29,785 - utils.prompt_client - INFO - model: meta-llama/Llama-3.2-1B-Instruct
2025-03-26 19:32:10,867 - utils.prompt_client - INFO - Text trace completed: tokens_generated=61, TTFT=39373.863ms, TPOT=28.472ms
...
```
5. run benchmarks:
```log
2025-03-26 19:35:07,223 - run_benchmarks.py:188 - INFO: Running benchmark Llama-3.2-1B-Instruct: 1/18

2025-03-26 19:35:09,226 - utils.py:130 - INFO: Running command: /home/tstesco/projects/tt-inference-server/.workflow_venvs/.venv_benchmarks_vllm/bin/serve --backend vllm --model meta-llama/Llama-3.2-1B-Instruct --port 8000 --dataset-name random --max-concurrency 1 --num-prompts 8 --random-input-len 128 --random-output-len 128 --ignore-eos --percentile-metrics ttft,tpot,itl,e2el --save-result --result-filename /home/tstesco/projects/tt-inference-server/workflow_logs/benchmarks_output/benchmark_Llama-3.2-1B-Instruct_n300_2025-03-26_19-35-09_isl-128_osl-128_maxcon-1_n-8.json
Namespace(backend='vllm', base_url=None, host='127.0.0.1', port=8000, endpoint='/v1/completions', dataset=None, dataset_name='random', dataset_path=None, max_concurrency=1, model='meta-llama/Llama-3.2-1B-Instruct', tokenizer=None, best_of=1, use_beam_search=False, num_prompts=8, logprobs=None, request_rate=inf, burstiness=1.0, seed=0, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=True, metadata=None, result_dir=None, result_filename='/home/tstesco/projects/tt-inference-server/workflow_logs/benchmarks_output/benchmark_Llama-3.2-1B-Instruct_n300_2025-03-26_19-35-09_isl-128_osl-128_maxcon-1_n-8.json', ignore_eos=True, percentile_metrics='ttft,tpot,itl,e2el', metric_percentiles='99', goodput=None, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, sharegpt_output_len=None, random_input_len=128, random_output_len=128, random_range_ratio=1.0, random_prefix_len=0, hf_subset=None, hf_split=None, hf_output_len=None, tokenizer_mode='auto', served_model_name=None, lora_modules=None)

Starting initial single prompt test run...
Initial test run completed. Starting main benchmark run...
Traffic request rate: inf
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: 1
0%|          | 0/8 [00:00<?, ?it/s]
12%|█▎        | 1/8 [00:02<00:15,  2.17s/it]
25%|██▌       | 2/8 [00:04<00:13,  2.18s/it]
38%|███▊      | 3/8 [00:06<00:10,  2.18s/it]
50%|█████     | 4/8 [00:08<00:08,  2.18s/it]
62%|██████▎   | 5/8 [00:10<00:06,  2.17s/it]
75%|███████▌  | 6/8 [00:13<00:04,  2.17s/it]
88%|████████▊ | 7/8 [00:15<00:02,  2.17s/it]
100%|██████████| 8/8 [00:17<00:00,  2.18s/it]
100%|██████████| 8/8 [00:17<00:00,  2.18s/it]
============ Serving Benchmark Result ============
...
==================================================
```

Outputs are stored in this case: `workflow_logs/evals_output/eval_Llama-3.2-1B-Instruct_n300/meta-llama__Llama-3.2-1B-Instruct`

For example, above in the logs it states the output filename:= `/home/tstesco/projects/tt-inference-server/workflow_logs/benchmarks_output/benchmark_Llama-3.2-1B-Instruct_n300_2025-03-26_19-35-09_isl-128_osl-128_maxcon-1_n-8.json`

See [benchmarking docs](../benchmarking/README.md) for more detail on code.

# Accuracy evaluations

The `evals` workflow follows the same pattern as the `benchmarks` workflow, it will set itself up and wait for the inference server to be ready, then send HTTP requests to it.
The venv setup functions are defined in `workflows/workflow_venvs.py`. Each evaluations will use a different venv which is important to allow for multiple different eval repos and different versions of e.g. https://github.com/EleutherAI/lm-evaluation-harness.

```bash
python3 run.py --model Llama-3.2-1B-Instruct --device n300 --workflow evals
```

Outputs are stored in this case: `workflow_logs/evals_output/eval_Llama-3.2-1B-Instruct_n300/meta-llama__Llama-3.2-1B-Instruct`

See [evals docs](../evals/README.md) for more detail on code.

# Reports

The `reports` workflow generates `reports_outputs` log files summarizing the raw data collected from `benchmarks` and `evals` workflows.

```bash
python3 run.py --model Llama-3.2-1B-Instruct --device n300 --workflow reports
```

This report contains and summarizes metrics and uses defined tolerance thresholds to determine if models pass or fail validation.

See [Logs](#logs) section below for example format of the report files generated.

# Logs

Log types:
- run_logs: the stdout and stderr output you see from `run.py` stored for debugging
- docker_server: the logs from the Docker container running the vLLM inference server for the model
- benchmarks_output: the raw data output from `benchmarks` workflow
- evals_output: the raw data output from `evals` workflow
- reports_output: for each workflow, the markdown (.md) summary output and `/data` summary data. The `release` workflow output is has a summary report of both `benchmarks` and `evals` results. This is used to determine if a model has expected accuracy performance for release. An example of the release report can be seen at https://github.com/tenstorrent/tt-inference-server/issues/164.


In this example for: 
- `model_name`:=Llama-3.2-1B-Instruct
- `device`:=n300

The logs have the following structure:
```log
./workflow_logs
├── benchmarks_output
│   ├── benchmark_Llama-3.2-1B-Instruct_n300_2025-03-25_04-23-40_isl-128_osl-128_maxcon-1_n-8.json
│   ├── ...
│   └── benchmark_Llama-3.2-1B-Instruct_n300_2025-03-25_04-48-11_isl-16000_osl-64_maxcon-32_n-256.json
├── docker_server
│   └── vllm_2025-03-25_20-58-29_Llama-3.2-1B-Instruct_n300_benchmarks.log
├── evals_output
│   └── eval_Llama-3.2-1B-Instruct_n300/meta-llama__Llama-3.2-1B-Instruct
│       ├── results_2025-03-25T04-57-53.064778.json
│       └── samples_meta_gpqa_2025-03-25T04-57-53.064778.jsonl
├── reports_output
│   ├── benchmarks
│   │   ├── data
│   │   │   └── benchmark_stats_Llama-3.2-1B-Instruct_n300.csv
│   │   └── benchmark_display_Llama-3.2-1B-Instruct_n300.md
│   ├── evals
│   │   ├── data
│   │   │   └── eval_data_DeepSeek-R1-Distill-Llama-70B_t3k.json
│   │   ├── summary_Llama-3.2-1B-Instruct_n300.md
│   └── release
│       ├── data
│       │   └── report_data_Llama-3.2-1B-Instruct_n300.json
│       └── report_Llama-3.2-3B-Instruct_n300.md
└── run_logs
    └── run_2025-03-26_02-09-13_Llama-3.2-1B-Instruct_n300_evals.log
```

# Additional Documentation

- [Development](development.md)
- [Benchmarking](../benchmarking/README.md)
- [Evals](../evals/README.md)
- [tests](../tests/README.md)
