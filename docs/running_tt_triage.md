# Running TT-Triage on TT-Inference-Server Docker Images

## Motivation

`tt-triage` is a debugging tool from the `tt-metal` repository designed to diagnose system hangs on Tenstorrent hardware. When a workload hangs or becomes unresponsive, `tt-triage` captures and analyzes the callstacks from each core on the device, providing crucial information to identify where and why the hang occurred.

This is particularly useful when:
- A Docker container running inference becomes unresponsive
- Model execution hangs during serving
- You need to debug performance issues or deadlocks
- The system appears stuck but hasn't crashed

The `run_tt_triage.py` script automates the process of running `tt-triage` inside a running Docker container, making it easy to debug hangs without manually entering the container or setting up the debugging environment.

## Prerequisites

- A running Docker container with tt-inference-server
- The container must be accessible via its service port
- Python 3.8+ on the host machine
- Docker installed and accessible from the host

## Usage Workflow

### Step 1: Start the Docker Server

First, start your inference server using the standard workflow:

```bash
# Example: Start server for Llama-3.1-8B on N300
python run.py --model Llama-3.1-8B-Instruct --device n300 --workflow server --docker-server
```

The server will start and bind to a service port (default: 8000).

### Step 2: Run Your Workload

Start sending requests to your inference server to trigger the workload:

```bash
# Example: Run benchmarks or evals
python run.py --model Llama-3.1-8B-Instruct --device n300 --workflow benchmarks --docker-server
```

### Step 3: Wait for Hang to Occur

Monitor your workload. If it hangs or becomes unresponsive:
- The container is still running but not making progress
- Requests timeout or receive no response
- System appears stuck without error messages

**Do not stop the container!** The triage tool needs to analyze the live running state.

### Step 4: Run TT-Triage

While the container is still running in the hung state, execute the triage script from the host:

```bash
python scripts/run_tt_triage.py \
    --service-port 8000 \
    --model Llama-3.1-8B-Instruct \
    --device n300
```

## Command Line Arguments

### Required Arguments

| Argument | Type | Description | Example |
|----------|------|-------------|---------|
| `--service-port` | int | Service port number used to identify the Docker container | `8000` |
| `--model` | str | Model name exactly as specified when starting the server | `Llama-3.1-8B-Instruct` |
| `--device` | str | Device type (must match what was used to start the server) | `n300`, `n150`, `t3000` |

### Optional Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--impl` | str | `tt-transformers` | Implementation name (only change if using non-default impl) |

## Examples

### Basic Usage (Default Implementation)

```bash
# For Llama-3.1-8B on N300
python scripts/run_tt_triage.py \
    --service-port 8000 \
    --model Llama-3.1-8B-Instruct \
    --device n300

# For Qwen2.5-VL-3B on N300
python scripts/run_tt_triage.py \
    --service-port 8000 \
    --model Qwen2.5-VL-3B-Instruct \
    --device n300

# For Mistral-7B on N150
python scripts/run_tt_triage.py \
    --service-port 8000 \
    --model Mistral-7B-Instruct-v0.3 \
    --device n150
```

### Multiple Containers

If running multiple containers on different ports:

```bash
# Container 1 on port 8000
python scripts/run_tt_triage.py \
    --service-port 8000 \
    --model Llama-3.1-8B-Instruct \
    --device n300

# Container 2 on port 8001
python scripts/run_tt_triage.py \
    --service-port 8001 \
    --model Mistral-7B-Instruct-v0.3 \
    --device n300
```

## Output

### Output Location

The triage report is automatically saved to:

```
workflow_logs/triage_output/triage_{model_spec_id}_{timestamp}.txt
```

Where:
- `{model_spec_id}` is generated from the impl, model, and device (e.g., `id_tt-transformers_Llama-3.1-8B-Instruct_n300`)
- `{timestamp}` is in format `YYYYMMDD_HHMMSS` (e.g., `20251014_143022`)

**Example output path:**
```
workflow_logs/triage_output/triage_id_tt-transformers_Llama-3.1-8B-Instruct_n300_20251014_143022.txt
```

### Output Contents

The triage report contains:

1. **Header Information**
   - Container ID
   - Execution timestamp
   - Model and device information

2. **Debugging Sequence Output**
   - Environment setup confirmation
   - Debugger tools installation logs
   - Python dependency installation status
   - **Core callstacks from triage.py** - This is the critical information showing what each core was doing when the hang occurred

3. **Return Codes**
   - Success/failure status of each step


## What the Script Does

The script performs the following sequence of operations inside the Docker container:

1. **Locates the container** by matching the service port
2. **Navigates** to the tt-metal debugging scripts directory
3. **Activates** the tt-metal Python environment
4. **Unsets** `LD_LIBRARY_PATH` to avoid library conflicts
5. **Installs** debugger tools using `install_debugger.sh`
6. **Installs** Python dependencies required for triage
7. **Runs** `triage.py` to capture core callstacks
8. **Saves** all output to the host filesystem

All commands are chained together in a single shell session to ensure environment variables and directory changes persist.

## Additional Resources

- [tt-triage tool documentation](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tools/triage.html)
- [Workflows User Guide](./workflows_user_guide.md)
- [Development Guide](./development.md)

