# LiveCodeBench Integration with TT-Inference Server

## Overview

This document explains how the lm-evaluation-harness repository (specifically the LiveCodeBench evaluation) integrates with the TT-Inference Server system. The integration allows for automated evaluation of code generation capabilities of language models running on Tenstorrent hardware.

## Architecture Overview

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Docker Container  │    │  TT-Inference       │    │  lm-evaluation      │
│   (Model Server)    │◄───┤  Server (run.py)    │◄───┤  harness            │
│                     │    │                     │    │  (LiveCodeBench)    │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
         │                           │                           │
         │                           │                           │
         ▼                           ▼                           ▼
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Model Weights     │    │  Workflow Engine    │    │  Evaluation Tasks   │
│   & Configurations  │    │  & Logging          │    │  & Scoring          │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

## Key Components

### 1. Model Configuration (`model_config.py`)

Defines model specifications including:
- **Device configurations**: Which Tenstorrent devices the model supports
- **Implementation details**: Which backend implementation to use
- **Resource requirements**: Memory, disk space, context limits
- **Performance targets**: Expected throughput and latency metrics

Example for Llama-3.2-3B-Instruct:
```python
ModelConfig(
    device_configurations={DeviceTypes.N150, DeviceTypes.N300, DeviceTypes.T3K},
    weights=["meta-llama/Llama-3.2-3B-Instruct"],
    max_context_map={
        DeviceTypes.N300: 32 * 1024,  # Context window limits per device
    },
    status="ready"
)
```

### 2. Evaluation Configuration (`eval_config.py`)

Defines evaluation tasks and scoring criteria:
- **Task definitions**: Which evaluation benchmarks to run
- **Scoring functions**: How to calculate final scores
- **Model-specific settings**: Batch size, generation parameters, etc.

Example LiveCodeBench configuration:
```python
EvalTask(
    task_name="livecodebench",
    workflow_venv_type=WorkflowVenvType.EVALS,
    score=EvalTaskScore(
        published_score=5.4,  # Published benchmark score
        gpu_reference_score=5,  # Expected GPU reference
        score_func=score_task_single_key,
        score_func_kwargs={
            "result_keys": ["acc,none"],  # Accuracy metric
            "unit": "percent",
        },
    ),
)
```

### 3. lm-evaluation-harness Integration

The system uses a custom lm-evaluation-harness setup located in:
```
/home/aroberge/lm-evaluation-harness/lm_eval/tasks/livecodebench/
├── livecodebench.yaml      # Task configuration
├── utils.py               # Utility functions
├── testing_util.py        # Testing utilities
├── pass_k_utils.py        # Pass@k scoring utilities
└── test_pipeline.py       # Test execution pipeline
```

## Workflow Process

### Step 1: Docker Build
```bash
python3 workflows/build_release_docker_images.py --build-metal-commit v0.57.0-rc71
```

**What happens:**
- Builds Docker image with TT-Metal backend
- Includes all necessary dependencies for model inference
- Creates versioned image tagged with commit hashes

### Step 2: Server Launch
```bash
python3 run.py --model Llama-3.2-3B-Instruct --device n300 --workflow server --dev-mode --docker-server
```

**What happens:**
1. **Model Resolution**: System looks up model config for `Llama-3.2-3B-Instruct`
2. **Device Validation**: Confirms N300 device is supported for this model
3. **Docker Container Launch**: Starts inference server in container
4. **Model Loading**: Downloads and loads model weights onto Tenstorrent hardware
5. **Server Ready**: HTTP server starts listening on port 8000 (default)

**Key files involved:**
- `run.py`: Main orchestration
- `model_config.py`: Model specifications
- `workflows/run_docker_server.py`: Docker container management
- `workflows/setup_host.py`: Host environment setup

### Step 3: Evaluation Execution
```bash
python3 run.py --model Llama-3.2-3B-Instruct --device n300 --workflow evals
```

**What happens:**
1. **Task Discovery**: System finds `livecodebench` task in eval config
2. **Environment Setup**: Creates isolated Python environment for evaluations
3. **lm-eval Integration**: Calls lm-evaluation-harness with custom configuration
4. **Code Generation**: Model generates code solutions for programming problems
5. **Execution & Scoring**: Generated code is executed and scored using pass@k metrics

**Evaluation Flow:**
```
livecodebench.yaml → lm_eval → HTTP requests → TT-Inference Server → Model → Responses
                                     ↓
                             testing_util.py → Code execution → pass_k_utils.py → Scores
```

### Step 4: Report Generation
```bash
python3 run.py --model Llama-3.2-3B-Instruct --device n300 --workflow reports
```

**What happens:**
- Aggregates evaluation results from previous runs
- Compares against published benchmarks and GPU references
- Generates formatted reports with performance metrics

## LiveCodeBench Specific Details

### Task Configuration (`livecodebench.yaml`)

The YAML configuration defines the complete evaluation setup:

```yaml
task: livecodebench
dataset_path: livecodebench/code_generation_lite
output_type: generate_until
validation_split: test
doc_to_text: "### Question:\n{{question_content}}\n\n### Answer: (Please provide your answer in a Python code block)\n\n"
doc_to_target: !function utils.doc_to_target
process_results: !function utils.process_results
num_fewshot: 0
generation_kwargs:
  until:
  - "<|end_of_text|>"
  - "<|endoftext|>"
  - "<|im_end|>"
  do_sample: false
  max_gen_toks: 2048
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
```

**Key Configuration Points:**
- **Dataset**: Uses `livecodebench/code_generation_lite` from HuggingFace
- **Prompt Format**: Structures coding problems with clear question/answer format
- **Generation Control**: Uses deterministic generation (`do_sample: false`) for consistency
- **Token Limits**: Allows up to 2048 tokens for code generation
- **Stop Sequences**: Multiple end-of-text tokens to handle different model formats

### Code Generation and Extraction

The system uses sophisticated code extraction logic:

1. **Model Output Processing**: Handles both base and chat model formats
2. **Code Block Detection**: Looks for markdown code blocks (```python ... ```)
3. **Fallback Extraction**: If no code blocks found, uses heuristics to detect Python code
4. **Common Python Indicators**: Searches for keywords like `def`, `import`, `class`, `if`, etc.

```python
def extract_code_generation(model_output: str, model_type: str = 'chat'):
    outputlines = model_output.split('\n')
    
    if model_type == 'chat':
        indexlines = [i for i, line in enumerate(outputlines) if '```' in line]
        
        if len(indexlines) >= 2:
            return '\n'.join(outputlines[indexlines[0] + 1:indexlines[1]])
        
        # Fallback: detect raw Python code
        stripped_output = model_output.strip()
        python_indicators = ['def ', 'import ', 'from ', 'class ', 'if ', 'for ', 'while ']
        if any(indicator in stripped_output for indicator in python_indicators):
            return stripped_output
            
    return ''
```

### Test Case Processing

LiveCodeBench handles both public and private test cases:

1. **Public Test Cases**: Visible test cases for validation
2. **Private Test Cases**: Hidden test cases (base64 + zlib compressed)
3. **Test Format Conversion**: Converts to standardized input/output format
4. **Robust Decoding**: Multiple fallback strategies for private test case decoding

The `doc_to_target` function processes the dataset format:
- Parses JSON test case data
- Handles compressed private test cases
- Creates unified input/output format for execution

### Code Execution Pipeline

The `testing_util.py` implements a secure code execution environment:

1. **Isolation**: Uses separate execution namespace
2. **Timeout Protection**: Per-test-case timeout (default 6 seconds)
3. **Input/Output Handling**: Captures stdin/stdout for test case execution
4. **Exception Handling**: Graceful failure for syntax errors or runtime exceptions

```python
def run_test(sample, test, debug=False, timeout=6):
    # Set up timeout handler
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    
    try:
        # Capture stdout and provide stdin
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        sys.stdin = io.StringIO(input_str)
        
        # Execute in isolated namespace
        exec_globals = {'__name__': '__main__', '__builtins__': __builtins__}
        exec(test, exec_globals)
        
        # Compare outputs
        actual_output = captured_output.getvalue().strip()
        test_passed = actual_output == expected_output_str
        
    finally:
        # Cleanup and restore
        sys.stdout = old_stdout
        signal.alarm(0)
```

### Scoring Methodology

LiveCodeBench uses **pass@k** scoring through the `pass_k_utils.py` module:

- **pass@1**: Percentage of problems solved on first attempt
- **pass@k**: Percentage solvable with k attempts (for multiple generations)
- **Execution-Based**: Only counts solutions that pass all test cases
- **Strict Matching**: Output must exactly match expected results

The scoring process:
1. **Code Extraction**: Extract Python code from model output
2. **Test Execution**: Run code against all test cases for each problem
3. **Result Aggregation**: Collect pass/fail results across all problems
4. **Metric Calculation**: Compute final accuracy percentage

### Error Handling and Robustness

The system includes comprehensive error handling:

- **Syntax Errors**: Invalid Python code gets zero score
- **Runtime Exceptions**: Caught and logged, problem marked as failed
- **Timeouts**: Long-running code is terminated and marked as failed
- **Missing Test Cases**: Graceful degradation with minimal test setup
- **Decoding Failures**: Fallback strategies for corrupted private test cases

## Configuration Deep Dive

### Model-Eval Mapping
The system automatically maps models to their evaluation tasks:

```python
# In eval_config.py
EVAL_CONFIGS = {
    model_config.model_name: _eval_config_map[model_config.hf_model_repo]
    for _, model_config in MODEL_CONFIGS.items()
    if model_config.hf_model_repo in _eval_config_map
}
```

This ensures that:
- Only models with defined evaluations can run eval workflows
- Evaluation parameters are model-specific
- Scoring is compared against appropriate benchmarks

### Workflow Validation
Before execution, the system validates:
- Model supports the specified device type
- Evaluation tasks are defined for the model
- Required environment variables are set (HF_TOKEN, JWT_SECRET)
- Sufficient disk space and memory are available

## Environment Setup

### Python Virtual Environments
The system uses multiple isolated environments:
- **EVALS**: Standard evaluation environment with lm-evaluation-harness
- **EVALS_REASON**: Enhanced environment for reasoning tasks
- **EVALS_META**: Meta's evaluation framework
- **EVALS_VISION**: Vision-language model evaluations

### Dependencies
Key dependencies include:
- `lm-evaluation-harness`: Core evaluation framework
- `torch`: PyTorch for model operations
- `transformers`: HuggingFace model interfaces
- Custom TT-specific evaluation adapters

## Monitoring and Logging

### Log Files
All operations are logged to structured files:
```
~/tt-inference-server-logs/
├── run_logs/
│   └── run_{timestamp}_{model}_{workflow}.log
├── workflow_logs/
│   └── {workflow_name}_{run_id}/
└── server_logs/
    └── docker_server_{run_id}.log
```

### Metrics Tracked
- **Performance**: Inference latency, throughput
- **Accuracy**: Task-specific scoring metrics
- **Resource Usage**: Memory, GPU utilization
- **Error Rates**: Failed requests, timeout issues

### Debug Information
For troubleshooting, the system can provide detailed logs:
- Individual test case results
- Code execution traces
- Model generation samples
- Timing information per problem

## Troubleshooting Common Issues

### 1. Model Not Found in Eval Config
**Error**: `Model:=Llama-3.2-3B-Instruct not found in EVAL_CONFIGS`
**Solution**: Check if model has evaluation tasks defined in `eval_config.py`

### 2. Device Not Supported
**Error**: `model:=Llama-3.2-3B does not support device:=n300`
**Solution**: Verify device compatibility in `model_config.py`

### 3. Server Connection Issues
**Error**: Connection refused to localhost:8000
**Solution**: 
- Ensure Docker server is running
- Check port availability
- Verify JWT_SECRET is set for production

### 4. Evaluation Timeouts
**Error**: Task execution exceeds time limits
**Solution**:
- Increase timeout values in task configuration
- Check model inference performance
- Verify test case complexity

### 5. Code Execution Issues
**Error**: Generated code fails to execute
**Common Causes**:
- Incomplete code generation (increase `max_gen_toks`)
- Missing imports in generated code
- Syntax errors from model
- Timeout on complex algorithms

### 6. Test Case Decoding Failures
**Error**: Cannot decode private test cases
**Solution**:
- Check dataset integrity
- Verify base64/zlib decoding libraries
- Use debug mode to inspect test case format

## Performance Considerations

### Resource Requirements
- **N300 Device**: Typically requires 16-32GB RAM
- **Model Loading**: Can take 5-15 minutes depending on model size
- **Evaluation Runtime**: LiveCodeBench typically takes 30-60 minutes
- **Code Execution**: Each test case has 6-second timeout

### Optimization Tips
1. **Pre-warm Models**: Keep server running between evaluations
2. **Batch Processing**: Use appropriate batch sizes for your hardware
3. **Context Management**: Monitor context window usage
4. **Concurrent Requests**: Balance parallelism with resource constraints
5. **Debug Mode**: Use selectively as it significantly slows evaluation

### Expected Performance Benchmarks

Based on the evaluation configurations:

| Model | Expected Accuracy | Reference Hardware |
|-------|------------------|-------------------|
| Llama-3.2-3B-Instruct | 5.4% | Published Benchmark |
| Qwen2.5-7B-Instruct | 14.2% | Published Benchmark |

Note: LiveCodeBench is a challenging benchmark with generally low accuracy scores even for large models.

## Next Steps

To get started with LiveCodeBench evaluation:

1. **Verify Setup**: Ensure your environment meets all requirements
2. **Run Sample Evaluation**: Start with a small model like Llama-3.2-3B
3. **Monitor Logs**: Watch for any configuration or runtime issues
4. **Compare Results**: Validate against published benchmarks
5. **Scale Up**: Try larger models or different evaluation tasks
6. **Debug Issues**: Use debug mode to inspect individual problem solutions

For additional support, refer to the GitHub issues tracker and community documentation. 