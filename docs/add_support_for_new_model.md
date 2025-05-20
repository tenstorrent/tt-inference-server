# How To Add Support for a New Model

## Step 1: Add a new GitHub issue for tracking model readiness support

Add a new GitHub [tt-inference-server issue](https://github.com/tenstorrent/tt-inference-server/issues).

- use issue name format: `Model readiness support: {MODEL_NAME}`
- creat sub-issues for:
    - `Evals: {MODEL_NAME}`
    - `Benchmark targets: {MODEL_NAME}`

Examples of Model readiness support tickets:
* https://github.com/tenstorrent/tt-inference-server/issues/248
* https://github.com/tenstorrent/tt-inference-server/issues/237
* https://github.com/tenstorrent/tt-inference-server/issues/233


## Add Accuracy Evals





### eval_config.py

#### Step 1: Define an EvalConfig
Each model gets one EvalConfig. The primary field to update is the hf_model_repo, which is the HuggingFace repo or unique identifier of your model. Below is an example for Qwen2.5-70B model:

```python
EvalConfig(
    hf_model_repo="Qwen/Qwen2.5-72B-Instruct",
    tasks=[
        # Add one or more EvalTask entries here
        EvalTask(
            # Check Step 2 for EvalTask()
        );  
    ],
)
```
#### Step 2: Add EvalTask Entries
Each EvalTask specifies a benchmark/task (e.g., AIME, GPQA, IFEval) the model will be evaluated on. You may add multiple tasks per model. Below is an example for AIME24:
```python
EvalTask(
    task_name="r1_aime24",
    score=EvalTaskScore(
        published_score=70.00,
        published_score_ref="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        gpu_reference_score=70.00,
        gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/112",
        score_func=score_task_single_key,
        score_func_kwargs={
            "result_keys": [
                "exact_match,none",
            ],
            "unit": "percent",
        },
    ),
    workflow_venv_type=WorkflowVenvType.EVALS_REASON,
    include_path="work_dir",
    max_concurrent=None,
    apply_chat_template=True,
    model_kwargs={
        "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "base_url": "http://127.0.0.1:8000/v1/completions",
        "tokenizer_backend": "huggingface",
        "max_concurrent": 32,
        "max_length": 65536,
    },
    gen_kwargs={"stream": "false", "max_gen_toks": "32768"},
    seed=42,
    num_fewshot=0,
    batch_size=32,
    log_samples=True,
),
```

**Parameter notes:**
- `task_name`: Name of the evaluation task.
- `score`: Scoring configuration, including published and reference scores, and scoring function.
- `workflow_venv_type`: Type of virtual environment used for the workflow.
- `include_path`: Path to include for the evaluation task.
- `max_concurrent`: Maximum number of concurrent runs (None uses default).
- `apply_chat_template`: Whether to apply a chat template to the input.
- `model_kwargs`: Model-specific arguments (e.g., model name, server URL, tokenizer backend, etc.).
- `gen_kwargs`: Generation-specific arguments (e.g., streaming, max tokens to generate).
- `seed`: Random seed for reproducibility.
- `num_fewshot`: Number of few-shot examples to use.
- `batch_size`: Number of samples per batch.
- `log_samples`: Whether to log sample outputs.


## Add Performance Benchmarks

