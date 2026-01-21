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


## Step 2: Add Accuracy Evals

### Selecting Tasks for Model Evaluation

This section provides guidance on how to select and curate new tasks for model evaluation within our infrastructure.


#### What Makes a Good Task Suite?

- **Comprehensive Coverage:**  A good task suite should thoroughly assess all major capabilities of a model.

- **Published Baselines:**  Tasks included in the suite should have established, published baselines for comparison.

- **Diversity:**  Ensure that selected tasks vary in complexity and domain, offering both broad and deep insights into the model’s strengths and weaknesses.

- **Reproducibility:**  Prefer tasks where data, evaluation metrics, and methodology are well-documented and publicly available.



#### Where to Look for Evaluation Tasks

##### 1. Official Benchmarks from Model Developers

- **Model Papers:**  
  Start with the official research papers by model developers. These will often include the most prominent and relevant evaluation tasks.

- **Blog Posts and Official Announcements:**  
  Model developers often highlight benchmarks and key evaluation results in their blogs or press releases.

- **Documentation and GitHub Repositories:**  
  Official repositories may contain scripts, datasets, or pointers to the tasks used in evaluations.

##### 2. Popular Third-Party Publishers

- **LLM Leaderboards (e.g., Hugging Face):**  
  Third-party curated leaderboards aggregate standard tasks for evaluating broad model capabilities. Although some, like the Hugging Face Leaderboard, may be archived, they remain a useful reference for widely adopted benchmarks.


#### Task Selection Guidelines

- **Reuse Supported Tasks:**  
  If an evaluation task is already supported in `tt-inference-server`, **reuse it** to ensure consistency and comparability.

- **Adding New Tasks:**  
  If meaningful gaps exist, select new tasks according to the criteria above. Make sure new tasks:
  - Cover unique or underrepresented capabilities
  - Possess reproducible methodologies and baseline results

- **Coverage Recommendation:**  
  - **Minimum:** At least **2 tasks per model**
  - **Comprehensiveness:** The tasks should cover all major modalities and capabilities of the model.
    - Example: For a multi-modal model, select both chat (text) evaluation tasks and vision-language tasks.

### Adding GPU Reference Scores

In addition to the published benchmarks we also care about the scores achieved for the tasks on a GPU. This usually tends to provide a better reference for variance in hyperparameter tuning. You can obtain these scores by running the vLLM server on a GPU and running the eval workflow on that.


### eval_config.py

#### Step 2A: Define an EvalConfig
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
#### Step 2B: Add EvalTask Entries
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
    workflow_venv_type=_
    WorkflowVenvType.EVALS_COMMON,
    include_path="work_dir",
    apply_chat_template=True,
    model_kwargs={
        "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "base_url": "http://127.0.0.1:8000/v1/completions",
        "tokenizer_backend": "huggingface",
        "max_length": 65536,
    },
    gen_kwargs={"stream": "false", "max_gen_toks": "32768"},
    seed=42,
    num_fewshot=0,
    log_samples=True,
),
```

**Parameter notes:**
- `task_name` (str): Name of the evaluation task (e.g., 'r1_aime24').
- `score` (EvalTaskScore): Scoring configuration for the task, including:
    - `published_score` (float): Reference score from published results (if available).
    - `published_score_ref` (str): URL or citation for the published score.
    - `gpu_reference_score` (float, optional): Reference score for GPU (if available).
    - `gpu_reference_score_ref` (str, optional): URL or citation for the GPU reference score.
    - `score_func` (Callable): Function used to compute the score (e.g., `score_task_single_key`).
    - `score_func_kwargs` (dict): Arguments for the scoring function (e.g., result keys, unit).
    - `tolerance` (float, optional): Allowed tolerance for score comparison (default: 0.1).
- `workflow_venv_type` (WorkflowVenvType): Type of virtual environment used for the workflow. Options include:
    - `WorkflowVenvType.EVALS_COMMON`: Standard evaluation environment (default).
    - `WorkflowVenvType.EVALS_META`: Meta-specific evaluations.
    - `WorkflowVenvType.EVALS_VISION`: Vision model evaluations.
- `eval_class` (str): Evaluation class to use (default: 'local-completions').
- `include_path` (str): Path to include for the evaluation task, relative to the venv (e.g., 'work_dir').
- `max_concurrent` (int, optional): Maximum number of concurrent runs (default: 32; None uses default for the environment).
- `tokenizer_backend` (str): Tokenizer backend to use (default: 'huggingface').
- `num_fewshot` (int): Number of few-shot examples to use (default: 0).
- `seed` (int): Random seed for reproducibility (default: 42).
- `use_chat_api` (bool): Whether to use the chat API endpoint (default: False).
- `apply_chat_template` (bool): Whether to apply a chat template to the input (default: True; should be False if using chat API).
- `log_samples` (bool): Whether to log sample outputs (default: True).
- `batch_size` (int): Number of samples per batch (default: 32).
- `gen_kwargs` (dict): Generation-specific arguments (e.g., streaming, max tokens to generate). Default: `{ 'stream': 'False' }`.
- `model_kwargs` (dict): Model-specific arguments (e.g., model name, server URL, tokenizer backend, max length, etc.). Default: `{}`.


## Step 3: Add Performance Targets

The performance targets are all in https://github.com/tenstorrent/tt-inference-server/blob/main/benchmarking/benchmark_targets/model_performance_reference.json, each model has list of benchmark targets. For example LLMs specific specific points on ISL/OSL/concurrency curve. The theoretical targets are then checked against those measured points. In the example below for `Llama-3.3-70B-Instruct` for `galaxy` the 2 points ISL=128,OSL=128,concurrency=1 and ISL=2048,OSL=128,concurrency=1. These then become the checkpoints for the performance pass fail in Models CI.

```json
"Llama-3.3-70B-Instruct": {
        "t3k": [
            {
                "isl": 128,
                "osl": 128,
                "max_concurrency": 1,
                "num_prompts": 8,
                "task_type": "text",
                "image_height": null,
                "image_width": null,
                "images_per_prompt": null,
                "targets": {
                    "theoretical": {
                        "ttft_ms": 113.0,
                        "tput_user": 19.0,
                        "tput": 610.0
                    }
                }
            }
        ],
        "galaxy": [
            {
                "isl": 128,
                "osl": 128,
                "max_concurrency": 1,
                "num_prompts": 8,
                "task_type": "text",
                "image_height": null,
                "image_width": null,
                "images_per_prompt": null,
                "targets": {
                    "theoretical": {
                        "ttft_ms": 50.0,
                        "tput_user": 73.0,
                        "tput": 2438.0
                    }
                }
            },
            {
                "isl": 2048,
                "osl": 128,
                "max_concurrency": 1,
                "num_prompts": 8,
                "targets": {
                    "theoretical": {
                        "ttft_ms": 800,
                        "tput_user": 80
                    }
                }
            }
        ],
```

This defines the pass/fail checks you can see in the report:
```
#### Text-to-Text Performance Benchmark Targets Llama-3.3-70B-Instruct on galaxy

| ISL  | OSL | Concurrency | TTFT (ms) | Tput User (TPS) | Tput Decode (TPS) | Functional TTFT Check | Functional Tput User Check | Complete TTFT Check | Complete Tput User Check | Target TTFT Check | Target Tput User Check | Functional TTFT (ms) | Functional Tput User (TPS) | Complete TTFT (ms) | Complete Tput User (TPS) | Target TTFT (ms) | Target Tput User (TPS) |
|------|-----|-------------|-----------|-----------------|-------------------|-----------------------|----------------------------|---------------------|--------------------------|-------------------|------------------------|----------------------|----------------------------|--------------------|--------------------------|------------------|------------------------|
|  128 | 128 |           1 |      94.0 |           53.79 |              53.8 | PASS ✅               | PASS ✅                    | PASS ✅             | PASS ✅                  | FAIL ⛔           | FAIL ⛔                |               500.00 |                       8.00 |             100.00 |                    40.00 |            50.00 |                  80.00 |
| 2048 | 128 |           1 |     397.0 |           49.98 |              50.0 | PASS ✅               | PASS ✅                    | PASS ✅             | PASS ✅                  | PASS ✅           | FAIL ⛔                |              8000.00 |                       8.00 |            1600.00 |                    40.00 |           800.00 |                  80.00 |

```
