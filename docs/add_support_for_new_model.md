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

