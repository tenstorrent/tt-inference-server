# Accuracy Evals

Accuracy evaluations for Tenstorrent LLM (model) implementations. These evaluations determine if a LLM implementation has correct output.

## Usage: `--workflow evals`

See [Model Readiness Workflows User Guide](../docs/workflows_user_guide.md#accuracy-evaluations)

### `run_evals.py`

Purpose: Main script for accuracy evals defined in `EVAL_CONFIGS`, called by `run.py` via `run_workflows.py`.

### Workflow

1. Parse CLI runtime arguments
2. Model & Device Validation
3. Wait for vLLM Inference Server to be ready
4. Run all EvalTask in EvalConfig for given model as a subprocess with own python environment

### `evals_config.py`

Purpose: defines all static information known ahead of run time for evaluations to be run for each model implementation including: python environment, eval parameters, scoring methods, and expected results.

#### Components

- **`EvalConfig`**: a set of tasks for a specific model implementation. All different devices that support that model have the same evaluations.
- **`EvalTask`**: defines python environment to use, what eval tasks to run e.g. using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), and what parameters to use.
- **`EvalTaskScore`**: Defines how a task should be scored.
- **`EVAL_CONFIGS`**: Final dictionary mapping all internal model names to their EvalConfig.


### `eval_utils.py`

Implements scoring functions to compute task results.

## Add evals for a new Model

For instructions on how to add evals for a new model see: [add_support_for_new_model.md](../docs/add_support_for_new_model.md)


## Unique task_name values present in EVAL_CONFIGS
- chartqa
- docvqa_val
- gpqa_diamond_generative_n_shot
- humaneval_instruct
- ifeval
- leaderboard_ifeval
- leaderboard_math_hard
- librispeech_test_other
- livecodebench
- load_image
- mbpp_instruct
- meta_gpqa
- meta_gpqa_cot
- meta_ifeval
- meta_math
- mmlu_pro
- mmmu_val
- r1_aime24
- r1_gpqa_diamond
- r1_math500
- ruler
