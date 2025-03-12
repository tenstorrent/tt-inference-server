# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from dataclasses import dataclass
from typing import List

from workflows.workflow_config import WorkflowVenvType


@dataclass(frozen=True)
class LMEvalConfig:
    task: List[str]
    model: str = "local-completions"
    max_concurrent: int = 32
    tokenizer_backend: str = "huggingface"
    num_fewshot: int = 0
    seed: int = 42
    use_chat_api: bool = False
    apply_chat_template: bool = True
    log_samples: bool = True
    batch_size: int = 32
    # Note: include_path is specified relative to the respective venv
    include_path: str = None


@dataclass(frozen=True)
class EvalConfig:
    hf_model_repo: str
    lm_eval_tasks: List[LMEvalConfig]
    workflow_venv_type: WorkflowVenvType = WorkflowVenvType.EVALS


_eval_config_list = [
    EvalConfig(
        hf_model_repo="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        lm_eval_tasks=[
            LMEvalConfig(task="leaderboard_ifeval"),
            LMEvalConfig(task="gpqa_diamond_generative_n_shot", num_fewshot=5),
            LMEvalConfig(task="mmlu_pro"),
        ],
    ),
    EvalConfig(
        hf_model_repo="Qwen/Qwen2.5-72B-Instruct",
        lm_eval_tasks=[
            LMEvalConfig(task="mmlu_pro", num_fewshot=5),
            LMEvalConfig(task="gpqa_diamond_generative_n_shot", num_fewshot=5),
            LMEvalConfig(task="leaderboard_ifeval"),
        ],
    ),
    EvalConfig(
        hf_model_repo="meta-llama/Llama-3.3-70B-Instruct",
        workflow_venv_type=WorkflowVenvType.EVALS_META,
        lm_eval_tasks=[
            LMEvalConfig(
                task="meta_ifeval",
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
            ),
            LMEvalConfig(
                task="meta_gpqa",
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="meta-llama/Llama-3.2-11B-Vision-Instruct",
        workflow_venv_type=WorkflowVenvType.EVALS_META,
        lm_eval_tasks=[
            LMEvalConfig(
                task="meta_math",
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
            ),
            LMEvalConfig(
                task="meta_gpqa",
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="meta-llama/Llama-3.2-3B-Instruct",
        workflow_venv_type=WorkflowVenvType.EVALS_META,
        lm_eval_tasks=[
            LMEvalConfig(
                task="meta_math",
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
            ),
            LMEvalConfig(
                task="meta_gpqa",
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="meta-llama/Llama-3.2-1B-Instruct",
        workflow_venv_type=WorkflowVenvType.EVALS_META,
        lm_eval_tasks=[
            LMEvalConfig(
                task="meta_math",
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
            ),
            LMEvalConfig(
                task="meta_gpqa",
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="meta-llama/Llama-3.1-70B-Instruct",
        workflow_venv_type=WorkflowVenvType.EVALS_META,
        lm_eval_tasks=[
            LMEvalConfig(
                task="meta_ifeval",
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
            ),
            LMEvalConfig(
                task="meta_gpqa",
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="meta-llama/Llama-3.1-8B-Instruct",
        workflow_venv_type=WorkflowVenvType.EVALS_META,
        lm_eval_tasks=[
            LMEvalConfig(
                task="meta_ifeval",
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
            ),
            LMEvalConfig(
                task="meta_gpqa",
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
            ),
        ],
    ),
]
EVAL_CONFIGS = {config.hf_model_repo: config for config in _eval_config_list}
