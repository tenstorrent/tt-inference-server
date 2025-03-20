# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from dataclasses import dataclass, field
from typing import List, Dict

from workflows.workflow_types import WorkflowVenvType
from workflows.utils import map_configs_by_attr
from workflows.model_config import MODEL_CONFIGS


@dataclass(frozen=True)
class EvalTask:
    task: str
    workflow_venv_type: WorkflowVenvType = WorkflowVenvType.EVALS
    eval_class: str = "local-completions"
    max_concurrent: int = 32
    tokenizer_backend: str = "huggingface"
    num_fewshot: int = 0
    seed: int = 42
    use_chat_api: bool = False
    apply_chat_template: bool = True
    log_samples: bool = True
    batch_size: int = 32
    gen_kwargs: Dict[str, str] = field(default_factory=lambda: {"stream": "False"})
    # Note: include_path is specified relative to the respective venv
    include_path: str = None

    def __post_init__(self):
        self.validate_data()
        self._infer_data()

    def _infer_data(self):
        if self.use_chat_api and self.eval_class == "local-completions":
            object.__setattr__(self, "eval_class", "local-chat-completions")

    def validate_data(self):
        assert not (
            self.use_chat_api and self.apply_chat_template
        ), "Chat API applies chat template."


@dataclass(frozen=True)
class EvalConfig:
    hf_model_repo: str
    tasks: List[EvalTask]


# Note: meta evals defined in: https://github.com/meta-llama/llama-cookbook/blob/main/end-to-end-use-cases/benchmarks/llm_eval_harness/meta_eval/eval_config.yaml
# Note: meta_math_hard for Llama 3.1 models has a bug: see https://github.com/tenstorrent/tt-inference-server/issues/155


_eval_config_list = [
    EvalConfig(
        hf_model_repo="Qwen/QwQ-32B",
        tasks=[
            EvalTask(task="leaderboard_ifeval"),
            EvalTask(task="gpqa_diamond_generative_n_shot", num_fewshot=5),
            EvalTask(task="mmlu_pro", num_fewshot=5),
        ],
    ),
    EvalConfig(
        hf_model_repo="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        tasks=[
            EvalTask(task="leaderboard_ifeval"),
            EvalTask(task="gpqa_diamond_generative_n_shot", num_fewshot=5),
            EvalTask(task="mmlu_pro"),
        ],
    ),
    EvalConfig(
        hf_model_repo="Qwen/Qwen2.5-72B-Instruct",
        tasks=[
            EvalTask(task="leaderboard_ifeval"),
            EvalTask(task="gpqa_diamond_generative_n_shot", num_fewshot=5),
            EvalTask(task="mmlu_pro", num_fewshot=5),
        ],
    ),
    EvalConfig(
        hf_model_repo="Qwen/Qwen2.5-7B-Instruct",
        tasks=[
            EvalTask(task="leaderboard_ifeval"),
            EvalTask(task="gpqa_diamond_generative_n_shot", num_fewshot=5),
            EvalTask(task="mmlu_pro", num_fewshot=5),
        ],
    ),
    EvalConfig(
        hf_model_repo="meta-llama/Llama-3.3-70B-Instruct",
        tasks=[
            EvalTask(
                task="meta_ifeval",
                workflow_venv_type=WorkflowVenvType.EVALS_META,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
            ),
            EvalTask(
                task="meta_gpqa_cot",
                workflow_venv_type=WorkflowVenvType.EVALS_META,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="meta-llama/Llama-3.2-11B-Vision-Instruct",
        tasks=[
            EvalTask(
                task="meta_gpqa",
                workflow_venv_type=WorkflowVenvType.EVALS_META,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
            ),
            EvalTask(
                task="meta_math",
                workflow_venv_type=WorkflowVenvType.EVALS_META,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
            ),
            EvalTask(
                eval_class="local-mm-chat-completions",
                task="mmmu_val",
                workflow_venv_type=WorkflowVenvType.EVALS_VISION,
                max_concurrent=16,
                apply_chat_template=False,
                use_chat_api=True,
                batch_size=16,
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="meta-llama/Llama-3.2-3B-Instruct",
        tasks=[
            EvalTask(
                task="meta_gpqa",
                workflow_venv_type=WorkflowVenvType.EVALS_META,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
            ),
            EvalTask(
                task="meta_math",
                workflow_venv_type=WorkflowVenvType.EVALS_META,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="meta-llama/Llama-3.2-1B-Instruct",
        tasks=[
            EvalTask(
                task="meta_gpqa",
                workflow_venv_type=WorkflowVenvType.EVALS_META,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
            ),
            EvalTask(
                task="meta_math",
                workflow_venv_type=WorkflowVenvType.EVALS_META,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="meta-llama/Llama-3.1-70B-Instruct",
        tasks=[
            EvalTask(
                task="meta_ifeval",
                workflow_venv_type=WorkflowVenvType.EVALS_META,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
            ),
            EvalTask(
                task="meta_gpqa_cot",
                workflow_venv_type=WorkflowVenvType.EVALS_META,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="meta-llama/Llama-3.1-8B-Instruct",
        tasks=[
            EvalTask(
                task="meta_ifeval",
                workflow_venv_type=WorkflowVenvType.EVALS_META,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
            ),
            EvalTask(
                task="meta_gpqa_cot",
                workflow_venv_type=WorkflowVenvType.EVALS_META,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
            ),
        ],
    ),
]


_eval_config_map = map_configs_by_attr(
    config_list=_eval_config_list, attr="hf_model_repo"
)
EVAL_CONFIGS = {
    model_name: _eval_config_map[model_config.hf_model_repo]
    for model_name, model_config in MODEL_CONFIGS.items()
    if model_config.hf_model_repo in _eval_config_map
}
