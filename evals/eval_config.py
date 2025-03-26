# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from dataclasses import dataclass, field
from typing import List, Dict, Callable

from workflows.workflow_types import WorkflowVenvType
from workflows.utils import map_configs_by_attr
from workflows.model_config import MODEL_CONFIGS
from evals.eval_utils import (
    score_task_keys_mean,
    score_task_single_key,
    score_multilevel_keys_mean,
)


@dataclass(frozen=True)
class EvalTaskScore:
    expected_score: float
    expected_score_ref: str
    score_func: Callable
    score_func_kwargs: Dict[str, str] = field(default_factory=dict)
    tolerance: float = 0.1


@dataclass(frozen=True)
class EvalTask:
    task_name: str
    score: EvalTaskScore = None
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
    model_kwargs: Dict[str, str] = field(default_factory=lambda: {})
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
# Note: reasoning models (QwQ-32B, DeepSeek-R1-Distill-Llama-70B) need evals allowing more tokens generated


_eval_config_list = [
    EvalConfig(
        hf_model_repo="Qwen/QwQ-32B",
        tasks=[
            EvalTask(
                task_name="leaderboard_ifeval",
                score=EvalTaskScore(
                    expected_score=40.35,
                    expected_score_ref="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/?search=QwQ-32B&official=true",
                    score_func=score_task_keys_mean,
                    score_func_kwargs={
                        "result_keys": [
                            "prompt_level_strict_acc,none",
                            "inst_level_strict_acc,none",
                            "prompt_level_loose_acc,none",
                            "inst_level_loose_acc,none",
                        ]
                    },
                ),
            ),
            EvalTask(
                task_name="leaderboard_math_hard",
                num_fewshot=4,
                score=EvalTaskScore(
                    expected_score=16.09,
                    expected_score_ref="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/?search=QwQ-32B&official=true",
                    score_func=score_multilevel_keys_mean,
                    score_func_kwargs={
                        "result_keys": [
                            ("leaderboard_math_algebra_hard", "exact_match,none"),
                            (
                                "leaderboard_math_counting_and_prob_hard",
                                "exact_match,none",
                            ),
                            ("leaderboard_math_geometry_hard", "exact_match,none"),
                            (
                                "leaderboard_math_intermediate_algebra_hard",
                                "exact_match,none",
                            ),
                            ("leaderboard_math_num_theory_hard", "exact_match,none"),
                            ("leaderboard_math_prealgebra_hard", "exact_match,none"),
                            ("leaderboard_math_precalculus_hard", "exact_match,none"),
                        ],
                        "unit": "percent",
                    },
                ),
            ),
            EvalTask(task_name="gpqa_diamond_generative_n_shot", num_fewshot=5),
            EvalTask(task_name="mmlu_pro", num_fewshot=5),
        ],
    ),
    EvalConfig(
        hf_model_repo="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        tasks=[
            EvalTask(
                task_name="leaderboard_ifeval",
                gen_kwargs={"max_gen_toks": "32768"},
                score=EvalTaskScore(
                    expected_score=83.3,
                    expected_score_ref="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B#deepseek-r1-evaluation",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "prompt_level_strict_acc,none",
                        ],
                        "unit": "percent",
                    },
                ),
            ),
            EvalTask(
                task_name="leaderboard_math_hard",
                num_fewshot=4,
                score=EvalTaskScore(
                    expected_score=30.74,
                    expected_score_ref="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/?search=DeepSeek-R1-Distill-Llama-70B&official=true",
                    score_func=score_multilevel_keys_mean,
                    score_func_kwargs={
                        "result_keys": [
                            ("leaderboard_math_algebra_hard", "exact_match,none"),
                            (
                                "leaderboard_math_counting_and_prob_hard",
                                "exact_match,none",
                            ),
                            ("leaderboard_math_geometry_hard", "exact_match,none"),
                            (
                                "leaderboard_math_intermediate_algebra_hard",
                                "exact_match,none",
                            ),
                            ("leaderboard_math_num_theory_hard", "exact_match,none"),
                            ("leaderboard_math_prealgebra_hard", "exact_match,none"),
                            ("leaderboard_math_precalculus_hard", "exact_match,none"),
                        ],
                        "unit": "percent",
                    },
                ),
            ),
            EvalTask(
                task_name="gpqa_diamond_generative_n_shot",
                num_fewshot=5,
                gen_kwargs={"max_gen_toks": "32768"},
            ),
            EvalTask(
                task_name="mmlu_pro",
                gen_kwargs={"max_gen_toks": "32768"},
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="Qwen/Qwen2.5-72B-Instruct",
        tasks=[
            EvalTask(
                task_name="leaderboard_ifeval",
                score=EvalTaskScore(
                    expected_score=84.1,
                    expected_score_ref="https://arxiv.org/abs/2412.15115",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "prompt_level_strict_acc,none",
                        ],
                        "unit": "percent",
                    },
                ),
            ),
            EvalTask(
                task_name="leaderboard_math_hard",
                num_fewshot=4,
                score=EvalTaskScore(
                    expected_score=59.82,
                    expected_score_ref="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/?search=qwen2.5-72B-Instruct&official=true",
                    score_func=score_multilevel_keys_mean,
                    score_func_kwargs={
                        "result_keys": [
                            ("leaderboard_math_algebra_hard", "exact_match,none"),
                            (
                                "leaderboard_math_counting_and_prob_hard",
                                "exact_match,none",
                            ),
                            ("leaderboard_math_geometry_hard", "exact_match,none"),
                            (
                                "leaderboard_math_intermediate_algebra_hard",
                                "exact_match,none",
                            ),
                            ("leaderboard_math_num_theory_hard", "exact_match,none"),
                            ("leaderboard_math_prealgebra_hard", "exact_match,none"),
                            ("leaderboard_math_precalculus_hard", "exact_match,none"),
                        ],
                        "unit": "percent",
                    },
                ),
            ),
            EvalTask(task_name="gpqa_diamond_generative_n_shot", num_fewshot=5),
            EvalTask(task_name="mmlu_pro", num_fewshot=5),
        ],
    ),
    EvalConfig(
        hf_model_repo="Qwen/Qwen2.5-7B-Instruct",
        tasks=[
            EvalTask(
                task_name="leaderboard_ifeval",
                score=EvalTaskScore(
                    expected_score=71.2,
                    expected_score_ref="https://arxiv.org/abs/2412.15115",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "prompt_level_strict_acc,none",
                        ],
                        "unit": "percent",
                    },
                ),
            ),
            EvalTask(
                task_name="leaderboard_math_hard",
                num_fewshot=4,
                score=EvalTaskScore(
                    expected_score=50.0,
                    expected_score_ref="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/?search=qwen2.5-7B-Instruct&official=true",
                    score_func=score_multilevel_keys_mean,
                    score_func_kwargs={
                        "result_keys": [
                            ("leaderboard_math_algebra_hard", "exact_match,none"),
                            (
                                "leaderboard_math_counting_and_prob_hard",
                                "exact_match,none",
                            ),
                            ("leaderboard_math_geometry_hard", "exact_match,none"),
                            (
                                "leaderboard_math_intermediate_algebra_hard",
                                "exact_match,none",
                            ),
                            ("leaderboard_math_num_theory_hard", "exact_match,none"),
                            ("leaderboard_math_prealgebra_hard", "exact_match,none"),
                            ("leaderboard_math_precalculus_hard", "exact_match,none"),
                        ],
                        "unit": "percent",
                    },
                ),
            ),
            EvalTask(task_name="gpqa_diamond_generative_n_shot", num_fewshot=5),
            EvalTask(task_name="mmlu_pro", num_fewshot=5),
        ],
    ),
    EvalConfig(
        hf_model_repo="meta-llama/Llama-3.3-70B-Instruct",
        tasks=[
            EvalTask(
                task_name="meta_ifeval",
                workflow_venv_type=WorkflowVenvType.EVALS_META,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
                score=EvalTaskScore(
                    expected_score=92.1,
                    expected_score_ref="https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct#instruction-tuned-models",
                    score_func=score_task_keys_mean,
                    score_func_kwargs={
                        "result_keys": [
                            "prompt_level_strict_acc,none",
                            "inst_level_strict_acc,none",
                            "prompt_level_loose_acc,none",
                            "inst_level_loose_acc,none",
                        ],
                        "unit": "percent",
                    },
                ),
            ),
            EvalTask(
                task_name="meta_gpqa_cot",
                workflow_venv_type=WorkflowVenvType.EVALS_META,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
                score=EvalTaskScore(
                    expected_score=50.5,
                    expected_score_ref="https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct#instruction-tuned-models",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "exact_match,strict-match",
                        ],
                        "unit": "percent",
                    },
                ),
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="meta-llama/Llama-3.2-11B-Vision-Instruct",
        tasks=[
            EvalTask(
                task_name="meta_gpqa",
                workflow_venv_type=WorkflowVenvType.EVALS_META,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
                score=EvalTaskScore(
                    expected_score=46.7,
                    expected_score_ref="https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct#instruction-tuned-models",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "exact_match,strict-match",
                        ],
                        "unit": "percent",
                    },
                ),
            ),
            EvalTask(
                task_name="meta_math",
                workflow_venv_type=WorkflowVenvType.EVALS_META,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
                score=EvalTaskScore(
                    expected_score=68.0,
                    expected_score_ref="https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct#instruction-tuned-models",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "exact_match,none",
                        ],
                        "unit": "percent",
                    },
                ),
            ),
            EvalTask(
                eval_class="local-mm-chat-completions",
                task_name="mmmu_val",
                workflow_venv_type=WorkflowVenvType.EVALS_VISION,
                max_concurrent=16,
                apply_chat_template=False,
                use_chat_api=True,
                batch_size=16,
                model_kwargs={
                    "num_concurrent": 16,
                    "max_retries": 1,
                    "tokenized_requests": "False",
                    "add_bos_token": "True",
                    "timeout": "9999",
                    "eos_string": "<|end_of_text|>",
                },
                gen_kwargs={
                    "stop": "<|eot_id|>",
                    "stream": "False",
                },
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="meta-llama/Llama-3.2-3B-Instruct",
        tasks=[
            EvalTask(
                task_name="meta_gpqa",
                workflow_venv_type=WorkflowVenvType.EVALS_META,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
                score=EvalTaskScore(
                    expected_score=32.8,
                    expected_score_ref="https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct#instruction-tuned-models",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "exact_match,strict-match",
                        ],
                        "unit": "percent",
                    },
                ),
            ),
            EvalTask(
                task_name="meta_math",
                workflow_venv_type=WorkflowVenvType.EVALS_META,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
                score=EvalTaskScore(
                    expected_score=48.0,
                    tolerance=0.15,
                    expected_score_ref="https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct#instruction-tuned-models",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "exact_match,none",
                        ],
                        "unit": "percent",
                    },
                ),
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="meta-llama/Llama-3.2-1B-Instruct",
        tasks=[
            EvalTask(
                task_name="meta_gpqa",
                workflow_venv_type=WorkflowVenvType.EVALS_META,
                include_path="work_dir",
                apply_chat_template=False,
                max_concurrent=None,  # not supported in lm-eval==0.4.3
                model_kwargs={},  # not supported in lm-eval==0.4.3
                score=EvalTaskScore(
                    expected_score=27.2,
                    expected_score_ref="https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct#instruction-tuned-models",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "exact_match,strict-match",
                        ],
                        "unit": "percent",
                    },
                ),
            ),
            # NOTE: blocked by https://github.com/tenstorrent/tt-inference-server/issues/163
            # EvalTask(
            #     task_name="meta_math",
            #     workflow_venv_type=WorkflowVenvType.EVALS_META,
            #     include_path="work_dir",
            #     max_concurrent=None,
            #     apply_chat_template=False,
            #     score=EvalTaskScore(
            #         expected_score=30.6,
            #         expected_score_ref="https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct#instruction-tuned-models",
            #         score_func=score_task_single_key,
            #         score_func_kwargs={
            #             "result_keys": [
            #                 "exact_match,none",
            #             ],
            #             "unit": "percent",
            #         },
            #     ),
            # ),
        ],
    ),
    EvalConfig(
        hf_model_repo="meta-llama/Llama-3.1-70B-Instruct",
        tasks=[
            EvalTask(
                task_name="meta_ifeval",
                workflow_venv_type=WorkflowVenvType.EVALS_META,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
                score=EvalTaskScore(
                    expected_score=87.5,
                    expected_score_ref="https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct#instruction-tuned-models",
                    score_func=score_task_keys_mean,
                    score_func_kwargs={
                        "result_keys": [
                            "prompt_level_strict_acc,none",
                            "inst_level_strict_acc,none",
                            "prompt_level_loose_acc,none",
                            "inst_level_loose_acc,none",
                        ],
                        "unit": "percent",
                    },
                ),
            ),
            EvalTask(
                task_name="meta_gpqa_cot",
                workflow_venv_type=WorkflowVenvType.EVALS_META,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
                score=EvalTaskScore(
                    expected_score=46.7,
                    expected_score_ref="https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct#instruction-tuned-models",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "exact_match,strict-match",
                        ],
                        "unit": "percent",
                    },
                ),
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="meta-llama/Llama-3.1-8B-Instruct",
        tasks=[
            EvalTask(
                task_name="meta_ifeval",
                workflow_venv_type=WorkflowVenvType.EVALS_META,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
                score=EvalTaskScore(
                    expected_score=80.4,
                    expected_score_ref="https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct#instruction-tuned-models",
                    score_func=score_task_keys_mean,
                    score_func_kwargs={
                        "result_keys": [
                            "prompt_level_strict_acc,none",
                            "inst_level_strict_acc,none",
                            "prompt_level_loose_acc,none",
                            "inst_level_loose_acc,none",
                        ],
                        "unit": "percent",
                    },
                ),
            ),
            EvalTask(
                task_name="meta_gpqa_cot",
                workflow_venv_type=WorkflowVenvType.EVALS_META,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
                score=EvalTaskScore(
                    expected_score=30.4,
                    expected_score_ref="https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct#instruction-tuned-models",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "exact_match,strict-match",
                        ],
                        "unit": "percent",
                    },
                ),
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
