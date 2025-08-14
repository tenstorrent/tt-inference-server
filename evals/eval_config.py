# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from dataclasses import dataclass, field
from typing import List, Dict, Callable

from workflows.workflow_types import WorkflowVenvType
from workflows.utils import map_configs_by_attr
from workflows.model_spec import MODEL_SPECS
from evals.eval_utils import (
    score_task_keys_mean,
    score_task_single_key,
    score_multilevel_keys_mean,
)


@dataclass(frozen=True)
class EvalTaskScore:
    published_score: float
    published_score_ref: str
    score_func: Callable
    gpu_reference_score: float = None
    gpu_reference_score_ref: str = None
    score_func_kwargs: Dict[str, str] = field(default_factory=dict)
    tolerance: float = 0.0


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
    # Optional: limit the number of samples passed to lm_eval (--limit)
    limit_samples: int = None

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
        hf_model_repo="Qwen/Qwen3-8B",
        tasks=[
            EvalTask(
                task_name="r1_gpqa_diamond",
                score=EvalTaskScore(
                    published_score=62.0,  
                    published_score_ref="https://arxiv.org/pdf/2505.09388",
                    gpu_reference_score=64.14, 
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/384#issuecomment-3129960933",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "exact_match,none",
                        ],
                        "unit": "percent",
                    },
                ),
                workflow_venv_type=WorkflowVenvType.EVALS,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=True,
                model_kwargs={
                    "model": "Qwen/Qwen3-8B",
                    "base_url": "http://127.0.0.1:8000/v1/completions",
                    "tokenizer_backend": "huggingface",
                    "max_length": 65536,
                },
                # gen_kwargs chosen according to https://huggingface.co/Qwen/Qwen3-8B#best-practices
                gen_kwargs={
                    "stream": "false",
                    "max_gen_toks": 32768,
                    "until": [],
                    "do_sample": "true",
                    "temperature": 0.6,
                    "top_k": 20,
                    "top_p": 0.95,
                },
                seed=42,
                num_fewshot=0,
                batch_size=32,
                log_samples=True,
            ),
            EvalTask(
                task_name="mmlu_pro",
                num_fewshot=5,
                score=EvalTaskScore(
                    published_score=56.73,
                    published_score_ref="https://arxiv.org/pdf/2505.09388",
                    gpu_reference_score=None,
                    gpu_reference_score_ref="TBD",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "exact_match,custom-extract",
                        ],
                        "unit": "percent",
                    },
                ),
                workflow_venv_type=WorkflowVenvType.EVALS,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=True,
                model_kwargs={
                    "model": "Qwen/Qwen3-8B",
                    "base_url": "http://127.0.0.1:8000/v1/completions",
                    "tokenizer_backend": "huggingface",
                    "max_length": 65536,
                },
                # gen_kwargs chosen according to https://huggingface.co/Qwen/Qwen3-8B#best-practices
                gen_kwargs={
                    "stream": "false",
                    "max_gen_toks": 32768,
                    "until": [],
                    "do_sample": "true",
                    "temperature": 0.6,
                    "top_k": 20,
                    "top_p": 0.95,
                },
                seed=42,
                batch_size=32,
                log_samples=True,
                limit_samples=100,
            )
        ],
    ),
    EvalConfig(
        hf_model_repo="Qwen/Qwen3-32B",
        tasks=[
            EvalTask(
                task_name="r1_aime24",
                score=EvalTaskScore(
                    published_score=81.40,  
                    published_score_ref="https://qwenlm.github.io/blog/qwen3/",
                    gpu_reference_score=80.00,  # Estimate - needs to be validated
                    gpu_reference_score_ref="TBD",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "exact_match,none",
                        ],
                        "unit": "percent",
                    },
                ),
                workflow_venv_type=WorkflowVenvType.EVALS,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=True,
                model_kwargs={
                    "model": "Qwen/Qwen3-32B",
                    "base_url": "http://127.0.0.1:8000/v1/completions",
                    "tokenizer_backend": "huggingface",
                    "max_length": 65536,
                },
                # gen_kwargs chosen according to https://huggingface.co/Qwen/Qwen3-32B#best-practices
                gen_kwargs={
                    "stream": "false",
                    "max_gen_toks": 32768,
                    "until": [],
                    "do_sample": "true",
                    "temperature": 0.6,
                    "top_k": 20,
                    "top_p": 0.95,
                },
                seed=42,
                num_fewshot=0,
                batch_size=32,
                log_samples=True,
            ),
            EvalTask(
                task_name="r1_math500",
                score=EvalTaskScore(
                    published_score=96.1,  
                    published_score_ref="https://artificialanalysis.ai/models/comparisons/qwen3-32b-instruct-reasoning-vs-qwen3-4b-instruct",
                    gpu_reference_score=96.10,  # Estimate - needs to be validated
                    gpu_reference_score_ref="TBD",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "exact_match,none",
                        ],
                        "unit": "percent",
                    },
                ),
                workflow_venv_type=WorkflowVenvType.EVALS,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=True,
                model_kwargs={
                    "model": "Qwen/Qwen3-32B",
                    "base_url": "http://127.0.0.1:8000/v1/completions",
                    "tokenizer_backend": "huggingface",
                    "max_length": 65536,
                },
                # gen_kwargs chosen according to https://huggingface.co/Qwen/Qwen3-32B#best-practices
                gen_kwargs={
                    "stream": "false",
                    "max_gen_toks": 32768,
                    "until": [],
                    "do_sample": "true",
                    "temperature": 0.6,
                    "top_k": 20,
                    "top_p": 0.95,
                },
                seed=42,
                num_fewshot=0,
                batch_size=32,
                log_samples=True,
            ),
            EvalTask(
                task_name="r1_gpqa_diamond",
                score=EvalTaskScore(
                    published_score=66.80,  
                    published_score_ref="https://artificialanalysis.ai/models/comparisons/qwen3-32b-instruct-reasoning-vs-qwen3-4b-instruct",
                    gpu_reference_score=66.80,  # Estimate - needs to be validated
                    gpu_reference_score_ref="TBD",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "exact_match,none",
                        ],
                        "unit": "percent",
                    },
                ),
                workflow_venv_type=WorkflowVenvType.EVALS,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=True,
                model_kwargs={
                    "model": "Qwen/Qwen3-32B",
                    "base_url": "http://127.0.0.1:8000/v1/completions",
                    "tokenizer_backend": "huggingface",
                    "max_length": 65536,
                },
                # gen_kwargs chosen according to https://huggingface.co/Qwen/Qwen3-32B#best-practices
                gen_kwargs={
                    "stream": "false",
                    "max_gen_toks": 32768,
                    "until": [],
                    "do_sample": "true",
                    "temperature": 0.6,
                    "top_k": 20,
                    "top_p": 0.95,
                },
                seed=42,
                num_fewshot=0,
                batch_size=32,
                log_samples=True,
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="mistralai/Mistral-7B-Instruct-v0.3",
        tasks=[
            EvalTask(
                task_name="ifeval",
                score=EvalTaskScore(
                    published_score=54.65,
                    published_score_ref="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/?search=mistralai%2FMistral-7B-Instruct-v0.3&official=true",
                    gpu_reference_score=48.24,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/248#issuecomment-2922880818",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "prompt_level_strict_acc,none",
                            "inst_level_strict_acc,none",
                        ],
                        "unit": "percent",
                    },
                ),
            ),
            EvalTask(
                task_name="mmlu_pro",
                num_fewshot=5,
                score=EvalTaskScore(
                    published_score=23.06,
                    published_score_ref="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/?search=mistralai%2FMistral-7B-Instruct-v0.3&official=true",
                    gpu_reference_score=29.12,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/248#issuecomment-2922880818",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "exact_match,custom-extract",
                        ],
                        "unit": "percent",
                    },
                ),
            )
        ],
    ),
    EvalConfig(
        hf_model_repo="Qwen/QwQ-32B",
        tasks=[
            EvalTask(
                task_name="r1_aime24",
                score=EvalTaskScore(
                    published_score=80.00,
                    published_score_ref="https://qwenlm.github.io/blog/qwq-32b/",
                    gpu_reference_score=80.00,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/141",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "exact_match,none",
                        ],
                        "unit": "percent",
                    },
                ),
                workflow_venv_type=WorkflowVenvType.EVALS,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=True,
                model_kwargs={
                    "model": "Qwen/QwQ-32B",
                    "base_url": "http://127.0.0.1:8000/v1/completions",
                    "tokenizer_backend": "huggingface",
                    "max_concurrent": 32,
                    "max_length": 65536,
                },
                gen_kwargs={
                    "stream": "false",
                },
                seed=42,
                num_fewshot=0,
                batch_size=32,
                log_samples=True,
            ),
            EvalTask(
                task_name="r1_math500",
                score=EvalTaskScore(
                    published_score=96.05,
                    published_score_ref="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/?search=QwQ-32B&official=true",
                    gpu_reference_score=96.00,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/141",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "exact_match,none",
                        ],
                        "unit": "percent",
                    },
                ),
                workflow_venv_type=WorkflowVenvType.EVALS,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=True,
                model_kwargs={
                    "model": "Qwen/QwQ-32B",
                    "base_url": "http://127.0.0.1:8000/v1/completions",
                    "tokenizer_backend": "huggingface",
                    "max_concurrent": 32,
                    "max_length": 65536,
                },
                gen_kwargs={
                    "stream": "false",
                },
                seed=42,
                num_fewshot=0,
                batch_size=32,
                log_samples=True,
            ),
            EvalTask(
                task_name="r1_gpqa_diamond",
                score=EvalTaskScore(
                    published_score=67.17,
                    published_score_ref="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/?search=QwQ-32B&official=true",
                    gpu_reference_score=63.63,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/141",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "exact_match,none",
                        ],
                        "unit": "percent",
                    },
                ),
                workflow_venv_type=WorkflowVenvType.EVALS,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=True,
                model_kwargs={
                    "model": "Qwen/QwQ-32B",
                    "base_url": "http://127.0.0.1:8000/v1/completions",
                    "tokenizer_backend": "huggingface",
                    "max_concurrent": 32,
                    "max_length": 65536,
                },
                gen_kwargs={
                    "stream": "false",
                },
                seed=42,
                num_fewshot=0,
                batch_size=32,
                log_samples=True,
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        tasks=[
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
                workflow_venv_type=WorkflowVenvType.EVALS,
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
            EvalTask(
                task_name="r1_gpqa_diamond",
                score=EvalTaskScore(
                    published_score=65.20,
                    published_score_ref="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
                    gpu_reference_score=55.05,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/112",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "exact_match,none",
                        ],
                        "unit": "percent",
                    },
                ),
                workflow_venv_type=WorkflowVenvType.EVALS,
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
        ],
    ),
    EvalConfig(
        hf_model_repo="Qwen/Qwen2.5-72B-Instruct",
        tasks=[
            EvalTask(
                task_name="leaderboard_ifeval",
                score=EvalTaskScore(
                    published_score=84.1,
                    published_score_ref="https://arxiv.org/abs/2412.15115",
                    gpu_reference_score=82.99,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/143#issuecomment-2770711161",
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
                    published_score=None,
                    published_score_ref=None,
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
                score=EvalTaskScore(
                    published_score=None,
                    published_score_ref=None,
                    gpu_reference_score=42.93,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/143#issuecomment-2770711161",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "exact_match,flexible-extract",
                        ],
                        "unit": "percent",
                    },
                ),
            ),
            EvalTask(
                task_name="mmlu_pro",
                num_fewshot=5,
                score=EvalTaskScore(
                    published_score=None,
                    published_score_ref=None,
                    gpu_reference_score=34.79,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/143#issuecomment-2770711161",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "exact_match,custom-extract",
                        ],
                        "unit": "percent",
                    },
                ),
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="Qwen/Qwen2.5-7B-Instruct",
        tasks=[
            EvalTask(
                task_name="leaderboard_ifeval",
                score=EvalTaskScore(
                    published_score=71.2,
                    published_score_ref="https://arxiv.org/abs/2412.15115",
                    gpu_reference_score=69.13,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/125#issuecomment-2762236580",
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
                    published_score=None,
                    published_score_ref=None,
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
                score=EvalTaskScore(
                    published_score=None,
                    published_score_ref=None,
                    gpu_reference_score=33.8,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/125#issuecomment-2762236580",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "exact_match,flexible-extract",
                        ],
                        "unit": "percent",
                    },
                ),
            ),
            EvalTask(
                task_name="mmlu_pro",
                num_fewshot=5,
                score=EvalTaskScore(
                    published_score=None,
                    published_score_ref=None,
                    gpu_reference_score=28.09,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/125#issuecomment-2762236580",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "exact_match,custom-extract",
                        ],
                        "unit": "percent",
                    },
                ),
            ),
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
                    gpu_reference_score=91.35,
                    gpu_reference_score_ref="https://docs.google.com/spreadsheets/d/1kFIUj9Bp5WJ0lW3QPwQRRWDyLRieedKrFZqfxWBfeNw/edit?gid=0#gid=0&range=J86",
                    published_score=92.1,
                    published_score_ref="https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct#instruction-tuned-models",
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
                    gpu_reference_score=60.04,
                    gpu_reference_score_ref="https://docs.google.com/spreadsheets/d/1kFIUj9Bp5WJ0lW3QPwQRRWDyLRieedKrFZqfxWBfeNw/edit?gid=0#gid=0&range=J87",
                    published_score=50.5,
                    published_score_ref="https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct#instruction-tuned-models",
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
                    published_score=32.8,
                    published_score_ref="https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct#instruction-tuned-models",
                    gpu_reference_score=33.035,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/131#issuecomment-2769531835",
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
                    published_score=51.9,
                    published_score_ref="https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct#instruction-tuned-models",
                    gpu_reference_score=47.06,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/131#issuecomment-2769531835",
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
                score=EvalTaskScore(
                    published_score=50.7,
                    published_score_ref="https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct#instruction-tuned-models",
                    gpu_reference_score=43.11,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/131#issuecomment-2769531835",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "acc,none",
                        ],
                        "unit": "percent",
                    },
                ),
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
        hf_model_repo="meta-llama/Llama-3.2-90B-Vision-Instruct",
        tasks=[
            EvalTask(
                task_name="meta_gpqa",
                workflow_venv_type=WorkflowVenvType.EVALS_META,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
                score=EvalTaskScore(
                    published_score=46.7,
                    published_score_ref="https://huggingface.co/meta-llama/Llama-3.2-90B-Vision-Instruct#instruction-tuned-models",
                    gpu_reference_score=44.0,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/131#issuecomment-2769531835",
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
                    published_score=68.0,
                    published_score_ref="https://huggingface.co/meta-llama/Llama-3.2-90B-Vision-Instruct#instruction-tuned-models",
                    gpu_reference_score=65.0,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/131#issuecomment-2769531835",
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
                score=EvalTaskScore(
                    published_score=60.3,
                    published_score_ref="https://huggingface.co/meta-llama/Llama-3.2-90B-Vision-Instruct#instruction-tuned-models",
                    gpu_reference_score=48.1,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/379#issuecomment-3071570950",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "acc,none",
                        ],
                        "unit": "percent",
                    },
                ),
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
                    published_score=32.8,
                    published_score_ref="https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct#instruction-tuned-models",
                    gpu_reference_score=32.59,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/139#issuecomment-2761649617",
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
                    published_score=48.0,
                    published_score_ref="https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct#instruction-tuned-models",
                    gpu_reference_score=40.70,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/139#issuecomment-2761649617",
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
                task_name="livecodebench",
                workflow_venv_type=WorkflowVenvType.EVALS,
                score=EvalTaskScore(
                    published_score=None,
                    published_score_ref=None,
                    gpu_reference_score=13.93,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/311#issuecomment-2991859987",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "acc,none",
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
                    published_score=27.2,
                    published_score_ref="https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct#instruction-tuned-models",
                    gpu_reference_score=27.01,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/139#issuecomment-2761649617",
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
            #         published_score=30.6,
            #         published_score_ref="https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct#instruction-tuned-models",
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
                    published_score=87.5,
                    published_score_ref="https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct#instruction-tuned-models",
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
                    published_score=46.7,
                    published_score_ref="https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct#instruction-tuned-models",
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
                    published_score=80.4,
                    published_score_ref="https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct#instruction-tuned-models",
                    gpu_reference_score=81.38,
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
                    published_score=30.4,
                    published_score_ref="https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct#instruction-tuned-models",
                    gpu_reference_score=28.34,
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
    model_spec.model_name: _eval_config_map[model_spec.hf_model_repo]
    for _, model_spec in MODEL_SPECS.items()
    if model_spec.hf_model_repo in _eval_config_map
}

