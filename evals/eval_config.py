# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Callable

from workflows.workflow_types import WorkflowVenvType
from workflows.utils import map_configs_by_attr, get_repo_root_path
from workflows.model_config import MODEL_CONFIGS
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
    eval_script: str = None  # workflow-specific eval script

    def __post_init__(self):
        self.validate_data()

    def validate_data(self):
        if self.eval_script is not None:
            path = Path(self.eval_script)
            assert path.exists(), f"eval_script must exist: {self.eval_script}"


# Note: meta evals defined in: https://github.com/meta-llama/llama-cookbook/blob/main/end-to-end-use-cases/benchmarks/llm_eval_harness/meta_eval/eval_config.yaml
# Note: meta_math_hard for Llama 3.1 models has a bug: see https://github.com/tenstorrent/tt-inference-server/issues/155
# Note: reasoning models (QwQ-32B, DeepSeek-R1-Distill-Llama-70B) need evals allowing more tokens generated


# Helper function to get appropriate result keys based on dataset
def _get_whisper_audio_eval_result_keys(dataset: str):
    """Get result keys for whisper audio evaluation based on dataset."""
    if dataset == "openslr_librispeech":
        return [("openslr_librispeech_other", "wer,none")]
    elif dataset == "librispeech_test_other":
        return [("librispeech_test_other", "wer,none")]
    elif dataset == "librispeech_full":
        return [
            ("librispeech_dev_clean", "wer,none"),
            ("librispeech_dev_other", "wer,none"), 
            ("librispeech_test_clean", "wer,none"),
            ("librispeech_test_other", "wer,none"),
        ]
    else:
        raise ValueError(f"Invalid dataset: {dataset}")

# Legacy function for backward compatibility
def _get_whisper_librispeech_result_keys(scope: str):
    """Get result keys for whisper LibriSpeech evaluation based on scope. (Legacy function)"""
    if scope == "test_other":
        return _get_whisper_audio_eval_result_keys("librispeech_test_other")
    elif scope == "full":
        return _get_whisper_audio_eval_result_keys("librispeech_full")
    else:
        raise ValueError(f"Invalid scope: {scope}")


_eval_config_list = [
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
                workflow_venv_type=WorkflowVenvType.EVALS_REASON,
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
                workflow_venv_type=WorkflowVenvType.EVALS_REASON,
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
                workflow_venv_type=WorkflowVenvType.EVALS_REASON,
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
                    published_score=46.7,
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
                    published_score=68.0,
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
                    tolerance=0.15,
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
    # TODO: Probably create DockerEvalConfig because this doesn't make sense to
    # mix with these "vLLM" eval configs
    EvalConfig(
        hf_model_repo="openai/whisper-large-v3",
        eval_script=get_repo_root_path()
        / "evals"
        / "run_docker_evals_scripts"
        / "whisper_eval.sh",
        tasks=[
            EvalTask(
                task_name="librispeech",
                eval_class="whisper_tt",
                batch_size=1,
                max_concurrent=1,
                apply_chat_template=False,
                workflow_venv_type=WorkflowVenvType.DOCKER_EVALS_LMMS_EVAL,
                score=EvalTaskScore(
                    published_score=(100 - 5.25),  # Will be updated at runtime
                    gpu_reference_score=(100 - 3.805),  # Will be updated at runtime
                    published_score_ref="https://arxiv.org/pdf/2311.00430",
                    score_func=score_multilevel_keys_mean,
                    score_func_kwargs={
                        "result_keys": [  # Will be updated at runtime based on scope
                            ("librispeech_test_other", "wer,none"),
                        ],
                        "unit": "WER",
                    },
                ),
            )
        ],
    ),
    EvalConfig(
        hf_model_repo="distil-whisper/distil-large-v3",
        eval_script=get_repo_root_path()
        / "evals"
        / "run_docker_evals_scripts"
        / "whisper_eval.sh",
        tasks=[
            EvalTask(
                task_name="librispeech",
                eval_class="whisper_tt",
                batch_size=1,
                max_concurrent=1,
                apply_chat_template=False,
                workflow_venv_type=WorkflowVenvType.DOCKER_EVALS_LMMS_EVAL,
                score=EvalTaskScore(
                    published_score=(100 - 5.25),  # Will be updated at runtime
                    gpu_reference_score=(100 - 3.805),  # Will be updated at runtime
                    published_score_ref="https://arxiv.org/pdf/2311.00430",
                    score_func=score_multilevel_keys_mean,
                    score_func_kwargs={
                        "result_keys": [  # Will be updated at runtime based on scope
                            ("librispeech_test_other", "wer,none"),
                        ],
                        "unit": "WER",
                    },
                ),
            )
        ],
    ),
]


_eval_config_map = map_configs_by_attr(
    config_list=_eval_config_list, attr="hf_model_repo"
)
EVAL_CONFIGS = {
    model_config.model_name: _eval_config_map[model_config.hf_model_repo]
    for _, model_config in MODEL_CONFIGS.items()
    if model_config.hf_model_repo in _eval_config_map
}
