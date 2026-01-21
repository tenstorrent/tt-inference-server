# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Union

from evals.eval_utils import (
    score_multilevel_keys_mean,
    score_task_keys_mean,
    score_task_single_key,
)
from workflows.model_spec import MODEL_SPECS
from workflows.utils import map_configs_by_attr
from workflows.workflow_types import EvalLimitMode, WorkflowVenvType


@dataclass(frozen=True)
class EvalTaskScore:
    published_score: float
    published_score_ref: str
    score_func: Callable
    gpu_reference_score: float = None
    gpu_reference_score_ref: str = None
    score_func_kwargs: Dict[str, str] = field(default_factory=dict)
    tolerance: float = 0.05


@dataclass(frozen=True)
class EvalTask:
    task_name: str
    score: EvalTaskScore = None
    workflow_venv_type: WorkflowVenvType = WorkflowVenvType.EVALS_COMMON
    eval_class: str = "local-completions"
    tokenizer_backend: str = "huggingface"
    # Note: batch_size is set to 1 because max_concurrent is set to 32
    # this means that 32 requests are sent concurrently by lm-eval / lmms-eval
    # for clarity, the client side eval scripts cannot control the batch size
    # so setting just multiplys the max_concurrent which is misleading
    batch_size: int = 1
    max_concurrent: int = 32
    num_fewshot: int = 0
    seed: int = 42
    use_chat_api: bool = False
    apply_chat_template: bool = True
    log_samples: bool = True
    gen_kwargs: Dict[str, str] = field(default_factory=lambda: {"stream": "False"})
    model_kwargs: Dict[str, str] = field(default_factory=lambda: {})
    # Note: include_path is specified relative to the respective venv
    include_path: str = None
    # Optional: kwargs passed to task custom_dataset loaders (e.g., RULER sequence length configs)
    custom_dataset_kwargs: Dict[str, Union[str, List[int]]] = None
    # Optional: limit the number of samples passed to lm_eval (--limit)
    # Limit the number of examples per task.
    # If <1, limit is a percentage of the total number of examples.
    limit_samples_map: Dict[EvalLimitMode, Union[float, int]] = field(
        default_factory=lambda: {
            # this defines smoke test limit to 1% for all models unless overridden
            EvalLimitMode.SMOKE_TEST: 0.01,
        }
    )

    def __post_init__(self):
        self.validate_data()
        self._infer_data()

    def _infer_data(self):
        if self.use_chat_api and self.eval_class == "local-completions":
            object.__setattr__(self, "eval_class", "local-chat-completions")

        if self.workflow_venv_type == WorkflowVenvType.EVALS_META:
            # max_concurrent is not supported in lm-eval==0.4.3
            object.__setattr__(self, "batch_size", self.max_concurrent)
            object.__setattr__(self, "max_concurrent", None)
            if self.model_kwargs:
                raise ValueError("model_kwargs are not supported in lm-eval==0.4.3")

    def validate_data(self):
        assert not (self.use_chat_api and self.apply_chat_template), (
            "Chat API applies chat template."
        )


@dataclass(frozen=True)
class EvalConfig:
    hf_model_repo: str
    tasks: List[EvalTask]


# Note: meta evals defined in: https://github.com/meta-llama/llama-cookbook/blob/main/end-to-end-use-cases/benchmarks/llm_eval_harness/meta_eval/eval_config.yaml
# Note: meta_math_hard for Llama 3.1 models has a bug: see https://github.com/tenstorrent/tt-inference-server/issues/155
# Note: reasoning models (QwQ-32B, DeepSeek-R1-Distill-Llama-70B) need evals allowing more tokens generated


_eval_config_list = [
    EvalConfig(
        hf_model_repo="arcee-ai/AFM-4.5B",
        tasks=[
            EvalTask(
                task_name="ifeval",
                score=EvalTaskScore(
                    published_score=None,
                    published_score_ref=None,
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
                    published_score=None,
                    published_score_ref=None,
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
        hf_model_repo="google/gemma-3-4b-it",
        tasks=[
            EvalTask(
                task_name="ifeval",
                score=EvalTaskScore(
                    published_score=90.2,
                    published_score_ref="https://storage.googleapis.com/deepmind-media/gemma/Gemma3Report.pdf",
                    gpu_reference_score=79.5,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/521#issuecomment-3249524785",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "prompt_level_strict_acc,none",
                            "inst_level_strict_acc,none",
                        ],
                        "unit": "percent",
                    },
                ),
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.5,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
            ),
            EvalTask(
                task_name="mbpp_instruct",
                workflow_venv_type=WorkflowVenvType.EVALS_COMMON,
                score=EvalTaskScore(
                    published_score=46.0,
                    published_score_ref="https://storage.googleapis.com/deepmind-media/gemma/Gemma3Report.pdf",
                    gpu_reference_score=58.4,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/521#issuecomment-3533832922",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "pass_at_1,extract_code",
                        ],
                        "unit": "percent",
                    },
                ),
                apply_chat_template=True,
                gen_kwargs={
                    "max_gen_toks": "256",
                    "do_sample": "false",
                    "stream": "false",
                },
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.5,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
            ),
            EvalTask(
                eval_class="openai_compatible",
                task_name="chartqa",
                workflow_venv_type=WorkflowVenvType.EVALS_VISION,
                apply_chat_template=False,
                use_chat_api=True,
                score=EvalTaskScore(
                    published_score=63.6,
                    published_score_ref="https://storage.googleapis.com/deepmind-media/gemma/Gemma3Report.pdf",
                    gpu_reference_score=40.0,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/521#issuecomment-3249524785",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "relaxed_overall,none",
                        ],
                        "unit": "percent",
                    },
                ),
                model_kwargs={
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
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.2,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
            ),
            EvalTask(
                task_name="ruler",
                workflow_venv_type=WorkflowVenvType.EVALS_COMMON,
                score=EvalTaskScore(
                    published_score=61.4,  # 61.4% on 32k tokens
                    published_score_ref="https://arxiv.org/html/2503.19786v1",
                    gpu_reference_score=None,
                    gpu_reference_score_ref="TBD",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "4096,none",
                            "8192,none",
                            "16384,none",
                            "32768,none",
                            "65536,none",
                        ],
                        "unit": "percent",
                    },
                ),
                model_kwargs={
                    # "max_length": 131072,  # Support long context as recommended for RULER
                    "max_length": 65536,  # Support long context as recommended for RULER
                },
                gen_kwargs={
                    "stream": "false",
                    "max_gen_toks": 256,  # Reasonable limit for RULER responses
                    "do_sample": "false",  # Deterministic for evaluation
                },
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: None,
                    EvalLimitMode.SMOKE_TEST: None,  # No global limit - we apply per-length limiting
                },
                custom_dataset_kwargs={
                    # "max_seq_lengths": [4096, 8192, 16384, 32768, 65536, 131072],
                    "max_seq_lengths": [4096, 8192, 16384, 32768, 65536],
                    "pretrained": "google/gemma-3-4b-it",  # Provide model name for RULER tokenizer
                    "num_samples_per_length": 50,  # Number of samples per sequence length per sub-task in full evaluation mode
                    "limit_factor": 0.1,  # Smoke/CI test multiplier: reduces to 5 samples per sequence length
                },
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="google/gemma-3-27b-it",
        tasks=[
            EvalTask(
                task_name="ifeval",
                score=EvalTaskScore(
                    published_score=90.4,
                    published_score_ref="https://storage.googleapis.com/deepmind-media/gemma/Gemma3Report.pdf",
                    gpu_reference_score=83.3,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/607#issuecomment-3250668712",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "prompt_level_strict_acc,none",
                            "inst_level_strict_acc,none",
                        ],
                        "unit": "percent",
                    },
                ),
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.5,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
            ),
            EvalTask(
                task_name="mbpp_instruct",
                workflow_venv_type=WorkflowVenvType.EVALS_COMMON,
                score=EvalTaskScore(
                    published_score=65.6,
                    published_score_ref="https://storage.googleapis.com/deepmind-media/gemma/Gemma3Report.pdf",
                    gpu_reference_score=69.2,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/607#issuecomment-3524037012",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "pass_at_1,extract_code",
                        ],
                        "unit": "percent",
                    },
                ),
                apply_chat_template=True,
                gen_kwargs={
                    "max_gen_toks": "256",
                    "do_sample": "false",
                    "stream": "false",
                },
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.5,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
            ),
            EvalTask(
                eval_class="openai_compatible",
                task_name="chartqa",
                workflow_venv_type=WorkflowVenvType.EVALS_VISION,
                apply_chat_template=False,
                use_chat_api=True,
                score=EvalTaskScore(
                    published_score=76.3,
                    published_score_ref="https://storage.googleapis.com/deepmind-media/gemma/Gemma3Report.pdf",
                    gpu_reference_score=47.6,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/607#issuecomment-3250668712",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "relaxed_overall,none",
                        ],
                        "unit": "percent",
                    },
                ),
                model_kwargs={
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
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.2,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
            ),
            EvalTask(
                task_name="ruler",
                workflow_venv_type=WorkflowVenvType.EVALS_COMMON,
                score=EvalTaskScore(
                    published_score=91.1,  # 91.1% on 32k tokens
                    published_score_ref="https://arxiv.org/html/2503.19786v1",
                    gpu_reference_score=None,
                    gpu_reference_score_ref="TBD",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "4096,none",
                            "8192,none",
                            "16384,none",
                            "32768,none",
                            "65536,none",
                        ],
                        "unit": "percent",
                    },
                ),
                model_kwargs={
                    # "max_length": 131072,  # Support long context as recommended for RULER
                    "max_length": 65536,  # Support long context as recommended for RULER
                },
                gen_kwargs={
                    "stream": "false",
                    "max_gen_toks": 256,  # Reasonable limit for RULER responses
                    "do_sample": "false",  # Deterministic for evaluation
                },
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: None,
                    EvalLimitMode.SMOKE_TEST: None,  # No global limit - we apply per-length limiting
                },
                custom_dataset_kwargs={
                    # "max_seq_lengths": [4096, 8192, 16384, 32768, 65536, 131072],
                    "max_seq_lengths": [4096, 8192, 16384, 32768, 65536],
                    "pretrained": "google/gemma-3-27b-it",  # Provide model name for RULER tokenizer
                    "num_samples_per_length": 50,  # Number of samples per sequence length per sub-task in full evaluation mode
                    "limit_factor": 0.1,  # Smoke/CI test multiplier: reduces to 5 samples per sequence length
                },
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="Qwen/Qwen2.5-VL-3B-Instruct",
        tasks=[
            EvalTask(
                eval_class="openai_compatible",
                task_name="chartqa",
                workflow_venv_type=WorkflowVenvType.EVALS_VISION,
                apply_chat_template=False,
                use_chat_api=True,
                score=EvalTaskScore(
                    published_score=84.0,
                    published_score_ref="https://arxiv.org/pdf/2502.13923",
                    gpu_reference_score=83.6,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/392#issuecomment-3422979962",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "relaxed_overall,none",
                        ],
                        "unit": "percent",
                    },
                ),
                model_kwargs={
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
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.2,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
            ),
            EvalTask(
                eval_class="openai_compatible",
                task_name="docvqa_val",
                workflow_venv_type=WorkflowVenvType.EVALS_VISION,
                apply_chat_template=False,
                use_chat_api=True,
                score=EvalTaskScore(
                    published_score=93.9,
                    published_score_ref="https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct#image-benchmark",
                    gpu_reference_score=92.5,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/392#issuecomment-3429715261",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "anls,none",
                        ],
                        "unit": "percent",
                    },
                ),
                model_kwargs={
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
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.05,
                    EvalLimitMode.SMOKE_TEST: 0.001,
                },
            ),
            EvalTask(
                eval_class="openai_compatible",
                task_name="mmmu_val",
                workflow_venv_type=WorkflowVenvType.EVALS_VISION,
                apply_chat_template=False,
                use_chat_api=True,
                score=EvalTaskScore(
                    published_score=53.1,
                    published_score_ref="https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct#image-benchmark",
                    gpu_reference_score=46.4,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/392#issuecomment-3422979962",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "mmmu_acc,none",
                        ],
                        "unit": "percent",
                    },
                ),
                model_kwargs={
                    "num_concurrent": 32,
                    "max_retries": 1,
                    "tokenized_requests": "False",
                    "add_bos_token": "True",
                    "timeout": "9999",
                    "eos_string": "<|end_of_text|>",
                },
                gen_kwargs={
                    "stop": "<|eot_id|>",
                    "stream": "False",
                    "max_new_tokens": "512",
                },
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="Qwen/Qwen2.5-VL-7B-Instruct",
        tasks=[
            EvalTask(
                eval_class="openai_compatible",
                task_name="chartqa",
                workflow_venv_type=WorkflowVenvType.EVALS_VISION,
                apply_chat_template=False,
                use_chat_api=True,
                score=EvalTaskScore(
                    published_score=87.3,
                    published_score_ref="https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct#image-benchmark",
                    gpu_reference_score=84.0,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/391#issuecomment-3423157480",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "relaxed_overall,none",
                        ],
                        "unit": "percent",
                    },
                ),
                model_kwargs={
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
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.2,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
            ),
            EvalTask(
                eval_class="openai_compatible",
                task_name="docvqa_val",
                workflow_venv_type=WorkflowVenvType.EVALS_VISION,
                apply_chat_template=False,
                use_chat_api=True,
                score=EvalTaskScore(
                    published_score=95.7,
                    published_score_ref="https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct#image-benchmark",
                    gpu_reference_score=94.97,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/391#issuecomment-3429870349",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "anls,none",
                        ],
                        "unit": "percent",
                    },
                ),
                model_kwargs={
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
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.05,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
            ),
            EvalTask(
                eval_class="openai_compatible",
                task_name="mmmu_val",
                workflow_venv_type=WorkflowVenvType.EVALS_VISION,
                apply_chat_template=False,
                use_chat_api=True,
                score=EvalTaskScore(
                    published_score=58.6,
                    published_score_ref="https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct#image-benchmark",
                    gpu_reference_score=50.78,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/391#issuecomment-3423157480",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "mmmu_acc,none",
                        ],
                        "unit": "percent",
                    },
                ),
                model_kwargs={
                    "num_concurrent": 32,
                    "max_retries": 1,
                    "tokenized_requests": "False",
                    "add_bos_token": "True",
                    "timeout": "9999",
                    "eos_string": "<|end_of_text|>",
                },
                gen_kwargs={
                    "stop": "<|eot_id|>",
                    "stream": "False",
                    "max_new_tokens": "512",
                },
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="Qwen/Qwen2.5-VL-32B-Instruct",
        tasks=[
            EvalTask(
                eval_class="openai_compatible",
                task_name="chartqa",
                workflow_venv_type=WorkflowVenvType.EVALS_VISION,
                apply_chat_template=False,
                use_chat_api=True,
                score=EvalTaskScore(
                    published_score=83.29,
                    published_score_ref="https://arxiv.org/html/2509.07966v1",
                    gpu_reference_score=66.04,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/839#issuecomment-3423691284",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "relaxed_overall,none",
                        ],
                        "unit": "percent",
                    },
                ),
                model_kwargs={
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
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.2,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
            ),
            EvalTask(
                eval_class="openai_compatible",
                task_name="docvqa_val",
                workflow_venv_type=WorkflowVenvType.EVALS_VISION,
                apply_chat_template=False,
                use_chat_api=True,
                score=EvalTaskScore(
                    published_score=94.8,
                    published_score_ref="https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct#image-benchmark",
                    gpu_reference_score=92.71,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/839#issuecomment-3451730277",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "anls,none",
                        ],
                        "unit": "percent",
                    },
                ),
                model_kwargs={
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
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.05,
                    EvalLimitMode.SMOKE_TEST: 0.001,
                },
            ),
            EvalTask(
                eval_class="openai_compatible",
                task_name="mmmu_val",
                workflow_venv_type=WorkflowVenvType.EVALS_VISION,
                apply_chat_template=False,
                use_chat_api=True,
                score=EvalTaskScore(
                    published_score=70,
                    published_score_ref="https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct#image-benchmark",
                    gpu_reference_score=58.22,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/839#issuecomment-3423691284",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "mmmu_acc,none",
                        ],
                        "unit": "percent",
                    },
                ),
                model_kwargs={
                    "num_concurrent": 32,
                    "max_retries": 1,
                    "tokenized_requests": "False",
                    "add_bos_token": "True",
                    "timeout": "9999",
                    "eos_string": "<|end_of_text|>",
                },
                gen_kwargs={
                    "stop": "<|eot_id|>",
                    "stream": "False",
                    "max_new_tokens": "512",
                },
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="Qwen/Qwen2.5-VL-72B-Instruct",
        tasks=[
            EvalTask(
                eval_class="openai_compatible",
                task_name="chartqa",
                workflow_venv_type=WorkflowVenvType.EVALS_VISION,
                apply_chat_template=False,
                use_chat_api=True,
                score=EvalTaskScore(
                    published_score=89.6,
                    published_score_ref="https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct#image-benchmark",
                    gpu_reference_score=77.12,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/235#issuecomment-2902002942",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "relaxed_overall,none",
                        ],
                        "unit": "percent",
                    },
                ),
                model_kwargs={
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
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.2,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
            ),
            EvalTask(
                eval_class="openai_compatible",
                task_name="docvqa_val",
                workflow_venv_type=WorkflowVenvType.EVALS_VISION,
                apply_chat_template=False,
                use_chat_api=True,
                score=EvalTaskScore(
                    published_score=96.4,
                    published_score_ref="https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct#image-benchmark",
                    gpu_reference_score=81.4,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/235#issuecomment-2902002942",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "anls,none",
                        ],
                        "unit": "percent",
                    },
                ),
                model_kwargs={
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
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.05,
                    EvalLimitMode.SMOKE_TEST: 0.001,
                },
            ),
            EvalTask(
                eval_class="openai_compatible",
                task_name="mmmu_val",
                workflow_venv_type=WorkflowVenvType.EVALS_VISION,
                apply_chat_template=False,
                use_chat_api=True,
                score=EvalTaskScore(
                    published_score=70.2,
                    published_score_ref="https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct#image-benchmark",
                    gpu_reference_score=59.56,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/235#issuecomment-2902002942",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "mmmu_acc,none",
                        ],
                        "unit": "percent",
                    },
                ),
                model_kwargs={
                    "num_concurrent": 32,
                    "max_retries": 1,
                    "tokenized_requests": "False",
                    "add_bos_token": "True",
                    "timeout": "9999",
                    "eos_string": "<|end_of_text|>",
                },
                gen_kwargs={
                    "stop": "<|eot_id|>",
                    "stream": "False",
                    "max_new_tokens": "512",
                },
            ),
        ],
    ),
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
                workflow_venv_type=WorkflowVenvType.EVALS_COMMON,
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
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.2,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
            ),
            EvalTask(
                task_name="mmlu_pro",
                num_fewshot=5,
                score=EvalTaskScore(
                    published_score=56.73,
                    published_score_ref="https://arxiv.org/pdf/2505.09388",
                    gpu_reference_score=66.07,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/384#issuecomment-3176953494",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "exact_match,custom-extract",
                        ],
                        "unit": "percent",
                    },
                ),
                workflow_venv_type=WorkflowVenvType.EVALS_COMMON,
                model_kwargs={
                    "model": "Qwen/Qwen3-8B",
                    "base_url": "http://127.0.0.1:8000/v1/completions",
                    "tokenizer_backend": "huggingface",
                    "max_length": 65536,
                    "timeout": "3600",
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
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.05,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
            ),
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
                workflow_venv_type=WorkflowVenvType.EVALS_COMMON,
                model_kwargs={
                    "model": "Qwen/Qwen3-32B",
                    "base_url": "http://127.0.0.1:8000/v1/completions",
                    "tokenizer_backend": "huggingface",
                    "max_length": 65536,
                    "timeout": "3600",
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
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.5,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
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
                workflow_venv_type=WorkflowVenvType.EVALS_COMMON,
                model_kwargs={
                    "model": "Qwen/Qwen3-32B",
                    "base_url": "http://127.0.0.1:8000/v1/completions",
                    "tokenizer_backend": "huggingface",
                    "max_length": 65536,
                    "timeout": "3600",
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
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.2,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
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
                max_concurrent=16,
                workflow_venv_type=WorkflowVenvType.EVALS_COMMON,
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
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.2,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
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
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.2,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
            ),
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
                workflow_venv_type=WorkflowVenvType.EVALS_COMMON,
                model_kwargs={
                    "model": "Qwen/QwQ-32B",
                    "base_url": "http://127.0.0.1:8000/v1/completions",
                    "tokenizer_backend": "huggingface",
                    "max_length": 65536,
                    "timeout": "3600",
                },
                gen_kwargs={
                    "stream": "false",
                },
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.5,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
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
                workflow_venv_type=WorkflowVenvType.EVALS_COMMON,
                model_kwargs={
                    "model": "Qwen/QwQ-32B",
                    "base_url": "http://127.0.0.1:8000/v1/completions",
                    "tokenizer_backend": "huggingface",
                    "max_length": 65536,
                    "timeout": "3600",
                },
                gen_kwargs={
                    "stream": "false",
                },
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.2,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
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
                workflow_venv_type=WorkflowVenvType.EVALS_COMMON,
                model_kwargs={
                    "model": "Qwen/QwQ-32B",
                    "base_url": "http://127.0.0.1:8000/v1/completions",
                    "tokenizer_backend": "huggingface",
                    "max_length": 65536,
                },
                gen_kwargs={
                    "stream": "false",
                },
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.2,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
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
                workflow_venv_type=WorkflowVenvType.EVALS_COMMON,
                model_kwargs={
                    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
                    "base_url": "http://127.0.0.1:8000/v1/completions",
                    "tokenizer_backend": "huggingface",
                    "max_length": 65536,
                },
                gen_kwargs={"stream": "false", "max_gen_toks": "32768"},
                seed=42,
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.2,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
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
                workflow_venv_type=WorkflowVenvType.EVALS_COMMON,
                model_kwargs={
                    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
                    "base_url": "http://127.0.0.1:8000/v1/completions",
                    "tokenizer_backend": "huggingface",
                    "max_length": 65536,
                },
                gen_kwargs={"stream": "false", "max_gen_toks": "32768"},
                seed=42,
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="deepseek-ai/DeepSeek-R1-0528",
        tasks=[
            EvalTask(
                task_name="r1_aime24",
                score=EvalTaskScore(
                    published_score=None,
                    published_score_ref=None,
                    gpu_reference_score=None,
                    gpu_reference_score_ref=None,
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "exact_match,none",
                        ],
                        "unit": "percent",
                    },
                ),
                workflow_venv_type=WorkflowVenvType.EVALS_COMMON,
                model_kwargs={
                    "model": "deepseek-ai/DeepSeek-R1-0528",
                    "base_url": "http://127.0.0.1:8000/v1/completions",
                    "tokenizer_backend": "huggingface",
                    "max_length": 65536,
                },
                gen_kwargs={"stream": "false", "max_gen_toks": "32768"},
                seed=42,
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.2,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
            ),
            EvalTask(
                task_name="r1_gpqa_diamond",
                score=EvalTaskScore(
                    published_score=None,
                    published_score_ref=None,
                    gpu_reference_score=None,
                    gpu_reference_score_ref=None,
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "exact_match,none",
                        ],
                        "unit": "percent",
                    },
                ),
                workflow_venv_type=WorkflowVenvType.EVALS_COMMON,
                model_kwargs={
                    "model": "deepseek-ai/DeepSeek-R1-0528",
                    "base_url": "http://127.0.0.1:8000/v1/completions",
                    "tokenizer_backend": "huggingface",
                    "max_length": 65536,
                },
                gen_kwargs={"stream": "false", "max_gen_toks": "32768"},
                seed=42,
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.2,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
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
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.2,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
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
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.2,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
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
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.2,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
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
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.2,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
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
                eval_class="openai_compatible",
                task_name="chartqa",
                workflow_venv_type=WorkflowVenvType.EVALS_VISION,
                max_concurrent=16,
                apply_chat_template=False,
                use_chat_api=True,
                score=EvalTaskScore(
                    published_score=83.4,
                    published_score_ref="https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct#instruction-tuned-models",
                    gpu_reference_score=81.4,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/131#issuecomment-2769531835",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "relaxed_overall,none",
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
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.2,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
            ),
            EvalTask(
                eval_class="openai_compatible",
                task_name="docvqa_val",
                workflow_venv_type=WorkflowVenvType.EVALS_VISION,
                max_concurrent=16,
                apply_chat_template=False,
                use_chat_api=True,
                score=EvalTaskScore(
                    published_score=88.4,
                    published_score_ref="https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct#instruction-tuned-models",
                    gpu_reference_score=81.4,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/131#issuecomment-2769531835",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "anls,none",
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
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.2,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
            ),
            EvalTask(
                eval_class="openai_compatible",
                task_name="mmmu_val",
                workflow_venv_type=WorkflowVenvType.EVALS_VISION,
                max_concurrent=16,
                apply_chat_template=False,
                use_chat_api=True,
                score=EvalTaskScore(
                    published_score=50.7,
                    published_score_ref="https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct#instruction-tuned-models",
                    gpu_reference_score=43.11,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/131#issuecomment-2769531835",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "mmmu_acc,none",
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
                    "max_new_tokens": "512",
                },
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.2,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="meta-llama/Llama-3.2-90B-Vision-Instruct",
        tasks=[
            EvalTask(
                eval_class="openai_compatible",
                task_name="chartqa",
                workflow_venv_type=WorkflowVenvType.EVALS_VISION,
                max_concurrent=16,
                apply_chat_template=False,
                use_chat_api=True,
                score=EvalTaskScore(
                    published_score=85.5,
                    published_score_ref="https://huggingface.co/meta-llama/Llama-3.2-90B-Vision-Instruct#instruction-tuned-models",
                    gpu_reference_score=33.68,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/379#issuecomment-3071570950",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "relaxed_overall,none",
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
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.2,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
            ),
            EvalTask(
                eval_class="openai_compatible",
                task_name="docvqa_val",
                workflow_venv_type=WorkflowVenvType.EVALS_VISION,
                max_concurrent=16,
                apply_chat_template=False,
                use_chat_api=True,
                score=EvalTaskScore(
                    published_score=90.1,
                    published_score_ref="https://huggingface.co/meta-llama/Llama-3.2-90B-Vision-Instruct#instruction-tuned-models",
                    gpu_reference_score=79.7,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/379#issuecomment-3071570950",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "anls,none",
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
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.2,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
            ),
            EvalTask(
                eval_class="openai_compatible",
                task_name="mmmu_val",
                workflow_venv_type=WorkflowVenvType.EVALS_VISION,
                max_concurrent=16,
                apply_chat_template=False,
                use_chat_api=True,
                score=EvalTaskScore(
                    published_score=60.3,
                    published_score_ref="https://huggingface.co/meta-llama/Llama-3.2-90B-Vision-Instruct#instruction-tuned-models",
                    gpu_reference_score=48.1,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/379#issuecomment-3071570950",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "mmmu_acc,none",
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
                    "max_new_tokens": "512",
                },
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.2,
                    EvalLimitMode.SMOKE_TEST: 0.01,
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
                workflow_venv_type=WorkflowVenvType.EVALS_COMMON,
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
    EvalConfig(
        hf_model_repo="stabilityai/stable-diffusion-xl-base-1.0",
        tasks=[
            EvalTask(
                task_name="load_image",
                workflow_venv_type=WorkflowVenvType.EVALS_META,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
                score=EvalTaskScore(
                    published_score=14.0,
                    published_score_ref="",
                    score_func=lambda results: 0.0,
                ),
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="stabilityai/stable-diffusion-3.5-large",
        tasks=[
            EvalTask(
                task_name="load_image",
                workflow_venv_type=WorkflowVenvType.EVALS_META,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
                score=EvalTaskScore(
                    published_score=14.0,
                    published_score_ref="",
                    score_func=lambda results: 0.0,
                ),
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="stabilityai/stable-diffusion-xl-base-1.0-img-2-img",
        tasks=[
            EvalTask(
                task_name="load_image",
                workflow_venv_type=WorkflowVenvType.EVALS_META,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
                score=EvalTaskScore(
                    published_score=14.0,
                    published_score_ref="",
                    score_func=lambda results: 0.0,
                ),
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        tasks=[
            EvalTask(
                task_name="load_image",
                workflow_venv_type=WorkflowVenvType.EVALS_META,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
                score=EvalTaskScore(
                    published_score=14.0,
                    published_score_ref="",
                    score_func=lambda results: 0.0,
                ),
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="openai/whisper-large-v3",
        tasks=[
            EvalTask(
                task_name="librispeech_test_other",
                eval_class="whisper_tt",
                batch_size=1,
                max_concurrent=1,
                apply_chat_template=False,
                workflow_venv_type=WorkflowVenvType.EVALS_AUDIO,
                score=EvalTaskScore(
                    published_score=(100 - 3.91),
                    published_score_ref="https://huggingface.co/spaces/hf-audio/open_asr_leaderboard",
                    score_func=score_multilevel_keys_mean,
                    score_func_kwargs={
                        "result_keys": [
                            ("librispeech_test_other", "wer,none"),
                        ],
                        "unit": "WER",
                    },
                ),
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.20,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
            )
        ],
    ),
    EvalConfig(
        hf_model_repo="distil-whisper/distil-large-v3",
        tasks=[
            EvalTask(
                task_name="librispeech_test_other",
                eval_class="whisper_tt",
                batch_size=1,
                max_concurrent=1,
                apply_chat_template=False,
                workflow_venv_type=WorkflowVenvType.EVALS_AUDIO,
                score=EvalTaskScore(
                    published_score=(100 - 5.19),
                    gpu_reference_score=(100 - 5.1208),
                    published_score_ref="https://huggingface.co/spaces/hf-audio/open_asr_leaderboard",
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/194#issuecomment-2791159501",
                    score_func=score_multilevel_keys_mean,
                    score_func_kwargs={
                        "result_keys": [
                            ("librispeech_test_other", "wer,none"),
                        ],
                        "unit": "WER",
                    },
                ),
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.50,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
            )
        ],
    ),
    EvalConfig(
        hf_model_repo="genmo/mochi-1-preview",
        tasks=[
            EvalTask(
                task_name="load_video",
                workflow_venv_type=WorkflowVenvType.EVALS_VIDEO,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
                score=EvalTaskScore(
                    published_score=72.0,
                    published_score_ref="https://arxiv.org/abs/1801.04381",
                    score_func=lambda results: 0.0,
                ),
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        tasks=[
            EvalTask(
                task_name="load_video",
                workflow_venv_type=WorkflowVenvType.EVALS_VIDEO,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
                score=EvalTaskScore(
                    published_score=72.0,
                    published_score_ref="https://arxiv.org/abs/1801.04381",
                    score_func=lambda results: 0.0,
                ),
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="Qwen/Qwen2.5-Coder-32B-Instruct",
        tasks=[
            EvalTask(
                task_name="mbpp_instruct",
                workflow_venv_type=WorkflowVenvType.EVALS_COMMON,
                score=EvalTaskScore(
                    published_score=90.2,
                    published_score_ref="https://qwen.ai/blog?id=60a9025af59d5f27f1d4f0cc149725393e5f9130&from=research.research-list",
                    gpu_reference_score=68.8,
                    gpu_reference_score_ref="A100 GPU benchmark results",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "pass_at_1,extract_code",
                        ],
                        "unit": "percent",
                    },
                ),
                apply_chat_template=True,
                batch_size=16,
                gen_kwargs={
                    "max_gen_toks": "256",
                    "do_sample": "false",
                    "stream": "false",
                },
            ),
            EvalTask(
                task_name="humaneval_instruct",
                workflow_venv_type=WorkflowVenvType.EVALS_COMMON,
                score=EvalTaskScore(
                    published_score=92.7,
                    published_score_ref="https://qwen.ai/blog?id=60a9025af59d5f27f1d4f0cc149725393e5f9130&from=research.research-list",
                    gpu_reference_score=92.68,
                    gpu_reference_score_ref="A100 GPU benchmark results",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "pass@1,create_test",
                        ],
                        "unit": "percent",
                    },
                ),
                apply_chat_template=True,
                batch_size=16,
                gen_kwargs={
                    "max_gen_toks": "256",
                    "do_sample": "false",
                    "stream": "false",
                },
            ),
            # EvalTask(
            #     task_name="livecodebench",
            #     workflow_venv_type=WorkflowVenvType.EVALS_COMMON,
            #     score=EvalTaskScore(
            #         published_score=31.4,
            #         published_score_ref="https://qwen.ai/blog?id=60a9025af59d5f27f1d4f0cc149725393e5f9130&from=research.research-list",
            #         gpu_reference_score=42.46,
            #         gpu_reference_score_ref="A100 GPU benchmark results",
            #         score_func=score_task_single_key,
            #         score_func_kwargs={
            #             "result_keys": [
            #                 "acc",
            #             ],
            #             "unit": "percent",
            #         },
            #     ),
            #     apply_chat_template=True,
            #     model_kwargs={
            #         "timeout": "9999",
            #     },
            #     batch_size=16,
            # ),
        ],
    ),
    EvalConfig(
        hf_model_repo="BAAI/bge-large-en-v1.5",
        tasks=[
            EvalTask(
                task_name="embedding",
                workflow_venv_type=WorkflowVenvType.EVALS_META,  # Using META as a placeholder
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
                score=EvalTaskScore(
                    published_score=85.2,
                    published_score_ref="https://huggingface.co/BAAI/bge-large-en-v1.5",
                    score_func=lambda results: 0.0,
                ),
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="Qwen/Qwen3-Embedding-8B",
        tasks=[
            EvalTask(
                task_name="embedding",
                workflow_venv_type=WorkflowVenvType.EVALS_EMBEDDING,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
                score=EvalTaskScore(
                    published_score=85.2,
                    published_score_ref="https://huggingface.co/Qwen/Qwen3-Embedding-4B",
                    score_func=lambda results: 0.0,
                ),
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="Qwen/Qwen3-Embedding-4B",
        tasks=[
            EvalTask(
                task_name="embedding",
                workflow_venv_type=WorkflowVenvType.EVALS_EMBEDDING,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
                score=EvalTaskScore(
                    published_score=85.2,
                    published_score_ref="https://huggingface.co/Qwen/Qwen3-Embedding-4B",
                    score_func=lambda results: 0.0,
                ),
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="resnet-50",
        tasks=[
            EvalTask(
                task_name="load_image",
                workflow_venv_type=WorkflowVenvType.EVALS_META,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
                score=EvalTaskScore(
                    published_score=76.1,
                    published_score_ref="https://pytorch.org/vision/stable/models.html#torchvision.models.resnet50",
                    score_func=lambda results: 0.0,
                ),
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="vovnet",
        tasks=[
            EvalTask(
                task_name="load_image",
                workflow_venv_type=WorkflowVenvType.EVALS_META,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
                score=EvalTaskScore(
                    published_score=76.9,
                    published_score_ref="https://arxiv.org/abs/1904.09730",
                    score_func=lambda results: 0.0,
                ),
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="mobilenetv2",
        tasks=[
            EvalTask(
                task_name="load_image",
                workflow_venv_type=WorkflowVenvType.EVALS_META,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
                score=EvalTaskScore(
                    published_score=72.0,
                    published_score_ref="https://arxiv.org/abs/1801.04381",
                    score_func=lambda results: 0.0,
                ),
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="segformer",
        tasks=[
            EvalTask(
                task_name="load_image",
                workflow_venv_type=WorkflowVenvType.EVALS_META,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
                score=EvalTaskScore(
                    published_score=None,
                    published_score_ref="https://arxiv.org/abs/2105.15203",
                    score_func=lambda results: 0.0,
                ),
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="unet",
        tasks=[
            EvalTask(
                task_name="load_image",
                workflow_venv_type=WorkflowVenvType.EVALS_META,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
                score=EvalTaskScore(
                    published_score=None,
                    published_score_ref="https://arxiv.org/abs/1505.04597",
                    score_func=lambda results: 0.0,
                ),
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="vit",
        tasks=[
            EvalTask(
                task_name="load_image",
                workflow_venv_type=WorkflowVenvType.EVALS_META,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
                score=EvalTaskScore(
                    published_score=None,
                    published_score_ref="https://arxiv.org/abs/2010.11929",
                    score_func=lambda results: 0.0,
                ),
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="efficientnet",
        tasks=[
            EvalTask(
                task_name="load_image",
                workflow_venv_type=WorkflowVenvType.EVALS_META,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
                score=EvalTaskScore(
                    published_score=84.3,
                    published_score_ref="https://arxiv.org/abs/1905.11946",
                    score_func=lambda results: 0.0,
                ),
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="microsoft/speecht5_tts",
        tasks=[
            EvalTask(
                task_name="tts_generation",
                workflow_venv_type=WorkflowVenvType.EVALS_COMMON,
                include_path=None,
                max_concurrent=32,
                apply_chat_template=False,
                score=EvalTaskScore(
                    published_score=None,
                    published_score_ref="https://arxiv.org/abs/2110.07205",
                    score_func=lambda results: 0.0,
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
