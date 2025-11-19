# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Callable, Union

from workflows.workflow_types import WorkflowVenvType, EvalLimitMode
from workflows.utils import map_configs_by_attr
from workflows.model_spec import MODEL_SPECS
from evals.eval_utils import (
    score_task_keys_mean,
    score_task_single_key,
    score_multilevel_keys_mean,
)


@dataclass(frozen=True)
class TestTask:
    task_name: str
    test_path: Path
    workflow_venv_type: WorkflowVenvType = WorkflowVenvType.TESTS_RUN_SCRIPT
    test_args: Dict[str, str] = field(default_factory=dict)
    # eval_class: str = "local-completions"
    # tokenizer_backend: str = "huggingface"
    # # Note: batch_size is set to 1 because max_concurrent is set to 32
    # # this means that 32 requests are sent concurrently by lm-eval / lmms-eval
    # # for clarity, the client side eval scripts cannot control the batch size
    # # so setting just multiplys the max_concurrent which is misleading
    # batch_size: int = 1
    # max_concurrent: int = 32
    # num_fewshot: int = 0
    # seed: int = 42
    # use_chat_api: bool = False
    # apply_chat_template: bool = True
    # log_samples: bool = True
    # gen_kwargs: Dict[str, str] = field(default_factory=lambda: {"stream": "False"})
    # model_kwargs: Dict[str, str] = field(default_factory=lambda: {})
    # # Note: include_path is specified relative to the respective venv
    # include_path: str = None
    # # Optional: kwargs passed to task custom_dataset loaders (e.g., RULER sequence length configs)
    # custom_dataset_kwargs: Dict[str, Union[str, List[int]]] = None
    # # Optional: limit the number of samples passed to lm_eval (--limit)
    # # Limit the number of examples per task.
    # # If <1, limit is a percentage of the total number of examples.
    # limit_samples_map: Dict[EvalLimitMode, Union[float, int]] = field(
    #     default_factory=lambda: {
    #         # this defines smoke test limit to 1% for all models unless overridden
    #         EvalLimitMode.SMOKE_TEST: 0.01,
    #     }
    # )

    def __post_init__(self):
        self.validate_data()
        self._infer_data()

    def _infer_data(self):
        pass
        # if self.use_chat_api and self.eval_class == "local-completions":
        #     object.__setattr__(self, "eval_class", "local-chat-completions")

        # if self.workflow_venv_type == WorkflowVenvType.EVALS_META:
        #     # max_concurrent is not supported in lm-eval==0.4.3
        #     object.__setattr__(self, "batch_size", self.max_concurrent)
        #     object.__setattr__(self, "max_concurrent", None)
        #     if self.model_kwargs:
        #         raise ValueError("model_kwargs are not supported in lm-eval==0.4.3")

    def validate_data(self):
        pass
        # assert not (
        #     self.use_chat_api and self.apply_chat_template
        # ), "Chat API applies chat template."


@dataclass(frozen=True)
class TestConfig:
    hf_model_repo: str
    tasks: List[TestTask]


# Note: meta evals defined in: https://github.com/meta-llama/llama-cookbook/blob/main/end-to-end-use-cases/benchmarks/llm_eval_harness/meta_eval/eval_config.yaml
# Note: meta_math_hard for Llama 3.1 models has a bug: see https://github.com/tenstorrent/tt-inference-server/issues/155
# Note: reasoning models (QwQ-32B, DeepSeek-R1-Distill-Llama-70B) need evals allowing more tokens generated


_test_config_list = [
    TestConfig(
        hf_model_repo="Qwen/Qwen3-32B",
        tasks=[
            TestTask(
                task_name="vllm_params",
                test_path=Path("tests/server_tests/test_cases/test_vllm_server_parameters.py"),
                test_args=("s", "v"),
            ),
        ],
    ),
]


_test_config_map = map_configs_by_attr(
    config_list=_test_config_list, attr="hf_model_repo"
)
TEST_CONFIGS = {
    model_spec.model_name: _test_config_map[model_spec.hf_model_repo]
    for _, model_spec in MODEL_SPECS.items()
    if model_spec.hf_model_repo in _test_config_map
}
