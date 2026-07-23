# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from evals.eval_utils import (
    score_multilevel_keys_mean,
    score_task_keys_mean,
    score_task_single_key,
)
from workflows.model_spec import MODEL_SPECS
from workflows.utils import map_configs_by_attr
from workflows.workflow_types import EvalLimitMode, WorkflowVenvType


@dataclass(frozen=True)
class ModeReferenceScore:
    """Reference score measured on a specific EvalLimitMode's fixed subset.

    When a run uses --limit-samples-mode / --ci-mode, the acceptance check
    compares the subset score against the matching subset reference instead of
    the full-dataset gpu_reference_score (apples-to-apples). ``score`` is in the
    same unit as the task score (typically percent).

    The acceptance check for a subset reference is sample-count-aware (see
    ``accept_eval_score``): PASS when
    ``round(score/100 * n) >= floor(n * ref/100 * (1 - tolerance))``. The
    integer floor gives small subsets the right leniency automatically (a 5-item
    subset moves in 20% steps), so no separate absolute-margin knob is needed.
    """

    score: float
    ref: str = ""
    # Falls back to EvalTaskScore.tolerance when None.
    tolerance: Optional[float] = None


@dataclass(frozen=True)
class EvalTaskScore:
    published_score: float
    published_score_ref: str
    score_func: Callable
    gpu_reference_score: float = None
    gpu_reference_score_ref: str = None
    score_func_kwargs: Dict[str, str] = field(default_factory=dict)
    tolerance: float = 0.05
    # Per-limit-mode references measured on that mode's fixed subset. Used by
    # the release report's accuracy check when the run sets --limit-samples-mode
    # (or --ci-mode). The full-set gpu_reference_score remains the baseline for
    # unrestricted --workflow evals runs.
    mode_reference_scores: Dict["EvalLimitMode", "ModeReferenceScore"] = field(
        default_factory=dict
    )


def resolve_eval_reference(score_obj, limit_mode):
    """Select the reference the accuracy check should compare against.

    When a run uses a limit mode (--limit-samples-mode / --ci-mode) and the
    task defines a matching entry in ``mode_reference_scores``, return that
    subset-specific reference (apples-to-apples with the subset score).
    Otherwise return the full-dataset ``gpu_reference_score`` baseline.

    Shared by the v1 release report (workflows/run_reports.py) and the v2
    engine scorers (tt-inference-server-v2) so both paths agree.

    Returns a dict: reference_score, reference_ref, tolerance,
    is_subset_reference (bool: True when the returned reference is a
    limit-mode subset reference rather than the full-dataset baseline).
    """
    mode_ref = None
    if limit_mode is not None:
        mode_ref = (getattr(score_obj, "mode_reference_scores", None) or {}).get(
            limit_mode
        )

    full_ref_label = getattr(score_obj, "gpu_reference_score_ref", None)

    if mode_ref is not None:
        label = mode_ref.ref or full_ref_label or ""
        suffix = f"[{limit_mode.name} subset]"
        return {
            "reference_score": mode_ref.score,
            "reference_ref": f"{label} {suffix}".strip(),
            "tolerance": mode_ref.tolerance
            if mode_ref.tolerance is not None
            else score_obj.tolerance,
            "is_subset_reference": True,
        }

    return {
        "reference_score": score_obj.gpu_reference_score,
        "reference_ref": full_ref_label,
        "tolerance": score_obj.tolerance,
        "is_subset_reference": False,
    }


def accept_eval_score(ref, score, n_total=None):
    """Decide PASS/FAIL for an observed percent ``score`` against ``ref``.

    ``ref`` is a dict from ``resolve_eval_reference``. Returns True (PASS),
    False (FAIL), or None when no reference is defined (caller renders N/A).

    For a subset (mode) reference with a known sample count ``n_total`` the
    check is sample-count-aware:

        n_correct_obs = round(score/100 * n_total)
        threshold     = floor(n_total * reference/100 * (1 - tolerance))
        PASS iff n_correct_obs >= threshold

    The integer floor lets tiny subsets tolerate one flipped item without a
    hand-tuned absolute margin. When ``n_total`` is unknown, or for the
    full-dataset reference path, it falls back to the percent ratio check
    (score / reference >= 1 - tolerance), preserving prior behavior.
    """
    reference = ref["reference_score"]
    tolerance = ref["tolerance"]
    if not reference:
        return None
    assert reference > 0, "Reference score is not > 0"
    if ref.get("is_subset_reference") and n_total:
        ref_rate = reference / 100.0
        threshold = math.floor(n_total * ref_rate * (1.0 - tolerance))
        observed = round(score / 100.0 * n_total)
        return observed >= threshold
    return (score / reference) >= (1.0 - tolerance)


@dataclass(frozen=True)
class TerminalBenchEvalConfig:
    dataset: str
    agent: str
    model: Optional[str] = None
    n_concurrent_trials: int = 1
    n_attempts: int = 1
    n_tasks: Optional[int] = None
    task_names: List[str] = field(default_factory=list)
    exclude_task_names: List[str] = field(default_factory=list)
    agent_kwargs: Dict[str, Any] = field(default_factory=dict)
    environment_type: str = "docker"
    override_cpus: Optional[int] = None
    override_memory_mb: Optional[int] = None
    timeout_multiplier: Optional[float] = None
    agent_timeout_sec: Optional[float] = None
    quiet: bool = True
    yes: bool = True
    task_names_map: Dict[EvalLimitMode, List[str]] = field(default_factory=dict)
    agent_import_path: Optional[str] = None
    environment_env: Dict[str, str] = field(default_factory=dict)
    verifier_env: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SWEbenchEvalConfig:
    dataset_name: str
    sweagent_subset: str = "verified"
    dataset_split: str = "test"
    agent_backend: str = "mini-swe-agent"
    model: Optional[str] = None
    n_concurrent_trials: int = 1
    max_workers: int = 1
    n_tasks: Optional[int] = None
    temperature: float = 1.0
    top_p: float = 0.95
    max_input_tokens: int = 200 * 1024
    max_output_tokens: Optional[int] = None
    completion_kwargs: Dict[str, Any] = field(default_factory=dict)
    sweagent_config: str = "config/default.yaml"
    mini_config: str = "swebench.yaml"
    mini_model_class: str = "litellm"
    mini_environment_class: str = "docker"
    swebench_timeout_sec: Optional[int] = None
    shuffle: bool = True
    random_delay_multiplier: float = 0.3
    instance_ids_map: Dict[EvalLimitMode, List[str]] = field(default_factory=dict)


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
    # Skip task when device.max_context < this. Avoids hours of HTTP 400
    # retry-burn on long-context evals (longbench, RULER) at small ctx.
    min_context_required: Optional[int] = None
    # Optional: limit the number of samples passed to lm_eval (--limit)
    # Limit the number of examples per task.
    # If <1, limit is a percentage of the total number of examples.
    limit_samples_map: Dict[EvalLimitMode, Union[float, int]] = field(
        default_factory=lambda: {
            # this defines smoke test limit to 1% for all models unless overridden
            EvalLimitMode.SMOKE_TEST: 0.01,
        }
    )
    agentic_eval_config: Optional[TerminalBenchEvalConfig] = None
    swebench_eval_config: Optional[SWEbenchEvalConfig] = None

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
            # lm-eval 0.4.4's API models default add_bos_token=False. Llama is trained with a
            # leading BOS and regresses badly without it (meta_ifeval strict-format failures,
            # ~3pt drop crossing the 0.95 gate). Force BOS on for all meta tasks.
            mk = dict(self.model_kwargs or {})
            mk.setdefault("add_bos_token", "True")
            object.__setattr__(self, "model_kwargs", mk)

    def validate_data(self):
        pass


@dataclass(frozen=True)
class EvalConfig:
    hf_model_repo: str
    tasks: List[EvalTask]


# Note: meta evals defined in: https://github.com/meta-llama/llama-cookbook/blob/main/end-to-end-use-cases/benchmarks/llm_eval_harness/meta_eval/eval_config.yaml
# Note: meta_math_hard for Llama 3.1 models has a bug: see https://github.com/tenstorrent/tt-inference-server/issues/155
# Note: reasoning models (QwQ-32B, DeepSeek-R1-Distill-Llama-70B) need evals allowing more tokens generated


_eval_config_list = [
    EvalConfig(
        hf_model_repo="moonshotai/Kimi-K2.6",
        tasks=[
            EvalTask(
                task_name="r1_gpqa_diamond",
                workflow_venv_type=WorkflowVenvType.EVALS_COMMON,
                max_concurrent=64,
                # The remote Tenstorrent console only exposes /v1/chat/completions
                # (text /v1/completions returns 404), so use the chat API.
                use_chat_api=True,
                score=EvalTaskScore(
                    published_score=90.5,
                    published_score_ref="https://huggingface.co/moonshotai/Kimi-K2.6",
                    gpu_reference_score=90.91,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/3752#issuecomment-4574524682",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "exact_match,none",
                        ],
                        "unit": "percent",
                    },
                ),
                model_kwargs={
                    "max_length": 256 * 1024,
                    # Per-request HTTP timeout (lm-eval default 1800s). Long
                    # reasoning generations on the shared console can exceed
                    # 30min under load, so allow up to 2h before giving up.
                    "timeout": 7200,
                },
                gen_kwargs={
                    "max_gen_toks": 256 * 1024,
                    "until": ["[EOS]"],
                    "do_sample": "true",
                    "temperature": 1.0,
                    # "top_k": 20,
                    "top_p": 1.0,
                    "stream": "true",
                },
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.2,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
            ),
            EvalTask(
                task_name="terminal_bench_2",
                workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC,
                score=EvalTaskScore(
                    published_score=66.7,
                    published_score_ref="https://huggingface.co/moonshotai/Kimi-K2.6",
                    gpu_reference_score=61.9,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/3752#issuecomment-4586446467",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": ["accuracy"],
                        "unit": "percent",
                    },
                ),
                agentic_eval_config=TerminalBenchEvalConfig(
                    dataset="terminal-bench/terminal-bench-2",
                    agent="terminus-2",
                    n_concurrent_trials=16,
                    n_attempts=1,
                    n_tasks=89,
                    override_cpus=16,
                    override_memory_mb=32 * 1024,
                    agent_timeout_sec=2 * 60 * 60,
                    agent_kwargs={
                        "parser_name": "json",
                        # "interleaved_thinking": True,  # Feeds reasoning content back into the message history
                        "temperature": 1.0,
                        "model_info": {
                            "max_input_tokens": 256 * 1024,
                            "max_output_tokens": 64 * 1024,
                        },
                        "llm_kwargs": {
                            "top_p": 1.0,
                            "max_tokens": 64 * 1024,
                            "timeout": 60 * 60,
                        },
                        # "llm_call_kwargs": {
                        #     "extra_body": {
                        #         "chat_template_kwargs": {
                        #             "thinking": True,
                        #             "preserve_thinking": True,
                        #         }
                        #     },
                        # },
                    },
                    task_names_map={
                        EvalLimitMode.CI_NIGHTLY: [
                            "terminal-bench/break-filter-js-from-html",
                            "terminal-bench/cobol-modernization",
                            "terminal-bench/compile-compcert",
                            "terminal-bench/feal-differential-cryptanalysis",
                            "terminal-bench/qemu-startup",
                        ],
                    },
                ),
                limit_samples_map={
                    EvalLimitMode.SMOKE_TEST: 5,
                },
            ),
            EvalTask(
                task_name="swe_bench_verified",
                workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC,
                score=EvalTaskScore(
                    published_score=80.2,
                    published_score_ref="https://huggingface.co/moonshotai/Kimi-K2.6",
                    gpu_reference_score=66.2,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/3752#issuecomment-4574524682",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": ["accuracy"],
                        "unit": "percent",
                    },
                ),
                swebench_eval_config=SWEbenchEvalConfig(
                    dataset_name="SWE-bench/SWE-bench_Verified",
                    sweagent_subset="verified",
                    dataset_split="test",
                    agent_backend="mini-swe-agent",
                    n_concurrent_trials=16,
                    max_workers=24,
                    n_tasks=None,
                    temperature=1.0,
                    top_p=1.0,
                    # max inputs tokens should be increased when we get a chance
                    max_input_tokens=256 * 1024,
                    max_output_tokens=64 * 1024,
                    # mini_last_n_observations is ommitted for now
                    # mini_last_n_observations=15,
                    # completion_kwargs={
                    #     "extra_body": {
                    #         "top_k": 20,
                    #     },
                    # },
                    instance_ids_map={
                        EvalLimitMode.CI_NIGHTLY: [
                            "django__django-12143",
                            "pytest-dev__pytest-5262",
                            "django__django-14672",
                            "sympy__sympy-13551",
                            "sphinx-doc__sphinx-9281",
                        ],
                    },
                ),
                limit_samples_map={
                    EvalLimitMode.SMOKE_TEST: 5,
                },
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="moonshotai/Kimi-K2.7-Code",
        tasks=[
            EvalTask(
                task_name="r1_gpqa_diamond",
                workflow_venv_type=WorkflowVenvType.EVALS_COMMON,
                max_concurrent=64,
                # This vLLM server only exposes /v1/chat/completions; the legacy
                # text /v1/completions endpoint returns 404. use_chat_api switches
                # lm-eval's eval_class from "local-completions" to
                # "local-chat-completions" so requests go to the right route.
                use_chat_api=True,
                score=EvalTaskScore(
                    published_score=89.6,
                    published_score_ref="https://artificialanalysis.ai/evaluations/gpqa-diamond?models=kimi-k2-7-code",
                    gpu_reference_score=85.3,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/4271#issuecomment-4841263402",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "exact_match,none",
                        ],
                        "unit": "percent",
                    },
                ),
                model_kwargs={
                    "max_length": 256 * 1024,
                    # Per-request HTTP timeout (lm-eval default 1800s). Long
                    # reasoning generations on the shared console can exceed
                    # 30min under load, so allow up to 2h before giving up.
                    "timeout": 7200,
                },
                gen_kwargs={
                    "max_gen_toks": 256 * 1024,
                    "until": ["[EOS]"],
                    "do_sample": "true",
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "stream": "true",
                },
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.2,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
            ),
            EvalTask(
                task_name="terminal_bench_2_1",
                workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC,
                score=EvalTaskScore(
                    published_score=67.4,
                    published_score_ref="https://artificialanalysis.ai/evaluations/terminalbench-v2-1?models=kimi-k2-7-code",
                    gpu_reference_score=65.2,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/4271#issuecomment-4854374547",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": ["accuracy"],
                        "unit": "percent",
                    },
                ),
                agentic_eval_config=TerminalBenchEvalConfig(
                    dataset="terminal-bench/terminal-bench-2-1",
                    agent="terminus-2",
                    n_concurrent_trials=4,
                    n_attempts=1,
                    n_tasks=89,
                    override_cpus=16,
                    override_memory_mb=32 * 1024,
                    agent_timeout_sec=2 * 60 * 60,
                    agent_kwargs={
                        "parser_name": "json",
                        "temperature": 1.0,
                        "model_info": {
                            "max_input_tokens": 256 * 1024,
                            "max_output_tokens": 64 * 1024,
                        },
                        "llm_kwargs": {
                            "top_p": 0.95,
                            "max_tokens": 64 * 1024,
                            "timeout": 60 * 60,
                        },
                    },
                    task_names_map={
                        EvalLimitMode.CI_NIGHTLY: [
                            "terminal-bench/break-filter-js-from-html",
                            "terminal-bench/cobol-modernization",
                            "terminal-bench/compile-compcert",
                            "terminal-bench/feal-differential-cryptanalysis",
                            "terminal-bench/qemu-startup",
                        ],
                    },
                ),
                limit_samples_map={
                    EvalLimitMode.SMOKE_TEST: 5,
                },
            ),
            EvalTask(
                task_name="tau3_bench_banking",
                workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC,
                score=EvalTaskScore(
                    published_score=18.1,
                    published_score_ref="https://artificialanalysis.ai/evaluations/tau3-banking?models=kimi-k2-7-code",
                    gpu_reference_score=11.3,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/4271#issuecomment-4950368694",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": ["accuracy"],
                        "unit": "percent",
                    },
                    tolerance=0.10,
                ),
                agentic_eval_config=TerminalBenchEvalConfig(
                    dataset="sierra-research/tau3-bench",
                    agent="tau3_llm_agent",
                    agent_import_path="adapters.tau3-bench.tau3_llm_agent:Tau3LLMAgent",
                    task_names=["sierra-research/tau3-bench__tau3-banking_knowledge-*"],
                    # A single served instance is shared by the agent,
                    # the simulated user, and the Natural Language verifier.
                    n_concurrent_trials=4,
                    n_attempts=1,
                    n_tasks=97,
                    override_cpus=4,
                    override_memory_mb=8 * 1024,
                    agent_timeout_sec=3600,
                    agent_kwargs={
                        "tau2_trial_index": 0,
                        "temperature": 1.0,
                        "max_steps": 200,
                        # Default is 120s; a single reasoning user-sim turn under
                        # load can exceed that and trip an MCP request timeout.
                        "tool_timeout_sec": 900,
                        "read_timeout_sec": 120,
                    },
                    # NOTE: values injected here are passed to the Harbor
                    # container verbatim. Unlike the task.toml env, the
                    # "${VAR:-default}" template syntax is NOT resolved on this
                    # path, so use literal values -- a templated model name
                    # reaches litellm unexpanded and fails with "LLM Provider
                    # NOT provided". OPENAI_BASE_URL / OPENAI_API_KEY are
                    # intentionally omitted: the task's docker-compose already
                    # substitutes those from the launching shell env.
                    environment_env={
                        "TAU2_USER_MODEL": "openai/moonshotai/Kimi-K2.7-Code",
                    },
                    verifier_env={
                        "TAU2_NL_ASSERTIONS_MODEL": "openai/moonshotai/Kimi-K2.7-Code",
                    },
                    task_names_map={
                        EvalLimitMode.CI_NIGHTLY: [
                            "sierra-research/tau3-bench__tau3-banking_knowledge-task-001",
                            "sierra-research/tau3-bench__tau3-banking_knowledge-task-022",
                            "sierra-research/tau3-bench__tau3-banking_knowledge-task-050",
                            "sierra-research/tau3-bench__tau3-banking_knowledge-task-075",
                            "sierra-research/tau3-bench__tau3-banking_knowledge-task-100",
                        ],
                    },
                ),
                limit_samples_map={
                    EvalLimitMode.SMOKE_TEST: 3,
                },
            ),
            EvalTask(
                task_name="swe_bench_verified",
                workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC,
                score=EvalTaskScore(
                    published_score=None,
                    published_score_ref=None,
                    gpu_reference_score=69.0,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/4271#issuecomment-4950368694",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": ["accuracy"],
                        "unit": "percent",
                    },
                ),
                swebench_eval_config=SWEbenchEvalConfig(
                    dataset_name="SWE-bench/SWE-bench_Verified",
                    sweagent_subset="verified",
                    dataset_split="test",
                    agent_backend="mini-swe-agent",
                    n_concurrent_trials=6,
                    max_workers=24,
                    n_tasks=None,
                    temperature=1.0,
                    top_p=0.95,
                    max_input_tokens=256 * 1024,
                    max_output_tokens=64 * 1024,
                    instance_ids_map={
                        EvalLimitMode.CI_NIGHTLY: [
                            "django__django-12143",
                            "pytest-dev__pytest-5262",
                            "django__django-14672",
                            "sympy__sympy-13551",
                            "sphinx-doc__sphinx-9281",
                        ],
                    },
                ),
                limit_samples_map={
                    EvalLimitMode.SMOKE_TEST: 5,
                },
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="MiniMaxAI/MiniMax-M2.7",
        tasks=[
            EvalTask(
                task_name="r1_gpqa_diamond",
                workflow_venv_type=WorkflowVenvType.EVALS_COMMON,
                max_concurrent=64,
                # The remote Tenstorrent console only exposes /v1/chat/completions
                # (text /v1/completions returns 404), so use the chat API.
                use_chat_api=True,
                score=EvalTaskScore(
                    published_score=87.4,
                    published_score_ref="https://artificialanalysis.ai/models?models=minimax-m2-7",
                    gpu_reference_score=85.35,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/4324#issuecomment-4809200160",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "exact_match,none",
                        ],
                        "unit": "percent",
                    },
                ),
                model_kwargs={
                    "max_length": 200 * 1024,
                    # Per-request HTTP timeout (lm-eval default 1800s). Long
                    # reasoning generations on the shared console can exceed
                    # 30min under load, so allow up to 2h before giving up.
                    "timeout": 7200,
                },
                gen_kwargs={
                    "max_gen_toks": 200 * 1024,
                    "until": ["[e~["],
                    "do_sample": "true",
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "stream": "true",
                },
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.999,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
            ),
            EvalTask(
                task_name="terminal_bench_2_1",
                workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC,
                score=EvalTaskScore(
                    published_score=51.1,
                    published_score_ref="https://huggingface.co/MiniMaxAI/MiniMax-M3",
                    gpu_reference_score=52.8,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/4324#issuecomment-4815788892",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": ["accuracy"],
                        "unit": "percent",
                    },
                ),
                agentic_eval_config=TerminalBenchEvalConfig(
                    dataset="terminal-bench/terminal-bench-2-1",
                    agent="terminus-2",
                    n_concurrent_trials=8,
                    n_attempts=1,
                    n_tasks=89,
                    override_cpus=16,
                    override_memory_mb=32 * 1024,
                    agent_timeout_sec=2 * 60 * 60,
                    agent_kwargs={
                        "parser_name": "json",
                        "temperature": 1.0,
                        "model_info": {
                            "max_input_tokens": 200 * 1024,
                            "max_output_tokens": 64 * 1024,
                        },
                        "llm_kwargs": {
                            "top_p": 0.95,
                            "max_tokens": 64 * 1024,
                            "timeout": 60 * 60,
                        },
                    },
                    task_names_map={
                        EvalLimitMode.CI_NIGHTLY: [
                            "terminal-bench/break-filter-js-from-html",
                            "terminal-bench/cobol-modernization",
                            "terminal-bench/compile-compcert",
                            "terminal-bench/feal-differential-cryptanalysis",
                            "terminal-bench/qemu-startup",
                        ],
                    },
                ),
                limit_samples_map={
                    EvalLimitMode.SMOKE_TEST: 5,
                },
            ),
            EvalTask(
                task_name="tau3_bench_banking",
                workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC,
                score=EvalTaskScore(
                    published_score=8.9,
                    published_score_ref="https://artificialanalysis.ai/models?models=minimax-m2-7",
                    gpu_reference_score=11.3,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/4324#issuecomment-4815788892",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": ["accuracy"],
                        "unit": "percent",
                    },
                    tolerance=0.10,
                ),
                agentic_eval_config=TerminalBenchEvalConfig(
                    dataset="sierra-research/tau3-bench",
                    agent="tau3_llm_agent",
                    agent_import_path="adapters.tau3-bench.tau3_llm_agent:Tau3LLMAgent",
                    task_names=["sierra-research/tau3-bench__tau3-banking_knowledge-*"],
                    # A single served instance is shared by the agent,
                    # the simulated user, and the Natural Language verifier.
                    n_concurrent_trials=4,
                    n_attempts=1,
                    n_tasks=97,
                    override_cpus=4,
                    override_memory_mb=8 * 1024,
                    agent_timeout_sec=3600,
                    agent_kwargs={
                        "tau2_trial_index": 0,
                        "temperature": 1.0,
                        "max_steps": 200,
                        # Default is 120s; a single reasoning user-sim turn under
                        # load can exceed that and trip an MCP request timeout.
                        "tool_timeout_sec": 900,
                        "read_timeout_sec": 120,
                    },
                    # NOTE: values injected here are passed to the Harbor
                    # container verbatim. Unlike the task.toml env, the
                    # "${VAR:-default}" template syntax is NOT resolved on this
                    # path, so use literal values -- a templated model name
                    # reaches litellm unexpanded and fails with "LLM Provider
                    # NOT provided". OPENAI_BASE_URL / OPENAI_API_KEY are
                    # intentionally omitted: the task's docker-compose already
                    # substitutes those from the launching shell env.
                    environment_env={
                        "TAU2_USER_MODEL": "openai/MiniMaxAI/MiniMax-M2.7",
                    },
                    verifier_env={
                        "TAU2_NL_ASSERTIONS_MODEL": "openai/MiniMaxAI/MiniMax-M2.7",
                    },
                    task_names_map={
                        EvalLimitMode.CI_NIGHTLY: [
                            "sierra-research/tau3-bench__tau3-banking_knowledge-task-031",
                            "sierra-research/tau3-bench__tau3-banking_knowledge-task-032",
                            "sierra-research/tau3-bench__tau3-banking_knowledge-task-052",
                            "sierra-research/tau3-bench__tau3-banking_knowledge-task-002",
                        ],
                    },
                ),
                limit_samples_map={
                    EvalLimitMode.SMOKE_TEST: 3,
                },
            ),
            EvalTask(
                task_name="swe_bench_verified",
                workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC,
                score=EvalTaskScore(
                    published_score=79.9,
                    published_score_ref="https://huggingface.co/MiniMaxAI/MiniMax-M3",
                    gpu_reference_score=62.4,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/4324#issuecomment-4830558090",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": ["accuracy"],
                        "unit": "percent",
                    },
                ),
                swebench_eval_config=SWEbenchEvalConfig(
                    dataset_name="SWE-bench/SWE-bench_Verified",
                    sweagent_subset="verified",
                    dataset_split="test",
                    agent_backend="mini-swe-agent",
                    n_concurrent_trials=8,
                    max_workers=24,
                    n_tasks=None,
                    temperature=1.0,
                    top_p=0.95,
                    max_input_tokens=200 * 1024,
                    max_output_tokens=64 * 1024,
                    instance_ids_map={
                        EvalLimitMode.CI_NIGHTLY: [
                            "django__django-12143",
                            "pytest-dev__pytest-5262",
                            "django__django-14672",
                            "sympy__sympy-13551",
                            "sphinx-doc__sphinx-9281",
                        ],
                    },
                ),
                limit_samples_map={
                    EvalLimitMode.SMOKE_TEST: 5,
                },
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="MiniMaxAI/MiniMax-M3",
        tasks=[
            # NOTE: we had issues with outputs parsing on GPU with M3!!
            EvalTask(
                task_name="r1_gpqa_diamond",
                workflow_venv_type=WorkflowVenvType.EVALS_COMMON,
                max_concurrent=64,
                # The remote Tenstorrent console only exposes /v1/chat/completions
                # (text /v1/completions returns 404), so use the chat API.
                use_chat_api=True,
                score=EvalTaskScore(
                    published_score=92.9,
                    published_score_ref="https://artificialanalysis.ai/models?models=minimax-m3",
                    gpu_reference_score=93.9,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/4376#issuecomment-4901015676",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "exact_match,none",
                        ],
                        "unit": "percent",
                    },
                ),
                model_kwargs={
                    "max_length": 200 * 1024,
                    # Per-request HTTP timeout (lm-eval default 1800s). Long
                    # reasoning generations on the shared console can exceed
                    # 30min under load, so allow up to 2h before giving up.
                    "timeout": 7200,
                },
                gen_kwargs={
                    "max_gen_toks": 200 * 1024,
                    # https://huggingface.co/MiniMaxAI/MiniMax-M3/blob/main/special_tokens_map.json
                    "until": "[e~[",
                    "do_sample": "true",
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "stream": "true",
                },
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.2,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
            ),
            EvalTask(
                task_name="terminal_bench_2_1",
                workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC,
                score=EvalTaskScore(
                    published_score=66.0,
                    published_score_ref="https://huggingface.co/MiniMaxAI/MiniMax-M3",
                    gpu_reference_score=61.8,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/4376#issuecomment-4901015676",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": ["accuracy"],
                        "unit": "percent",
                    },
                ),
                agentic_eval_config=TerminalBenchEvalConfig(
                    dataset="terminal-bench/terminal-bench-2-1",
                    agent="terminus-2",
                    n_concurrent_trials=8,
                    n_attempts=1,
                    n_tasks=89,
                    override_cpus=8,
                    override_memory_mb=32 * 1024,
                    agent_timeout_sec=2 * 60 * 60,
                    agent_kwargs={
                        "parser_name": "json",
                        "temperature": 1.0,
                        "model_info": {
                            "max_input_tokens": 500 * 1024,
                            "max_output_tokens": 64 * 1024,
                        },
                        "llm_kwargs": {
                            "top_p": 0.95,
                            "max_tokens": 64 * 1024,
                            "timeout": 60 * 60,
                        },
                    },
                    task_names_map={
                        EvalLimitMode.CI_NIGHTLY: [
                            "terminal-bench/break-filter-js-from-html",
                            "terminal-bench/cobol-modernization",
                            "terminal-bench/compile-compcert",
                            "terminal-bench/feal-differential-cryptanalysis",
                            "terminal-bench/qemu-startup",
                        ],
                    },
                ),
                limit_samples_map={
                    EvalLimitMode.SMOKE_TEST: 5,
                },
            ),
            EvalTask(
                task_name="tau3_bench_banking",
                workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC,
                score=EvalTaskScore(
                    published_score=13,
                    published_score_ref="https://artificialanalysis.ai/models?models=minimax-m3",
                    gpu_reference_score=16.5,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/4376#issuecomment-4922484555",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": ["accuracy"],
                        "unit": "percent",
                    },
                    tolerance=0.10,
                ),
                agentic_eval_config=TerminalBenchEvalConfig(
                    dataset="sierra-research/tau3-bench",
                    agent="tau3_llm_agent",
                    agent_import_path="adapters.tau3-bench.tau3_llm_agent:Tau3LLMAgent",
                    task_names=["sierra-research/tau3-bench__tau3-banking_knowledge-*"],
                    # A single served instance is shared by the agent,
                    # the simulated user, and the Natural Language verifier.
                    n_concurrent_trials=4,
                    n_attempts=1,
                    n_tasks=97,
                    override_cpus=4,
                    override_memory_mb=8 * 1024,
                    agent_timeout_sec=3600,
                    agent_kwargs={
                        "tau2_trial_index": 0,
                        "temperature": 1.0,
                        "max_steps": 200,
                        # Default is 120s; a single reasoning user-sim turn under
                        # load can exceed that and trip an MCP request timeout.
                        "tool_timeout_sec": 900,
                        "read_timeout_sec": 120,
                    },
                    environment_env={
                        "TAU2_USER_MODEL": "openai/MiniMaxAI/MiniMax-M3",
                    },
                    verifier_env={
                        "TAU2_NL_ASSERTIONS_MODEL": "openai/MiniMaxAI/MiniMax-M3",
                    },
                    task_names_map={
                        EvalLimitMode.CI_NIGHTLY: [
                            "sierra-research/tau3-bench__tau3-banking_knowledge-task-031",
                            "sierra-research/tau3-bench__tau3-banking_knowledge-task-032",
                            "sierra-research/tau3-bench__tau3-banking_knowledge-task-052",
                            "sierra-research/tau3-bench__tau3-banking_knowledge-task-002",
                        ],
                    },
                ),
                limit_samples_map={
                    EvalLimitMode.SMOKE_TEST: 3,
                },
            ),
            EvalTask(
                task_name="swe_bench_verified",
                workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC,
                score=EvalTaskScore(
                    published_score=80.5,
                    published_score_ref="https://huggingface.co/MiniMaxAI/MiniMax-M3",
                    gpu_reference_score=65.4,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/4324#issuecomment-4830558090",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": ["accuracy"],
                        "unit": "percent",
                    },
                ),
                swebench_eval_config=SWEbenchEvalConfig(
                    dataset_name="SWE-bench/SWE-bench_Verified",
                    sweagent_subset="verified",
                    dataset_split="test",
                    agent_backend="mini-swe-agent",
                    n_concurrent_trials=8,
                    max_workers=24,
                    n_tasks=None,
                    temperature=1.0,
                    top_p=0.95,
                    max_input_tokens=500 * 1024,
                    max_output_tokens=64 * 1024,
                    instance_ids_map={
                        EvalLimitMode.CI_NIGHTLY: [
                            "django__django-12143",
                            "pytest-dev__pytest-5262",
                            "django__django-14672",
                            "sympy__sympy-13551",
                            "sphinx-doc__sphinx-9281",
                        ],
                    },
                ),
                limit_samples_map={
                    EvalLimitMode.SMOKE_TEST: 5,
                },
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="Qwen/Qwen3.6-27B",
        tasks=[
            EvalTask(
                task_name="terminal_bench_2",
                workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC,
                score=EvalTaskScore(
                    published_score=59.3,
                    published_score_ref="https://huggingface.co/Qwen/Qwen3.6-27B",
                    gpu_reference_score=53.9,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/3359#issuecomment-4450842511",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": ["accuracy"],
                        "unit": "percent",
                    },
                ),
                agentic_eval_config=TerminalBenchEvalConfig(
                    dataset="terminal-bench/terminal-bench-2",
                    agent="terminus-2",
                    n_concurrent_trials=1,  # TODO increase back to 5 when batch > 1 is supported
                    n_attempts=1,
                    n_tasks=89,
                    override_cpus=16,
                    override_memory_mb=48 * 1024,
                    agent_timeout_sec=3 * 60 * 60,
                    agent_kwargs={
                        "parser_name": "json",
                        "temperature": 1.0,
                        "model_info": {
                            "max_input_tokens": 256 * 1024,
                            "max_output_tokens": 80 * 1024,
                        },
                        "llm_kwargs": {
                            "top_p": 0.95,
                            "max_tokens": 80 * 1024,
                            "timeout": 60 * 60,
                            "extra_body": {
                                "top_k": 20,
                            },
                        },
                    },
                    task_names_map={
                        EvalLimitMode.CI_NIGHTLY: [
                            "terminal-bench/break-filter-js-from-html",
                            "terminal-bench/cobol-modernization",
                            "terminal-bench/compile-compcert",
                            "terminal-bench/feal-differential-cryptanalysis",
                            "terminal-bench/qemu-startup",
                        ],
                    },
                ),
                limit_samples_map={
                    EvalLimitMode.SMOKE_TEST: 5,
                },
            ),
            # TODO: swe_bench_verified disabled due to timeouts from model limitations,
            # re-enable once prefix cache or equivalent is enabled.
            # timeout: https://github.com/tenstorrent/tt-shield/actions/runs/29363864010/job/87190489361#step:11:5880
            # ticket to re-enable: https://github.com/tenstorrent/tt-inference-server/issues/4675
            # EvalTask(
            #     task_name="swe_bench_verified",
            #     workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC,
            #     score=EvalTaskScore(
            #         published_score=77.2,
            #         published_score_ref="https://huggingface.co/Qwen/Qwen3.6-27B",
            #         gpu_reference_score=62.0,
            #         gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/3359#issuecomment-4427941401",
            #         score_func=score_task_single_key,
            #         score_func_kwargs={
            #             "result_keys": ["accuracy"],
            #             "unit": "percent",
            #         },
            #     ),
            #     swebench_eval_config=SWEbenchEvalConfig(
            #         dataset_name="SWE-bench/SWE-bench_Verified",
            #         sweagent_subset="verified",
            #         # we will need to specify specific tasks
            #         # for CI runs to keep runtime reasonable
            #         dataset_split="test",
            #         # mini-swe-agent is preferred: simpler CLI
            #         # The swe-agent backend is kept as a fallback.
            #         agent_backend="mini-swe-agent",
            #         n_concurrent_trials=5,
            #         max_workers=8,
            #         n_tasks=None,
            #         temperature=1.0,
            #         top_p=0.95,
            #         max_input_tokens=200 * 1024,
            #         # max output tokens is not specifed in Qwen docs btw
            #         max_output_tokens=32 * 1024,
            #         completion_kwargs={
            #             "extra_body": {
            #                 "top_k": 20,
            #             },
            #         },
            #         instance_ids_map={
            #             EvalLimitMode.CI_NIGHTLY: [
            #                 "django__django-11299",
            #                 "astropy__astropy-14096",
            #                 "matplotlib__matplotlib-25332",
            #                 "sympy__sympy-13551",
            #                 "scikit-learn__scikit-learn-14629",
            #             ],
            #         },
            #     ),
            #     limit_samples_map={
            #         EvalLimitMode.SMOKE_TEST: 5,
            #     },
            # ),
        ],
    ),
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
        hf_model_repo="Qwen/Qwen3-VL-32B-Instruct",
        tasks=[
            EvalTask(
                eval_class="openai_compatible",
                task_name="chartqa",
                workflow_venv_type=WorkflowVenvType.EVALS_VISION,
                apply_chat_template=False,
                use_chat_api=True,
                score=EvalTaskScore(
                    published_score=88.5,
                    published_score_ref="https://arxiv.org/pdf/2511.21631",
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
                    published_score=96.9,
                    published_score_ref="https://arxiv.org/pdf/2511.21631",
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
                    published_score=76.0,
                    published_score_ref="https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct#model-performance",
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
                    "max_length": 65536,
                },
                # gen_kwargs chosen according to https://huggingface.co/Qwen/Qwen3-8B#best-practices
                # max_gen_toks restored 12288 -> 32768: fits the P150 max_model_len
                # 40960 (was 16384 when clamped); the clamp truncated reasoning.
                gen_kwargs={
                    "stream": "true",
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
                    "max_length": 65536,
                    "timeout": "3600",
                },
                # gen_kwargs chosen according to https://huggingface.co/Qwen/Qwen3-8B#best-practices
                # max_gen_toks restored 12288 -> 32768: fits the P150 max_model_len
                # 40960 (was 16384 when clamped). Tracked in #4000.
                gen_kwargs={
                    "stream": "true",
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
        hf_model_repo="Qwen/Qwen3-4B",
        tasks=[
            EvalTask(
                task_name="r1_gpqa_diamond",
                score=EvalTaskScore(
                    published_score=None,
                    published_score_ref=None,
                    gpu_reference_score=None,
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
                    "max_length": 65536,
                },
                # max_gen_toks restored 12288 -> 32768 (see Qwen3-8B gpqa above).
                gen_kwargs={
                    "stream": "true",
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
            EvalTask(
                task_name="mmlu_pro",
                num_fewshot=5,
                score=EvalTaskScore(
                    published_score=None,
                    published_score_ref=None,
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
                workflow_venv_type=WorkflowVenvType.EVALS_COMMON,
                model_kwargs={
                    "max_length": 65536,
                    "timeout": "3600",
                },
                # max_gen_toks restored 12288 -> 32768 (see Qwen3-8B above). #4000.
                gen_kwargs={
                    "stream": "true",
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
            # --- Non-thinking published benchmark suite (https://qwenlm.github.io/blog/qwen3/) ---
            # Added for the BH-galaxy DP+TP deploy. These are the NON-thinking
            # scores (e.g. GPQA 49.49 vs the thinking 66.80 above) -- do NOT reuse
            # the thinking gen_kwargs above. Only the framework-supported tasks are
            # added here; SuperGPQA / MultiPL-E / CRUX-O / INCLUDE / EvalPlus-bundle
            # have no stock lm-eval task and are omitted. Long-prompt few-shot tasks
            # carry min_context_required so they SKIP at the 128/4K first-light
            # context and only run once MAX_MODEL_LENGTH is raised.
            EvalTask(
                task_name="mbpp_instruct",
                workflow_venv_type=WorkflowVenvType.EVALS_COMMON,
                score=EvalTaskScore(
                    published_score=78.20,
                    published_score_ref="https://qwenlm.github.io/blog/qwen3/",
                    gpu_reference_score=None,
                    gpu_reference_score_ref="TBD",
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
                task_name="mmlu_pro",
                num_fewshot=5,
                min_context_required=16384,
                score=EvalTaskScore(
                    published_score=65.54,
                    published_score_ref="https://qwenlm.github.io/blog/qwen3/",
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
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.2,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
            ),
            EvalTask(
                task_name="gpqa_diamond_generative_n_shot",
                num_fewshot=5,
                use_chat_api=True,
                min_context_required=16384,
                score=EvalTaskScore(
                    published_score=49.49,
                    published_score_ref="https://qwenlm.github.io/blog/qwen3/",
                    gpu_reference_score=None,
                    gpu_reference_score_ref="TBD",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "exact_match,flexible-extract",
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
                task_name="mmlu_generative",
                use_chat_api=True,
                min_context_required=16384,
                score=EvalTaskScore(
                    published_score=83.61,
                    published_score_ref="https://qwenlm.github.io/blog/qwen3/",
                    gpu_reference_score=None,
                    gpu_reference_score_ref="TBD",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "exact_match,get_response",
                        ],
                        "unit": "percent",
                    },
                ),
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.15,
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
        hf_model_repo="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        tasks=[
            EvalTask(
                task_name="gpqa_diamond_generative_n_shot",
                num_fewshot=5,
                use_chat_api=True,
                score=EvalTaskScore(
                    published_score=45.96,
                    published_score_ref="https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503#instruction-evals",
                    gpu_reference_score=None,
                    gpu_reference_score_ref="TBD",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "exact_match,flexible-extract",
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
                task_name="humaneval_instruct",
                workflow_venv_type=WorkflowVenvType.EVALS_COMMON,
                score=EvalTaskScore(
                    published_score=88.41,
                    published_score_ref="https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503",
                    gpu_reference_score=None,
                    gpu_reference_score_ref="TBD",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "pass@1,create_test",
                        ],
                        "unit": "percent",
                    },
                ),
                apply_chat_template=False,
                max_concurrent=32,
                gen_kwargs={
                    "max_gen_toks": "256",
                    "do_sample": "false",
                    "stream": "false",
                },
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.5,
                    EvalLimitMode.SMOKE_TEST: 0.05,
                },
            ),
            EvalTask(
                eval_class="openai_compatible",
                task_name="chartqa",
                workflow_venv_type=WorkflowVenvType.EVALS_VISION,
                apply_chat_template=False,
                use_chat_api=True,
                score=EvalTaskScore(
                    published_score=86.24,
                    published_score_ref="https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503",
                    gpu_reference_score=None,
                    gpu_reference_score_ref="TBD",
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
                    "stop": "</s>",
                    "stream": "False",
                },
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
                    published_score=91.40,
                    published_score_ref="https://huggingface.co/deepseek-ai/DeepSeek-R1-0528#deepseek-r1-0528-1",
                    gpu_reference_score=83.33,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/357#issuecomment-3048350923",
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
                    "max_length": 32768,
                },
                gen_kwargs={
                    "stream": "false",
                    "max_gen_toks": 28 * 1024,  # Allow up to 4K prompt tokens
                },
                seed=42,
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.2,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
            ),
            EvalTask(
                task_name="r1_gpqa_diamond",
                score=EvalTaskScore(
                    published_score=81.00,
                    published_score_ref="https://huggingface.co/deepseek-ai/DeepSeek-R1-0528#deepseek-r1-0528-1",
                    gpu_reference_score=81.31,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/357#issuecomment-3048350923",
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
                    "max_length": 32768,
                },
                gen_kwargs={
                    "stream": "false",
                    "max_gen_toks": 28 * 1024,  # Allow up to 4K prompt tokens
                },
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
            EvalTask(
                task_name="longbench_code_e",
                min_context_required=16384,
                score=EvalTaskScore(
                    published_score=None,
                    published_score_ref=None,
                    gpu_reference_score=31.89,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/1925#issuecomment-3813050051",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": ["score,none"],
                        "unit": "percent",
                    },
                ),
            ),
            EvalTask(
                task_name="longbench_fewshot_e",
                min_context_required=16384,
                score=EvalTaskScore(
                    published_score=None,
                    published_score_ref=None,
                    gpu_reference_score=51.66,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/1925#issuecomment-3813050051",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": ["score,none"],
                        "unit": "percent",
                    },
                ),
            ),
            EvalTask(
                task_name="longbench_multi_e",
                min_context_required=16384,
                score=EvalTaskScore(
                    published_score=None,
                    published_score_ref=None,
                    gpu_reference_score=18.58,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/1925#issuecomment-3813050051",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": ["score,none"],
                        "unit": "percent",
                    },
                ),
            ),
            EvalTask(
                task_name="longbench_single_e",
                min_context_required=16384,
                score=EvalTaskScore(
                    published_score=None,
                    published_score_ref=None,
                    gpu_reference_score=18.75,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/1925#issuecomment-3813050051",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": ["score,none"],
                        "unit": "percent",
                    },
                ),
            ),
            EvalTask(
                task_name="longbench_summarization_e",
                min_context_required=16384,
                score=EvalTaskScore(
                    published_score=None,
                    published_score_ref=None,
                    gpu_reference_score=22.65,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/1925#issuecomment-3813050051",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": ["score,none"],
                        "unit": "percent",
                    },
                ),
            ),
            EvalTask(
                task_name="longbench_synthetic_e",
                min_context_required=16384,
                score=EvalTaskScore(
                    published_score=None,
                    published_score_ref=None,
                    gpu_reference_score=7.67,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/1925#issuecomment-3813050051",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": ["score,none"],
                        "unit": "percent",
                    },
                ),
            ),
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
            EvalTask(
                task_name="longbench_code_e",
                min_context_required=16384,
                score=EvalTaskScore(
                    published_score=None,
                    published_score_ref=None,
                    gpu_reference_score=48.12,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/1948#issuecomment-3821456040",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": ["score,none"],
                        "unit": "percent",
                    },
                ),
            ),
            EvalTask(
                task_name="longbench_fewshot_e",
                min_context_required=16384,
                score=EvalTaskScore(
                    published_score=None,
                    published_score_ref=None,
                    gpu_reference_score=63.34,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/1948#issuecomment-3821456040",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": ["score,none"],
                        "unit": "percent",
                    },
                ),
            ),
            EvalTask(
                task_name="longbench_multi_e",
                min_context_required=16384,
                score=EvalTaskScore(
                    published_score=None,
                    published_score_ref=None,
                    gpu_reference_score=20.84,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/1948#issuecomment-3821456040",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": ["score,none"],
                        "unit": "percent",
                    },
                ),
            ),
            EvalTask(
                task_name="longbench_single_e",
                min_context_required=16384,
                score=EvalTaskScore(
                    published_score=None,
                    published_score_ref=None,
                    gpu_reference_score=22.22,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/1948#issuecomment-3821456040",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": ["score,none"],
                        "unit": "percent",
                    },
                ),
            ),
            EvalTask(
                task_name="longbench_summarization_e",
                min_context_required=16384,
                score=EvalTaskScore(
                    published_score=None,
                    published_score_ref=None,
                    gpu_reference_score=26.09,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/1948#issuecomment-3821456040",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": ["score,none"],
                        "unit": "percent",
                    },
                ),
            ),
            EvalTask(
                task_name="longbench_synthetic_e",
                min_context_required=16384,
                score=EvalTaskScore(
                    published_score=None,
                    published_score_ref=None,
                    gpu_reference_score=14.86,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/1948#issuecomment-3821456040",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": ["score,none"],
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
        hf_model_repo="black-forest-labs/FLUX.1-dev",
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
        hf_model_repo="black-forest-labs/FLUX.1-schnell",
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
        hf_model_repo="Motif-Technologies/Motif-Image-6B-Preview",
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
        hf_model_repo="Tongyi-MAI/Z-Image-Turbo",
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
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "wer,none",
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
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "wer,none",
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
    # VIDEO models (Mochi, Wan2.2 T2V/I2V) are served by the v2 engine (routed
    # via workflows/v2_bridge.can_route_to_v2); the actual eval runs in v2's
    # test_module, and ModelType.VIDEO is not in EVAL_TASK_TYPES (evals/run_evals.py),
    # so the v1 lm-eval path never dispatches this "load_video" task and the
    # workflow_venv_type below is never provisioned for video. These entries must
    # still exist: workflows/validate_setup.py asserts every EVALS/RELEASE model
    # is registered in EVAL_CONFIGS *before* v2 routing, so removing them breaks
    # `run.py --workflow evals/release` for video at the validation gate.
    EvalConfig(
        hf_model_repo="genmo/mochi-1-preview",
        tasks=[
            EvalTask(
                task_name="load_video",
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
        hf_model_repo="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        tasks=[
            EvalTask(
                task_name="load_video",
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
        hf_model_repo="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        tasks=[
            EvalTask(
                task_name="load_video",
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
        hf_model_repo="BAAI/bge-m3",
        tasks=[
            EvalTask(
                task_name="embedding",
                workflow_venv_type=WorkflowVenvType.EVALS_EMBEDDING,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
                score=EvalTaskScore(
                    published_score="",
                    published_score_ref="",
                    gpu_reference_score=0.7873,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/2892",
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
                    published_score=70.58,
                    published_score_ref="https://huggingface.co/Qwen/Qwen3-Embedding-8B",
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
        hf_model_repo="yolox_nano",
        tasks=[
            EvalTask(
                task_name="load_image",
                workflow_venv_type=WorkflowVenvType.EVALS_META,
                include_path="work_dir",
                max_concurrent=None,
                apply_chat_template=False,
                score=EvalTaskScore(
                    published_score=25.8,
                    published_score_ref="https://arxiv.org/abs/2107.08430",
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
        hf_model_repo="openai/gpt-oss-20b",
        tasks=[
            EvalTask(
                task_name="aime25",
                limit_samples_map={
                    EvalLimitMode.SMOKE_TEST: 0.05,  # 30 samples * 0.05 ~= 1 sample
                    EvalLimitMode.CI_NIGHTLY: 0.2,  # 30 samples * 0.2 = 6 samples
                },
                score=EvalTaskScore(
                    published_score=91.7,  # AIME 2025 score (without tools)
                    published_score_ref="https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf",
                    gpu_reference_score=88.8,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/1323#issuecomment-3842033753",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "exact_match,none",
                        ],
                        "unit": "percent",
                    },
                ),
                use_chat_api=True,
                max_concurrent=16,
                model_kwargs={
                    "timeout": "7200",
                },
                gen_kwargs={
                    "reasoning_effort": "high",
                    "do_sample": "true",
                    "temperature": 1.0,
                    "max_gen_toks": 64 * 1024,
                },
            ),
            EvalTask(
                task_name="gpqa_diamond_cot_zeroshot",
                limit_samples_map={
                    EvalLimitMode.SMOKE_TEST: 0.006,  # 198 samples * 0.006 ~= 1 sample
                    EvalLimitMode.CI_NIGHTLY: 0.035,  # 198 samples * 0.035 ~= 6 samples
                },
                score=EvalTaskScore(
                    published_score=71.5,  # GPQA Diamond score (without tools)
                    published_score_ref="https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf",
                    gpu_reference_score=72.8,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/1323#issuecomment-3848121656",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "exact_match,flexible-extract",
                        ],
                        "unit": "percent",
                    },
                ),
                use_chat_api=True,
                max_concurrent=16,
                model_kwargs={
                    "timeout": "7200",
                },
                gen_kwargs={
                    "reasoning_effort": "high",
                    "do_sample": "true",
                    "temperature": 1.0,
                    "max_gen_toks": 64 * 1024,
                },
            ),
            EvalTask(
                task_name="mmlu_generative",  # base MMLU task in lm-eval-harness uses loglikelihood evaluation
                limit_samples_map={
                    EvalLimitMode.SMOKE_TEST: 0.000063,  # 15,908 samples * 0.00006286 ~= 1 sample per sub-task
                    EvalLimitMode.CI_NIGHTLY: 0.15,  # 15% of 15,902 samples ~= 42 samples per sub-task
                },
                score=EvalTaskScore(
                    published_score=80.4,  # MMLU score "low" reasoning level (without tools)
                    published_score_ref="https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf",
                    gpu_reference_score=80.4,  # TODO: MEASURE THIS https://github.com/tenstorrent/tt-inference-server/issues/1323
                    gpu_reference_score_ref="DUMMY VALUE",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "exact_match,get_response",
                        ],
                        "unit": "percent",
                    },
                ),
                use_chat_api=True,
                max_concurrent=128,
                model_kwargs={
                    "timeout": "7200",
                },
                gen_kwargs={
                    "reasoning_effort": "low",
                    "do_sample": "true",
                    "temperature": 1.0,
                    "max_gen_toks": 64 * 1024,
                    "until": ["</s>"],
                },
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="openai/gpt-oss-120b",
        tasks=[
            EvalTask(
                task_name="aime25",
                limit_samples_map={
                    EvalLimitMode.SMOKE_TEST: 0.05,  # 30 samples * 0.05 ~= 1 sample
                    EvalLimitMode.CI_NIGHTLY: 0.50,  # 30 samples * 0.5 = 15 samples
                },
                score=EvalTaskScore(
                    published_score=92.5,  # AIME 2025 score (without tools)
                    published_score_ref="https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf",
                    gpu_reference_score=90.4,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/1322#issuecomment-3801635211",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "exact_match,none",
                        ],
                        "unit": "percent",
                    },
                ),
                use_chat_api=True,
                max_concurrent=32,
                model_kwargs={
                    "timeout": "14400",
                },
                gen_kwargs={
                    # lm-eval-harness' SSE consumer only parses
                    # /v1/completions chunks, not /v1/chat/completions; keep
                    # stream=false to avoid empty resps + KeyError: 'message'.
                    "stream": "false",
                    "reasoning_effort": "high",
                    "do_sample": "true",
                    "temperature": 1.0,
                    # Must stay strictly below max_context (131072); equal
                    # values leave zero headroom, the Harmony path schedules
                    # a 1-token prefill, and every response comes back empty.
                    "max_gen_toks": 120 * 1024,
                },
            ),
            EvalTask(
                task_name="gpqa_diamond_cot_zeroshot",
                limit_samples_map={
                    EvalLimitMode.SMOKE_TEST: 0.006,  # 198 samples * 0.006 ~= 1 sample
                    EvalLimitMode.CI_NIGHTLY: 0.035,  # 198 samples * 0.035 ~= 6 samples
                },
                score=EvalTaskScore(
                    published_score=80.1,  # GPQA Diamond score (without tools)
                    published_score_ref="https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf",
                    gpu_reference_score=79.7,
                    gpu_reference_score_ref="https://github.com/tenstorrent/tt-inference-server/issues/1322#issuecomment-3801635211",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "exact_match,flexible-extract",
                        ],
                        "unit": "percent",
                    },
                ),
                use_chat_api=True,
                max_concurrent=32,
                model_kwargs={
                    "timeout": "14400",
                },
                gen_kwargs={
                    "stream": "false",
                    "reasoning_effort": "high",
                    "do_sample": "true",
                    "temperature": 1.0,
                    "max_gen_toks": 120 * 1024,
                },
            ),
            EvalTask(
                task_name="mmlu_generative",  # base MMLU task in lm-eval-harness uses loglikelihood evaluation
                limit_samples_map={
                    EvalLimitMode.SMOKE_TEST: 0.000063,  # 15,908 samples * 0.00006286 ~= 1 sample per sub-task
                    EvalLimitMode.CI_NIGHTLY: 0.15,  # 15% of 15,902 samples ~= 42 samples per sub-task
                },
                score=EvalTaskScore(
                    published_score=85.9,  # MMLU score "low" reasoning level (without tools)
                    published_score_ref="https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf",
                    gpu_reference_score=85.9,  # TODO: MEASURE THIS https://github.com/tenstorrent/tt-inference-server/issues/1322
                    gpu_reference_score_ref="DUMMY VALUE",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "exact_match,get_response",
                        ],
                        "unit": "percent",
                    },
                ),
                use_chat_api=True,
                max_concurrent=128,
                model_kwargs={
                    "timeout": "7200",
                },
                gen_kwargs={
                    "stream": "false",
                    "reasoning_effort": "low",
                    "do_sample": "true",
                    "temperature": 1.0,
                    "max_gen_toks": 64 * 1024,
                    "until": ["</s>"],
                },
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="tiiuae/Falcon3-7B-Instruct",
        tasks=[
            # tokenizer_backend defaults to "huggingface" — matches the Qwen3-*
            # configs below. Don't set "none" here: without a tokenizer loaded
            # lm-eval can't apply_chat_template host-side and instead sends the
            # raw chat message list, which the server's CompletionRequest schema
            # rejects with HTTP 422.
            EvalTask(
                task_name="ifeval",
                score=EvalTaskScore(
                    published_score=None,
                    published_score_ref="https://huggingface.co/tiiuae/Falcon3-7B-Instruct",
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
                    EvalLimitMode.CI_NIGHTLY: 0.05,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
            ),
            EvalTask(
                task_name="gpqa_diamond_generative_n_shot",
                num_fewshot=5,
                score=EvalTaskScore(
                    published_score=None,
                    published_score_ref="https://huggingface.co/tiiuae/Falcon3-7B-Instruct",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "exact_match,flexible-extract",
                        ],
                        "unit": "percent",
                    },
                ),
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.05,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
            ),
        ],
    ),
    # =========================================================================
    # Gemma 4 family - GPU reference eval configs.
    #
    # Mirrors the Qwen/Qwen3.6-27B agentic block above and adds GPQA-Diamond.
    # Recipe follows the footnotes of the eval table on the Qwen3.6-27B HF page
    # (https://huggingface.co/Qwen/Qwen3.6-27B):
    #   - SWE-Bench: temp=1.0, top_p=0.95.
    #   - Terminal-Bench 2.0: Terminus-2 harness; temp=1.0, top_p=0.95,
    #     top_k=20; 3h timeout; 32 CPU / 48 GB RAM.
    # Published reference scores exist only for Gemma4-31B (the only gemma-4
    # column in that table); other variants record GPU reference scores only.
    #
    # Context window per the Gemma 4 model card
    # (https://ai.google.dev/gemma/docs/core/model_card_4): the medium models
    # 31B / 26B-A4B / 12B support 256K tokens; the small E2B / E4B support 128K.
    # Agentic max_input_tokens + max_output_tokens are sized to fit the model's
    # native window (the agent sends ~input+output per request); run the vLLM
    # server with a matching --max-model-len (262144 for 256K models, 131072 for
    # the E-models). NOTE: only the 31B agentic tasks are bumped to 256K so far;
    # 26B-A4B / 12B still carry the 128K (96K+32K) placeholder budgets.
    # =========================================================================
    EvalConfig(
        hf_model_repo="google/gemma-4-31B-it",
        tasks=[
            EvalTask(
                # R1-style zero-shot reasoning GPQA Diamond. This matches the
                # thinking-mode methodology behind the Qwen3.6-27B table's
                # "GPQA Diamond" column (model emits reasoning, then a final
                # answer; the task's own extractor scores exact_match,none).
                # The gpqa_diamond_generative_n_shot variant is wrong for a
                # reasoning model: its 5-shot examples demonstrate bare "(C)"
                # answers, suppressing reasoning (gemma-4 scored only ~53%).
                task_name="r1_gpqa_diamond",
                score=EvalTaskScore(
                    published_score=84.3,
                    published_score_ref="https://huggingface.co/Qwen/Qwen3.6-27B",
                    # Full 198-sample r1_gpqa_diamond, single run, on an H100
                    # reference vLLM server (vllm 0.23.1rc1.dev, max-model-len
                    # 131072) with thinking enabled, temp=1.0/top_p=0.95/
                    # top_k=20, 124K output budget. 83.33% +/- 2.66 via the
                    # orchestrated run.py --workflow evals path, 2026-06-16
                    # (a prior direct lm-eval run scored 82.32% +/- 2.72; both
                    # CIs cover the published 84.3). (Without thinking the same
                    # task/sampling scored only 75.76 -- enable_thinking is the
                    # key: gemma-4's template otherwise injects an empty thought
                    # channel that suppresses native reasoning.)
                    gpu_reference_score=83.33,
                    gpu_reference_score_ref="run.py --workflow evals r1_gpqa_diamond full (198), H100 gemma-4-31B-it bring-your-own vLLM w/ enable_thinking=true, 2026-06-16",
                    # CI subset (--ci-mode -> ci-nightly limit 0.2 = doc_ids
                    # 0-39). The full run scores only ~75% on these same 40
                    # harder-than-average questions, and temp=1.0 adds ~+/-2.5
                    # pts jitter, so compare against the subset measurement with
                    # a looser 10% tolerance instead of the full-set 83.33.
                    mode_reference_scores={
                        EvalLimitMode.CI_NIGHTLY: ModeReferenceScore(
                            score=80.00,
                            ref="ci-nightly r1_gpqa_diamond (doc_ids 0-39), H100 gemma-4-31B-it, 2026-06-29",
                            tolerance=0.10,
                        ),
                    },
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": [
                            "exact_match,none",
                        ],
                        "unit": "percent",
                    },
                ),
                workflow_venv_type=WorkflowVenvType.EVALS_COMMON,
                # Use the chat endpoint so the server applies the chat template
                # (and thus --default-chat-template-kwargs '{"enable_thinking":
                # true}'); client-side apply_chat_template on /v1/completions
                # would render with the default enable_thinking=false.
                use_chat_api=True,
                model_kwargs={
                    "max_length": 131072,
                },
                # Thinking-mode sampling (Qwen3.6 page, general tasks):
                # temperature=1.0, top_p=0.95, top_k=20.
                # stream=false is REQUIRED: lm-eval's local-chat-completions
                # streaming parser raises KeyError 'message' on every response.
                # max_gen_toks must stay strictly below max_length (131072)
                # minus the prompt (server 400s if output+prompt > ctx);
                # 124K leaves ~4K headroom for prompt + chat template.
                gen_kwargs={
                    "stream": "false",
                    "max_gen_toks": 124 * 1024,
                    "until": [],
                    "do_sample": "true",
                    "temperature": 1.0,
                    "top_k": 20,
                    "top_p": 0.95,
                },
                # CI_NIGHTLY 0.05 (~10 samples), not the usual 0.2: reasoning
                # eval (~48K tokens/sample) served batch-1, so samples run
                # sequentially and dominate CI runtime. Widen EvalTaskScore
                # tolerance if the small-N accuracy gate flakes.
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.05,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
            ),
            EvalTask(
                task_name="terminal_bench_2",
                workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC,
                score=EvalTaskScore(
                    published_score=42.9,
                    published_score_ref="https://huggingface.co/Qwen/Qwen3.6-27B",
                    # Full terminal-bench-2 (89 tasks), terminus-2, single
                    # H100 NVL bring-your-own vLLM (gemma-4-31B-it, max-model-len
                    # 204800, enable_thinking=true), temp=1.0/top_p=0.95/
                    # top_k=20, 112K in / 80K out, 2026-06-17. 40/89 solved =
                    # 44.94%, which exceeds the published 42.9. 16 tasks hit
                    # timeouts (15 AgentTimeoutError at the 3h/task limit + 1
                    # VerifierTimeoutError) and scored 0, so 44.94 is a floor;
                    # raising agent_timeout_sec could recover a few.
                    gpu_reference_score=44.94,
                    gpu_reference_score_ref="run.py --workflow evals terminal_bench_2 full (89), H100 gemma-4-31B-it bring-your-own vLLM w/ enable_thinking=true, 2026-06-17",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": ["accuracy"],
                        "unit": "percent",
                    },
                ),
                agentic_eval_config=TerminalBenchEvalConfig(
                    dataset="terminal-bench/terminal-bench-2",
                    agent="terminus-2",
                    n_concurrent_trials=1,  # TODO increase back to 5 when batch > 1 is supported
                    n_attempts=1,
                    n_tasks=None,  # full dataset
                    # QB2 release runners expose only 16 CPUs; docker compose
                    # rejects a higher --cpus reservation ("range of CPUs is
                    # from 0.01 to 16.00"). The H100 reference run used 32.
                    override_cpus=16,
                    override_memory_mb=48 * 1024,
                    agent_timeout_sec=3 * 60 * 60,
                    agent_kwargs={
                        "parser_name": "json",
                        "temperature": 1.0,
                        "model_info": {
                            # gemma-4-31B native ctx is 256K (model card), but a
                            # single H100 NVL (94GB, bf16 KV) only holds a
                            # 210,605-token KV cache, so a request can't exceed
                            # ~205K. We serve at --max-model-len 204800 (200K).
                            # The agent sends ~max_input + max_output per
                            # request, so keep them under 204800: 112K + 80K =
                            # 196K (~8K headroom for chat template + tool defs).
                            # SWE/Terminal prompts rarely approach 200K, so the
                            # 256K->200K cap should not affect scores.
                            "max_input_tokens": 112 * 1024,
                            "max_output_tokens": 80 * 1024,
                        },
                        "llm_kwargs": {
                            "top_p": 0.95,
                            "max_tokens": 80 * 1024,
                            "timeout": 60 * 60,
                            "extra_body": {
                                "top_k": 20,
                            },
                        },
                    },
                    task_names_map={
                        EvalLimitMode.CI_NIGHTLY: [
                            "terminal-bench/break-filter-js-from-html",
                            "terminal-bench/cobol-modernization",
                            "terminal-bench/compile-compcert",
                            "terminal-bench/feal-differential-cryptanalysis",
                            "terminal-bench/qemu-startup",
                        ],
                    },
                ),
                limit_samples_map={
                    EvalLimitMode.SMOKE_TEST: 5,
                },
            ),
            EvalTask(
                task_name="swe_bench_verified",
                workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC,
                score=EvalTaskScore(
                    published_score=52.0,
                    published_score_ref="https://huggingface.co/Qwen/Qwen3.6-27B",
                    # Full SWE-bench Verified (500), mini-swe-agent, single
                    # H100 NVL bring-your-own vLLM (gemma-4-31B-it, max-model-len
                    # 204800, enable_thinking=true), temp=1.0/top_p=0.95/
                    # top_k=20, 160K in / 32K out, 2026-06-18. 324/500 resolved
                    # = 64.80%, which exceeds the published 52.0 (ratio 1.25).
                    gpu_reference_score=64.80,
                    gpu_reference_score_ref="run.py --workflow evals swe_bench_verified full (500), H100 gemma-4-31B-it bring-your-own vLLM w/ enable_thinking=true, 2026-06-18",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": ["accuracy"],
                        "unit": "percent",
                    },
                ),
                swebench_eval_config=SWEbenchEvalConfig(
                    dataset_name="SWE-bench/SWE-bench_Verified",
                    sweagent_subset="verified",
                    dataset_split="test",
                    agent_backend="mini-swe-agent",
                    n_concurrent_trials=5,
                    max_workers=8,
                    n_tasks=None,  # full dataset
                    temperature=1.0,
                    top_p=0.95,
                    # gemma-4-31B native ctx is 256K (model card), but a single
                    # H100 NVL (94GB, bf16 KV) only holds a 210,605-token KV
                    # cache, so a request can't exceed ~205K. We serve at
                    # --max-model-len 204800 (200K). mini-swe-agent sends
                    # ~max_input + max_output per request, so keep under 204800:
                    # 160K + 32K = 192K (~8K headroom). SWE prompts rarely
                    # approach 200K, so the 256K->200K cap should not affect
                    # scores.
                    max_input_tokens=160 * 1024,
                    max_output_tokens=32 * 1024,
                    completion_kwargs={
                        "extra_body": {
                            "top_k": 20,
                        },
                    },
                    instance_ids_map={
                        EvalLimitMode.CI_NIGHTLY: [
                            "django__django-11299",
                            "astropy__astropy-14096",
                            "matplotlib__matplotlib-25332",
                            "sympy__sympy-13551",
                            "scikit-learn__scikit-learn-14629",
                        ],
                    },
                ),
                limit_samples_map={
                    EvalLimitMode.SMOKE_TEST: 5,
                },
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="google/gemma-4-26B-A4B-it",
        tasks=[
            EvalTask(
                # R1-style zero-shot reasoning GPQA Diamond (see gemma-4-31B-it
                # note above). Matches the Qwen3.6-27B table's thinking-mode
                # "GPQA Diamond" methodology; scores exact_match,none.
                task_name="r1_gpqa_diamond",
                score=EvalTaskScore(
                    published_score=None,
                    published_score_ref="https://huggingface.co/Qwen/Qwen3.6-27B",
                    gpu_reference_score=None,
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
                # Use the chat endpoint so the server applies the chat template
                # (and thus --default-chat-template-kwargs '{"enable_thinking":
                # true}'); client-side apply_chat_template on /v1/completions
                # would render with the default enable_thinking=false and
                # suppress native reasoning (see gemma-4-31B-it note above).
                use_chat_api=True,
                model_kwargs={
                    "max_length": 131072,
                },
                # Thinking-mode sampling (Qwen3.6 page, general tasks):
                # temperature=1.0, top_p=0.95, top_k=20.
                # stream=false is REQUIRED: lm-eval's local-chat-completions
                # streaming parser raises KeyError 'message' on every response.
                gen_kwargs={
                    "stream": "false",
                    "max_gen_toks": 32 * 1024,
                    "until": [],
                    "do_sample": "true",
                    "temperature": 1.0,
                    "top_k": 20,
                    "top_p": 0.95,
                },
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.2,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
            ),
            EvalTask(
                task_name="terminal_bench_2",
                workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC,
                score=EvalTaskScore(
                    published_score=None,
                    published_score_ref="https://huggingface.co/Qwen/Qwen3.6-27B",
                    gpu_reference_score=None,
                    gpu_reference_score_ref="TBD",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": ["accuracy"],
                        "unit": "percent",
                    },
                ),
                agentic_eval_config=TerminalBenchEvalConfig(
                    dataset="terminal-bench/terminal-bench-2",
                    agent="terminus-2",
                    n_concurrent_trials=5,
                    n_attempts=1,
                    n_tasks=None,
                    override_cpus=32,
                    override_memory_mb=48 * 1024,
                    agent_timeout_sec=3 * 60 * 60,
                    agent_kwargs={
                        "parser_name": "json",
                        "temperature": 1.0,
                        "model_info": {
                            "max_input_tokens": 96 * 1024,
                            "max_output_tokens": 32 * 1024,
                        },
                        "llm_kwargs": {
                            "top_p": 0.95,
                            "max_tokens": 32 * 1024,
                            "timeout": 60 * 60,
                            "extra_body": {
                                "top_k": 20,
                            },
                        },
                    },
                    task_names_map={
                        EvalLimitMode.CI_NIGHTLY: [
                            "terminal-bench/caffe-cifar-10",
                            "terminal-bench/password-recovery",
                            "terminal-bench/portfolio-optimization",
                            "terminal-bench/hf-model-inference",
                            "terminal-bench/financial-document-processor",
                        ],
                    },
                ),
                limit_samples_map={
                    EvalLimitMode.SMOKE_TEST: 5,
                },
            ),
            EvalTask(
                task_name="swe_bench_verified",
                workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC,
                score=EvalTaskScore(
                    published_score=None,
                    published_score_ref="https://huggingface.co/Qwen/Qwen3.6-27B",
                    gpu_reference_score=None,
                    gpu_reference_score_ref="TBD",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": ["accuracy"],
                        "unit": "percent",
                    },
                ),
                swebench_eval_config=SWEbenchEvalConfig(
                    dataset_name="SWE-bench/SWE-bench_Verified",
                    sweagent_subset="verified",
                    dataset_split="test",
                    agent_backend="mini-swe-agent",
                    n_concurrent_trials=5,
                    max_workers=8,
                    n_tasks=None,
                    temperature=1.0,
                    top_p=0.95,
                    # Clamped so max_input + max_output (96K + 32K = 128K) fits gemma-4's 131072 ctx.
                    max_input_tokens=96 * 1024,
                    max_output_tokens=32 * 1024,
                    completion_kwargs={
                        "extra_body": {
                            "top_k": 20,
                        },
                    },
                    instance_ids_map={
                        EvalLimitMode.CI_NIGHTLY: [
                            "django__django-11299",
                            "astropy__astropy-14096",
                            "matplotlib__matplotlib-25332",
                            "sympy__sympy-13551",
                            "scikit-learn__scikit-learn-14629",
                        ],
                    },
                ),
                limit_samples_map={
                    EvalLimitMode.SMOKE_TEST: 5,
                },
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="google/gemma-4-12B-it",
        tasks=[
            EvalTask(
                # R1-style zero-shot reasoning GPQA Diamond (see gemma-4-31B-it
                # note above). Matches the Qwen3.6-27B table's thinking-mode
                # "GPQA Diamond" methodology; scores exact_match,none.
                task_name="r1_gpqa_diamond",
                score=EvalTaskScore(
                    published_score=None,
                    published_score_ref="https://huggingface.co/Qwen/Qwen3.6-27B",
                    gpu_reference_score=None,
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
                # Use the chat endpoint so the server applies the chat template
                # (and thus --default-chat-template-kwargs '{"enable_thinking":
                # true}'); client-side apply_chat_template on /v1/completions
                # would render with the default enable_thinking=false and
                # suppress native reasoning (see gemma-4-31B-it note above).
                use_chat_api=True,
                model_kwargs={
                    "max_length": 131072,
                },
                # Thinking-mode sampling (Qwen3.6 page, general tasks):
                # temperature=1.0, top_p=0.95, top_k=20.
                # stream=false is REQUIRED: lm-eval's local-chat-completions
                # streaming parser raises KeyError 'message' on every response.
                gen_kwargs={
                    "stream": "false",
                    "max_gen_toks": 32 * 1024,
                    "until": [],
                    "do_sample": "true",
                    "temperature": 1.0,
                    "top_k": 20,
                    "top_p": 0.95,
                },
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.2,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
            ),
            EvalTask(
                task_name="terminal_bench_2",
                workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC,
                score=EvalTaskScore(
                    published_score=None,
                    published_score_ref="https://huggingface.co/Qwen/Qwen3.6-27B",
                    gpu_reference_score=None,
                    gpu_reference_score_ref="TBD",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": ["accuracy"],
                        "unit": "percent",
                    },
                ),
                agentic_eval_config=TerminalBenchEvalConfig(
                    dataset="terminal-bench/terminal-bench-2",
                    agent="terminus-2",
                    n_concurrent_trials=5,
                    n_attempts=1,
                    n_tasks=None,
                    override_cpus=32,
                    override_memory_mb=48 * 1024,
                    agent_timeout_sec=3 * 60 * 60,
                    agent_kwargs={
                        "parser_name": "json",
                        "temperature": 1.0,
                        "model_info": {
                            "max_input_tokens": 96 * 1024,
                            "max_output_tokens": 32 * 1024,
                        },
                        "llm_kwargs": {
                            "top_p": 0.95,
                            "max_tokens": 32 * 1024,
                            "timeout": 60 * 60,
                            "extra_body": {
                                "top_k": 20,
                            },
                        },
                    },
                    task_names_map={
                        EvalLimitMode.CI_NIGHTLY: [
                            "terminal-bench/caffe-cifar-10",
                            "terminal-bench/password-recovery",
                            "terminal-bench/portfolio-optimization",
                            "terminal-bench/hf-model-inference",
                            "terminal-bench/financial-document-processor",
                        ],
                    },
                ),
                limit_samples_map={
                    EvalLimitMode.SMOKE_TEST: 5,
                },
            ),
            EvalTask(
                task_name="swe_bench_verified",
                workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC,
                score=EvalTaskScore(
                    published_score=None,
                    published_score_ref="https://huggingface.co/Qwen/Qwen3.6-27B",
                    gpu_reference_score=None,
                    gpu_reference_score_ref="TBD",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": ["accuracy"],
                        "unit": "percent",
                    },
                ),
                swebench_eval_config=SWEbenchEvalConfig(
                    dataset_name="SWE-bench/SWE-bench_Verified",
                    sweagent_subset="verified",
                    dataset_split="test",
                    agent_backend="mini-swe-agent",
                    n_concurrent_trials=5,
                    max_workers=8,
                    n_tasks=None,
                    temperature=1.0,
                    top_p=0.95,
                    # Clamped so max_input + max_output (96K + 32K = 128K) fits gemma-4's 131072 ctx.
                    max_input_tokens=96 * 1024,
                    max_output_tokens=32 * 1024,
                    completion_kwargs={
                        "extra_body": {
                            "top_k": 20,
                        },
                    },
                    instance_ids_map={
                        EvalLimitMode.CI_NIGHTLY: [
                            "django__django-11299",
                            "astropy__astropy-14096",
                            "matplotlib__matplotlib-25332",
                            "sympy__sympy-13551",
                            "scikit-learn__scikit-learn-14629",
                        ],
                    },
                ),
                limit_samples_map={
                    EvalLimitMode.SMOKE_TEST: 5,
                },
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="google/gemma-4-E4B-it",
        tasks=[
            EvalTask(
                # R1-style zero-shot reasoning GPQA Diamond (see gemma-4-31B-it
                # note above). Matches the Qwen3.6-27B table's thinking-mode
                # "GPQA Diamond" methodology; scores exact_match,none.
                task_name="r1_gpqa_diamond",
                score=EvalTaskScore(
                    published_score=None,
                    published_score_ref="https://huggingface.co/Qwen/Qwen3.6-27B",
                    gpu_reference_score=None,
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
                # Use the chat endpoint so the server applies the chat template
                # (and thus --default-chat-template-kwargs '{"enable_thinking":
                # true}'); client-side apply_chat_template on /v1/completions
                # would render with the default enable_thinking=false and
                # suppress native reasoning (see gemma-4-31B-it note above).
                use_chat_api=True,
                model_kwargs={
                    "max_length": 131072,
                },
                # Thinking-mode sampling (Qwen3.6 page, general tasks):
                # temperature=1.0, top_p=0.95, top_k=20.
                # stream=false is REQUIRED: lm-eval's local-chat-completions
                # streaming parser raises KeyError 'message' on every response.
                gen_kwargs={
                    "stream": "false",
                    "max_gen_toks": 32 * 1024,
                    "until": [],
                    "do_sample": "true",
                    "temperature": 1.0,
                    "top_k": 20,
                    "top_p": 0.95,
                },
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.2,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
            ),
            EvalTask(
                task_name="terminal_bench_2",
                workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC,
                score=EvalTaskScore(
                    published_score=None,
                    published_score_ref="https://huggingface.co/Qwen/Qwen3.6-27B",
                    gpu_reference_score=None,
                    gpu_reference_score_ref="TBD",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": ["accuracy"],
                        "unit": "percent",
                    },
                ),
                agentic_eval_config=TerminalBenchEvalConfig(
                    dataset="terminal-bench/terminal-bench-2",
                    agent="terminus-2",
                    n_concurrent_trials=5,
                    n_attempts=1,
                    n_tasks=None,
                    override_cpus=32,
                    override_memory_mb=48 * 1024,
                    agent_timeout_sec=3 * 60 * 60,
                    agent_kwargs={
                        "parser_name": "json",
                        "temperature": 1.0,
                        "model_info": {
                            "max_input_tokens": 96 * 1024,
                            "max_output_tokens": 32 * 1024,
                        },
                        "llm_kwargs": {
                            "top_p": 0.95,
                            "max_tokens": 32 * 1024,
                            "timeout": 60 * 60,
                            "extra_body": {
                                "top_k": 20,
                            },
                        },
                    },
                    task_names_map={
                        EvalLimitMode.CI_NIGHTLY: [
                            "terminal-bench/caffe-cifar-10",
                            "terminal-bench/password-recovery",
                            "terminal-bench/portfolio-optimization",
                            "terminal-bench/hf-model-inference",
                            "terminal-bench/financial-document-processor",
                        ],
                    },
                ),
                limit_samples_map={
                    EvalLimitMode.SMOKE_TEST: 5,
                },
            ),
            EvalTask(
                task_name="swe_bench_verified",
                workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC,
                score=EvalTaskScore(
                    published_score=None,
                    published_score_ref="https://huggingface.co/Qwen/Qwen3.6-27B",
                    gpu_reference_score=None,
                    gpu_reference_score_ref="TBD",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": ["accuracy"],
                        "unit": "percent",
                    },
                ),
                swebench_eval_config=SWEbenchEvalConfig(
                    dataset_name="SWE-bench/SWE-bench_Verified",
                    sweagent_subset="verified",
                    dataset_split="test",
                    agent_backend="mini-swe-agent",
                    n_concurrent_trials=5,
                    max_workers=8,
                    n_tasks=None,
                    temperature=1.0,
                    top_p=0.95,
                    # Clamped so max_input + max_output (96K + 32K = 128K) fits gemma-4's 131072 ctx.
                    max_input_tokens=96 * 1024,
                    max_output_tokens=32 * 1024,
                    completion_kwargs={
                        "extra_body": {
                            "top_k": 20,
                        },
                    },
                    instance_ids_map={
                        EvalLimitMode.CI_NIGHTLY: [
                            "django__django-11299",
                            "astropy__astropy-14096",
                            "matplotlib__matplotlib-25332",
                            "sympy__sympy-13551",
                            "scikit-learn__scikit-learn-14629",
                        ],
                    },
                ),
                limit_samples_map={
                    EvalLimitMode.SMOKE_TEST: 5,
                },
            ),
        ],
    ),
    EvalConfig(
        hf_model_repo="google/gemma-4-E2B-it",
        tasks=[
            EvalTask(
                # R1-style zero-shot reasoning GPQA Diamond (see gemma-4-31B-it
                # note above). Matches the Qwen3.6-27B table's thinking-mode
                # "GPQA Diamond" methodology; scores exact_match,none.
                task_name="r1_gpqa_diamond",
                score=EvalTaskScore(
                    published_score=None,
                    published_score_ref="https://huggingface.co/Qwen/Qwen3.6-27B",
                    gpu_reference_score=None,
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
                # Use the chat endpoint so the server applies the chat template
                # (and thus --default-chat-template-kwargs '{"enable_thinking":
                # true}'); client-side apply_chat_template on /v1/completions
                # would render with the default enable_thinking=false and
                # suppress native reasoning (see gemma-4-31B-it note above).
                use_chat_api=True,
                model_kwargs={
                    "max_length": 131072,
                },
                # Thinking-mode sampling (Qwen3.6 page, general tasks):
                # temperature=1.0, top_p=0.95, top_k=20.
                # stream=false is REQUIRED: lm-eval's local-chat-completions
                # streaming parser raises KeyError 'message' on every response.
                gen_kwargs={
                    "stream": "false",
                    "max_gen_toks": 32 * 1024,
                    "until": [],
                    "do_sample": "true",
                    "temperature": 1.0,
                    "top_k": 20,
                    "top_p": 0.95,
                },
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 0.2,
                    EvalLimitMode.SMOKE_TEST: 0.01,
                },
            ),
            EvalTask(
                task_name="terminal_bench_2",
                workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC,
                score=EvalTaskScore(
                    published_score=None,
                    published_score_ref="https://huggingface.co/Qwen/Qwen3.6-27B",
                    gpu_reference_score=None,
                    gpu_reference_score_ref="TBD",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": ["accuracy"],
                        "unit": "percent",
                    },
                ),
                agentic_eval_config=TerminalBenchEvalConfig(
                    dataset="terminal-bench/terminal-bench-2",
                    agent="terminus-2",
                    n_concurrent_trials=5,
                    n_attempts=1,
                    n_tasks=None,
                    override_cpus=32,
                    override_memory_mb=48 * 1024,
                    agent_timeout_sec=3 * 60 * 60,
                    agent_kwargs={
                        "parser_name": "json",
                        "temperature": 1.0,
                        "model_info": {
                            "max_input_tokens": 96 * 1024,
                            "max_output_tokens": 32 * 1024,
                        },
                        "llm_kwargs": {
                            "top_p": 0.95,
                            "max_tokens": 32 * 1024,
                            "timeout": 60 * 60,
                            "extra_body": {
                                "top_k": 20,
                            },
                        },
                    },
                    task_names_map={
                        EvalLimitMode.CI_NIGHTLY: [
                            "terminal-bench/caffe-cifar-10",
                            "terminal-bench/password-recovery",
                            "terminal-bench/portfolio-optimization",
                            "terminal-bench/hf-model-inference",
                            "terminal-bench/financial-document-processor",
                        ],
                    },
                ),
                limit_samples_map={
                    EvalLimitMode.SMOKE_TEST: 5,
                },
            ),
            EvalTask(
                task_name="swe_bench_verified",
                workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC,
                score=EvalTaskScore(
                    published_score=None,
                    published_score_ref="https://huggingface.co/Qwen/Qwen3.6-27B",
                    gpu_reference_score=None,
                    gpu_reference_score_ref="TBD",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": ["accuracy"],
                        "unit": "percent",
                    },
                ),
                swebench_eval_config=SWEbenchEvalConfig(
                    dataset_name="SWE-bench/SWE-bench_Verified",
                    sweagent_subset="verified",
                    dataset_split="test",
                    agent_backend="mini-swe-agent",
                    n_concurrent_trials=5,
                    max_workers=8,
                    n_tasks=None,
                    temperature=1.0,
                    top_p=0.95,
                    # Clamped so max_input + max_output (96K + 32K = 128K) fits gemma-4's 131072 ctx.
                    max_input_tokens=96 * 1024,
                    max_output_tokens=32 * 1024,
                    completion_kwargs={
                        "extra_body": {
                            "top_k": 20,
                        },
                    },
                    instance_ids_map={
                        EvalLimitMode.CI_NIGHTLY: [
                            "django__django-11299",
                            "astropy__astropy-14096",
                            "matplotlib__matplotlib-25332",
                            "sympy__sympy-13551",
                            "scikit-learn__scikit-learn-14629",
                        ],
                    },
                ),
                limit_samples_map={
                    EvalLimitMode.SMOKE_TEST: 5,
                },
            ),
        ],
    ),
    # =========================================================================
    # Devstral-2-123B (forge / tt-xla, BH Galaxy 4x8 DP+TP).
    # Devstral 2 is an agentic *coding* model (256k native context, FP8). Its
    # published benchmarks are all agentic SWE/terminal tasks -- there is no
    # published GPQA/MMLU/AIME score -- so the eval suite mirrors those exactly:
    # swe_bench_verified, swe_bench (multilingual), and terminal_bench_2, modeled
    # on the gemma-4-31B agentic block above.
    #
    # published_score values are from the HF model card benchmark table
    # (https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512).
    # gpu_reference_score is left None (a supported "TBD" state -- run_reports
    # skips the ratio check when it is falsy) -- TODO(CSE): fill from a
    # bring-your-own vLLM reference run, as was done for gemma-4-31B.
    #
    # Token budgets mirror gemma (same 256k native context). These agentic evals
    # send ~192k-token requests, so min_context_required guards them: at the
    # 128-ctx first-light spec (workflows/model_specs/dev/cnn.yaml) they SKIP
    # cleanly (llm_eval_tests.py honors min_context_required) and start running
    # automatically once max_context is raised toward the model max.
    # Sampling follows the model card (temperature=0.15); Devstral is not a
    # thinking model, so no top_k / enable_thinking (unlike gemma/Qwen3.6).
    # =========================================================================
    EvalConfig(
        hf_model_repo="mistralai/Devstral-2-123B-Instruct-2512",
        tasks=[
            EvalTask(
                task_name="swe_bench_verified",
                workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC,
                min_context_required=192 * 1024,
                score=EvalTaskScore(
                    published_score=72.2,
                    published_score_ref="https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512 (SWE-bench Verified)",
                    gpu_reference_score=None,  # TODO(CSE): bring-your-own vLLM reference run
                    gpu_reference_score_ref="TBD -- TODO(CSE): bring-your-own vLLM reference run",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": ["accuracy"],
                        "unit": "percent",
                    },
                ),
                swebench_eval_config=SWEbenchEvalConfig(
                    dataset_name="SWE-bench/SWE-bench_Verified",
                    sweagent_subset="verified",
                    dataset_split="test",
                    agent_backend="mini-swe-agent",
                    n_concurrent_trials=5,
                    max_workers=8,
                    n_tasks=None,  # full dataset (500)
                    temperature=0.15,  # model card recommendation
                    top_p=0.95,
                    # Devstral native ctx is 256k; keep max_input + max_output
                    # under the served max-model-len. 160K + 32K = 192K.
                    max_input_tokens=160 * 1024,
                    max_output_tokens=32 * 1024,
                    instance_ids_map={
                        # Generic SWE-bench Verified instances (model-agnostic),
                        # same nightly subset used by the gemma-4-31B block.
                        EvalLimitMode.CI_NIGHTLY: [
                            "django__django-11299",
                            "astropy__astropy-14096",
                            "matplotlib__matplotlib-25332",
                            "sympy__sympy-13551",
                            "scikit-learn__scikit-learn-14629",
                        ],
                    },
                ),
                limit_samples_map={
                    EvalLimitMode.SMOKE_TEST: 5,
                },
            ),
            EvalTask(
                task_name="swe_bench_multilingual",
                workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC,
                min_context_required=192 * 1024,
                score=EvalTaskScore(
                    published_score=61.3,
                    published_score_ref="https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512 (SWE-bench Multilingual)",
                    gpu_reference_score=None,  # TODO(CSE): bring-your-own vLLM reference run
                    gpu_reference_score_ref="TBD -- TODO(CSE): bring-your-own vLLM reference run",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": ["accuracy"],
                        "unit": "percent",
                    },
                ),
                swebench_eval_config=SWEbenchEvalConfig(
                    # SWE-bench Multilingual reuses the SWEbench runner with the
                    # multilingual dataset + mini-swe-agent "multilingual" subset.
                    dataset_name="SWE-bench/SWE-bench_Multilingual",
                    sweagent_subset="multilingual",
                    dataset_split="test",
                    agent_backend="mini-swe-agent",
                    n_concurrent_trials=5,
                    max_workers=8,
                    n_tasks=None,  # full dataset
                    temperature=0.15,  # model card recommendation
                    top_p=0.95,
                    max_input_tokens=160 * 1024,
                    max_output_tokens=32 * 1024,
                    # No pinned nightly instance_ids (verified IDs don't exist in
                    # the multilingual set); CI_NIGHTLY slices via limit below.
                    # TODO(CSE): pin real multilingual instance_ids for a
                    # deterministic nightly subset.
                ),
                limit_samples_map={
                    EvalLimitMode.CI_NIGHTLY: 5,
                    EvalLimitMode.SMOKE_TEST: 5,
                },
            ),
            EvalTask(
                task_name="terminal_bench_2",
                workflow_venv_type=WorkflowVenvType.EVALS_AGENTIC,
                min_context_required=192 * 1024,
                score=EvalTaskScore(
                    published_score=32.6,
                    published_score_ref="https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512 (Terminal Bench 2)",
                    gpu_reference_score=None,  # TODO(CSE): bring-your-own vLLM reference run
                    gpu_reference_score_ref="TBD -- TODO(CSE): bring-your-own vLLM reference run",
                    score_func=score_task_single_key,
                    score_func_kwargs={
                        "result_keys": ["accuracy"],
                        "unit": "percent",
                    },
                ),
                agentic_eval_config=TerminalBenchEvalConfig(
                    dataset="terminal-bench/terminal-bench-2",
                    agent="terminus-2",
                    n_concurrent_trials=5,
                    n_attempts=1,
                    n_tasks=None,  # full dataset (89)
                    override_cpus=32,
                    override_memory_mb=48 * 1024,
                    agent_timeout_sec=3 * 60 * 60,
                    agent_kwargs={
                        "parser_name": "json",
                        "temperature": 0.15,  # model card recommendation
                        "model_info": {
                            # 112K + 80K = 192K, under the served max-model-len.
                            "max_input_tokens": 112 * 1024,
                            "max_output_tokens": 80 * 1024,
                        },
                        "llm_kwargs": {
                            "top_p": 0.95,
                            "max_tokens": 80 * 1024,
                            "timeout": 60 * 60,
                        },
                    },
                    task_names_map={
                        # Generic terminal-bench-2 tasks (model-agnostic), same
                        # nightly subset used by the gemma-4-31B block.
                        EvalLimitMode.CI_NIGHTLY: [
                            "terminal-bench/caffe-cifar-10",
                            "terminal-bench/password-recovery",
                            "terminal-bench/portfolio-optimization",
                            "terminal-bench/hf-model-inference",
                            "terminal-bench/financial-document-processor",
                        ],
                    },
                ),
                limit_samples_map={
                    EvalLimitMode.SMOKE_TEST: 5,
                },
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
