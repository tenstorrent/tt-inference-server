# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

"""Engine-owned workflow types.

These enums are generic validation-framework concepts (workflow kinds, model
modalities, check/status classifications, limit modes). They live in the
engine so engine packages never import from the host/adapter layer
(``workflows/``). Tenstorrent-specific taxonomies (``DeviceTypes``,
``InferenceEngine``, ``WorkflowVenvType``, ...) remain in
``workflows/workflow_types.py``, which re-exports everything defined here
for backward compatibility.
"""

from enum import IntEnum, auto
from typing import List, Optional


class WorkflowType(IntEnum):
    BENCHMARKS = auto()
    EVALS = auto()
    STRESS_TESTS = auto()
    SERVER = auto()
    RELEASE = auto()
    SPEC_TESTS = auto()
    AGENTIC = auto()
    AGENTIC_TRACES = auto()
    SERVING_BENCH = auto()
    PREFILL_DECODE = auto()
    TRAINING_TESTS = auto()

    @classmethod
    def from_string(cls, name: str):
        try:
            return cls[name.upper()]
        except KeyError:
            raise ValueError(f"Invalid TaskType: {name}")


class BenchmarkTaskType(IntEnum):
    HTTP_CLIENT_VLLM_API = auto()
    HTTP_CLIENT_CNN_API = auto()
    HTTP_CLIENT_VLLM_STRUCTURED_OUTPUT_API = auto()


class ReportCheckTypes(IntEnum):
    NA = auto()
    PASS = auto()
    FAIL = auto()

    @classmethod
    def from_result(cls, result: bool):
        res_map = {
            None: ReportCheckTypes.NA,
            True: ReportCheckTypes.PASS,
            False: ReportCheckTypes.FAIL,
        }
        return res_map[result]

    @classmethod
    def to_display_string(cls, check_type: str):
        disp_map = {
            ReportCheckTypes.NA: "N/A",
            ReportCheckTypes.PASS: "PASS ✅",
            ReportCheckTypes.FAIL: "FAIL ⛔",
        }
        return disp_map[check_type]


class ModelStatusTypes(IntEnum):
    """
    EXPERIMENTAL: Model implementation is available, but is unstable or has performance issues.
    FUNCTIONAL: Model runs functionally without issue, but performance is lower than expected.
    COMPLETE: Operationally complete, performance is usable for most applications.
    TOP_PERF: Performance close to theoretical peak, nearly fully optimized.
    """

    EXPERIMENTAL = auto()
    FUNCTIONAL = auto()
    COMPLETE = auto()
    TOP_PERF = auto()

    @property
    def display_string(self) -> str:
        return {
            ModelStatusTypes.EXPERIMENTAL: "🛠️ Experimental",
            ModelStatusTypes.FUNCTIONAL: "🟡 Functional",
            ModelStatusTypes.COMPLETE: "🟢 Complete",
            ModelStatusTypes.TOP_PERF: "🚀 Top Performance",
        }[self]

    @property
    def required_target_tiers(self) -> List[str]:
        """Tiers that MUST pass for a model at this status level.

        Tiers not in this list are still computed and reported but
        treated as informational -- failures are accepted and do not
        block a release. This enables programmatic masking: e.g. an
        EXPERIMENTAL model (forge, new bring-up) can fail every
        performance benchmark and still be released.
        """
        tier_map = {
            ModelStatusTypes.EXPERIMENTAL: [],
            ModelStatusTypes.FUNCTIONAL: ["functional"],
            ModelStatusTypes.COMPLETE: ["functional", "complete"],
            ModelStatusTypes.TOP_PERF: ["functional", "complete", "target"],
        }
        return tier_map[self]

    @property
    def evals_enforced(self) -> bool:
        """Whether eval accuracy failures block acceptance at this status.

        Reuses required_target_tiers' signal (empty only for EXPERIMENTAL) so
        a model still in bring-up isn't blocked on eval accuracy either.
        """
        return bool(self.required_target_tiers)

    @classmethod
    def resolve(cls, name: Optional[str]) -> Optional["ModelStatusTypes"]:
        """Best-effort ``name`` -> member lookup, or ``None`` if missing/unrecognized.

        Unlike :meth:`EvalLimitMode.from_string`, this never raises: callers
        use a missing/garbled status as a signal to fall back to the
        strictest (fully-enforced) behavior rather than crash.
        """
        if not name:
            return None
        try:
            return cls[name]
        except KeyError:
            return None


class EvalLimitMode(IntEnum):
    SMOKE_TEST = auto()
    CI_COMMIT = auto()
    CI_NIGHTLY = auto()
    CI_LONG = auto()

    @classmethod
    def from_string(cls, name: str):
        if name is None:
            return None
        try:
            return cls[name.upper().replace("-", "_")]
        except KeyError:
            raise ValueError(f"Invalid EvalLimitMode: {name}")


class AgenticTracesMode(IntEnum):
    """Duration profile for the ``agentic_traces`` workflow.

    Deliberately separate from :class:`EvalLimitMode`: agentic trace replay is
    bounded by wall-clock profiling time rather than a dataset sample count, and
    the InferenceX scenario enforces its own duration floor (see
    ``AGENTIC_TRACES_MIN_PROFILE_SECONDS``), so the eval limit modes do not
    translate.

    ``FULL`` is the reference run used for reportable numbers; ``CI`` is the
    shortest run the scenario still permits.
    """

    FULL = auto()
    CI = auto()

    @classmethod
    def from_string(cls, name: str):
        if name is None:
            return None
        try:
            return cls[name.upper().replace("-", "_")]
        except KeyError:
            raise ValueError(f"Invalid AgenticTracesMode: {name}")

    def to_string(self) -> str:
        return self.name.lower()


class VersionMode(IntEnum):
    """Defines the enforcement mode for a version requirement."""

    STRICT = auto()  # Requirement must be met, raises an error otherwise.
    SUGGESTED = auto()  # A warning is issued if the requirement is not met.


class ModelType(IntEnum):
    LLM = auto()
    CNN = auto()
    AUDIO = auto()
    IMAGE = auto()
    EMBEDDING = auto()
    TEXT_TO_SPEECH = auto()
    VIDEO = auto()
    VLM = auto()  # Vision-Language Models (text+image-to-text)
    TRAINING = auto()

    @property
    def display_name(self) -> str:
        display_names = {
            ModelType.LLM: "Large Language Model",
            ModelType.CNN: "Convolutional Neural Network",
            ModelType.AUDIO: "Audio",
            ModelType.IMAGE: "Image",
            ModelType.EMBEDDING: "Embedding",
            ModelType.TEXT_TO_SPEECH: "Text-to-Speech",
            ModelType.VIDEO: "Video",
            ModelType.VLM: "Vision-Language Model",
            ModelType.TRAINING: "Training",
        }
        return display_names[self]

    @property
    def short_name(self) -> str:
        short_names = {
            ModelType.LLM: "LLM",
            ModelType.VLM: "VLM",
            ModelType.AUDIO: "Audio",
            ModelType.IMAGE: "Image",
            ModelType.CNN: "CNN",
            ModelType.EMBEDDING: "Embedding",
            ModelType.TEXT_TO_SPEECH: "TTS",
            ModelType.VIDEO: "Video",
            ModelType.TRAINING: "Training",
        }
        return short_names[self]

    @property
    def task_type(self) -> str:
        task_types = {
            ModelType.LLM: "text",
            ModelType.VLM: "vlm",
            ModelType.AUDIO: "asr",  # Automatic Speech Recognition
            ModelType.IMAGE: "image",
            ModelType.CNN: "cnn",
            ModelType.EMBEDDING: "embedding",
            ModelType.TEXT_TO_SPEECH: "tts",
            ModelType.VIDEO: "video",
            ModelType.TRAINING: "training",
        }
        return task_types[self]


class WorkflowVenvType(IntEnum):
    """Named tool-environment keys.

    The *keys* are engine-generic (a standalone framework needs named tool
    environments); the *content* — what each environment installs and where
    it lives — is adapter business behind
    :class:`workflow_module.venv_provisioner.VenvProvisioner`.
    """

    SYSTEM_SOFTWARE_VALIDATION = auto()
    STRESS_TESTS_RUN_SCRIPT = auto()
    STRESS_TESTS = auto()
    EVALS_RUN_SCRIPT = auto()
    TESTS_RUN_SCRIPT = auto()
    BENCHMARKS_RUN_SCRIPT = auto()
    REPORTS_RUN_SCRIPT = auto()
    WORKFLOW_RUN_SCRIPT = auto()
    PREFIX_CACHE = auto()
    AGENTIC_TRACES = auto()
    LLM_VLLM = auto()
    LLM_GUIDELLM = auto()
    LLM_AIPERF = auto()
    SPEC_DECODE = auto()
    EVALS_COMMON = auto()
    EVALS_META = auto()
    EVALS_VISION = auto()
    EVALS_AUDIO = auto()
    EVALS_EMBEDDING = auto()
    EVALS_AGENTIC = auto()
    BENCHMARKS_VLLM = auto()
    BENCHMARKS_VLLM_FORGE = auto()
    BENCHMARKS_GENAI_PERF = auto()
    HF_SETUP = auto()
    SERVER = auto()
    TT_SMI = auto()
    TT_TOPOLOGY = auto()
