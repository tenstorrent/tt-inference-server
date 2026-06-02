# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Tuple

from workflows.bootstrap_uv import UV_EXEC
from workflows.utils import (
    get_repo_root_path,
    map_configs_by_attr,
    run_command,
)
from workflows.workflow_types import ModelType, WorkflowVenvType

if TYPE_CHECKING:
    from workflows.model_spec import ModelSpec

logger = logging.getLogger("run_log")


# Parent directory for every workflow venv; `uv venv` creates it on first use.
default_venv_path = get_repo_root_path() / ".workflow_venvs"

# Per-venv pip lists live under <repo_root>/requirements/, sharing constraints.txt.
REQUIREMENTS_DIR = get_repo_root_path() / "requirements"


def install_requirements(
    venv_config: "VenvConfig",  # noqa: F821
    requirements_file: str,
) -> bool:
    """Install pip deps from requirements/<requirements_file> into the venv.

    Always passes ``--index-strategy unsafe-best-match`` so per-file
    ``--extra-index-url`` directives resolve against all configured indexes
    (e.g. PyPI + the PyTorch CPU index).
    """
    requirements_path = REQUIREMENTS_DIR / requirements_file
    if not requirements_path.is_file():
        raise FileNotFoundError(
            f"Requirements file not found: {requirements_path}. "
            f"Expected one of the per-venv files under {REQUIREMENTS_DIR}."
        )
    return_code = run_command(
        f"{UV_EXEC} pip install --managed-python "
        f"--python {venv_config.venv_python} "
        f"--index-strategy unsafe-best-match "
        f"-r {requirements_path}",
        logger=logger,
    )
    return return_code == 0


@dataclass(frozen=True)
class VenvConfig:
    """Declarative description of a workflow virtual environment.

    ``setup()`` runs in fixed order: ``uv venv`` → mkdir ``extra_dirs`` →
    ``install_requirements(requirements_file)`` → ``setup_function`` hook.
    """

    venv_type: WorkflowVenvType
    requirements_file: Optional[str] = None
    extra_dirs: Tuple[str, ...] = field(default_factory=tuple)
    setup_function: Optional[Callable[["VenvConfig", "ModelSpec"], bool]] = None
    name: Optional[str] = None
    python_version: Optional[str] = "3.10"
    venv_path: Optional[Path] = None
    venv_python: Optional[Path] = None
    venv_pip: Optional[Path] = None

    def __post_init__(self):
        self.validate_data()
        self._infer_data()

    def validate_data(self):
        pass

    def _infer_data(self):
        if self.name is None:
            object.__setattr__(self, "name", self.venv_type.name.lower())

        if self.venv_path is None:
            object.__setattr__(
                self, "venv_path", default_venv_path / f".venv_{self.name}"
            )

        if self.venv_python is None:
            object.__setattr__(self, "venv_python", self.venv_path / "bin" / "python")

        if self.venv_pip is None:
            object.__setattr__(self, "venv_pip", self.venv_path / "bin" / "pip")

    def setup(self, model_spec: "ModelSpec") -> bool:
        """Create the venv (if missing) and install/configure it.

        Raises ``RuntimeError`` if any step fails.
        """
        if not self.venv_path.exists():
            # https://docs.astral.sh/uv/reference/cli/#uv-venv
            run_command(
                f"{str(UV_EXEC)} venv --managed-python --python={self.python_version} {self.venv_path} --allow-existing",
                logger=logger,
                check=True,
            )

        for sub_dir in self.extra_dirs:
            target = self.venv_path / sub_dir
            if target.exists():
                logger.info(f"sub-dir already exists for {self.name}: {target}")
            else:
                logger.info(f"creating sub-dir for {self.name}: {target}")
                target.mkdir(parents=True, exist_ok=True)

        if self.requirements_file is not None:
            if not install_requirements(self, self.requirements_file):
                raise RuntimeError(
                    f"Failed to install requirements for venv {self.venv_type.name} "
                    f"from {self.requirements_file}"
                )

        if self.setup_function is not None:
            if not self.setup_function(self, model_spec=model_spec):
                raise RuntimeError(f"Failed to setup venv: {self.venv_type.name}")

        return True


def setup_evals_agentic(
    venv_config: VenvConfig,
    model_spec: "ModelSpec",  # noqa: F821
) -> bool:
    """Hook for EVALS_AGENTIC: clone SWE-agent and install it as editable.

    Other deps (harbor, mini-swe-agent, epoch SWE-bench) are in requirements/evals-agentic.txt.
    """
    sweagent_dir = venv_config.venv_path / "SWE-agent"
    if not sweagent_dir.exists():
        clone_return_code = run_command(
            f"git clone https://github.com/SWE-agent/SWE-agent.git {sweagent_dir}",
            logger=logger,
        )
        if clone_return_code != 0:
            return False

    return_code = run_command(
        f"{UV_EXEC} pip install --managed-python --python {venv_config.venv_python} "
        f"-e {sweagent_dir}",
        logger=logger,
    )
    return return_code == 0


def check_docker_available(
    venv_config: VenvConfig,
    model_spec: "ModelSpec",
) -> bool:
    """Hook for BENCHMARKS_GENAI_PERF: assert ``docker --version`` succeeds."""
    run_command("docker --version", logger=logger, check=True)
    return True


def setup_evals_meta(
    venv_config: VenvConfig,
    model_spec: "ModelSpec",
) -> bool:
    """Hook for EVALS_META: clone llama-cookbook (LLM only) and prep datasets.

    Non-LLM model types reuse this venv only for ``work_dir`` placement.
    """
    if (
        model_spec.model_type == ModelType.AUDIO
        or model_spec.model_type == ModelType.CNN
        or model_spec.model_type == ModelType.IMAGE
        or model_spec.model_type == ModelType.EMBEDDING
        or model_spec.model_type == ModelType.TEXT_TO_SPEECH
    ):
        return True

    setup_succeeded = True
    cookbook_dir = venv_config.venv_path / "llama-cookbook"
    original_dir = os.getcwd()
    if cookbook_dir.is_dir():
        logger.info(f"The directory {cookbook_dir} exists.")
    else:
        logger.info(f"The directory {cookbook_dir} does not exist. Setting up ...")
        clone_cmd = (
            f"git clone https://github.com/meta-llama/llama-cookbook.git {cookbook_dir}"
        )
        setup_succeeded = run_command(clone_cmd, logger=logger) == 0 and setup_succeeded
        # cookbook editable install needs modern setuptools
        setup_succeeded = (
            run_command(
                f"{UV_EXEC} pip install --managed-python --python {venv_config.venv_python} -U pip setuptools",
                logger=logger,
            )
            == 0
            and setup_succeeded
        )
        # editable install is cwd-dependent, so it can't live in a requirements file
        os.chdir(cookbook_dir)
        setup_succeeded = (
            run_command(
                f"{UV_EXEC} pip install --managed-python --python {venv_config.venv_python} -e .",
                logger=logger,
            )
            == 0
            and setup_succeeded
        )
        logger.warning("this might take 5 to 15+ minutes to install on first run ...")
        setup_succeeded = (
            install_requirements(venv_config, "evals-meta.txt") and setup_succeeded
        )
    meta_eval_dir = (
        cookbook_dir
        / "end-to-end-use-cases"
        / "benchmarks"
        / "llm_eval_harness"
        / "meta_eval"
    )
    meta_eval_data_dir = meta_eval_dir / f"work_dir_{model_spec.model_name}"
    if not meta_eval_data_dir.exists():
        # PyYAML is only needed by this meta-eval setup hook, not by every
        # caller of ``workflow_venvs``.
        import yaml

        logger.info(f"preparing meta eval datasets for: {meta_eval_data_dir}")
        os.chdir(meta_eval_dir)
        yaml_path = meta_eval_dir / "eval_config.yaml"
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)

        # 3.3 reuses 3.1 evals; vision variants fall back to 3.2-3B
        _model_name = model_spec.hf_model_repo
        if _model_name == "meta-llama/Llama-3.2-11B-Vision-Instruct":
            _model_name = _model_name.replace("-3.2-11B-Vision-", "-3.2-3B-")
        elif _model_name == "meta-llama/Llama-3.2-90B-Vision-Instruct":
            _model_name = _model_name.replace("-3.2-90B-Vision-", "-3.2-3B-")
        _model_name = _model_name.replace("-3.3-", "-3.1-")
        logger.info(f"model_name: {_model_name}")

        config["work_dir"] = str(meta_eval_data_dir)
        config["model_name"] = _model_name
        config["evals_dataset"] = f"{_model_name}-evals"

        with open(yaml_path, "w") as f:
            yaml.safe_dump(config, f)

        # requires HF AUTH
        return_code = run_command(
            f"{venv_config.venv_python} prepare_meta_eval.py --config_path ./eval_config.yaml",
            logger=logger,
        )
        if return_code != 0:
            logger.warning(
                f"Failed to prepare meta eval datasets for: {meta_eval_data_dir}, continuing..."
            )
    # The model-specific data lives at meta_eval_data_dir (work_dir_<model_name>/).
    # IFEval (and likely others) hard-code ./work_dir relative to lm-eval's cwd,
    # so run_evals.py creates a per-PID staging dir with a 'work_dir' symlink
    # pointing here at command-build time. We do NOT write to a shared
    # .venv_evals_meta/work_dir/ here — that previously raced across parallel
    # model invocations and produced spurious FileNotFoundError for tasks (e.g.
    # meta_ifeval) when a sibling model's data overwrote the shared dir.
    os.chdir(original_dir)
    return setup_succeeded


# Pinned vLLM tag used both by `requirements/benchmarks-vllm.txt` (where
# vllm==<VLLM_PIN_VERSION>) and as the source for the structured-output
# benchmark scripts fetched at venv setup time. Keep these in sync.
VLLM_PIN_VERSION = "0.13.0"
VLLM_BENCHMARKS_RAW_BASE = f"https://raw.githubusercontent.com/vllm-project/vllm/v{VLLM_PIN_VERSION}/benchmarks"

# (relative_path_in_vllm_repo, relative_path_in_work_dir)
STRUCTURED_OUTPUT_FETCH_FILES = (
    (
        "benchmark_serving_structured_output.py",
        "benchmark_serving_structured_output.py",
    ),
    ("backend_request_func.py", "backend_request_func.py"),
    (
        "structured_schemas/structured_schema_1.json",
        "structured_schemas/structured_schema_1.json",
    ),
)

# Filename of the structured-output benchmark script downloaded into the
# BENCHMARKS_VLLM venv work_dir by fetch_structured_output_scripts().
# Imported by benchmarking/run_benchmarks.py to locate the script at run time.
STRUCTURED_OUTPUT_SCRIPT_NAME = "benchmark_serving_structured_output.py"


def fetch_structured_output_scripts(
    venv_config: "VenvConfig",
    model_spec: "ModelSpec",
) -> bool:
    """Hook for BENCHMARKS_VLLM: download structured-output benchmark scripts.

    Pip-installable deps (vllm/torch/datasets/pandas/xgrammar) come from
    requirements/benchmarks-vllm.txt. The benchmark driver scripts are not
    published on PyPI, so they're fetched at venv setup time from the pinned
    vLLM source tag rather than vendored into this repo.
    """
    logger.info("running fetch_structured_output_scripts() ...")
    work_dir = venv_config.venv_path / "work_dir"
    for src_rel, dst_rel in STRUCTURED_OUTPUT_FETCH_FILES:
        dst = work_dir / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        url = f"{VLLM_BENCHMARKS_RAW_BASE}/{src_rel}"
        return_code = run_command(
            f"curl -fSL --retry 3 --retry-delay 5 --retry-connrefused {url} -o {dst}",
            logger=logger,
        )
        if return_code != 0:
            return False
    return True


_venv_config_list = [
    # Pure pip install
    VenvConfig(
        venv_type=WorkflowVenvType.EVALS_RUN_SCRIPT,
        requirements_file="evals-run-script.txt",
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.STRESS_TESTS_RUN_SCRIPT,
        requirements_file="stress-tests-run-script.txt",
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.STRESS_TESTS,
        requirements_file="stress-tests-run-script.txt",
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.BENCHMARKS_RUN_SCRIPT,
        requirements_file="benchmarks-run-script.txt",
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.TESTS_RUN_SCRIPT,
        requirements_file="tests-run-script.txt",
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.EVALS_COMMON,
        requirements_file="evals-common.txt",
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.EVALS_VISION,
        requirements_file="evals-vision.txt",
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.EVALS_AUDIO,
        requirements_file="evals-audio.txt",
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.EVALS_EMBEDDING,
        requirements_file="evals-embedding.txt",
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.REPORTS_RUN_SCRIPT,
        requirements_file="reports-run-script.txt",
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.EVALS_AGENTIC,
        requirements_file="evals-agentic.txt",
        python_version="3.12",
        setup_function=setup_evals_agentic,
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.V2_RUN_SCRIPT,
        requirements_file="v2-run-script.txt",
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.V2_PREFIX_CACHE,
        requirements_file="v2-prefix-cache.txt",
        extra_dirs=("artifacts",),
        python_version="3.11",
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.HF_SETUP,
        requirements_file="hf-setup.txt",
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.SYSTEM_SOFTWARE_VALIDATION,
        requirements_file="system-software-validation.txt",
        python_version="3.11",
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.TT_SMI,
        requirements_file="tt-smi.txt",
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.TT_TOPOLOGY,
        requirements_file="tt-topology.txt",
    ),
    # Pip install + sub-directory
    VenvConfig(
        venv_type=WorkflowVenvType.EVALS_VIDEO,
        requirements_file="evals-video.txt",
        extra_dirs=("work_dir",),
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.BENCHMARKS_VLLM,
        requirements_file="benchmarks-vllm.txt",
        extra_dirs=("work_dir",),
        python_version="3.11",
        setup_function=fetch_structured_output_scripts,
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.BENCHMARKS_AIPERF,
        requirements_file="benchmarks-aiperf.txt",
        extra_dirs=("artifacts",),
        python_version="3.11",
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.BENCHMARKS_GUIDELLM,
        requirements_file="benchmarks-guidellm.txt",
        python_version="3.11",
    ),
    # No pip; directory and/or runtime check
    VenvConfig(
        venv_type=WorkflowVenvType.BENCHMARKS_VIDEO,
        extra_dirs=("work_dir",),
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.BENCHMARKS_GENAI_PERF,
        extra_dirs=("artifacts",),
        setup_function=check_docker_available,
    ),
    # Custom Python work; pip handled inside the hook (model-type dependent).
    # No extra_dirs — `run_evals.py` materializes a per-invocation staging
    # dir at command-build time (see EVALS_META branch in build_eval_command).
    VenvConfig(
        venv_type=WorkflowVenvType.EVALS_META,
        setup_function=setup_evals_meta,
    ),
]

VENV_CONFIGS = map_configs_by_attr(config_list=_venv_config_list, attr="venv_type")
