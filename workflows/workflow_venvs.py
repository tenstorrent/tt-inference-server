# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import yaml

from workflows.bootstrap_uv import UV_EXEC
from workflows.model_spec import ModelType
from workflows.utils import (
    ensure_readwriteable_dir,
    get_repo_root_path,
    map_configs_by_attr,
    run_command,
)
from workflows.workflow_types import WorkflowVenvType

logger = logging.getLogger("run_log")


def default_setup(venv_config: "VenvConfig", model_spec: "ModelSpec") -> bool:  # noqa: F821
    return True


default_venv_path = get_repo_root_path() / ".workflow_venvs"
ensure_readwriteable_dir(default_venv_path)


@dataclass(frozen=True)
class VenvConfig:
    venv_type: WorkflowVenvType
    setup_function: Callable[["VenvConfig"], None] = default_setup  # noqa: F821
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

    def setup(self, model_spec: "ModelSpec") -> bool:  # noqa: F821
        """Run the setup using the instance's provided setup_function."""
        # setup venv using uv if not exists
        if not self.venv_path.exists():
            # uv venv: https://docs.astral.sh/uv/reference/cli/#uv-venv
            # --python: set the python interpreter version in venv
            # --allow-existing: if venv exists, check if it has correct package versions
            # --seed: Install seed packages (one or more of: pip, setuptools, and wheel)
            # --managed-python: explicitly use uv managed python versions
            run_command(
                f"{str(UV_EXEC)} venv --managed-python --python={self.python_version} {self.venv_path} --allow-existing",
                logger=logger,
                check=True,
            )
        # uv will verify deps if venv exists
        venv_setup_succeeded = self.setup_function(self, model_spec=model_spec)
        if not venv_setup_succeeded:
            raise RuntimeError(f"Failed to setup venv: {self.venv_type.name}")
        return venv_setup_succeeded


def setup_evals_common(
    venv_config: VenvConfig,
    model_spec: "ModelSpec",  # noqa: F821
) -> bool:
    logger.warning("this might take 5 to 15+ minutes to install on first run ...")
    return_code = run_command(
        f"{UV_EXEC} pip install --managed-python --python {venv_config.venv_python} "
        "--index-strategy unsafe-best-match "
        "--extra-index-url https://download.pytorch.org/whl/cpu "
        "git+https://github.com/tstescoTT/lm-evaluation-harness.git@evals-common#egg=lm-eval[api,ifeval,math,sentencepiece,r1_evals,ruler,longbench,hf] "
        "protobuf pillow==11.1 pyjwt==2.7.0 datasets==3.1.0",
        logger=logger,
    )
    setup_succeeded = return_code == 0
    return setup_succeeded


def setup_venv(venv_config: VenvConfig) -> bool:
    """Setup a generic virtual environment.
    Args:
        venv_config: Virtual environment configuration

    Returns:
        True if setup was successful
    """
    work_dir = venv_config.venv_path / "work_dir"
    if not work_dir.exists():
        logger.info(f"Creating work_dir for generic server testing: {work_dir}")
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        logger.info(f"work_dir already exists for generic server testing: {work_dir}")
    return True


def setup_evals_meta(
    venv_config: VenvConfig,
    model_spec: "ModelSpec",  # noqa: F821
) -> bool:
    if (
        model_spec.model_type == ModelType.AUDIO
        or model_spec.model_type == ModelType.CNN
        or model_spec.model_type == ModelType.IMAGE
        or model_spec.model_type == ModelType.EMBEDDING
        or model_spec.model_type == ModelType.TEXT_TO_SPEECH
    ):
        return setup_venv(venv_config)

    # Default: Llama-specific setup
    setup_succeeded = True
    cookbook_dir = venv_config.venv_path / "llama-cookbook"
    original_dir = os.getcwd()
    if cookbook_dir.is_dir():
        logger.info(f"The directory {cookbook_dir} exists.")
    else:
        logger.info(f"The directory {cookbook_dir} does not exist. Setting up ...")
        # Clone the repository
        clone_cmd = (
            f"git clone https://github.com/meta-llama/llama-cookbook.git {cookbook_dir}"
        )
        setup_succeeded = run_command(clone_cmd, logger=logger) == 0 and setup_succeeded
        # Upgrade pip and setuptools
        setup_succeeded = (
            run_command(
                f"{UV_EXEC} pip install --managed-python --python {venv_config.venv_python} -U pip setuptools",
                logger=logger,
            )
            == 0
            and setup_succeeded
        )
        # Install the package in editable mode
        os.chdir(cookbook_dir)
        setup_succeeded = (
            run_command(
                f"{UV_EXEC} pip install --managed-python --python {venv_config.venv_python} -e .",
                logger=logger,
            )
            == 0
            and setup_succeeded
        )
        # Install specific dependencies
        setup_succeeded = (
            run_command(
                f"{UV_EXEC} pip install --managed-python --python {venv_config.venv_python} -U antlr4_python3_runtime==4.11",
                logger=logger,
            )
            == 0
            and setup_succeeded
        )
        logger.warning("this might take 5 to 15+ minutes to install on first run ...")
        setup_succeeded = (
            run_command(
                f"{UV_EXEC} pip install --managed-python --python {venv_config.venv_python} "
                "--index-strategy unsafe-best-match "
                "--extra-index-url https://download.pytorch.org/whl/cpu "
                "lm-eval[math,ifeval,sentencepiece,vllm]==0.4.3 pyjwt==2.7.0 pillow==11.1 datasets==3.1.0",
                logger=logger,
            )
            == 0
            and setup_succeeded
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
        logger.info(f"preparing meta eval datasets for: {meta_eval_data_dir}")
        # Change directory to meta_eval and run the preparation script
        os.chdir(meta_eval_dir)
        # need to edit yaml file
        yaml_path = meta_eval_dir / "eval_config.yaml"
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)

        # handle 3.3 having the same evals as 3.1
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

        # Write the updated configuration back to the YAML file.
        with open(yaml_path, "w") as f:
            yaml.safe_dump(config, f)

        # this requires HF AUTH
        return_code = run_command(
            f"{venv_config.venv_python} prepare_meta_eval.py --config_path ./eval_config.yaml",
            logger=logger,
        )
        if return_code != 0:
            logger.warning(
                f"Failed to prepare meta eval datasets for: {meta_eval_data_dir}, continuing..."
            )
    # Note: likely a bug, some evals, e.g. IFEval always look for the default ./work_dir
    # to deal with this and make downstream simpler, hotswap dirs
    work_dir = venv_config.venv_path / "work_dir"
    logger.info(f"moving {str(meta_eval_data_dir)} to {str(work_dir)}")
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    shutil.copytree(meta_eval_data_dir, work_dir)
    os.chdir(original_dir)
    return setup_succeeded


def setup_benchmarks_vllm(
    venv_config: VenvConfig,
    model_spec: "ModelSpec",  # noqa: F821
) -> bool:
    logger.info("running setup_benchmarks_vllm() ...")
    work_dir = venv_config.venv_path / "work_dir"
    if not work_dir.exists():
        logger.info(f"Creating work_dir for generic server testing: {work_dir}")
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        logger.info(f"work_dir already exists for generic server testing: {work_dir}")
    # pin vllm==0.13.0 for reproducibility and potential regressions
    setup_succeeded = (
        run_command(
            f"{UV_EXEC} pip install --managed-python --python {venv_config.venv_python} -U pip vllm==0.13.0 torch",
            logger=logger,
        )
        == 0
    )

    return setup_succeeded


def setup_benchmarks_video(
    venv_config: VenvConfig,
    model_spec: "ModelSpec",  # noqa: F821
) -> bool:
    """Setup video benchmarking environment."""
    logger.info("running setup_benchmarks_video() ...")
    return setup_venv(venv_config)


def setup_evals_vision(
    venv_config: VenvConfig,
    model_spec: "ModelSpec",  # noqa: F821
) -> bool:
    # use https://github.com/tstescoTT/lm-evaluation-harness/tree/tstesco/add-local-multimodal
    # for local-mm-completions model
    logger.warning("this might take 5 to 15+ minutes to install on first run ...")
    setup_succeeded = (
        run_command(
            f"{UV_EXEC} pip install --managed-python --python {venv_config.venv_python} git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git@v0.4.1 pyjwt==2.7.0 pillow==11.1 qwen_vl_utils",
            logger=logger,
        )
        == 0
    )
    return setup_succeeded


def setup_evals_audio(
    venv_config: VenvConfig,
    model_spec: "ModelSpec",  # noqa: F821
) -> bool:
    """
    Setup audio evaluation environment on HOST using lmms-eval.
    Uses TT-specific fork with whisper_tt model support.
    """
    logger.warning(
        "Installing lmms-eval for audio - this might take 5 to 15+ minutes on first run ..."
    )
    setup_succeeded = (
        run_command(
            f"{UV_EXEC} pip install --managed-python --python {venv_config.venv_python} "
            f"'git+https://github.com/bgoelTT/lmms-eval.git@ben/samt/whisper-tt#egg=lmms-eval[audio]' "
            f"pyjwt==2.7.0 pillow==11.1",
            logger=logger,
        )
        == 0
    )
    return setup_succeeded


def setup_evals_embedding(
    venv_config: VenvConfig,
    model_spec: "ModelSpec",  # noqa: F821
) -> bool:
    logger.info("running setup_evals_embedding() ...")
    setup_succeeded = (
        run_command(
            f"{UV_EXEC} pip install --managed-python --python {venv_config.venv_python} 'mteb>=2.6.6' openai",
            logger=logger,
        )
        == 0
    )
    return setup_succeeded


def setup_evals_video(
    venv_config: VenvConfig,
    model_spec: "ModelSpec",  # noqa: F821
) -> bool:
    """
    Setup video evaluation environment on HOST.
    Video models need dependencies for CLIP scoring, video frame extraction, and server interaction.
    """
    logger.info("Installing dependencies for video evaluation...")
    setup_venv(venv_config)
    setup_succeeded = (
        run_command(
            command=f"{UV_EXEC} pip install --managed-python --python {venv_config.venv_python} --index-url https://download.pytorch.org/whl/cpu torch torchvision",
            logger=logger,
        )
        == 0
    )
    setup_succeeded = (
        run_command(
            command=f"{UV_EXEC} pip install --managed-python --python {venv_config.venv_python} requests datasets open-clip-torch pyjwt==2.7.0 pillow==11.1 imageio imageio-ffmpeg",
            logger=logger,
        )
        == 0
        and setup_succeeded
    )
    return setup_succeeded


def setup_stress_tests_run_script(
    venv_config: VenvConfig,
    model_spec: "ModelSpec",  # noqa: F821
) -> bool:
    logger.info("running setup_stress_tests_run_script() ...")
    setup_succeeded = (
        run_command(
            command=f"{UV_EXEC} pip install --python {venv_config.venv_python} --index-url https://download.pytorch.org/whl/cpu torch numpy",
            logger=logger,
        )
        == 0
    )
    setup_succeeded = (
        run_command(
            command=f"{UV_EXEC} pip install --python {venv_config.venv_python} requests transformers datasets pyjwt==2.7.0 pillow==11.1 aiohttp",
            logger=logger,
        )
        == 0
        and setup_succeeded
    )
    # Remove the redundant download section since we now use stress_tests/stress_tests_benchmarking_script.py
    # The old benchmark_serving.py downloads are no longer needed
    return setup_succeeded


def setup_evals_run_script(
    venv_config: VenvConfig,
    model_spec: "ModelSpec",  # noqa: F821
) -> bool:  # noqa: F821
    logger.info("running setup_evals_run_script() ...")
    setup_succeeded = (
        run_command(
            command=f"{UV_EXEC} pip install --managed-python --python {venv_config.venv_python} numpy scipy",
            logger=logger,
        )
        == 0
    )
    setup_succeeded = (
        run_command(
            command=f"{UV_EXEC} pip install --managed-python --python {venv_config.venv_python} --index-url https://download.pytorch.org/whl/cpu torch torchvision",
            logger=logger,
        )
        == 0
        and setup_succeeded
    )
    setup_succeeded = (
        run_command(
            command=f"{UV_EXEC} pip install --managed-python --python {venv_config.venv_python} requests transformers protobuf sentencepiece datasets open-clip-torch pyjwt==2.7.0 pillow==11.1 imageio imageio-ffmpeg",
            logger=logger,
        )
        == 0
        and setup_succeeded
    )
    setup_succeeded = (
        run_command(
            f"{UV_EXEC} pip install --managed-python --python {venv_config.venv_python} "
            f"--extra-index-url https://download.pytorch.org/whl/cpu "
            f"'mteb[openai]>=2.6.6' tiktoken openai",
            logger=logger,
        )
        == 0
        and setup_succeeded
    )
    return setup_succeeded


def setup_benchmarks_run_script(
    venv_config: VenvConfig,
    model_spec: "ModelSpec",  # noqa: F821
) -> bool:
    logger.info("running setup_benchmarks_run_script() ...")
    setup_succeeded = (
        run_command(
            command=f"{UV_EXEC} pip install --managed-python --python {venv_config.venv_python} numpy scipy",
            logger=logger,
        )
        == 0
    )
    setup_succeeded = (
        run_command(
            command=f"{UV_EXEC} pip install --managed-python --python {venv_config.venv_python} --index-url https://download.pytorch.org/whl/cpu torch torchvision",
            logger=logger,
        )
        == 0
        and setup_succeeded
    )
    setup_succeeded = (
        run_command(
            command=f"{UV_EXEC} pip install --managed-python --python {venv_config.venv_python} requests sentencepiece protobuf transformers datasets open-clip-torch pyjwt==2.7.0 pillow==11.1",
            logger=logger,
        )
        == 0
        and setup_succeeded
    )
    return setup_succeeded


def setup_reports_run_script(
    venv_config: VenvConfig,
    model_spec: "ModelSpec",  # noqa: F821
) -> bool:
    logger.info("running setup_reports_run_script() ...")
    setup_succeeded = (
        run_command(
            command=f"{UV_EXEC} pip install --managed-python --python {venv_config.venv_python} requests numpy",
            logger=logger,
        )
        == 0
    )
    return setup_succeeded


def setup_hf_setup(
    venv_config: VenvConfig,
    model_spec: "ModelSpec",  # noqa: F821
) -> bool:
    logger.info("running setup_hf_setup() ...")
    # Install a modern version that provides the 'hf' CLI entrypoint
    setup_succeeded = (
        run_command(
            command=f"{UV_EXEC} pip install --managed-python --python {venv_config.venv_python} 'huggingface_hub>=1.0.0'",
            logger=logger,
        )
        == 0
    )
    return setup_succeeded


def setup_benchmarks_genai_perf(
    venv_config: VenvConfig,
    model_spec: "ModelSpec",  # noqa: F821
) -> bool:
    """Setup for genai-perf benchmarks (Docker-based, minimal local setup)."""
    logger.info("running setup_benchmarks_genai_perf() ...")
    # Ensure Docker is available
    run_command("docker --version", logger=logger, check=True)
    # Create artifacts directory
    artifacts_dir = venv_config.venv_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return True


def setup_benchmarks_aiperf(
    venv_config: VenvConfig,
    model_spec: "ModelSpec",  # noqa: F821
) -> bool:
    """Setup for aiperf benchmarks (pip-based installation).

    AIPerf is a comprehensive benchmarking tool for generative AI models.
    Repository: https://github.com/ai-dynamo/aiperf
    """
    logger.info("running setup_benchmarks_aiperf() ...")

    # Install torch CPU for dependencies that require it (like prompt_client)
    setup_succeeded = (
        run_command(
            command=f"{UV_EXEC} pip install --managed-python --python {venv_config.venv_python} 'torch==2.4.0+cpu' --index-url https://download.pytorch.org/whl/cpu",
            logger=logger,
        )
        == 0
    )

    # Install aiperf from PyPI
    setup_succeeded = (
        run_command(
            command=f"{UV_EXEC} pip install --managed-python --python {venv_config.venv_python} aiperf",
            logger=logger,
        )
        == 0
        and setup_succeeded
    )

    # Install additional dependencies for tokenization and prompt client
    setup_succeeded = (
        run_command(
            command=f"{UV_EXEC} pip install --managed-python --python {venv_config.venv_python} transformers pyjwt requests datasets",
            logger=logger,
        )
        == 0
        and setup_succeeded
    )

    # Create artifacts directory for benchmark outputs
    artifacts_dir = venv_config.venv_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    return setup_succeeded


def setup_system_software_validation(
    venv_config: VenvConfig,
    model_spec: "ModelSpec",  # noqa: F821
) -> bool:
    logger.info("running setup_system_software_validation() ...")

    setup_succeeded = (
        run_command(
            command=f"{UV_EXEC} pip install --managed-python --python {venv_config.venv_python} packaging==25.0 PyYAML",
            logger=logger,
        )
        == 0
    )
    return setup_succeeded


def setup_tt_smi(
    venv_config: VenvConfig,
    model_spec: "ModelSpec",  # noqa: F821
) -> bool:
    logger.info("running setup_tt_smi() ...")
    setup_succeeded = (
        run_command(
            command=f"{UV_EXEC} pip install --managed-python --python {venv_config.venv_python} tt-smi==3.0.39",
            logger=logger,
        )
        == 0
    )
    return setup_succeeded


def setup_tt_topology(
    venv_config: VenvConfig,
    model_spec: "ModelSpec",  # noqa: F821
) -> bool:
    logger.info("running setup_tt_topology() ...")
    setup_succeeded = (
        run_command(
            command=f"{UV_EXEC} pip install --managed-python --python {venv_config.venv_python} tt-topology==1.2.16",
            logger=logger,
        )
        == 0
    )
    return setup_succeeded


def setup_tests_run_script(
    venv_config: VenvConfig,
    model_spec: "ModelSpec",  # noqa: F821
) -> bool:
    logger.info("running setup_tests_run_script() ...")
    setup_succeeded = (
        run_command(
            command=f"{UV_EXEC} pip install --managed-python --python {venv_config.venv_python} --index-url https://download.pytorch.org/whl/cpu torch torchvision",
            logger=logger,
        )
        == 0
    )
    setup_succeeded = (
        run_command(
            command=f"{UV_EXEC} pip install --managed-python --python {venv_config.venv_python} datasets transformers==4.57.1 pyyaml==6.0.3 pytest==8.3.5 pytest-asyncio==1.3.0 requests==2.32.5 pyjwt==2.7.0",
            logger=logger,
        )
        == 0
        and setup_succeeded
    )
    return setup_succeeded


_venv_config_list = [
    VenvConfig(
        venv_type=WorkflowVenvType.EVALS_RUN_SCRIPT,
        setup_function=setup_evals_run_script,
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.STRESS_TESTS_RUN_SCRIPT,
        setup_function=setup_stress_tests_run_script,
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.STRESS_TESTS,
        setup_function=setup_stress_tests_run_script,
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.BENCHMARKS_RUN_SCRIPT,
        setup_function=setup_benchmarks_run_script,
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.TESTS_RUN_SCRIPT,
        setup_function=setup_tests_run_script,
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.EVALS_COMMON, setup_function=setup_evals_common
    ),
    VenvConfig(venv_type=WorkflowVenvType.EVALS_META, setup_function=setup_evals_meta),
    VenvConfig(
        venv_type=WorkflowVenvType.EVALS_VISION, setup_function=setup_evals_vision
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.EVALS_AUDIO, setup_function=setup_evals_audio
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.EVALS_EMBEDDING,
        setup_function=setup_evals_embedding,
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.EVALS_VIDEO, setup_function=setup_evals_video
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.BENCHMARKS_VLLM,
        setup_function=setup_benchmarks_vllm,
        python_version="3.11",
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.BENCHMARKS_AIPERF,
        setup_function=setup_benchmarks_aiperf,
        python_version="3.11",
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.BENCHMARKS_VIDEO,
        setup_function=setup_benchmarks_video,
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.REPORTS_RUN_SCRIPT,
        setup_function=setup_reports_run_script,
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.HF_SETUP,
        setup_function=setup_hf_setup,
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.BENCHMARKS_GENAI_PERF,
        setup_function=setup_benchmarks_genai_perf,
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.SYSTEM_SOFTWARE_VALIDATION,
        setup_function=setup_system_software_validation,
        python_version="3.11",
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.TT_SMI,
        setup_function=setup_tt_smi,
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.TT_TOPOLOGY,
        setup_function=setup_tt_topology,
    ),
]

VENV_CONFIGS = map_configs_by_attr(config_list=_venv_config_list, attr="venv_type")
