#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import argparse
import getpass
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
from torch import hub

# Add the script's directory to the Python path for zero-setup.
project_root = Path(__file__).resolve().parent.parent
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))

from workflows.model_spec import (
    ModelSpec,
    ModelSource,
)
from workflows.run_workflows import WorkflowSetup
from workflows.utils import (
    get_default_hf_home_path,
    get_default_th_home_path,
    get_weights_hf_cache_dir,
)
from workflows.workflow_types import WorkflowVenvType
from workflows.workflow_venvs import VENV_CONFIGS

logger = logging.getLogger("run_log")


@dataclass
class SetupConfig:
    # Environment configuration parameters
    model_spec: ModelSpec
    host_hf_home: str = ""  # Host HF cache directory (set interactively or via env)
    host_th_home: str = ""  # Host TorchHub cache directory (set interactively or via env)
    model_source: str = os.getenv(
        "MODEL_SOURCE", ModelSource.HUGGINGFACE.value
    )  # Either 'huggingface', 'local' or 'noaction'
    persistent_volume_root: Path = None
    host_model_volume_root: Path = None
    host_tt_metal_cache_dir: Path = None
    host_model_weights_snapshot_dir: Path = None
    host_model_weights_mount_dir: Path = None
    containter_user_home: Path = Path("/home/container_app_user/")
    cache_root: Path = containter_user_home / "cache_root"
    container_model_spec_dir: Path = containter_user_home / "model_spec"
    container_tt_metal_cache_dir: Path = None
    container_model_weights_snapshot_dir: Path = None
    container_model_weights_mount_dir: Path = None
    container_model_weights_path: Path = None
    repo_root: str = ""
    repacked_str: str = ""
    model_weights_format: str = ""
    model_weights_dir_format: str = "hf_cache"
    container_readonly_model_weights_dir: Path = None

    def __post_init__(self):
        self._infer_data()
        self._validate_data()

    def _infer_data(self):
        self.repo_root = str(Path(__file__).resolve().parent.parent)
        self.persistent_volume_root = Path(
            os.getenv(
                "PERSISTENT_VOLUME_ROOT",
                str(Path(self.repo_root) / "persistent_volume"),
            )
        )
        volume_name = f"volume_id_{self.model_spec.impl.impl_id}-{self.model_spec.model_name}-v{self.model_spec.version}"
        # host paths
        self.host_model_volume_root = self.persistent_volume_root / volume_name
        self.host_tt_metal_cache_dir = (
            self.host_model_volume_root
            / "tt_metal_cache"
            / f"cache_{self.model_spec.model_name}"
        )
        # container paths
        self.container_tt_metal_cache_dir = (
            self.cache_root / "tt_metal_cache" / f"cache_{self.model_spec.model_name}"
        )
        self.container_readonly_model_weights_dir = (
            self.containter_user_home / "readonly_weights_mount"
        )
        self.container_model_weights_mount_dir = (
            self.container_readonly_model_weights_dir / f"{self.model_spec.model_name}"
        )
        if self.model_source == ModelSource.HUGGINGFACE.value:
            repo_path_filter = None
            self.update_host_model_weights_snapshot_dir(
                get_weights_hf_cache_dir(self.model_spec.hf_model_repo),
                repo_path_filter=repo_path_filter,
            )
        elif self.model_source == ModelSource.TORCHHUB.value:

        elif self.model_source == ModelSource.LOCAL.value:
            self.update_host_model_weights_mount_dir(
                Path(os.getenv("MODEL_WEIGHTS_DIR"))
            )
        elif self.model_source == ModelSource.NOACTION.value:
            pass

    def _validate_data(self):
        # Validate that model_source is a valid enum value
        try:
            ModelSource(self.model_source)
        except ValueError:
            raise ValueError("⛔ Invalid model source.")

    def update_host_model_weights_snapshot_dir(
        self, host_model_weights_snapshot_dir, repo_path_filter=None
    ):
        assert self.model_source == ModelSource.HUGGINGFACE.value, (
            "⛔ update_host_model_weights_snapshot_dir only supported for huggingface model source."
        )
        if host_model_weights_snapshot_dir:
            if repo_path_filter:
                self.host_model_weights_snapshot_dir = (
                    host_model_weights_snapshot_dir / repo_path_filter
                )
                self.host_model_weights_mount_dir = (
                    self.host_model_weights_snapshot_dir.parent.parent.parent
                )
            else:
                self.host_model_weights_snapshot_dir = host_model_weights_snapshot_dir
                self.host_model_weights_mount_dir = (
                    self.host_model_weights_snapshot_dir.parent.parent
                )
            self.container_model_weights_snapshot_dir = (
                self.container_model_weights_mount_dir
                / self.host_model_weights_snapshot_dir.relative_to(
                    self.host_model_weights_mount_dir
                )
            )
            # container_model_weights_path is where weights are loaded from
            self.container_model_weights_path = (
                self.container_model_weights_snapshot_dir
            )

    def update_host_model_weights_mount_dir(self, host_model_weights_mount_dir):
        assert self.model_source == ModelSource.LOCAL.value, (
            "⛔ update_host_model_weights_mount_dir only supported for local model source."
        )
        self.host_model_weights_mount_dir = host_model_weights_mount_dir
        if self.host_model_weights_mount_dir.exists():
            self.container_model_weights_mount_dir = (
                self.container_readonly_model_weights_dir
                / self.host_model_weights_mount_dir.name
            )
            # container_model_weights_path is where weights are loaded from
            self.container_model_weights_path = self.container_model_weights_mount_dir


def http_request(
    url: str, method: str = "GET", headers: Dict[str, str] = None
) -> Tuple[bytes, int, dict]:
    headers = headers or {}
    req = urllib.request.Request(url, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.read(), resp.getcode(), dict(resp.headers)
    except urllib.error.HTTPError as e:
        return e.read(), e.getcode(), dict(e.headers)
    except Exception as e:
        logger.error(f"⛔ Error making {method} request to {url}: {e}")
        raise Exception from e


class HostSetupManager:
    def __init__(
        self,
        model_spec: ModelSpec,
    ):
        self.model_spec = model_spec
        self.automatic_setup = self._load_automatic_setup()
        self.setup_config = SetupConfig(model_spec=model_spec)
        self.jwt_secret = self._load_jwt_secret()
        self.models_repo_token = self._load_models_repo_token()
        self._set_hf_home()
        self._set_th_home()
    
    def _load_automatic_setup(self) -> bool:
        return os.getenv("AUTOMATIC_HOST_SETUP") == "1"

    def _load_jwt_secret(self) -> Optional[str]:
        if not self.automatic_setup:
            jwt_secret = getpass.getpass("Enter your JWT_SECRET: ").strip()
        else:
            jwt_secret = os.getenv("JWT_SECRET")
        assert jwt_secret, "⛔ JWT_SECRET cannot be empty."
        return jwt_secret
    
    def _load_models_repo_token(self) -> Optional[str]:
        models_repo_token = None

        if self.setup_config.model_source == ModelSource.HUGGINGFACE.value and not self.automatic_setup:
            models_repo_token = getpass.getpass("Enter your HF_TOKEN: ").strip()
        else:
            models_repo_token = os.getenv("HF_TOKEN")
        if self.setup_config.model_source == ModelSource.TORCHHUB.value and not self.automatic_setup:
            models_repo_token = getpass.getpass("Enter your TORCHHUB_TOKEN: ").strip()
        else:    
            models_repo_token = os.getenv("TORCHHUB_TOKEN")
        
        if models_repo_token is not None:
            assert self._validate_models_repo_access(models_repo_token), \
            "⛔ Models repository token validation failed."

        return models_repo_token
    
    def _set_cache_home(
        self, 
        model_source: ModelSource, 
        setup_config_attr: str, 
        prompt_name: str, 
        default_value: str
    ) -> None:
        """Generic method to set cache home directory for different model sources."""
        if self.setup_config.model_source != model_source.value:
            return
        
        if self.automatic_setup:
            setattr(self.setup_config, setup_config_attr, default_value)
            return
        
        # Interactive mode: prompt user
        user_input = (
            input(f"Enter host {prompt_name} [default: {default_value}]: ").strip()
            or default_value
        )
        setattr(self.setup_config, setup_config_attr, user_input)
    
    def _set_hf_home(self) -> None:
        self._set_cache_home(
            model_source=ModelSource.HUGGINGFACE,
            setup_config_attr="host_hf_home",
            prompt_name="HF_HOME",
            default_value=get_default_hf_home_path()
        )
    
    def _set_th_home(self) -> None:
        self._set_cache_home(
            model_source=ModelSource.TORCHHUB,
            setup_config_attr="host_th_home",
            prompt_name="TORCH_HOME",
            default_value=get_default_th_home_path()
        )

    def check_model_weights_dir(self, host_weights_dir: Path) -> bool:
        if not host_weights_dir or not host_weights_dir.exists():
            logger.info(
                f"Weights directory does not exist for {self.model_spec.model_name}."
            )
            return False

        # Define supported model formats
        model_formats = [
            {
                "format_name": "meta",
                "weights_format": "*.pth",
                "tokenizer_format": "tokenizer.model",
                "params_format": "params.json",
            },
            {
                "format_name": "hf",
                "weights_format": "model*.safetensors",
                "tokenizer_format": "tokenizer.json",
                "params_format": "config.json",
            },
        ]

        # Check each format
        for fmt in model_formats:
            has_weights = bool(list(host_weights_dir.glob(fmt["weights_format"])))
            has_tokenizer = bool(list(host_weights_dir.glob(fmt["tokenizer_format"])))
            has_params = bool(list(host_weights_dir.glob(fmt["params_format"])))

            if has_weights and has_tokenizer and has_params:
                self.setup_config.model_weights_format = fmt["format_name"]
                logger.info(f"detected {fmt['format_name']} model format")
                logger.info(
                    f"✅ Setup already completed for model {self.model_spec.model_name}."
                )
                return True
        logger.info(
            f"Incomplete model setup for {self.model_spec.model_name}. "
            f"checked: {host_weights_dir}"
        )
        return False

    def check_model_weights_exist(self) -> bool:
        if self.setup_config.model_source in [ModelSource.HUGGINGFACE.value, ModelSource.TORCHHUB.value]:
            return self.check_model_weights_dir(
                self.setup_config.host_model_weights_snapshot_dir
            )
        if self.setup_config.model_source == ModelSource.LOCAL.value:
            return self.check_model_weights_dir(
                self.setup_config.host_model_weights_mount_dir
            )
        if self.setup_config.model_source == ModelSource.NOACTION.value:
            logger.info(f"Model source: {self.setup_config.model_source}. \
                        Assuming that server self-provides the model weights...")
            return True

    def check_disk_space(self) -> bool:
        if not self.setup_config.model_source == ModelSource.HUGGINGFACE.value:
            return True
        assert self.setup_config.host_hf_home, "⛔ HOST_HF_HOME not set."
        total, used, free = shutil.disk_usage(self.setup_config.host_hf_home)
        free_gb = free // (1024**3)
        if free_gb >= self.model_spec.min_disk_gb:
            logger.info(
                f"✅ Sufficient disk space: {free_gb}GB (Required: {self.model_spec.min_disk_gb}GB)"
            )
            return True
        logger.error(
            f"❌ Insufficient disk space: {free_gb}GB (Required: {self.model_spec.min_disk_gb}GB)"
        )
        return False

    def check_ram(self) -> bool:
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        available_kb = int(line.split()[1])
                        available_gb = available_kb / (1024 * 1024)
                        break
                else:
                    logger.error("❌ Could not determine available RAM.")
                    return False
        except Exception as e:
            logger.error(f"❌ Error reading /proc/meminfo: {e}")
            return False
        if available_gb >= self.model_spec.min_ram_gb:
            logger.info(
                f"✅ Sufficient RAM: {int(available_gb)}GB (Required: {self.model_spec.min_ram_gb}GB)"
            )
            return True
        logger.error(
            f"❌ Insufficient RAM: {int(available_gb)}GB (Required: {self.model_spec.min_ram_gb}GB)"
        )
        return False

    def _set_model_repo_env_vars(
        self,
        model_repo_home_attr: str,
        model_repo_env_var: str
    ) -> None:
        """Generic method to set model repository environment variables for different model sources."""
        model_repo_home_str = getattr(self.setup_config, model_repo_home_attr)
        model_repo_home = Path(model_repo_home_str)
        
        assert os.access(model_repo_home, os.W_OK), (
            f"⛔ {model_repo_env_var}={model_repo_home_str} is not writable."
        )
        model_repo_home.mkdir(parents=True, exist_ok=True)
        
        # Set environment variable so that cache is used correctly
        os.environ[model_repo_env_var] = str(model_repo_home)
        logger.info(f"✅ Host {model_repo_env_var} set to {model_repo_home_str}")
    
    def set_hf_env_vars(self):
        self._set_model_repo_env_vars(
            model_repo_home_attr="host_hf_home",
            model_repo_env_var="HF_HOME"
        )
    
    def set_torchhub_env_vars(self):
        self._set_model_repo_env_vars(
            model_repo_home_attr="host_th_home",
            model_repo_env_var="TORCH_HOME"
        )

    def check_hf_access(self, token: str) -> int:
        if not token or not token.startswith("hf_"):
            logger.error("⛔ Invalid HF_TOKEN.")
            return False
        data, status, _ = http_request(
            "https://huggingface.co/api/whoami-v2",
            headers={"Authorization": f"Bearer {token}"},
        )
        if status != 200:
            logger.error("⛔ HF_TOKEN rejected by Hugging Face.")
            return False
        model_url = f"https://huggingface.co/api/models/{self.model_spec.hf_model_repo}"
        data, status, _ = http_request(
            model_url, headers={"Authorization": f"Bearer {token}"}
        )
        try:
            json_data = json.loads(data.decode("utf-8"))
        except json.JSONDecodeError:
            logger.error("⛔ Invalid JSON response from Hugging Face.")
            return False
        siblings = json_data.get("siblings", [])
        if not siblings:
            logger.error("⛔ No files found in repository.")
            return False
        first_file = siblings[0].get("rfilename")
        if not first_file:
            logger.error("⛔ Unexpected repository structure.")
            return False
        head_url = f"https://huggingface.co/{self.model_spec.hf_model_repo}/resolve/main/{first_file}"
        _, _, head_headers = http_request(
            head_url, method="HEAD", headers={"Authorization": f"Bearer {token}"}
        )
        if head_headers.get("x-error-code"):
            logger.error("⛔ The model is gated and you don't have access.")
            return False
        logger.info("✅ HF_TOKEN is valid.")
        return True
    
    def check_torchhub_access(self) -> bool:
        try:
            _ = torch.hub.list(self.model_spec.th_model_repo, skip_validation=False)
            logger.info("✅ TorchHub access check succeeded.")
            return True
        except Exception as e:
            logger.error(f"⛔ TorchHub access check failed: {e}")
            return False

    def _validate_models_repo_access(self, token: str) -> bool:
        if self.setup_config.model_source == ModelSource.HUGGINGFACE.value:
            return self.check_hf_access(token)
        elif self.setup_config.model_source == ModelSource.TORCHHUB.value:
            return self.check_torchhub_access(token)

    def setup_model_environment(self):
        assert self.check_ram(), "⛔ Insufficient host RAM."

        if self.automatic_setup:
            self.setup_config.persistent_volume_root = Path(
                os.getenv(
                    "PERSISTENT_VOLUME_ROOT",
                    str(self.setup_config.persistent_volume_root),
                )
            )

        if not self.setup_config.persistent_volume_root.exists():
            pv_input = input(
                f"Enter persistent_volume_root [default: {self.setup_config.persistent_volume_root}]: "
            ).strip()
            if pv_input:
                self.setup_config.persistent_volume_root = Path(pv_input)

        if not self.automatic_setup and os.getenv("MODEL_SOURCE") is None:
            print("\nHow do you want to provide a model?")
            print("1) Download from Hugging Face (default)")
            print("2) Download from TorchHub")
            print("3) Local folder")
            choice = input("Enter your choice: ").strip() or "1"
            # Map numeric choice to source type
            if choice == "1":
                self.setup_config.model_source = ModelSource.HUGGINGFACE.value
            elif choice == "2":
                self.setup_config.model_source = ModelSource.TORCHHUB.value
            elif choice == "3":
                self.setup_config.model_source = ModelSource.LOCAL.value
            else:
                raise ValueError("⛔ Invalid model source choice.")

        if self.setup_config.model_source == ModelSource.HUGGINGFACE.value:
            self.set_hf_env_vars()
        elif self.setup_config.model_source == ModelSource.TORCHHUB.value:
            self.set_torchhub_env_vars()
        elif self.setup_config.model_source == ModelSource.LOCAL.value:
            if self.automatic_setup:
                _host_model_weights_mount_dir = os.getenv("MODEL_WEIGHTS_DIR")
                assert _host_model_weights_mount_dir, (
                    "⛔ MODEL_WEIGHTS_DIR environment variable is required for local model source in automatic mode."
                )
            else:
                _host_model_weights_mount_dir = (
                    os.getenv("MODEL_WEIGHTS_DIR")
                    or input("Enter local model directory: ").strip()
                )
            self.setup_config.update_host_model_weights_mount_dir(
                Path(_host_model_weights_mount_dir)
            )
        # need to know where weights would be downloaded to before checking disk space
        assert self.check_disk_space(), "⛔ Insufficient disk space."

    def repack_weights(self, source_dir: Path, target_dir: Path):
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(source_dir / "tokenizer.model", target_dir / "tokenizer.model")
        shutil.copy(source_dir / "params.json", target_dir / "params.json")
        # Set up a temporary venv for repacking.
        venv_dir = Path(".venv_repack")
        subprocess.run(["python3", "-m", "venv", str(venv_dir)], check=True)
        venv_python = venv_dir / "bin" / "python"
        subprocess.run(
            [
                str(venv_python),
                "-m",
                "pip",
                "install",
                "--upgrade",
                "setuptools",
                "wheel",
                "pip==21.2.4",
                "tqdm",
            ],
            check=True,
        )
        subprocess.run(
            [
                str(venv_python),
                "-m",
                "pip",
                "install",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
                "torch==2.2.1",
            ],
            check=True,
        )
        repack_url = "https://github.com/tenstorrent/tt-metal/raw/refs/tags/v0.56.0-rc47/models/demos/t3000/llama2_70b/scripts/repack_weights.py"
        data, status, _ = http_request(repack_url)
        assert status == 200, "⛔ Failed to download repack_weights.py"
        repack_script = Path("repack_weights.py")
        with repack_script.open("wb") as f:
            f.write(data)
        subprocess.run(
            [
                str(venv_python),
                str(repack_script),
                str(source_dir),
                str(target_dir),
                "5",
            ],
            check=True,
        )
        shutil.rmtree(str(venv_dir))
        repack_script.unlink()
        logger.info("✅ Weight repacking completed.")

    def setup_weights_huggingface(self):
        assert self.hf_token and self.setup_config.host_hf_home, (
            "⛔ HF_TOKEN or HOST_HF_HOME not set."
        )
        if self.model_spec.repacked == 1:
            raise ValueError("⛔ Repacked models are not supported for Hugging Face.")
        # Bootstrap uv and create/use the managed HF setup venv
        WorkflowSetup.bootstrap_uv()
        uv_exec = WorkflowSetup.uv_exec
        venv_config = VENV_CONFIGS[WorkflowVenvType.HF_SETUP]
        if not venv_config.venv_path.exists():
            subprocess.run(
                [
                    str(uv_exec),
                    "venv",
                    "--managed-python",
                    f"--python={venv_config.python_version}",
                    str(venv_config.venv_path),
                    "--allow-existing",
                ],
                check=True,
            )
        # Ensure required packages are present in the venv
        venv_config.setup(model_spec=self.model_spec, uv_exec=uv_exec)
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
        os.environ["HF_TOKEN"] = self.hf_token
        # Require 'hf' CLI (no fallbacks). Ensure compatibility by installing huggingface_hub>=1.0.0.
        hf_exec = venv_config.venv_path / "bin" / "hf"
        assert hf_exec.exists(), (
            f"⛔ 'hf' CLI not found at: {hf_exec}. Check HF_SETUP venv installation."
        )
        base_cmd = [str(hf_exec)]
        hf_repo = self.model_spec.hf_model_repo
        # use default huggingface repo
        # fmt: off
        cmd = base_cmd + [
            "download", hf_repo,
            "--exclude", "original/"
        ]
        # fmt: on
        repo_path_filter = None
        logger.info(f"Downloading model from Hugging Face: {hf_repo}")
        logger.info(f"Command: {shlex.join(cmd)}")
        result = subprocess.run(cmd)
        assert result.returncode == 0, f"⛔ Error during: {' '.join(cmd)}"
        # need to update paths
        self.setup_config.update_host_model_weights_snapshot_dir(
            get_weights_hf_cache_dir(self.model_spec.hf_model_repo),
            repo_path_filter=repo_path_filter,
        )
        logger.info(
            f"✅ Using weights directory: {self.setup_config.host_model_weights_mount_dir}"
        )
    
    def setup_weights_torchhub(self):
        assert self.setup_config.host_th_home, "⛔ TORCH_HOME not set."

        torch.hub.set_dir(self.setup_config.host_th_home)
        try:
            if ':' in self.model_spec.th_model_repo and '/' in self.model_spec.th_model_repo.split(':')[-1]:
                repo_path, model_name = self.model_spec.th_model_repo.rsplit(':', 1)
                logger.info(f"Downloading model from TorchHub: repo={repo_path}, model={model_name}")
                model = torch.hub.load(
                    repo_path,
                    model_name,
                    pretrained=True,
                )
                logger.info(f"✅ Model {model_name} loaded successfully from TorchHub.")

                del model  # Free up memory
        except Exception as e:
            raise ValueError(f"⛔ Error loading model from TorchHub: {e}")

    def setup_weights_local(self):
        if self.setup_config.host_model_weights_mount_dir.exists():
            logger.info(
                f"✅ Using weights directory: {self.setup_config.host_model_weights_mount_dir}"
            )
            self.check_model_weights_dir(self.setup_config.host_model_weights_mount_dir)
        else:
            raise ValueError("⛔ Weights directory does not exist.")

    def make_host_dirs(self):
        logger.info(f"Trying to create host directories for model (skipping if they exist): \
                    {self.model_spec.model_name}")
        logger.info(f"Model volume root: {self.setup_config.host_model_volume_root}")
        self.setup_config.host_model_volume_root.mkdir(parents=True, exist_ok=True)
        logger.info(f"TT Metal cache directory: {self.setup_config.host_tt_metal_cache_dir}")
        self.setup_config.host_tt_metal_cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("✅ Host directories ready")

    def setup_weights(self):
        if not self.check_model_weights_exist():
            if self.setup_config.model_source == ModelSource.HUGGINGFACE.value:
                self.setup_weights_huggingface()
            elif self.setup_config.model_source == ModelSource.TORCHHUB.value:
                self.setup_weights_torchhub()
            elif self.setup_config.model_source == ModelSource.LOCAL.value:
                self.setup_weights_local()
            elif self.setup_config.model_source == ModelSource.NOACTION.value:
                logger.info(
                    "No action taken for model weights setup as per 'noaction' model source."
                )
            else:
                raise ValueError("⛔ Invalid model source.")
        logger.info("✅ done setup_weights")

    def run_setup(self):
        self.make_host_dirs()
        if self.check_model_weights_exist():
            return
        self.setup_model_environment()
        self.setup_weights()
        logger.info("✅ done run_setup")


def main():
    parser = argparse.ArgumentParser(description="Model setup script")
    parser.add_argument("model_name", help="Type of the model to setup")
    parser.add_argument("impl", help="Implementation to use")
    parser.add_argument(
        "--device",
        type=str,
        help="DeviceTypes str used to simulate different hardware configurations",
        required=True,
    )
    parser.add_argument(
        "--automatic",
        action="store_true",
        help="Bypass interactive prompts by using environment variables or defaults",
    )
    parser.add_argument(
        "--jwt-secret",
        type=str,
        help="JWT secret for generating token to set API_KEY",
        default=os.getenv("JWT_SECRET", ""),
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        help="HF_TOKEN",
        default=os.getenv("HF_TOKEN", ""),
    )
    parser.add_argument(
        "--impl",
        type=str,
        help="Model implementation to use",
        default=os.getenv("MODEL_IMPL", "tt-transformers"),
    )
    parser.add_argument(
        "--model-spec-json",
        type=str,
        help="Path to model spec JSON file",
        default=None,
    )

    args = parser.parse_args()
    model_spec = ModelSpec.from_json(args.model_spec_json)
    raise NotImplementedError("⛔ Not implemented")


if __name__ == "__main__":
    main()
