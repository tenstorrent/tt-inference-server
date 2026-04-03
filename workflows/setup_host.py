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
from typing import Dict, Tuple

# Add the script's directory to the Python path for zero-setup.
project_root = Path(__file__).resolve().parent.parent
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))

from workflows.model_spec import ModelSpec
from workflows.utils import (
    get_default_persistent_volume_root,
    resolve_hf_snapshot_dir,
)
from workflows.validate_setup import _try_fix_path_permissions_for_uid
from workflows.workflow_types import ModelSource, WorkflowVenvType
from workflows.workflow_venvs import VENV_CONFIGS

logger = logging.getLogger("run_log")


@dataclass
class SetupConfig:
    # Environment configuration parameters
    model_spec: ModelSpec
    host_volume: str = None  # --host-volume: host bind mount dir for cache_root
    host_hf_cache: str = None  # --host-hf-cache: host HF cache dir for readonly weights
    host_weights_dir: str = (
        None  # --host-weights-dir: host dir with pre-downloaded weights
    )
    local_server: bool = False
    model_source: str = os.getenv(
        "MODEL_SOURCE", ModelSource.HUGGINGFACE.value
    )  # Either 'huggingface', 'local' or 'noaction'
    docker_volume_name: str = ""
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
    container_readonly_model_weights_dir: Path = None

    def __post_init__(self):
        self._infer_data()
        self._validate_data()

    def _infer_data(self):
        self.repo_root = str(Path(__file__).resolve().parent.parent)
        self.docker_volume_name = (
            f"volume_id_{self.model_spec.impl.impl_id}-{self.model_spec.model_name}"
        )

        if self.local_server and not self.host_volume:
            self.host_volume = str(
                get_default_persistent_volume_root(Path(self.repo_root))
            )
        if self.host_volume:
            self.host_volume = str(Path(self.host_volume).expanduser().resolve())
            self.persistent_volume_root = Path(self.host_volume)
            volume_dir_name = (
                f"volume_id_{self.model_spec.impl.impl_id}"
                f"-{self.model_spec.model_name}"
                f"-v{self.model_spec.version}"
            )
            self.host_model_volume_root = self.persistent_volume_root / volume_dir_name
            self.host_tt_metal_cache_dir = (
                self.host_model_volume_root
                / "tt_metal_cache"
                / f"cache_{self.model_spec.model_name}"
            )

        if self.host_hf_cache:
            self.host_hf_cache = str(Path(self.host_hf_cache).expanduser().resolve())
        if self.host_weights_dir:
            self.host_weights_dir = str(
                Path(self.host_weights_dir).expanduser().resolve()
            )

        # container paths (always set)
        self.container_tt_metal_cache_dir = (
            self.cache_root / "tt_metal_cache" / f"cache_{self.model_spec.model_name}"
        )
        self.container_model_weights_path = (
            self.cache_root / "weights" / self.model_spec.model_name
        )

        # --host-hf-cache: readonly mount of host HF cache for weights
        if self.host_hf_cache and self.model_source == ModelSource.HUGGINGFACE.value:
            self.container_readonly_model_weights_dir = (
                self.containter_user_home / "readonly_weights_mount"
            )
            self.container_model_weights_mount_dir = (
                self.container_readonly_model_weights_dir / self.model_spec.model_name
            )
            snapshot_dir = resolve_hf_snapshot_dir(
                self.model_spec.hf_weights_repo, Path(self.host_hf_cache)
            )
            if snapshot_dir:
                self.update_host_model_weights_snapshot_dir(snapshot_dir)

        # --host-weights-dir: direct bind mount of pre-downloaded weights
        elif self.host_weights_dir:
            host_weights_path = Path(self.host_weights_dir)
            self.container_readonly_model_weights_dir = (
                self.containter_user_home / "readonly_weights_mount"
            )
            self.container_model_weights_mount_dir = (
                self.container_readonly_model_weights_dir / host_weights_path.name
            )
            self.host_model_weights_mount_dir = host_weights_path
            self.container_model_weights_path = self.container_model_weights_mount_dir

        # Local source: separate readonly bind mount
        elif self.model_source == ModelSource.LOCAL.value:
            self.container_readonly_model_weights_dir = (
                self.containter_user_home / "readonly_weights_mount"
            )
            self.container_model_weights_mount_dir = (
                self.container_readonly_model_weights_dir / self.model_spec.model_name
            )
            model_weights_dir = os.getenv("MODEL_WEIGHTS_DIR")
            if model_weights_dir:
                self.update_host_model_weights_mount_dir(Path(model_weights_dir))
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
        automatic: bool = False,
        jwt_secret: str = None,
        hf_token: str = None,
        host_volume: str = None,
        host_hf_cache: str = None,
        host_weights_dir: str = None,
        image_user: str = None,
        local_server: bool = False,
    ):
        self.model_spec = model_spec
        self.automatic = automatic
        self.setup_config = SetupConfig(
            model_spec=model_spec,
            host_volume=host_volume,
            host_hf_cache=host_hf_cache,
            host_weights_dir=host_weights_dir,
            local_server=local_server,
        )
        self.jwt_secret = jwt_secret
        self.hf_token = hf_token
        self.image_user = int(image_user) if image_user else None

    def check_model_weights_dir(self, host_weights_dir: Path) -> bool:
        if not host_weights_dir or not host_weights_dir.exists():
            logger.warning(
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
            {
                "format_name": "hf_bin",
                "weights_format": "pytorch_model*.bin",
                "tokenizer_format": "tokenizer_config.json",
                "params_format": "config.json",
            },
        ]

        # Check each format
        for fmt in model_formats:
            has_weights = bool(list(host_weights_dir.glob(fmt["weights_format"])))
            has_tokenizer = bool(list(host_weights_dir.glob(fmt["tokenizer_format"])))
            has_params = bool(list(host_weights_dir.glob(fmt["params_format"])))

            logger.info(f"has_weights: {has_weights}")
            logger.info(f"has_tokenizer: {has_tokenizer}")
            logger.info(f"has_params: {has_params}")

            if has_weights and has_tokenizer and has_params:
                self.setup_config.model_weights_format = fmt["format_name"]
                logger.info(f"Detected {fmt['format_name']} model format")
                logger.info(
                    f"✅ Setup already completed for model {self.model_spec.model_name}."
                )
                return True
        logger.warning(
            f"Incomplete model setup for {self.model_spec.model_name}. "
            f"checked: {host_weights_dir}"
        )
        return False

    def check_setup(self) -> bool:
        if self.setup_config.model_source == ModelSource.HUGGINGFACE.value:
            if (
                not self.setup_config.host_hf_cache
                and not self.setup_config.host_model_volume_root
                and not self.setup_config.host_weights_dir
            ):
                return True
            if self.setup_config.host_hf_cache:
                return self.check_model_weights_dir(
                    self.setup_config.host_model_weights_snapshot_dir
                )
            if self.setup_config.host_weights_dir:
                return self.check_model_weights_dir(
                    Path(self.setup_config.host_weights_dir)
                )
            if self.setup_config.host_model_volume_root:
                host_weights_dir = (
                    self.setup_config.host_model_volume_root
                    / "weights"
                    / self.model_spec.model_name
                )
                return self.check_model_weights_dir(host_weights_dir)
        elif self.setup_config.model_source == ModelSource.LOCAL.value:
            return self.check_model_weights_dir(
                self.setup_config.host_model_weights_mount_dir
            )
        elif self.setup_config.model_source == ModelSource.NOACTION.value:
            logger.info("Assuming that server self-provides the weights.")
            return True
        else:
            raise ValueError("⛔ Invalid model source.")

    def check_disk_space(self) -> bool:
        if self.setup_config.model_source != ModelSource.HUGGINGFACE.value:
            return True
        if self.setup_config.host_weights_dir:
            return True
        if self.setup_config.host_hf_cache:
            check_path = self.setup_config.host_hf_cache
        elif self.setup_config.host_model_volume_root:
            check_path = str(self.setup_config.host_model_volume_root)
        else:
            return True
        total, used, free = shutil.disk_usage(check_path)
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

    def get_hf_env_vars(self):
        self.hf_token = self.hf_token or os.getenv("HF_TOKEN", "")
        if not self.hf_token and not self.automatic:
            self.hf_token = getpass.getpass("Enter your HF_TOKEN: ").strip()
        assert self.check_hf_access(self.hf_token), "⛔ HF_TOKEN validation failed."

        if self.setup_config.host_hf_cache:
            hf_home = Path(self.setup_config.host_hf_cache)
            hf_home.mkdir(parents=True, exist_ok=True)
            os.environ["HOST_HF_HOME"] = str(hf_home)
            os.environ["HF_HOME"] = str(hf_home)
            logger.info(f"✅ HF_HOME set to {self.setup_config.host_hf_cache}")

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
        model_url = (
            f"https://huggingface.co/api/models/{self.model_spec.hf_weights_repo}"
        )
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
        head_url = f"https://huggingface.co/{self.model_spec.hf_weights_repo}/resolve/main/{first_file}"
        _, _, head_headers = http_request(
            head_url, method="HEAD", headers={"Authorization": f"Bearer {token}"}
        )
        if head_headers.get("x-error-code"):
            logger.error("⛔ The model is gated and you don't have access.")
            return False
        logger.info("✅ HF_TOKEN is valid.")
        return True

    def setup_model_environment(self):
        assert self.check_ram(), "⛔ Insufficient host RAM."

        if not self.automatic and os.getenv("MODEL_SOURCE") is None:
            print("\nHow do you want to provide a model?")
            print("1) Download from Hugging Face (default)")
            print("2) Local folder")
            choice = input("Enter your choice: ").strip() or "1"
            # Map numeric choice to source type
            if choice == "1":
                self.setup_config.model_source = ModelSource.HUGGINGFACE.value
            elif choice == "2":
                self.setup_config.model_source = ModelSource.LOCAL.value
            else:
                raise ValueError("⛔ Invalid model source choice.")

        if self.setup_config.model_source == ModelSource.HUGGINGFACE.value:
            self.get_hf_env_vars()
        elif self.setup_config.model_source == ModelSource.LOCAL.value:
            if self.automatic:
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
        assert self.check_disk_space(), "⛔ Insufficient disk space."

        if not self.jwt_secret:
            self.jwt_secret = os.getenv("JWT_SECRET", "")
        if not self.jwt_secret and not self.automatic:
            self.jwt_secret = getpass.getpass("Enter your JWT_SECRET: ").strip()

        assert self.jwt_secret, "⛔ JWT_SECRET cannot be empty."

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
        if self.setup_config.host_weights_dir:
            logger.info(
                f"Using pre-downloaded weights from --host-weights-dir: {self.setup_config.host_weights_dir}"
            )
            return
        if (
            not self.setup_config.host_hf_cache
            and not self.setup_config.host_model_volume_root
        ):
            logger.info(
                "Weights will be downloaded into Docker volume on first container start"
            )
            return

        assert self.hf_token, "⛔ HF_TOKEN not set."
        if self.model_spec.repacked == 1:
            raise ValueError("⛔ Repacked models are not supported for Hugging Face.")
        # setup venv using uv
        venv_config = VENV_CONFIGS[WorkflowVenvType.HF_SETUP]
        venv_config.setup(model_spec=self.model_spec)
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
        os.environ["HF_TOKEN"] = self.hf_token
        hf_exec = venv_config.venv_path / "bin" / "hf"
        assert hf_exec.exists(), (
            f"⛔ 'hf' CLI not found at: {hf_exec}. Check HF_SETUP venv installation."
        )
        hf_repo = self.model_spec.hf_weights_repo
        if self.setup_config.host_hf_cache:
            os.environ["HOST_HF_HOME"] = self.setup_config.host_hf_cache
            os.environ["HF_HOME"] = self.setup_config.host_hf_cache
            cmd = [
                str(hf_exec),
                "download",
                hf_repo,
                "--exclude",
                "original/**",
            ]
            logger.info(f"Downloading model to host HF cache: {hf_repo}")
            logger.info(f"Command: {shlex.join(cmd)}")
            result = subprocess.run(cmd)
            assert result.returncode == 0, f"⛔ Error during: {' '.join(cmd)}"
            snapshot_dir = resolve_hf_snapshot_dir(
                self.model_spec.hf_weights_repo, Path(self.setup_config.host_hf_cache)
            )
            self.setup_config.update_host_model_weights_snapshot_dir(snapshot_dir)
            logger.info(
                f"✅ Using weights directory: {self.setup_config.host_model_weights_mount_dir}"
            )
            return

        host_weights_dir = (
            self.setup_config.host_model_volume_root
            / "weights"
            / self.model_spec.model_name
        )
        if self.check_model_weights_dir(host_weights_dir):
            logger.info("Weights already exist in host volume, skipping download")
            return
        host_weights_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(hf_exec),
            "download",
            hf_repo,
            "--local-dir",
            str(host_weights_dir),
            "--exclude",
            "original/**",
        ]
        logger.info(f"Downloading model to host volume: {hf_repo}")
        logger.info(f"Command: {shlex.join(cmd)}")
        result = subprocess.run(cmd)
        assert result.returncode == 0, f"⛔ Error during: {' '.join(cmd)}"
        logger.info(f"✅ Using weights directory: {host_weights_dir}")

    def setup_weights_local(self):
        if self.setup_config.host_model_weights_mount_dir.exists():
            logger.info(
                f"✅ Using weights directory: {self.setup_config.host_model_weights_mount_dir}"
            )
            self.check_model_weights_dir(self.setup_config.host_model_weights_mount_dir)
        else:
            raise ValueError("⛔ Weights directory does not exist.")

    def make_host_dirs(self):
        dirs_to_create = [
            self.setup_config.host_model_volume_root,
            self.setup_config.host_tt_metal_cache_dir,
        ]
        for dir_path in dirs_to_create:
            if not dir_path:
                continue
            dir_path.mkdir(parents=True, exist_ok=True)

        if (
            self.setup_config.local_server
            or self.image_user is None
            or not self.setup_config.persistent_volume_root
        ):
            return
        volume_root = self.setup_config.persistent_volume_root
        for dir_path in dirs_to_create:
            if not dir_path:
                continue
            try:
                rel = dir_path.relative_to(volume_root)
            except ValueError:
                _try_fix_path_permissions_for_uid(
                    dir_path, self.image_user, need_write=True
                )
                continue
            cumulative = volume_root
            for part in rel.parts:
                cumulative = cumulative / part
                _try_fix_path_permissions_for_uid(
                    cumulative, self.image_user, need_write=True
                )

    def setup_weights(self):
        if not self.check_setup():
            if self.setup_config.model_source == ModelSource.HUGGINGFACE.value:
                self.setup_weights_huggingface()
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
        if self.setup_config.local_server:
            logger.info("✅ resolved local-server host storage paths")
            return
        setup_completed = self.check_setup()
        if setup_completed:
            return
        self.setup_model_environment()
        self.setup_weights()
        logger.info("✅ done run_setup")


def setup_host(
    model_spec,
    jwt_secret,
    hf_token,
    automatic_setup=False,
    host_volume=None,
    host_hf_cache=None,
    host_weights_dir=None,
    image_user=None,
    local_server=False,
):
    automatic = bool(automatic_setup)

    manager = HostSetupManager(
        model_spec=model_spec,
        jwt_secret=jwt_secret,
        hf_token=hf_token,
        automatic=automatic,
        host_volume=host_volume,
        host_hf_cache=host_hf_cache,
        host_weights_dir=host_weights_dir,
        image_user=image_user,
        local_server=local_server,
    )
    manager.run_setup()
    return manager.setup_config


def main():
    parser = argparse.ArgumentParser(description="Model setup script")
    parser.add_argument(
        "--check-weights",
        action="store_true",
        help="Replicate release workflow: run check_model_weights_dir and exit (for local/CI debug).",
    )
    parser.add_argument(
        "--model-spec-json",
        type=str,
        help="Path to model spec JSON file (required for --check-weights or full setup).",
        default=None,
    )
    parser.add_argument(
        "--weights-dir",
        type=str,
        default=None,
        help="Optional: test this directory instead of HF cache path (only with --check-weights).",
    )
    parser.add_argument("model_name", nargs="?", help="Type of the model to setup")
    parser.add_argument("impl", nargs="?", help="Implementation to use")
    parser.add_argument(
        "--device",
        type=str,
        help="DeviceTypes str used to simulate different hardware configurations",
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

    args = parser.parse_args()

    if args.check_weights:
        if not args.model_spec_json:
            print(
                "[setup_host] --check-weights requires --model-spec-json",
                file=sys.stderr,
                flush=True,
            )
            sys.exit(2)
        model_spec = ModelSpec.from_json(args.model_spec_json)
        manager = HostSetupManager(
            model_spec=model_spec,
            jwt_secret=args.jwt_secret or "",
            hf_token=args.hf_token or "",
            automatic=True,
        )
        if args.weights_dir:
            host_weights_dir = Path(args.weights_dir)
            print(
                f"[setup_host] Testing --weights-dir: {host_weights_dir}",
                file=sys.stderr,
                flush=True,
            )
        else:
            host_weights_dir = manager.setup_config.host_model_weights_snapshot_dir
            if manager.setup_config.model_source != ModelSource.HUGGINGFACE.value:
                host_weights_dir = manager.setup_config.host_model_weights_mount_dir
            if host_weights_dir is None:
                print(
                    "[setup_host] No weights path from setup_config (HF cache dir not found for this model).",
                    file=sys.stderr,
                    flush=True,
                )
                sys.exit(1)
            print(
                f"[setup_host] Using path from setup_config (same as release): {host_weights_dir}",
                file=sys.stderr,
                flush=True,
            )
        ok = manager.check_model_weights_dir(host_weights_dir)
        print(
            f"[setup_host] check_model_weights_dir result: {ok}",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(0 if ok else 1)

    if not args.model_spec_json:
        parser.error("Full setup requires --model-spec-json (or use --check-weights).")
    model_spec = ModelSpec.from_json(args.model_spec_json)
    raise NotImplementedError("⛔ Not implemented")
    setup_host(
        model_spec=model_spec,
        jwt_secret=args.jwt_secret,
        hf_token=args.hf_token,
    )


if __name__ == "__main__":
    main()
