#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import argparse
from dataclasses import dataclass
import getpass
import json
import os
import shutil
import logging
import subprocess
import sys
import urllib.request
import shlex
import urllib.error
from pathlib import Path
from typing import Tuple, Dict

# Add the script's directory to the Python path for zero-setup.
project_root = Path(__file__).resolve().parent.parent
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))

from workflows.model_config import (
    MODEL_CONFIGS,
    ModelConfig,
)

logger = logging.getLogger("run_log")


@dataclass
class SetupConfig:
    # Environment configuration parameters
    model_config: ModelConfig
    hf_home: str = ""  # Container-specific hf_home (derived later)
    host_hf_home: str = ""  # Host HF cache directory (set interactively or via env)
    model_source: str = "huggingface"  # Either 'huggingface' or 'local'
    repo_root: str = ""
    persistent_volume_root: Path = None
    cache_root: Path = Path("/home/container_app_user/cache_root")
    llama_host_weights_dir: str = ""
    repacked_str: str = ""

    def __post_init__(self):
        self._infer_data()

    def _infer_data(self):
        if not self.repacked_str:
            self.repacked_str = "repacked-" if self.model_config.repacked == 1 else ""

    @property
    def model_volume_root(self) -> Path:
        assert self.persistent_volume_root
        volume_name = f"volume_id_{self.model_config.impl_id}-{self.model_config.model_name}-v{self.model_config.version}/"
        return self.persistent_volume_root / volume_name

    @property
    def env_file(self) -> Path:
        assert self.persistent_volume_root
        return (
            self.persistent_volume_root
            / "model_envs"
            / f"{self.model_config.model_name}.env"
        )

    @property
    def tt_metal_cache_dir(self) -> Path:
        return (
            self.model_volume_root
            / "tt_metal_cache"
            / f"cache_{self.repacked_str}{self.model_config.model_name}"
        )

    @property
    def host_unrepacked_weights_dir(self) -> Path:
        return (
            self.model_volume_root / "model_weights" / f"{self.model_config.model_name}"
        )

    @property
    def host_weights_dir(self) -> Path:
        return (
            self.model_volume_root
            / "model_weights"
            / f"{self.repacked_str}{self.model_config.model_name}"
        )

    @property
    def model_weights_path(self) -> Path:
        # in docker container
        assert self.cache_root
        return (
            self.cache_root
            / "model_weights"
            / f"{self.repacked_str}{self.model_config.model_name}"
        )

    @property
    def model_tt_metal_cache_path(self) -> Path:
        # in docker container
        assert self.cache_root
        return (
            self.cache_root
            / "tt_metal_cache"
            / f"cache_{self.repacked_str}{self.model_config.model_name}"
        )


class HTTPError(Exception):
    pass


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
        raise HTTPError from e


class HostSetupManager:
    def __init__(
        self,
        model_config: ModelConfig,
        automatic: bool = False,
        jwt_secret: str = None,
        hf_token: str = None,
    ):
        self.model_config = model_config
        self.automatic = automatic
        self.setup_config = SetupConfig(model_config=model_config)
        self.setup_config.repo_root = str(Path(__file__).resolve().parent.parent)
        self.setup_config.persistent_volume_root = (
            Path(self.setup_config.repo_root) / "persistent_volume"
        )
        self.jwt_secret = jwt_secret
        self.hf_token = hf_token

    def _get_model_id(self) -> str:
        return f"id_{self.model_config.impl_id}-{self.model_config.model_name}-v{self.model_config.version}"

    def _get_env_file_path(self) -> Path:
        env_dir = self.setup_config.persistent_volume_root / "model_envs"
        env_dir.mkdir(parents=True, exist_ok=True)
        return env_dir / f"{self.model_config.model_name}.env"

    def check_setup(self) -> bool:
        env_file = self.setup_config.env_file
        if env_file.exists():
            self.load_env(load_setup=False)
            host_weights_dir = self.setup_config.host_weights_dir
            if host_weights_dir.exists() and any(host_weights_dir.iterdir()):
                logger.info(
                    f"✅ Setup already completed for model {self.model_config.model_name}."
                )
                return True
            logger.info(
                f"⚠️ Env file exists but weights directory is missing or empty for {self.model_config.model_name}."
            )
        return False

    def prompt_overwrite(self, path: Path) -> bool:
        if self.automatic:
            return True
        if path.exists():
            with path.open("r") as f:
                for line in f:
                    if line.startswith("MODEL_NAME="):
                        existing = line.split("=", 1)[1].strip()
                        logger.info(
                            f"Existing file {path} contains MODEL_NAME: {existing}"
                        )
                        break
            choice = input(f"Overwrite {path}? (y/n) [default: y]: ").strip() or "y"
            return choice.lower() in ("y", "yes")
        return True

    def check_disk_space(self) -> bool:
        total, used, free = shutil.disk_usage("/")
        free_gb = free // (1024**3)
        if free_gb >= self.model_config.min_disk_gb:
            logger.info(
                f"✅ Sufficient disk space: {free_gb}GB (Required: {self.model_config.min_disk_gb}GB)"
            )
            return True
        logger.error(
            f"❌ Insufficient disk space: {free_gb}GB (Required: {self.model_config.min_disk_gb}GB)"
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
        if available_gb >= self.model_config.min_ram_gb:
            logger.info(
                f"✅ Sufficient RAM: {int(available_gb)}GB (Required: {self.model_config.min_ram_gb}GB)"
            )
            return True
        logger.error(
            f"❌ Insufficient RAM: {int(available_gb)}GB (Required: {self.model_config.min_ram_gb}GB)"
        )
        return False

    def get_hf_env_vars(self):
        if self.automatic:
            self.hf_token = os.getenv("HF_TOKEN", "")
            if not self.check_hf_access(self.hf_token):
                logger.error("⛔ HF_TOKEN validation failed.")
                sys.exit(1)
            self.setup_config.host_hf_home = os.getenv(
                "CONTAINER_HF_HOME", str(Path.home() / ".cache" / "huggingface")
            )
            hf_home = Path(self.setup_config.host_hf_home)
            hf_home.mkdir(parents=True, exist_ok=True)
            assert os.access(hf_home, os.W_OK), "⛔ HOST_HF_HOME is not writable."
            return

        if not self.hf_token:
            self.hf_token = os.getenv("HF_TOKEN", "")

        if not self.hf_token and not self.automatic:
            self.hf_token = getpass.getpass("Enter your HF_TOKEN: ").strip()

        assert self.hf_token, "⛔ HF_TOKEN cannot be empty."

        if not self.setup_config.host_hf_home:
            default_hf_home = str(Path.home() / ".cache" / "huggingface")
            inp = (
                input(f"Enter host_hf_home [default: {default_hf_home}]: ").strip()
                or default_hf_home
            )
            hf_home = Path(inp)
            hf_home.mkdir(parents=True, exist_ok=True)
            if not os.access(hf_home, os.W_OK):
                logger.error("⛔ HOST_HF_HOME is not writable.")
                sys.exit(1)
            self.setup_config.host_hf_home = str(hf_home)

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
            f"https://huggingface.co/api/models/{self.model_config.hf_model_repo}"
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
        head_url = f"https://huggingface.co/{self.model_config.hf_model_repo}/resolve/main/{first_file}"
        _, _, head_headers = http_request(
            head_url, method="HEAD", headers={"Authorization": f"Bearer {token}"}
        )
        if head_headers.get("x-error-code"):
            logger.error("⛔ The model is gated and you don't have access.")
            return False
        logger.info("✅ HF_TOKEN is valid.")
        return True

    def setup_model_environment(self):
        if not self.check_disk_space() or not self.check_ram():
            sys.exit(1)
        if self.automatic:
            self.setup_config.persistent_volume_root = Path(
                os.environ.get(
                    "PERSISTENT_VOLUME_ROOT",
                    str(self.setup_config.persistent_volume_root),
                )
            )
        else:
            pv_input = input(
                f"Enter persistent_volume_root [default: {self.setup_config.persistent_volume_root}]: "
            ).strip()
            if pv_input:
                self.setup_config.persistent_volume_root = Path(pv_input)

        # Compute environment file path based on the selected model.
        self.setup_config.env_file.parent.mkdir(parents=True, exist_ok=True)

        if self.prompt_overwrite(self.setup_config.env_file):
            logger.info(f"Writing environment file to {self.setup_config.env_file}")
            if not self.automatic:
                print("\nHow do you want to provide a model?")
                print("1) Download from Hugging Face (default)")
                print("2) Local folder")
                choice = input("Enter your choice: ").strip() or "1"
            else:
                choice = os.environ.get("MODEL_SOURCE", "1")
            if choice == "1":
                self.setup_config.model_source = "huggingface"
                self.get_hf_env_vars()
            elif choice == "2":
                self.setup_config.model_source = "local"
                if self.automatic:
                    self.setup_config.llama_host_weights_dir = os.environ.get(
                        "LLAMA_DIR", ""
                    )
                    if not self.setup_config.llama_host_weights_dir:
                        logger.error(
                            "⛔ LLAMA_DIR environment variable is required for local model source in automatic mode."
                        )
                        sys.exit(1)
                else:
                    self.setup_config.llama_host_weights_dir = (
                        os.environ.get("LLAMA_DIR")
                        or input("Enter local Llama model directory: ").strip()
                    )
            else:
                logger.error("⛔ Invalid model source.")
                sys.exit(1)

            if not self.jwt_secret:
                self.jwt_secret = os.environ.get("JWT_SECRET", "")
            if not self.jwt_secret and not self.automatic:
                self.jwt_secret = getpass.getpass("Enter your JWT_SECRET: ").strip()

            assert self.jwt_secret, "⛔ JWT_SECRET cannot be empty."

            # Compute cache and weights paths on the fly.

            self.write_env_file(self.setup_config.env_file)
        else:
            logger.info(f"Using existing environment file {self.setup_config.env_file}")

    def write_env_file(self, env_file: Path):
        # Compute model-specific values on the fly.
        model_name = self.model_config.model_name
        hf_model_repo = self.model_config.hf_model_repo
        model_version = self.model_config.version
        env_vars = {
            "MODEL_SOURCE": self.setup_config.model_source,
            "MODEL_NAME": model_name,
            "MODEL_VERSION": model_version,
            "MODEL_ID": self._get_model_id(),
            # variables for user input
            "PERSISTENT_VOLUME_ROOT": self.setup_config.persistent_volume_root,
            "HOST_HF_HOME": self.setup_config.host_hf_home,
            # variables used in container
            "HF_MODEL_REPO_ID": hf_model_repo,
            "SERVICE_PORT": 7000,
            "CACHE_ROOT": self.setup_config.cache_root,
            "HF_HOME": f"{self.setup_config.cache_root}/huggingface",
            "MODEL_WEIGHTS_PATH": self.setup_config.model_weights_path,
            "LLAMA_DIR": self.setup_config.model_weights_path,
            "LLAMA3_CKPT_DIR": self.setup_config.model_weights_path,
            "LLAMA3_TOKENIZER_PATH": self.setup_config.model_weights_path
            / "tokenizer.model",
            "LLAMA3_CACHE_PATH": self.setup_config.model_tt_metal_cache_path,
            "JWT_SECRET": self.jwt_secret,
            "HF_TOKEN": self.hf_token,
        }

        with env_file.open("w") as f:
            f.write("# Environment variables for model setup\n")
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")

        logger.info(f"Environment file written: {env_file}")

    def load_env(self, load_config=True, load_setup=True):
        env_file = self.setup_config.env_file
        if not env_file.exists():
            logger.error(f"⛔ {env_file} not found. Run setup first.")
            sys.exit(1)

        load_keys_config = [
            "PERSISTENT_VOLUME_ROOT",
            "SERVICE_PORT",
            "HOST_HF_HOME",
        ]

        load_keys_setup = [
            "JWT_SECRET",
            "HF_TOKEN",
        ]
        with env_file.open("r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    if load_config and (key in load_keys_config):
                        setattr(self.setup_config, key.lower(), val)
                    if load_setup and (key in load_keys_setup):
                        setattr(self, key.lower(), val)
        self.setup_config.persistent_volume_root = Path(
            self.setup_config.persistent_volume_root
        )
        logger.info(f"Loaded environment from {env_file}")

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
        repack_url = "https://raw.githubusercontent.com/tenstorrent/tt-metal/refs/heads/main/models/demos/t3000/llama2_70b/scripts/repack_weights.py"
        data, status, _ = http_request(repack_url)
        if status != 200:
            logger.error("⛔ Failed to download repack_weights.py")
            sys.exit(1)
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
        if not (self.hf_token and self.setup_config.host_hf_home):
            logger.error("⛔ HF_TOKEN or HOST_HF_HOME not set.")
            sys.exit(1)
        venv_dir = Path(".venv_hf_setup")
        subprocess.run(["python3", "-m", "venv", str(venv_dir)], check=True)
        venv_python = venv_dir / "bin" / "python"
        subprocess.run(
            [
                str(venv_python),
                "-m",
                "pip",
                "install",
                "--upgrade",
                "pip",
                "setuptools",
                "wheel",
            ],
            check=True,
        )
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "huggingface_hub[cli]"],
            check=True,
        )
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"
        hf_repo = self.model_config.hf_model_repo
        hf_cli = venv_dir / "bin" / "huggingface-cli"
        if hf_repo in [
            "meta-llama/Llama-3.3-70B-Instruct",
            "meta-llama/Llama-3.3-70B",
            "meta-llama/Llama-3.1-70B-Instruct",
            "meta-llama/Llama-3.1-70B",
        ]:
            # fmt: off
            cmd = [
                str(hf_cli),
                "download", hf_repo,
                "original/params.json",
                "original/tokenizer.model",
                "original/consolidated.*",
                "--cache-dir", self.setup_config.host_hf_home,
                "--token", self.hf_token,
            ]
            # fmt: on
            repo_path_filter = "original/*"
        else:
            # use default huggingface repo
            # fmt: off
            cmd = [
                str(hf_cli), 
                "download", hf_repo, 
                "--exclude", "original/consolidated.*", 
                "--cache-dir", self.setup_config.host_hf_home,
                "--token", self.hf_token,
            ]
            # fmt: on
            repo_path_filter = "*"
        logger.info(f"Downloading model from Hugging Face: {hf_repo}")
        logger.info(f"Command: {shlex.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            logger.error(f"⛔ Error during: {' '.join(cmd)}")
            sys.exit(1)
        self.setup_config.host_weights_dir.mkdir(parents=True, exist_ok=True)
        self.setup_config.host_unrepacked_weights_dir.mkdir(parents=True, exist_ok=True)
        local_repo_name = hf_repo.replace("/", "--")
        snapshot_dir = (
            Path(self.setup_config.host_hf_home)
            / f"models--{local_repo_name}"
            / "snapshots"
        )
        if not snapshot_dir.is_dir():
            snapshot_dir = (
                Path(self.setup_config.host_hf_home)
                / "hub"
                / f"models--{local_repo_name}"
                / "snapshots"
            )
            if not snapshot_dir.is_dir():
                logger.error("⛔ Snapshot directory not found.")
                sys.exit(1)
        snapshots = list(snapshot_dir.glob("*"))
        if not snapshots:
            logger.error("⛔ No snapshot directories found.")
            sys.exit(1)
        most_recent = max(snapshots, key=lambda p: p.stat().st_mtime)
        for item in most_recent.glob(repo_path_filter):
            dest_item = self.setup_config.host_unrepacked_weights_dir / item.name
            if self.model_config.repacked == 1:
                try:
                    dest_item.symlink_to(item)
                except FileExistsError:
                    pass
            else:
                if item.is_symlink():
                    shutil.copy(item.resolve(), dest_item)
                elif item.is_dir():
                    shutil.copytree(item, dest_item, dirs_exist_ok=True)
                else:
                    shutil.copy(item, dest_item)
        if hf_repo == "meta-llama/Llama-3.2-11B-Vision-Instruct":
            old_path = self.setup_config.host_weights_dir / "consolidated.pth"
            new_path = self.setup_config.host_weights_dir / "consolidated.00.pth"
            if old_path.exists():
                old_path.rename(new_path)
        shutil.rmtree(str(venv_dir))
        if self.model_config.repacked == 1:
            self.repack_weights(
                self.setup_config.host_unrepacked_weights_dir,
                self.setup_config.host_weights_dir,
            )
        logger.info(f"✅ Using weights directory: {self.setup_config.host_weights_dir}")

    def setup_weights_local(self):
        persistent_host_weights_dir = self.setup_config.host_weights_dir
        persistent_host_weights_dir.mkdir(parents=True, exist_ok=True)
        for item in Path(self.setup_config.llama_host_weights_dir).glob("*"):
            dest_item = persistent_host_weights_dir / item.name
            if item.is_symlink():
                shutil.copy(item.resolve(), dest_item)
            elif item.is_dir():
                shutil.copytree(item, dest_item, dirs_exist_ok=True)
            else:
                shutil.copy(item, dest_item)
        logger.info(f"✅ Copied weights to: {persistent_host_weights_dir}")

    def setup_tt_metal_cache(self):
        self.setup_config.tt_metal_cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ tt_metal_cache set at: {self.setup_config.tt_metal_cache_dir}")

    def setup_weights(self):
        self.load_env()
        self.setup_tt_metal_cache()

        target_dir = self.setup_config.host_weights_dir

        if target_dir.exists() and any(target_dir.iterdir()):
            logger.info(f"Model weights already exist at {target_dir}")
        else:
            (self.setup_config.model_volume_root / "model_weights").mkdir(
                parents=True, exist_ok=True
            )
            if self.setup_config.model_source == "huggingface":
                self.setup_weights_huggingface()
            elif self.setup_config.model_source == "local":
                self.setup_weights_local()
            else:
                logger.error("⛔ Invalid model source.")
                sys.exit(1)
        logger.info("✅ done setup_weights")

    def run_setup(self):
        if self.check_setup():
            logger.info("Using existing setup.")
            return
        self.setup_model_environment()
        self.setup_weights()
        logger.info("✅ done run_setup")


def setup_host(model_name, jwt_secret, hf_token):
    model_config = MODEL_CONFIGS[model_name]
    manager = HostSetupManager(
        model_config=model_config, jwt_secret=jwt_secret, hf_token=hf_token
    )
    manager.run_setup()
    return manager.setup_config


def main():
    parser = argparse.ArgumentParser(description="Model setup script")
    parser.add_argument("model_name", help="Type of the model to setup")
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
    args = parser.parse_args()
    setup_host(
        model_name=args.model_name,
        jwt_secret=args.jwt_secret,
        hf_token=args.hf_token,
    )


if __name__ == "__main__":
    main()
