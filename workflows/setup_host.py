#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import argparse
import dataclasses
import getpass
import json
import logging
import os
import shutil
import subprocess
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Tuple, Dict

# Add the script's directory to the Python path
# this for 0 setup python setup script
project_root = Path(__file__).resolve().parent.parent
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))

from workflows.model_config import (
    MODEL_CONFIGS,
)  # expected to be a dict with model settings

logging.basicConfig(level=logging.INFO, format="%(message)s")


@dataclasses.dataclass
class SetupConfig:
    # Core configuration parameters (defaults provided where possible)
    HF_TOKEN: str = ""
    HF_HOME: str = ""
    MODEL_SOURCE: str = "huggingface"  # {huggingface, local}
    MODEL_VERSION: str = "0.0.1"
    MODEL_ID: str = ""
    REPO_ROOT: str = ""
    PERSISTENT_VOLUME_ROOT: Path = Path()
    ENV_FILE: Path = Path()
    TT_METAL_CACHE_DIR: Path = Path()
    WEIGHTS_DIR: Path = Path()
    # Model-specific configuration (populated from model_config)
    HF_MODEL_REPO_ID: str = ""
    MODEL_NAME: str = ""
    MIN_DISK: int = 10
    MIN_RAM: int = 8
    REPACKED: int = 0
    REPACKED_STR: str = ""
    IMPL_ID: str = "default"
    # Additional parameters
    JWT_SECRET: str = ""
    CACHE_ROOT: str = ""
    MODEL_WEIGHTS_PATH: str = ""
    LLAMA_WEIGHTS_DIR: str = ""


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
        logging.error(f"⛔ Error making {method} request to {url}: {e}")
        raise HTTPError from e


class HostSetupManager:
    def __init__(
        self, model_type: str, automatic: bool = False, jwt_secret: str = None
    ):
        self.model_type = model_type
        self.automatic = automatic
        self.config = SetupConfig()
        self.config.REPO_ROOT = str(Path(__file__).resolve().parent.parent)
        self.repo_root = Path(self.config.REPO_ROOT)
        self.config.PERSISTENT_VOLUME_ROOT = self.repo_root / "persistent_volume"
        self._apply_model_config()
        self.jwt_secret = jwt_secret

    def _apply_model_config(self):
        # Match the provided model_type against the keys in model_config.
        for key, cfg in MODEL_CONFIGS.items():
            if self.model_type.startswith(key):
                suffix = self.model_type[len(key) :]  # e.g., "-Instruct"
                self.config.MODEL_NAME = cfg.model_name + suffix
                self.config.HF_MODEL_REPO_ID = cfg.hf_model_repo + suffix
                self.config.MIN_DISK = int(cfg.min_disk_gb)
                self.config.MIN_RAM = int(cfg.min_ram_gb)
                self.config.REPACKED = int(cfg.repacked)
                self.config.IMPL_ID = cfg.impl_id
                return
        logging.error("⛔ Invalid model choice.")
        sys.exit(1)

    def check_setup(self) -> bool:
        env_file = self._get_env_file_path()
        if env_file.exists():
            weights_dir = (
                self.config.PERSISTENT_VOLUME_ROOT
                / "model_weights"
                / self.config.MODEL_NAME
            )
            if weights_dir.exists() and any(weights_dir.iterdir()):
                logging.info(
                    f"✅ Setup already completed for model {self.config.MODEL_NAME}."
                )
                return True
            logging.info(
                f"⚠️ Env file exists but weights directory is missing or empty for {self.config.MODEL_NAME}."
            )
        return False

    def _get_env_file_path(self) -> Path:
        env_dir = self.config.PERSISTENT_VOLUME_ROOT / "model_envs"
        env_dir.mkdir(parents=True, exist_ok=True)
        return env_dir / f"{self.config.MODEL_NAME}.env"

    def prompt_overwrite(self, path: Path) -> bool:
        if self.automatic:
            return True
        if path.exists():
            with path.open("r") as f:
                for line in f:
                    if line.startswith("MODEL_NAME="):
                        existing = line.split("=", 1)[1].strip()
                        logging.info(
                            f"Existing file {path} contains MODEL_NAME: {existing}"
                        )
                        break
            choice = input(f"Overwrite {path}? (y/n) [default: y]: ").strip() or "y"
            return choice.lower() in ("y", "yes")
        return True

    def check_disk_space(self) -> bool:
        total, used, free = shutil.disk_usage("/")
        free_gb = free // (1024**3)
        if free_gb >= self.config.MIN_DISK:
            logging.info(
                f"✅ Sufficient disk space: {free_gb}GB (Required: {self.config.MIN_DISK}GB)"
            )
            return True
        logging.error(
            f"❌ Insufficient disk space: {free_gb}GB (Required: {self.config.MIN_DISK}GB)"
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
                    logging.error("❌ Could not determine available RAM.")
                    return False
        except Exception as e:
            logging.error(f"❌ Error reading /proc/meminfo: {e}")
            return False
        if available_gb >= self.config.MIN_RAM:
            logging.info(
                f"✅ Sufficient RAM: {int(available_gb)}GB (Required: {self.config.MIN_RAM}GB)"
            )
            return True
        logging.error(
            f"❌ Insufficient RAM: {int(available_gb)}GB (Required: {self.config.MIN_RAM}GB)"
        )
        return False

    def get_hf_env_vars(self):
        if self.automatic:
            self.config.HF_TOKEN = os.environ.get("HF_TOKEN", "")
            if self.check_hf_access(self.config.HF_TOKEN) != 0:
                logging.error("⛔ HF_TOKEN validation failed.")
                sys.exit(1)
            self.config.HF_HOME = os.environ.get(
                "HF_HOME", str(Path.home() / ".cache" / "huggingface")
            )
            hf_home = Path(self.config.HF_HOME)
            hf_home.mkdir(parents=True, exist_ok=True)
            if not os.access(hf_home, os.W_OK):
                logging.error("⛔ HF_HOME is not writable.")
                sys.exit(1)
            return
        # Interactive mode
        if not self.config.HF_TOKEN:
            token = getpass.getpass("Enter your HF_TOKEN: ").strip()
            if self.check_hf_access(token) != 0:
                logging.error("⛔ HF_TOKEN validation failed.")
                sys.exit(1)
            self.config.HF_TOKEN = token
        if not self.config.HF_HOME:
            default_hf_home = str(Path.home() / ".cache" / "huggingface")
            inp = (
                input(f"Enter HF_HOME [default: {default_hf_home}]: ").strip()
                or default_hf_home
            )
            hf_home = Path(inp)
            hf_home.mkdir(parents=True, exist_ok=True)
            if not os.access(hf_home, os.W_OK):
                logging.error("⛔ HF_HOME is not writable.")
                sys.exit(1)
            self.config.HF_HOME = str(hf_home)

    def check_hf_access(self, token: str) -> int:
        if not token or not token.startswith("hf_"):
            logging.error("⛔ Invalid HF_TOKEN.")
            return 1
        data, status, _ = http_request(
            "https://huggingface.co/api/whoami-v2",
            headers={"Authorization": f"Bearer {token}"},
        )
        if status != 200:
            logging.error("⛔ HF_TOKEN rejected by Hugging Face.")
            return 2
        model_url = f"https://huggingface.co/api/models/{self.config.HF_MODEL_REPO_ID}"
        data, status, _ = http_request(
            model_url, headers={"Authorization": f"Bearer {token}"}
        )
        try:
            json_data = json.loads(data.decode("utf-8"))
        except json.JSONDecodeError:
            logging.error("⛔ Invalid JSON response from Hugging Face.")
            sys.exit(1)
        siblings = json_data.get("siblings", [])
        if not siblings:
            logging.error("⛔ No files found in repository.")
            sys.exit(1)
        first_file = siblings[0].get("rfilename")
        if not first_file:
            logging.error("⛔ Unexpected repository structure.")
            sys.exit(1)
        head_url = f"https://huggingface.co/{self.config.HF_MODEL_REPO_ID}/resolve/main/{first_file}"
        _, _, head_headers = http_request(
            head_url, method="HEAD", headers={"Authorization": f"Bearer {token}"}
        )
        if head_headers.get("x-error-code"):
            logging.error("⛔ The model is gated and you don't have access.")
            return 3
        logging.info("✅ HF_TOKEN is valid.")
        return 0

    def setup_model_environment(self):
        if not self.check_disk_space() or not self.check_ram():
            sys.exit(1)
        # Use env var or default for persistent volume root in automatic mode
        if self.automatic:
            self.config.PERSISTENT_VOLUME_ROOT = Path(
                os.environ.get(
                    "PERSISTENT_VOLUME_ROOT", str(self.config.PERSISTENT_VOLUME_ROOT)
                )
            )
        else:
            pv_input = input(
                f"Enter PERSISTENT_VOLUME_ROOT [default: {self.config.PERSISTENT_VOLUME_ROOT}]: "
            ).strip()
            if pv_input:
                self.config.PERSISTENT_VOLUME_ROOT = Path(pv_input)
        self.config.MODEL_VERSION = "0.0.1"
        self.config.MODEL_ID = f"id_{self.config.IMPL_ID}-{self.config.MODEL_NAME}-v{self.config.MODEL_VERSION}"
        # Set up the environment file path.
        env_file = self._get_env_file_path()
        self.config.ENV_FILE = env_file

        if self.prompt_overwrite(env_file):
            logging.info(f"Writing environment file to {env_file}")
            if not self.automatic:
                print("\nHow do you want to provide a model?")
                print("1) Download from Hugging Face (default)")
                print("2) Local folder")
                choice = input("Enter your choice: ").strip() or "1"
            else:
                choice = os.environ.get("MODEL_SOURCE", "1")
            if choice == "1":
                self.config.MODEL_SOURCE = "huggingface"
                self.get_hf_env_vars()
            elif choice == "2":
                self.config.MODEL_SOURCE = "local"
                if self.automatic:
                    self.config.LLAMA_WEIGHTS_DIR = os.environ.get("LLAMA_DIR", "")
                    if not self.config.LLAMA_WEIGHTS_DIR:
                        logging.error(
                            "⛔ LLAMA_DIR environment variable is required for local model source in automatic mode."
                        )
                        sys.exit(1)
                else:
                    self.config.LLAMA_WEIGHTS_DIR = (
                        os.environ.get("LLAMA_DIR")
                        or input("Enter local Llama model directory: ").strip()
                    )
            else:
                logging.error("⛔ Invalid model source.")
                sys.exit(1)
            if self.automatic:
                jwt_secret = os.environ.get("JWT_SECRET", self.jwt_secret)
            else:
                jwt_secret = getpass.getpass("Enter your JWT_SECRET: ").strip()
            if not jwt_secret:
                logging.error("⛔ JWT_SECRET cannot be empty.")
                sys.exit(1)
            self.config.JWT_SECRET = jwt_secret
            self.config.REPACKED_STR = "repacked-" if self.config.REPACKED == 1 else ""
            # Set additional paths used later (e.g. CACHE_ROOT, MODEL_WEIGHTS_PATH)
            self.config.CACHE_ROOT = str(Path("/home/container_app_user/cache_root"))
            self.config.MODEL_WEIGHTS_PATH = str(
                Path(self.config.CACHE_ROOT)
                / "model_weights"
                / f"{self.config.REPACKED_STR}{self.config.MODEL_NAME}"
            )
            self.write_env_file(env_file)
        else:
            logging.info(f"Using existing environment file {env_file}")

    def write_env_file(self, env_file: Path):
        with env_file.open("w") as f:
            f.write("# Environment variables for model setup\n")
            f.write(f"MODEL_SOURCE={self.config.MODEL_SOURCE}\n")
            f.write(f"HF_MODEL_REPO_ID={self.config.HF_MODEL_REPO_ID}\n")
            f.write(f"MODEL_NAME={self.config.MODEL_NAME}\n")
            f.write(f"MODEL_VERSION={self.config.MODEL_VERSION}\n")
            f.write(f"IMPL_ID={self.config.IMPL_ID}\n")
            f.write(f"MODEL_ID={self.config.MODEL_ID}\n")
            f.write(f"REPACKED={self.config.REPACKED}\n")
            f.write(f"REPACKED_STR={self.config.REPACKED_STR}\n")
            f.write("SERVICE_PORT=7000\n")
            f.write(f"HOST_HF_HOME={self.config.HF_HOME}\n")
            f.write(f"PERSISTENT_VOLUME_ROOT={self.config.PERSISTENT_VOLUME_ROOT}\n")
            f.write(f"CACHE_ROOT={self.config.CACHE_ROOT}\n")
            f.write(f"HF_HOME={self.config.CACHE_ROOT}/huggingface\n")
            f.write(f"MODEL_WEIGHTS_PATH={self.config.MODEL_WEIGHTS_PATH}\n")
            f.write(f"LLAMA_DIR={self.config.MODEL_WEIGHTS_PATH}\n")
            f.write(f"JWT_SECRET={self.config.JWT_SECRET}\n")
            f.write(f"HF_TOKEN={self.config.HF_TOKEN}\n")
        logging.info(f"Environment file written: {env_file}")

    def load_env(self):
        env_file = self.config.ENV_FILE
        if not env_file.exists():
            logging.error(f"⛔ {env_file} not found. Run setup first.")
            sys.exit(1)
        with env_file.open("r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    setattr(self.config, key, val)
        self.config.PERSISTENT_VOLUME_ROOT = Path(self.config.PERSISTENT_VOLUME_ROOT)
        logging.info(f"Loaded environment from {env_file}")

    def repack_weights(self, source_dir: Path, target_dir: Path):
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(source_dir / "tokenizer.model", target_dir / "tokenizer.model")
        shutil.copy(source_dir / "params.json", target_dir / "params.json")
        # Set up a temporary venv for repacking
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
            logging.error("⛔ Failed to download repack_weights.py")
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
        logging.info("✅ Weight repacking completed.")

    def setup_weights_huggingface(self):
        if not (self.config.HF_TOKEN and self.config.HF_HOME):
            logging.error("⛔ HF_TOKEN or HF_HOME not set.")
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
        hf_repo = self.config.HF_MODEL_REPO_ID
        hf_cli = venv_dir / "bin" / "huggingface-cli"
        if hf_repo in [
            "Qwen/Qwen2.5-72B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        ]:
            cmd = [str(hf_cli), "download", hf_repo]
            repo_path_filter = "*"
        else:
            cmd = [
                str(hf_cli),
                "download",
                hf_repo,
                "original/params.json",
                "original/tokenizer.model",
                "original/consolidated.*",
                "--cache-dir",
                self.config.HF_HOME,
                "--token",
                self.config.HF_TOKEN,
            ]
            repo_path_filter = "original/*"
        result = subprocess.run(cmd)
        if result.returncode != 0:
            logging.error(f"⛔ Error during: {' '.join(cmd)}")
            sys.exit(1)
        persistent_weights_dir = (
            self.config.PERSISTENT_VOLUME_ROOT
            / "model_weights"
            / self.config.MODEL_NAME
        )
        persistent_weights_dir.mkdir(parents=True, exist_ok=True)
        local_repo_name = hf_repo.replace("/", "--")
        snapshot_dir = (
            Path(self.config.HF_HOME) / f"models--{local_repo_name}" / "snapshots"
        )
        if not snapshot_dir.is_dir():
            snapshot_dir = (
                Path(self.config.HF_HOME)
                / "hub"
                / f"models--{local_repo_name}"
                / "snapshots"
            )
            if not snapshot_dir.is_dir():
                logging.error("⛔ Snapshot directory not found.")
                sys.exit(1)
        snapshots = list(snapshot_dir.glob("*"))
        if not snapshots:
            logging.error("⛔ No snapshot directories found.")
            sys.exit(1)
        most_recent = max(snapshots, key=lambda p: p.stat().st_mtime)
        for item in most_recent.glob(repo_path_filter):
            dest_item = persistent_weights_dir / item.name
            if self.config.REPACKED == 1:
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
            old_path = persistent_weights_dir / "consolidated.pth"
            new_path = persistent_weights_dir / "consolidated.00.pth"
            if old_path.exists():
                old_path.rename(new_path)
        shutil.rmtree(str(venv_dir))
        if self.config.REPACKED == 1:
            repacked_dir = (
                self.config.PERSISTENT_VOLUME_ROOT
                / "model_weights"
                / f"{self.config.REPACKED_STR}{self.config.MODEL_NAME}"
            )
            repacked_dir.mkdir(parents=True, exist_ok=True)
            self.repack_weights(persistent_weights_dir, repacked_dir)
            persistent_weights_dir = repacked_dir
        self.config.WEIGHTS_DIR = persistent_weights_dir
        logging.info(f"Using weights directory: {persistent_weights_dir}")

    def setup_weights_local(self):
        persistent_weights_dir = (
            self.config.PERSISTENT_VOLUME_ROOT
            / "model_weights"
            / self.config.MODEL_NAME
        )
        persistent_weights_dir.mkdir(parents=True, exist_ok=True)
        for item in Path(self.config.LLAMA_WEIGHTS_DIR).glob("*"):
            dest_item = persistent_weights_dir / item.name
            if item.is_symlink():
                shutil.copy(item.resolve(), dest_item)
            elif item.is_dir():
                shutil.copytree(item, dest_item, dirs_exist_ok=True)
            else:
                shutil.copy(item, dest_item)
        self.config.WEIGHTS_DIR = persistent_weights_dir
        logging.info(f"Copied weights to: {persistent_weights_dir}")

    def setup_tt_metal_cache(self):
        cache_dir = (
            self.config.PERSISTENT_VOLUME_ROOT
            / "tt_metal_cache"
            / f"cache_{self.config.REPACKED_STR}{self.config.MODEL_NAME}"
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.config.TT_METAL_CACHE_DIR = cache_dir
        logging.info(f"✅ tt_metal_cache set at: {cache_dir}")

    def setup_weights(self):
        self.load_env()
        self.setup_tt_metal_cache()
        target_dir = (
            self.config.PERSISTENT_VOLUME_ROOT
            / "model_weights"
            / f"{self.config.REPACKED_STR}{self.config.MODEL_NAME}"
        )
        if target_dir.exists() and any(target_dir.iterdir()):
            logging.info(f"Model weights already exist at {target_dir}")
        else:
            (self.config.PERSISTENT_VOLUME_ROOT / "model_weights").mkdir(
                parents=True, exist_ok=True
            )
            if self.config.MODEL_SOURCE == "huggingface":
                self.setup_weights_huggingface()
            elif self.config.MODEL_SOURCE == "local":
                self.setup_weights_local()
            else:
                logging.error("⛔ Invalid model source.")
                sys.exit(1)

    def run_setup(self):
        if self.check_setup():
            logging.info("Using existing setup. Exiting.")
            sys.exit(0)
        self.setup_model_environment()
        self.setup_weights()


def setup_host(model_type):
    manager = HostSetupManager(model_type)
    manager.run_setup()


def main():
    parser = argparse.ArgumentParser(description="Model setup script")
    parser.add_argument("model_type", help="Type of the model to setup")
    parser.add_argument(
        "--automatic",
        action="store_true",
        help="Bypass interactive prompts by using environment variables or defaults",
    )
    args = parser.parse_args()
    manager = HostSetupManager(args.model_type, automatic=args.automatic)
    manager.run_setup()


if __name__ == "__main__":
    main()
