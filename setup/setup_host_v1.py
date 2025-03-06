#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import argparse
import getpass
import json
import logging
import shutil
import subprocess
import sys
import urllib.request
import urllib.error
import os
from pathlib import Path

from setup.configs import model_config

logging.basicConfig(level=logging.INFO, format="%(message)s")


class HostSetupManager:
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.config = {}
        self.repo_root = Path(__file__).resolve().parent
        self.config["REPO_ROOT"] = str(self.repo_root)
        self.persistent_volume_root = self.repo_root / "persistent_volume"
        self.config["PERSISTENT_VOLUME_ROOT"] = str(self.persistent_volume_root)

    @staticmethod
    def usage() -> None:
        models_str = "\n  ".join(model_config.keys())
        usage_str = f"Usage: {sys.argv[0]} <model_type>\n\nAvailable model types:\n  {models_str}"
        logging.info(usage_str)
        sys.exit(1)

    def check_setup(self) -> bool:
        """
        Checks if setup is already done for the given model by verifying if the
        environment file exists and if the model weights directory is populated.
        """
        base_config = None
        for key, cfg in model_config.items():
            if self.model_type.startswith(key):
                base_config = cfg.copy()
                suffix = self.model_type[
                    len(key) :
                ]  # capture any extra suffix (e.g. "-Instruct")
                base_config["MODEL_NAME"] += suffix
                base_config["HF_MODEL_REPO_ID"] += suffix
                break

        if base_config is None:
            logging.error("Invalid model choice.")
            self.usage()

        model_env_dir = self.persistent_volume_root / "model_envs"
        env_file = model_env_dir / f"{base_config['MODEL_NAME']}.env"

        if env_file.exists():
            weights_dir = (
                self.persistent_volume_root
                / "model_weights"
                / base_config["MODEL_NAME"]
            )
            if weights_dir.exists() and any(weights_dir.iterdir()):
                logging.info(
                    f"✅ Setup already completed for model {base_config['MODEL_NAME']}."
                )
                return True
            else:
                logging.info(
                    f"⚠️ Environment file exists but weights directory is missing or empty for {base_config['MODEL_NAME']}."
                )
        return False

    def http_get(self, url: str, headers: dict = None):
        headers = headers or {}
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.read(), resp.getcode(), resp.headers
        except urllib.error.HTTPError as e:
            return e.read(), e.getcode(), e.headers
        except Exception as e:
            logging.error(f"⛔ Error making GET request to {url}: {e}")
            sys.exit(1)

    def http_head(self, url: str, headers: dict = None):
        headers = headers or {}
        req = urllib.request.Request(url, headers=headers, method="HEAD")
        try:
            with urllib.request.urlopen(req) as resp:
                return resp.read(), resp.getcode(), resp.headers
        except urllib.error.HTTPError as e:
            return e.read(), e.getcode(), e.headers
        except Exception as e:
            logging.error(f"⛔ Error making HEAD request to {url}: {e}")
            sys.exit(1)

    def check_and_prompt_env_file(self, env_file: Path) -> bool:
        if env_file.exists():
            logging.info(f"Found ENV_FILE: {env_file}")
            model_name = ""
            with env_file.open("r") as f:
                for line in f:
                    if line.startswith("MODEL_NAME="):
                        model_name = line.split("=", 1)[1].strip()
                        break
            if model_name:
                logging.info(
                    f"The existing file {env_file} contains MODEL_NAME: {model_name}"
                )
                choice = (
                    input(
                        f"Do you want to overwrite the existing file {env_file}? (y/n) [default: y]: "
                    ).strip()
                    or "y"
                )
                if choice.lower() in ("y", "yes"):
                    logging.info(f"Overwriting the {env_file} file ...")
                    return True
                elif choice.lower() in ("n", "no"):
                    return False
                else:
                    logging.error("⛔ Invalid option. Exiting.")
                    sys.exit(1)
            else:
                logging.info(f"MODEL_NAME not found in {env_file}. Overwriting.")
                return True
        else:
            logging.info(f"{env_file} does not exist. Proceeding to create a new one.")
            return True

    def check_hf_access(self, token: str, repo_id: str) -> int:
        if not token:
            logging.error("⛔ HF_TOKEN cannot be empty. Please try again.")
            return 1
        if not token.startswith("hf_"):
            logging.error("⛔ HF_TOKEN must start with 'hf_'. Please try again.")
            return 1

        whoami_url = "https://huggingface.co/api/whoami-v2"
        try:
            data, status_code, _ = self.http_get(
                whoami_url, headers={"Authorization": f"Bearer {token}"}
            )
        except Exception as e:
            logging.error(f"⛔ Error connecting to Hugging Face: {e}")
            return 2

        if status_code != 200:
            logging.error(
                "⛔ HF_TOKEN rejected by 🤗 Hugging Face. Please check your token and try again."
            )
            return 2

        model_url = f"https://huggingface.co/api/models/{repo_id}"
        data, status_code, _ = self.http_get(
            model_url, headers={"Authorization": f"Bearer {token}"}
        )
        try:
            json_data = json.loads(data.decode("utf-8"))
        except json.JSONDecodeError:
            logging.error("⛔ Could not decode JSON from Hugging Face response.")
            sys.exit(1)
        siblings = json_data.get("siblings", [])
        if not siblings:
            logging.error(
                f"⛔ No files found in the model repository. HF_MODEL_REPO_ID={repo_id}. Does your HF_TOKEN have access?"
            )
            sys.exit(1)
        first_file = siblings[0].get("rfilename")
        if not first_file:
            logging.error("⛔ Unexpected repository structure. Exiting.")
            sys.exit(1)

        head_url = f"https://huggingface.co/{repo_id}/resolve/main/{first_file}"
        _, head_status, head_headers = self.http_head(
            head_url, headers={"Authorization": f"Bearer {token}"}
        )
        if "x-error-code" in head_headers and head_headers["x-error-code"]:
            logging.error("⛔ The model is gated and you don't have access.")
            return 3

        logging.info("✅ HF_TOKEN is valid and has access to the model.")
        return 0

    def check_disk_space(self, min_disk: int) -> bool:
        total, used, free = shutil.disk_usage("/")
        free_gb = free // (1024**3)
        if free_gb >= min_disk:
            logging.info(
                f"✅ Sufficient disk space available: {free_gb}GB, Required: {min_disk}GB"
            )
            return True
        else:
            logging.error(
                f"❌ Insufficient disk space! Available: {free_gb}GB, Required: {min_disk}GB"
            )
            return False

    def check_ram(self, min_ram: int) -> bool:
        try:
            with Path("/proc/meminfo").open("r") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        available_kb = int(line.split()[1])
                        available_gb = available_kb / (1024 * 1024)
                        break
                else:
                    logging.error(
                        "❌ Could not determine available RAM from /proc/meminfo."
                    )
                    return False
        except Exception as e:
            logging.error(f"❌ Error reading /proc/meminfo: {e}")
            return False

        if available_gb >= min_ram:
            logging.info(
                f"✅ Sufficient RAM available: {int(available_gb)}GB, Required: {min_ram}GB"
            )
            return True
        else:
            logging.error(
                f"❌ Insufficient RAM! Available: {int(available_gb)}GB, Required: {min_ram}GB"
            )
            return False

    def get_hf_env_vars(self):
        if not self.config.get("HF_TOKEN"):
            token = getpass.getpass("Enter your HF_TOKEN: ").strip()
            if self.check_hf_access(token, self.config["HF_MODEL_REPO_ID"]) != 0:
                logging.error(
                    "⛔ Error occurred during HF_TOKEN validation. Please check the token and try again."
                )
                sys.exit(1)
            self.config["HF_TOKEN"] = token
            logging.info("✅ HF_TOKEN set.")
        if not self.config.get("HF_HOME"):
            default_hf_home = str(Path.home() / ".cache" / "huggingface")
            inp = input(
                f"HF_HOME environment variable is not set. Enter your HF_HOME [default: {default_hf_home}]: "
            ).strip()
            hf_home = inp or default_hf_home
            hf_home_path = Path(hf_home)
            try:
                hf_home_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logging.error(
                    "⛔ Failed to create HF_HOME directory. Please check permissions and try again."
                )
                sys.exit(1)
            if not os.access(hf_home_path, os.W_OK):
                logging.error(
                    "⛔ HF_HOME must be a valid directory and writable by the user. Please try again."
                )
                sys.exit(1)
            self.config["HF_HOME"] = hf_home
            logging.info("✅ HF_HOME set.")

    def setup_model_environment(self):
        base_config = None
        for key, cfg in model_config.items():
            if self.model_type.startswith(key):
                base_config = cfg.copy()
                suffix = self.model_type[len(key) :]
                base_config["MODEL_NAME"] += suffix
                base_config["HF_MODEL_REPO_ID"] += suffix
                break

        if base_config is None:
            logging.error("⛔ Invalid model choice.")
            self.usage()

        self.config.update(base_config)

        if not self.check_disk_space(int(self.config["MIN_DISK"])):
            sys.exit(1)
        if not self.check_ram(int(self.config["MIN_RAM"])):
            sys.exit(1)

        default_persistent_root = self.persistent_volume_root
        inp = input(
            f"Enter your PERSISTENT_VOLUME_ROOT [default: {default_persistent_root}]: "
        ).strip()
        self.config["PERSISTENT_VOLUME_ROOT"] = inp or str(default_persistent_root)

        self.config["MODEL_VERSION"] = "0.0.1"
        self.config["MODEL_ID"] = (
            f"id_{self.config['IMPL_ID']}-{self.config['MODEL_NAME']}-v{self.config['MODEL_VERSION']}"
        )
        self.config["PERSISTENT_VOLUME"] = str(
            Path(self.config["PERSISTENT_VOLUME_ROOT"])
            / ("volume_" + self.config["MODEL_ID"])
        )

        model_env_dir = Path(self.config["PERSISTENT_VOLUME_ROOT"]) / "model_envs"
        model_env_dir.mkdir(parents=True, exist_ok=True)
        env_file = model_env_dir / f"{self.config['MODEL_NAME']}.env"
        self.config["ENV_FILE"] = str(env_file)

        overwrite_env = self.check_and_prompt_env_file(env_file)
        if not overwrite_env:
            logging.info(f"✅ Using existing .env file: {env_file}.")
            return

        print("\nHow do you want to provide a model?")
        print("1) Download from 🤗 Hugging Face (default)")
        print("2) Local folder")
        choice = input("Enter your choice: ").strip() or "1"
        self.config["MODEL_SOURCE"] = choice

        if choice == "1":
            logging.info("Using 🤗 Hugging Face Token.")
            self.get_hf_env_vars()
        elif choice == "2":
            if "LLAMA_DIR" in os.environ:
                self.config["LLAMA_WEIGHTS_DIR"] = os.environ["LLAMA_DIR"]
            else:
                inp = input("Provide the path to the Llama model directory: ").strip()
                self.config["LLAMA_WEIGHTS_DIR"] = inp
            logging.info(f"Using local folder: {self.config.get('LLAMA_WEIGHTS_DIR')}")
        else:
            logging.error("⛔ Invalid option. Exiting.")
            sys.exit(1)

        jwt_secret = getpass.getpass("Enter your JWT_SECRET: ").strip()
        if not jwt_secret:
            logging.error("⛔ JWT_SECRET cannot be empty. Please try again.")
            sys.exit(1)
        self.config["JWT_SECRET"] = jwt_secret

        self.config["REPACKED_STR"] = (
            "repacked-" if int(self.config["REPACKED"]) == 1 else ""
        )
        self.config["CONTAINER_APP_USERNAME"] = "container_app_user"
        self.config["CONTAINER_HOME"] = f"/home/{self.config['CONTAINER_APP_USERNAME']}"
        self.config["CACHE_ROOT"] = str(
            Path(self.config["CONTAINER_HOME"]) / "cache_root"
        )
        self.config["MODEL_WEIGHTS_PATH"] = str(
            Path(self.config["CACHE_ROOT"])
            / "model_weights"
            / f"{self.config['REPACKED_STR']}{self.config['MODEL_NAME']}"
        )

        logging.info(f"Writing environment variables to {env_file} ...")
        with env_file.open("w") as f:
            f.write("# Environment variables for the model setup\n")
            f.write(f"MODEL_SOURCE={self.config['MODEL_SOURCE']}\n")
            f.write(f"HF_MODEL_REPO_ID={self.config['HF_MODEL_REPO_ID']}\n")
            f.write(f"MODEL_NAME={self.config['MODEL_NAME']}\n")
            f.write(f"MODEL_VERSION={self.config['MODEL_VERSION']}\n")
            f.write(f"IMPL_ID={self.config['IMPL_ID']}\n")
            f.write(f"MODEL_ID={self.config['MODEL_ID']}\n")
            f.write(f"REPACKED={self.config['REPACKED']}\n")
            f.write(f"REPACKED_STR={self.config['REPACKED_STR']}\n")
            f.write("# model runtime variables\n")
            f.write("SERVICE_PORT=7000\n")
            f.write(f"HOST_HF_HOME={self.config.get('HF_HOME','')}\n")
            f.write(f"LLAMA_REPO={self.config.get('LLAMA_REPO','')}\n")
            f.write(f"LLAMA_MODELS_DIR={self.config.get('LLAMA_MODELS_DIR','')}\n")
            f.write(f"LLAMA_WEIGHTS_DIR={self.config.get('LLAMA_WEIGHTS_DIR','')}\n")
            f.write(f"PERSISTENT_VOLUME_ROOT={self.config['PERSISTENT_VOLUME_ROOT']}\n")
            f.write(f"PERSISTENT_VOLUME={self.config['PERSISTENT_VOLUME']}\n")
            f.write(f"CACHE_ROOT={self.config['CACHE_ROOT']}\n")
            f.write(f"HF_HOME={self.config['CACHE_ROOT']}/huggingface\n")
            f.write(
                f"MODEL_WEIGHTS_ID=id_{self.config['REPACKED_STR']}{self.config['MODEL_NAME']}\n"
            )
            f.write(f"MODEL_WEIGHTS_PATH={self.config['MODEL_WEIGHTS_PATH']}\n")
            f.write(f"LLAMA_DIR={self.config['MODEL_WEIGHTS_PATH']}\n")
            f.write(f"LLAMA3_CKPT_DIR={self.config['MODEL_WEIGHTS_PATH']}\n")
            f.write(
                f"LLAMA3_TOKENIZER_PATH={self.config['MODEL_WEIGHTS_PATH']}/tokenizer.model\n"
            )
            f.write(
                f"LLAMA3_CACHE_PATH={self.config['CACHE_ROOT']}/tt_metal_cache/cache_{self.config['REPACKED_STR']}{self.config['MODEL_NAME']}\n"
            )
            f.write(
                "# These are secrets and must be stored securely for production environments\n"
            )
            f.write(f"JWT_SECRET={self.config['JWT_SECRET']}\n")
            f.write(f"HF_TOKEN={self.config.get('HF_TOKEN','')}\n")
        logging.info(f"Environment variables written to: {env_file}")
        logging.info("✅ setup_model_environment completed!")

    def load_env(self):
        env_file = Path(self.config.get("ENV_FILE"))
        if not env_file.exists():
            logging.error(f"⛔ {env_file} file not found. Please run the setup first.")
            sys.exit(1)
        with env_file.open("r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                self.config[key] = val
        logging.info(f"Sourced environment variables from {env_file}")

    def repack_weights(self, source_dir: str, target_dir: str):
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(
            Path(source_dir) / "tokenizer.model", target_path / "tokenizer.model"
        )
        shutil.copy(Path(source_dir) / "params.json", target_path / "params.json")

        venv_dir = Path(".venv_repack")
        logging.info(f"Setting up python venv for repacking: {venv_dir}")
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
        logging.info("Downloading repack_weights.py ...")
        data, status_code, _ = self.http_get(repack_url)
        if status_code != 200:
            logging.error("⛔ Failed to download repack_weights.py")
            sys.exit(1)
        repack_script = Path("repack_weights.py")
        with repack_script.open("wb") as f:
            f.write(data)

        logging.info("Repacking weights...")
        subprocess.run(
            [str(venv_python), str(repack_script), source_dir, target_dir, "5"],
            check=True,
        )

        shutil.rmtree(str(venv_dir))
        repack_script.unlink()
        logging.info("✅ Weight repacking completed!")

    def setup_weights_huggingface(self):
        if not self.config.get("HF_TOKEN") or not self.config.get("HF_HOME"):
            logging.error(
                "⛔ HF_TOKEN or HF_HOME not set. Please ensure both environment variables are set."
            )
            sys.exit(1)

        venv_dir = Path(".venv_hf_setup")
        logging.info(f"Setting up python venv for Hugging Face downloads: {venv_dir}")
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

        # Set a download timeout for the hub
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"

        hf_repo = self.config["HF_MODEL_REPO_ID"]
        if hf_repo in [
            "Qwen/Qwen2.5-72B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        ]:
            hf_repo_path_filter = "*"
            cmd = ["huggingface-cli", "download", hf_repo]
        else:
            hf_repo_path_filter = "original/*"
            cmd = [
                "huggingface-cli",
                "download",
                hf_repo,
                "original/params.json",
                "original/tokenizer.model",
                "original/consolidated.*",
                "--cache-dir",
                self.config["HF_HOME"],
                "--token",
                self.config["HF_TOKEN"],
            ]
        logging.info("Downloading model from Hugging Face Hub...")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            logging.error(f"⛔ Error occurred during: {' '.join(cmd)}")
            logging.error(
                "🔔 Check for common issues such as 401 Unauthorized and correct HF_TOKEN in the .env file."
            )
            sys.exit(1)

        persistent_weights_dir = (
            Path(self.config["PERSISTENT_VOLUME"])
            / "model_weights"
            / self.config["MODEL_NAME"]
        )
        persistent_weights_dir.mkdir(parents=True, exist_ok=True)
        local_repo_name = hf_repo.replace("/", "--")
        snapshot_dir = (
            Path(self.config["HF_HOME"]) / f"models--{local_repo_name}" / "snapshots"
        )
        if not snapshot_dir.is_dir():
            snapshot_dir = (
                Path(self.config["HF_HOME"])
                / "hub"
                / f"models--{local_repo_name}"
                / "snapshots"
            )
            if not snapshot_dir.is_dir():
                logging.error("⛔ Snapshot directory not found at expected locations.")
                sys.exit(1)
            logging.info(f"Using alternative snapshot directory: {snapshot_dir}")

        snapshots = list(snapshot_dir.glob("*"))
        if not snapshots:
            logging.error("⛔ No snapshot directories found.")
            sys.exit(1)
        most_recent = max(snapshots, key=lambda p: p.stat().st_mtime)

        for item in most_recent.glob(hf_repo_path_filter):
            dest_item = persistent_weights_dir / item.name
            if int(self.config["REPACKED"]) == 1:
                logging.info(
                    f"Creating symlink for: {item} in {persistent_weights_dir}"
                )
                try:
                    dest_item.symlink_to(item)
                except FileExistsError:
                    pass
            else:
                logging.info(f"Copying {item} to {persistent_weights_dir} ...")
                if item.is_symlink():
                    real_path = item.resolve()
                    shutil.copy(real_path, dest_item)
                else:
                    if item.is_dir():
                        shutil.copytree(item, dest_item, dirs_exist_ok=True)
                    else:
                        shutil.copy(item, dest_item)

        if hf_repo == "meta-llama/Llama-3.2-11B-Vision-Instruct":
            old_path = persistent_weights_dir / "consolidated.pth"
            new_path = persistent_weights_dir / "consolidated.00.pth"
            if old_path.exists():
                old_path.rename(new_path)

        shutil.rmtree(str(venv_dir))

        if int(self.config["REPACKED"]) == 1:
            repacked_dir = (
                Path(self.config["PERSISTENT_VOLUME"])
                / "model_weights"
                / f"{self.config['REPACKED_STR']}{self.config['MODEL_NAME']}"
            )
            repacked_dir.mkdir(parents=True, exist_ok=True)
            self.repack_weights(str(persistent_weights_dir), str(repacked_dir))
            persistent_weights_dir = repacked_dir

        self.config["WEIGHTS_DIR"] = str(persistent_weights_dir)
        logging.info(f"Using weights directory: {persistent_weights_dir}")
        logging.info("✅ setup_weights_huggingface completed!")

    def setup_weights_local(self):
        persistent_weights_dir = (
            Path(self.config["PERSISTENT_VOLUME"])
            / "model_weights"
            / self.config["MODEL_NAME"]
        )
        persistent_weights_dir.mkdir(parents=True, exist_ok=True)
        for item in Path(self.config["LLAMA_WEIGHTS_DIR"]).glob("*"):
            dest_item = persistent_weights_dir / item.name
            if item.is_symlink():
                real_path = item.resolve()
                shutil.copy(real_path, dest_item)
            else:
                if item.is_dir():
                    shutil.copytree(item, dest_item, dirs_exist_ok=True)
                else:
                    shutil.copy(item, dest_item)
        self.config["WEIGHTS_DIR"] = str(persistent_weights_dir)
        logging.info(f"Copied weights to: {persistent_weights_dir}")

    def setup_tt_metal_cache(self):
        cache_dir = (
            Path(self.config["PERSISTENT_VOLUME"])
            / "tt_metal_cache"
            / f"cache_{self.config['REPACKED_STR']}{self.config['MODEL_NAME']}"
        )
        if cache_dir.is_dir():
            logging.info(f"✅ tt_metal_cache already exists at: {cache_dir}.")
        else:
            cache_dir.mkdir(parents=True, exist_ok=True)
            logging.info("✅ setup_tt_metal_cache completed!")
        self.config["TT_METAL_CACHE_DIR"] = str(cache_dir)

    def setup_weights(self):
        self.load_env()
        self.setup_tt_metal_cache()
        target_dir = (
            Path(self.config["PERSISTENT_VOLUME"])
            / "model_weights"
            / f"{self.config['REPACKED_STR']}{self.config['MODEL_NAME']}"
        )
        if target_dir.is_dir() and any(target_dir.iterdir()):
            logging.info(f"Model weights already exist at {target_dir}")
            logging.info("🔔 Please verify the directory contents:")
            for item in target_dir.iterdir():
                logging.info(item.name)
            logging.info(
                "If incorrect, delete the directory to re-download or copy the model weights."
            )
        else:
            (Path(self.config["PERSISTENT_VOLUME"]) / "model_weights").mkdir(
                parents=True, exist_ok=True
            )
            if self.config["MODEL_SOURCE"] == "1":
                self.setup_weights_huggingface()
            elif self.config["MODEL_SOURCE"] == "2":
                self.setup_weights_local()
            else:
                logging.error("⛔ Invalid model source. Exiting.")
                sys.exit(1)

    def run_setup(self):
        if self.check_setup():
            logging.info("Using existing setup. Exiting setup process.")
            sys.exit(0)
        self.setup_model_environment()
        self.setup_weights()


def setup_host(model_type):
    manager = HostSetupManager(model_type)
    manager.run_setup()


def main():
    parser = argparse.ArgumentParser(description="Model setup script")
    parser.add_argument("model_type", help="Type of the model to setup")
    args = parser.parse_args()

    manager = HostSetupManager(args.model_type)
    manager.run_setup()


if __name__ == "__main__":
    main()
