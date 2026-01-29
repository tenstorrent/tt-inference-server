# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os

from config.settings import settings
from huggingface_hub import snapshot_download, try_to_load_from_cache
from huggingface_hub.utils import LocalEntryNotFoundError
from tt_model_runners.runner_fabric import get_device_runner

from utils.logger import TTLogger


class HuggingFaceUtils:
    def __init__(self):
        self.logger = TTLogger()

    def download_weights(self):
        model_name = settings.model_weights_path
        try:
            # get device runner instance and try to load weights
            pipeline_weights_loaded = get_device_runner("-1").load_weights()

            # if weights could be loaded from pipeline stop here
            # otherwise continue for models that don't use pipeline object
            if pipeline_weights_loaded:
                return

            if not model_name:
                self.logger.warning(
                    "No model_weights_path specified, skipping download"
                )
                return

            if os.path.exists(model_name):
                self.logger.info(f"Model already exists locally at: {model_name}")
                return

            if self._are_huggingface_weights_cached(model_name):
                self.logger.info(
                    f"Model {model_name} already cached, skipping download"
                )
                # Get the cached path
                cache_dir = os.environ.get("HF_HOME")
                try:
                    cached_path = snapshot_download(
                        repo_id=model_name,
                        cache_dir=cache_dir,
                        local_files_only=True,
                    )
                    settings.model_weights_path = cached_path
                    self.logger.info(f"Using cached model at: {cached_path}")
                    return
                except Exception:
                    # Cache check failed, proceed with download
                    pass

            self.logger.info(f"Downloading weights for model: {model_name}")

            cache_dir = os.environ.get("HF_HOME")
            downloaded_path = snapshot_download(repo_id=model_name, cache_dir=cache_dir)

            self.logger.info(
                f"Successfully downloaded model weights to: {downloaded_path}"
            )
            settings.model_weights_path = downloaded_path

        except ImportError:
            self.logger.error(
                "huggingface_hub not installed. Install with: pip install huggingface_hub"
            )
            raise RuntimeError("Missing required dependency: huggingface_hub")

        except Exception as e:
            self.logger.error(f"Failed to download model weights for {model_name}: {e}")
            self.logger.warning("Continuing with existing model weights path")

    def _are_huggingface_weights_cached(self, repo_id: str) -> bool:
        """Check if a HuggingFace repo is already cached locally"""
        try:
            cache_dir = os.environ.get("HF_HOME")

            # Try to find any file from the repo in cache
            try:
                cached_file = try_to_load_from_cache(
                    repo_id=repo_id,
                    filename="config.json",
                    cache_dir=cache_dir,
                )
                return cached_file is not None
            except LocalEntryNotFoundError:
                return False
        except ImportError:
            return False
        except Exception:
            return False
