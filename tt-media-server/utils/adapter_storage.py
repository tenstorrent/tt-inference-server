# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import json
import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AdapterInfo:
    base_model_name: str
    adapter_path: str


class AdapterStorage(ABC):
    """Abstract interface for saving/loading PEFT adapter checkpoints."""

    @abstractmethod
    def save_checkpoint(self, peft_model, job_id: str, checkpoint_id: str) -> str:
        """Save a PEFT checkpoint and return a reference to it."""

    @abstractmethod
    def resolve_adapter(self, adapter: str) -> AdapterInfo:
        """Resolve an adapter identifier (``{job_id}/{checkpoint_id}``) to an AdapterInfo."""

    @abstractmethod
    def get_checkpoint_path(self, job_id: str, checkpoint_id: str) -> Optional[str]:
        """Return a local download path for the checkpoint, or ``None``."""

    @abstractmethod
    def ensure_job_dir(self, job_id: str) -> str:
        """Ensure the output location for *job_id* exists and return a reference to it."""


class LocalAdapterStorage(AdapterStorage):
    def __init__(self, base_dir: str):
        self.base_dir = base_dir

    def save_checkpoint(self, peft_model, job_id: str, checkpoint_id: str) -> str:
        checkpoint_path = os.path.join(self.base_dir, job_id, checkpoint_id)
        os.makedirs(checkpoint_path, exist_ok=True)
        peft_model.save_pretrained(
            checkpoint_path,
            state_dict={k: v.cpu() for k, v in peft_model.state_dict().items()},
        )
        return checkpoint_path

    def resolve_adapter(self, adapter: str) -> AdapterInfo:
        adapter_path = os.path.join(self.base_dir, adapter)

        if not os.path.isdir(adapter_path):
            raise FileNotFoundError(f"Adapter not found at {adapter_path}")

        config_path = os.path.join(adapter_path, "adapter_config.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"adapter_config.json not found at {adapter_path}"
            )

        with open(config_path) as f:
            config = json.load(f)

        base_model_name = config.get("base_model_name_or_path")
        if not base_model_name:
            raise ValueError(f"base_model_name_or_path missing in {config_path}")

        return AdapterInfo(base_model_name=base_model_name, adapter_path=adapter_path)

    def get_checkpoint_path(self, job_id: str, checkpoint_id: str) -> Optional[str]:
        checkpoint_path = os.path.join(self.base_dir, job_id, checkpoint_id)
        if os.path.isdir(checkpoint_path):
            return checkpoint_path
        return None

    def ensure_job_dir(self, job_id: str) -> str:
        job_dir = os.path.join(self.base_dir, job_id)
        os.makedirs(job_dir, exist_ok=True)
        return job_dir


def _read_adapter_config(config_path: str, display_location: str) -> str:
    """Read adapter_config.json and return base_model_name_or_path."""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"adapter_config.json not found in {display_location}"
        )
    with open(config_path) as f:
        config = json.load(f)
    base_model_name = config.get("base_model_name_or_path")
    if not base_model_name:
        raise ValueError(
            f"base_model_name_or_path missing in {display_location}"
        )
    return base_model_name


class HfHubAdapterStorage(AdapterStorage):
    """Store adapters on the Hugging Face Hub.

    Each training job maps to an HF repo ``{hf_org}/{job_id}``.
    Checkpoints are stored as sub-folders inside the repo.
    """

    def __init__(self, hf_org: str, token: Optional[str] = None):
        if not hf_org:
            raise ValueError(
                "hf_adapter_org must be set when using hf_hub adapter storage backend"
            )
        from huggingface_hub import HfApi

        self.hf_org = hf_org
        self.api = HfApi(token=token)

    def _repo_id(self, job_id: str) -> str:
        return f"{self.hf_org}/{job_id}"

    def save_checkpoint(self, peft_model, job_id: str, checkpoint_id: str) -> str:
        repo_id = self._repo_id(job_id)
        self.api.create_repo(repo_id, exist_ok=True, private=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            peft_model.save_pretrained(
                tmp_dir,
                state_dict={
                    k: v.cpu() for k, v in peft_model.state_dict().items()
                },
            )
            self.api.upload_folder(
                repo_id=repo_id,
                folder_path=tmp_dir,
                path_in_repo=checkpoint_id,
            )
        return f"{repo_id}/{checkpoint_id}"

    def resolve_adapter(self, adapter: str) -> AdapterInfo:
        from huggingface_hub import snapshot_download

        job_id, checkpoint_id = adapter.split("/", 1)
        repo_id = self._repo_id(job_id)
        local_dir = snapshot_download(
            repo_id,
            allow_patterns=[f"{checkpoint_id}/*"],
        )
        adapter_path = os.path.join(local_dir, checkpoint_id)
        config_path = os.path.join(adapter_path, "adapter_config.json")
        base_model_name = _read_adapter_config(
            config_path, f"{repo_id}/{checkpoint_id}"
        )
        return AdapterInfo(base_model_name=base_model_name, adapter_path=adapter_path)

    def get_checkpoint_path(self, job_id: str, checkpoint_id: str) -> Optional[str]:
        from huggingface_hub import snapshot_download

        repo_id = self._repo_id(job_id)
        try:
            local_dir = snapshot_download(
                repo_id,
                allow_patterns=[f"{checkpoint_id}/*"],
            )
        except Exception:
            return None
        checkpoint_path = os.path.join(local_dir, checkpoint_id)
        if os.path.isdir(checkpoint_path):
            return checkpoint_path
        return None

    def ensure_job_dir(self, job_id: str) -> str:
        repo_id = self._repo_id(job_id)
        self.api.create_repo(repo_id, exist_ok=True, private=True)
        return repo_id


def get_adapter_storage() -> AdapterStorage:
    """Create an AdapterStorage from the current application settings."""
    from config.constants import TRAINING_STORE_ADAPTERS_DIR, AdapterStorageBackend
    from config.settings import get_settings

    settings = get_settings()
    if settings.adapter_storage_backend == AdapterStorageBackend.HF_HUB.value:
        return HfHubAdapterStorage(
            hf_org=settings.hf_adapter_org,
            token=os.environ.get("HF_TOKEN"),
        )
    return LocalAdapterStorage(base_dir=TRAINING_STORE_ADAPTERS_DIR)
