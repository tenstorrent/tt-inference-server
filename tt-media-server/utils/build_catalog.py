# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
import math

from config.constants import (
    MODEL_RUNNER_TO_MODEL_NAMES_MAP,
    DeviceTypes,
    ModelDisplayNames,
    ModelRunners,
    SupportedModels,
    TrainingOptimizers,
    TrainingTrainers,
)

from utils.dataset_loaders.dataset_resolver import AVAILABLE_DATASET_LOADERS

TRAINING_CATALOG_DATA = {
    "trainers": {
        TrainingTrainers.LORA: {"display_name": "LoRA", "supported": True},
        TrainingTrainers.SFT: {"display_name": "SFT", "supported": False},
    },
    "optimizers": {
        TrainingOptimizers.ADAMW: {"display_name": "AdamW", "supported": True},
    },
}


def _weights_path_matches(model_weights_path: str, repo_id: str) -> bool:
    """Check if model_weights_path matches the HF repo ID.

    model_weights_path may be the repo ID itself ("Qwen/Qwen3-8B") or a
    resolved local snapshot path that embeds it
    ("…/models--Qwen--Qwen3-8B/snapshots/…").
    """
    if model_weights_path == repo_id:
        return True
    cache_fragment = f"models--{repo_id.replace('/', '--')}"
    return cache_fragment in model_weights_path


def _build_models_catalog(model_runner: str, model_weights_path: str = ""):
    try:
        runner_enum = ModelRunners(model_runner)
    except ValueError:
        return []
    models = []
    model_names_set = MODEL_RUNNER_TO_MODEL_NAMES_MAP.get(runner_enum, set())
    for model_name in model_names_set:
        model_config = SupportedModels[model_name.name].value
        if model_weights_path and not _weights_path_matches(model_weights_path, model_config):
            continue
        try:
            display_name = ModelDisplayNames[model_name.name].value
        except KeyError:
            raise ValueError(
                f"Model '{model_name.name}' for runner '{model_runner}' "
                f"must have an entry in ModelDisplayNames"
            )
        models.append(
            {
                "id": model_name.value,
                "display_name": display_name,
                "supported": True,
                "model_config": model_config,
            }
        )
    return models


def _build_clusters_catalog(device: str, device_mesh_shape: tuple, num_workers: int):
    try:
        dt = DeviceTypes(device)
    except ValueError:
        return []
    mesh_shape = list(device_mesh_shape)
    chips_per_worker = math.prod(mesh_shape)
    total_devices = num_workers * chips_per_worker
    return [
        {
            "id": dt.value,
            "display_name": dt.value.upper(),
            "supported": True,
            "partition": None,
            "mesh_shape": mesh_shape,
            "num_workers": num_workers,
            "topology": {
                "mesh_shape": mesh_shape,
                "nodes": num_workers,
                "total_devices": total_devices,
            },
        }
    ]


def build_training_catalog(
    model_runner: str,
    device: str,
    device_mesh_shape: tuple,
    num_workers: int,
    model_weights_path: str = "",
):
    models = _build_models_catalog(model_runner, model_weights_path)
    clusters = _build_clusters_catalog(device, device_mesh_shape, num_workers)

    datasets = [
        {
            "id": loader.value,
            "display_name": loader.value,
            "supported": True,
        }
        for loader in AVAILABLE_DATASET_LOADERS
    ]

    trainers = [
        {
            "id": t.value,
            "display_name": meta["display_name"],
            "supported": meta["supported"],
        }
        for t, meta in TRAINING_CATALOG_DATA["trainers"].items()
    ]

    optimizers = [
        {
            "id": o.value,
            "display_name": meta["display_name"],
            "supported": meta["supported"],
        }
        for o, meta in TRAINING_CATALOG_DATA["optimizers"].items()
    ]

    supported_trainers = [
        t.value
        for t, meta in TRAINING_CATALOG_DATA["trainers"].items()
        if meta["supported"]
    ]
    supported_optimizers = [
        o.value
        for o, meta in TRAINING_CATALOG_DATA["optimizers"].items()
        if meta["supported"]
    ]

    return {
        "supported": {
            "trainers": supported_trainers,
            "optimizers": supported_optimizers,
        },
        "models": models,
        "datasets": datasets,
        "trainers": trainers,
        "optimizers": optimizers,
        "clusters": clusters,
    }
