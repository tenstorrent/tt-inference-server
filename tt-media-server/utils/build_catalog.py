# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
import math

from config.constants import (
    DeviceTypes,
    ModelNames,
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


def _build_models_catalog(model_runner: ModelRunners, model: str):
    try:
        model_name_enum = ModelNames(model).value
    except ValueError:
        return []
    supported_model = getattr(SupportedModels, model_name_enum.name, None)
    if supported_model:
        return [
            {
                "id": model_name_enum.value,
                "display_name": model,
                "supported": True,
                "model_config": supported_model.value,
            }
        ]
    return []


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
    model: str = "",
):
    models = _build_models_catalog(model_runner, model)
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
