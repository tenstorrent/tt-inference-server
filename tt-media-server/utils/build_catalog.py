# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
import math

from config.constants import (
    MODEL_RUNNER_TO_MODEL_NAMES_MAP,
    TRAINING_RUNNER_SUPPORTED_DEVICES,
    ModelDisplayNames,
    ModelRunners,
    SupportedModels,
    TrainingMeshShapes,
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


def _build_models_catalog(model_runner: str):
    try:
        runner_enum = ModelRunners(model_runner)
    except ValueError:
        return []
    models = []
    for model_name in MODEL_RUNNER_TO_MODEL_NAMES_MAP.get(runner_enum, set()):
        try:
            model_config = SupportedModels[model_name.name].value
            display_name = ModelDisplayNames[model_name.name].value
        except KeyError:
            raise ValueError(
                f"Model '{model_name.name}' for runner '{model_runner}' "
                f"must have an entry in SupportedModels and ModelDisplayNames"
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


def _build_clusters_catalog(model_runner: str):
    try:
        runner_enum = ModelRunners(model_runner)
    except ValueError:
        return []
    clusters = []
    for dt in TRAINING_RUNNER_SUPPORTED_DEVICES.get(runner_enum, set()):
        mesh_shape = list(TrainingMeshShapes[dt.name].value)
        total_devices = math.prod(mesh_shape)
        clusters.append(
            {
                "id": dt.value,
                "display_name": dt.value.upper(),
                "supported": True,
                "partition": None,
                "mesh_shape": mesh_shape,
                "topology": {
                    "mesh_shape": mesh_shape,
                    "nodes": total_devices,
                    "total_devices": total_devices,
                },
            }
        )
    return clusters


def build_training_catalog(model_runner: str):
    models = _build_models_catalog(model_runner)
    clusters = _build_clusters_catalog(model_runner)

    datasets = [
        {
            "id": loader.value,
            "display_name": loader.value.upper(),
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
