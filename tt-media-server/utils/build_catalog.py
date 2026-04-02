# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
from config.constants import (
    MODEL_RUNNER_TO_MODEL_NAMES_MAP,
    MODEL_SERVICE_RUNNER_MAP,
    ModelDisplayNames,
    ModelServices,
    SupportedModels,
    TrainingOptimizers,
    TrainingTrainers,
    DeviceTypes,
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
    "clusters": [
        {
            "id": DeviceTypes.P150.value,
            "display_name": "1× P150",
            "supported": True,
            "partition": None,
            "mesh_shape": [1, 1],
            "topology": {"mesh_shape": [1, 1], "nodes": 1, "total_devices": 1},
        },
        {
            "id": DeviceTypes.P300.value,
            "display_name": "1× P300",
            "supported": True,
            "partition": None,
            "mesh_shape": [1, 2],
            "topology": {"mesh_shape": [1, 2], "nodes": 2, "total_devices": 2},
        },
    ],
}


def _build_models_catalog():
    runners = MODEL_SERVICE_RUNNER_MAP.get(ModelServices.TRAINING, set())
    models = []
    for runner in runners:
        for model_name in MODEL_RUNNER_TO_MODEL_NAMES_MAP.get(runner, set()):
            try:
                model_config = SupportedModels[model_name.name].value
            except KeyError:
                model_config = model_name.value
            try:
                display_name = ModelDisplayNames[model_name.name].value
            except KeyError:
                display_name = model_name.value
            models.append(
                {
                    "id": model_name.value,
                    "display_name": display_name,
                    "supported": True,
                    "model_config": model_config,
                }
            )
    return models


def _build_training_catalog():
    models = _build_models_catalog()

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
        "clusters": TRAINING_CATALOG_DATA["clusters"],
    }


TRAINING_CATALOG = _build_training_catalog()
