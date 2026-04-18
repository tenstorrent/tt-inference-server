# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import enum
from unittest.mock import patch

import pytest
from config.constants import (
    DeviceTypes,
    ModelDisplayNames,
    ModelNames,
    ModelRunners,
    SupportedModels,
)
from utils.build_catalog import (
    _build_clusters_catalog,
    _build_models_catalog,
    build_training_catalog,
)
from utils.dataset_loaders.dataset_resolver import AVAILABLE_DATASET_LOADERS


class TestBuildModelsCatalog:
    """Tests for _build_models_catalog."""

    def test_invalid_runner_returns_empty(self):
        assert _build_models_catalog("nonexistent-runner") == []

    def test_training_gemma_runner_returns_model(self):
        models = _build_models_catalog(ModelRunners.TRAINING_GEMMA_LORA.value)
        required_keys = {"id", "display_name", "supported", "model_config"}
        for model in models:
            assert set(model.keys()) == required_keys
        assert len(models) == 1
        model = models[0]
        assert model["id"] == ModelNames.GEMMA_1_1_2B_IT.value
        assert model["display_name"] == ModelDisplayNames.GEMMA_1_1_2B_IT.value
        assert model["model_config"] == SupportedModels.GEMMA_1_1_2B_IT.value
        assert model["supported"] is True

    def test_training_llama_runner_returns_model(self):
        models = _build_models_catalog(ModelRunners.TRAINING_LLAMA_LORA.value)
        required_keys = {"id", "display_name", "supported", "model_config"}
        for model in models:
            assert set(model.keys()) == required_keys
        assert len(models) == 1
        model = models[0]
        assert model["id"] == ModelNames.LLAMA_3_1_8B.value
        assert model["display_name"] == ModelDisplayNames.LLAMA_3_1_8B.value
        assert model["model_config"] == SupportedModels.LLAMA_3_1_8B.value
        assert model["supported"] is True

    def test_raises_when_model_missing_from_enums(self):
        fake_model = enum.Enum("FakeModel", {"FAKE_MODEL": "fake-model"}).FAKE_MODEL
        fake_map = {ModelRunners.TRAINING_GEMMA_LORA: {fake_model}}
        with patch("utils.build_catalog.MODEL_RUNNER_TO_MODEL_NAMES_MAP", fake_map):
            with pytest.raises(
                ValueError, match="SupportedModels and ModelDisplayNames"
            ):
                _build_models_catalog(ModelRunners.TRAINING_GEMMA_LORA.value)


class TestBuildClustersCatalog:
    """Tests for _build_clusters_catalog."""

    def test_invalid_device_returns_empty(self):
        assert _build_clusters_catalog("nonexistent-device", (1, 1), 1) == []

    def test_single_worker_single_chip(self):
        clusters = _build_clusters_catalog(DeviceTypes.P150.value, (1, 1), 1)
        assert len(clusters) == 1
        cluster = clusters[0]
        assert cluster["id"] == DeviceTypes.P150.value
        assert cluster["display_name"] == DeviceTypes.P150.value.upper()
        assert cluster["supported"] is True
        assert cluster["mesh_shape"] == [1, 1]
        assert cluster["num_workers"] == 1
        assert cluster["topology"]["nodes"] == 1
        assert cluster["topology"]["total_devices"] == 1

    def test_single_worker_multichip(self):
        clusters = _build_clusters_catalog(DeviceTypes.P300.value, (1, 2), 1)
        assert len(clusters) == 1
        assert clusters[0]["mesh_shape"] == [1, 2]
        assert clusters[0]["num_workers"] == 1
        assert clusters[0]["topology"]["nodes"] == 1
        assert clusters[0]["topology"]["total_devices"] == 2

    def test_multiple_workers_multiply_total_devices(self):
        clusters = _build_clusters_catalog(DeviceTypes.P150.value, (1, 1), 3)
        assert len(clusters) == 1
        assert clusters[0]["num_workers"] == 3
        assert clusters[0]["topology"]["nodes"] == 3
        assert clusters[0]["topology"]["total_devices"] == 3

    def test_multiple_workers_multichip(self):
        clusters = _build_clusters_catalog(DeviceTypes.P300.value, (2, 2), 2)
        assert len(clusters) == 1
        assert clusters[0]["num_workers"] == 2
        assert clusters[0]["topology"]["nodes"] == 2
        assert clusters[0]["topology"]["total_devices"] == 8


class TestBuildTrainingCatalog:
    """Tests for build_training_catalog."""

    def test_returns_all_expected_keys(self):
        catalog = build_training_catalog(
            ModelRunners.TRAINING_GEMMA_LORA.value, DeviceTypes.P150.value, (1, 1), 1
        )
        expected_keys = {
            "supported",
            "models",
            "datasets",
            "trainers",
            "optimizers",
            "clusters",
        }
        assert set(catalog.keys()) == expected_keys

    def test_datasets_from_available_loaders(self):
        catalog = build_training_catalog(
            ModelRunners.TRAINING_GEMMA_LORA.value, DeviceTypes.P150.value, (1, 1), 1
        )
        dataset_ids = {d["id"] for d in catalog["datasets"]}
        for loader in AVAILABLE_DATASET_LOADERS:
            assert loader.value in dataset_ids

    def test_trainers_include_lora_and_sft(self):
        catalog = build_training_catalog(
            ModelRunners.TRAINING_GEMMA_LORA.value, DeviceTypes.P150.value, (1, 1), 1
        )
        trainers_by_id = {t["id"]: t for t in catalog["trainers"]}
        assert "lora" in trainers_by_id
        assert trainers_by_id["lora"]["supported"] is True
        assert "sft" in trainers_by_id
        assert trainers_by_id["sft"]["supported"] is False

    def test_only_supported_trainers_in_supported_dict(self):
        catalog = build_training_catalog(
            ModelRunners.TRAINING_GEMMA_LORA.value, DeviceTypes.P150.value, (1, 1), 1
        )
        assert "lora" in catalog["supported"]["trainers"]
        assert "sft" not in catalog["supported"]["trainers"]

    def test_optimizers_include_adamw(self):
        catalog = build_training_catalog(
            ModelRunners.TRAINING_GEMMA_LORA.value, DeviceTypes.P150.value, (1, 1), 1
        )
        opt_ids = {o["id"] for o in catalog["optimizers"]}
        assert "adamw" in opt_ids

    def test_supported_optimizers(self):
        catalog = build_training_catalog(
            ModelRunners.TRAINING_GEMMA_LORA.value, DeviceTypes.P150.value, (1, 1), 1
        )
        assert "adamw" in catalog["supported"]["optimizers"]

    def test_invalid_runner_has_empty_models_and_clusters(self):
        catalog = build_training_catalog(
            "nonexistent-runner", "nonexistent-device", (1, 1), 1
        )
        assert catalog["models"] == []
        assert catalog["clusters"] == []
        assert len(catalog["datasets"]) > 0
        assert len(catalog["trainers"]) > 0
        assert len(catalog["optimizers"]) > 0

    def test_models_populated_for_valid_runner(self):
        catalog = build_training_catalog(
            ModelRunners.TRAINING_GEMMA_LORA.value, DeviceTypes.P150.value, (1, 1), 1
        )
        assert len(catalog["models"]) > 0

    def test_clusters_populated_for_valid_device(self):
        catalog = build_training_catalog(
            ModelRunners.TRAINING_GEMMA_LORA.value, DeviceTypes.P150.value, (1, 1), 1
        )
        assert len(catalog["clusters"]) == 1
        assert catalog["clusters"][0]["id"] == DeviceTypes.P150.value

    def test_llama_catalog_has_correct_model_and_cluster(self):
        catalog = build_training_catalog(
            ModelRunners.TRAINING_LLAMA_LORA.value, DeviceTypes.P300.value, (1, 2), 1
        )
        assert catalog["models"][0]["id"] == ModelNames.LLAMA_3_1_8B.value
        assert catalog["clusters"][0]["id"] == DeviceTypes.P300.value
