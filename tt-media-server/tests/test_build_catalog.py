# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import math

import pytest
from config.constants import (
    MODEL_RUNNER_TO_MODEL_NAMES_MAP,
    DeviceTypes,
    ModelDisplayNames,
    ModelNames,
    ModelRunners,
    SupportedModels,
    TrainingMeshShapes,
)
from utils.build_catalog import (
    TRAINING_CATALOG_DATA,
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
        assert len(models) == 1
        model = models[0]
        assert model["id"] == ModelNames.GEMMA_1_1_2B_IT.value
        assert model["display_name"] == ModelDisplayNames.GEMMA_1_1_2B_IT.value
        assert model["model_config"] == SupportedModels.GEMMA_1_1_2B_IT.value
        assert model["supported"] is True

    def test_training_llama_runner_returns_model(self):
        models = _build_models_catalog(ModelRunners.TRAINING_LLAMA_LORA.value)
        assert len(models) == 1
        model = models[0]
        assert model["id"] == ModelNames.LLAMA_3_1_8B.value
        assert model["display_name"] == ModelDisplayNames.LLAMA_3_1_8B.value
        assert model["model_config"] == SupportedModels.LLAMA_3_1_8B.value
        assert model["supported"] is True

    def test_model_config_falls_back_when_not_in_supported_models(self):
        """MICROSOFT_RESNET_50 is in ModelNames but not SupportedModels."""
        models = _build_models_catalog(ModelRunners.TT_XLA_RESNET.value)
        assert len(models) == 1
        assert models[0]["model_config"] == ModelNames.MICROSOFT_RESNET_50.value

    def test_display_name_falls_back_when_not_in_display_names(self):
        """MICROSOFT_RESNET_50 is in ModelNames but not ModelDisplayNames."""
        models = _build_models_catalog(ModelRunners.TT_XLA_RESNET.value)
        assert len(models) == 1
        assert models[0]["display_name"] == ModelNames.MICROSOFT_RESNET_50.value

    def test_runner_with_no_models_returns_empty(self):
        """A runner not in MODEL_RUNNER_TO_MODEL_NAMES_MAP returns empty."""
        runner_without_models = None
        for runner in ModelRunners:
            if runner not in MODEL_RUNNER_TO_MODEL_NAMES_MAP:
                runner_without_models = runner
                break
        if runner_without_models is None:
            pytest.skip("all runners have model entries")
        assert _build_models_catalog(runner_without_models.value) == []

    def test_all_returned_models_have_required_keys(self):
        models = _build_models_catalog(ModelRunners.TRAINING_GEMMA_LORA.value)
        required_keys = {"id", "display_name", "supported", "model_config"}
        for model in models:
            assert set(model.keys()) == required_keys


class TestBuildClustersCatalog:
    """Tests for _build_clusters_catalog."""

    def test_invalid_runner_returns_empty(self):
        assert _build_clusters_catalog("nonexistent-runner") == []

    def test_runner_without_supported_devices_returns_empty(self):
        assert _build_clusters_catalog(ModelRunners.TT_XLA_RESNET.value) == []

    def test_gemma_runner_returns_p150_cluster(self):
        clusters = _build_clusters_catalog(ModelRunners.TRAINING_GEMMA_LORA.value)
        assert len(clusters) == 1
        cluster = clusters[0]
        assert cluster["id"] == DeviceTypes.P150.value
        assert cluster["display_name"] == DeviceTypes.P150.value.upper()
        assert cluster["supported"] is True
        assert cluster["partition"] is None
        expected_mesh = list(TrainingMeshShapes.P150.value)
        assert cluster["mesh_shape"] == expected_mesh
        expected_devices = math.prod(expected_mesh)
        assert cluster["topology"]["nodes"] == expected_devices
        assert cluster["topology"]["total_devices"] == expected_devices
        assert cluster["topology"]["mesh_shape"] == expected_mesh

    def test_llama_runner_returns_p300_cluster(self):
        clusters = _build_clusters_catalog(ModelRunners.TRAINING_LLAMA_LORA.value)
        assert len(clusters) == 1
        cluster = clusters[0]
        assert cluster["id"] == DeviceTypes.P300.value
        expected_mesh = list(TrainingMeshShapes.P300.value)
        assert cluster["mesh_shape"] == expected_mesh
        assert cluster["topology"]["total_devices"] == math.prod(expected_mesh)

    def test_all_returned_clusters_have_required_keys(self):
        clusters = _build_clusters_catalog(ModelRunners.TRAINING_GEMMA_LORA.value)
        required_keys = {
            "id",
            "display_name",
            "supported",
            "partition",
            "mesh_shape",
            "topology",
        }
        for cluster in clusters:
            assert set(cluster.keys()) == required_keys
            assert set(cluster["topology"].keys()) == {
                "mesh_shape",
                "nodes",
                "total_devices",
            }


class TestBuildTrainingCatalog:
    """Tests for build_training_catalog."""

    def test_returns_all_expected_keys(self):
        catalog = build_training_catalog(ModelRunners.TRAINING_GEMMA_LORA.value)
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
        catalog = build_training_catalog(ModelRunners.TRAINING_GEMMA_LORA.value)
        dataset_ids = {d["id"] for d in catalog["datasets"]}
        for loader in AVAILABLE_DATASET_LOADERS:
            assert loader.value in dataset_ids

    def test_trainers_include_lora_and_sft(self):
        catalog = build_training_catalog(ModelRunners.TRAINING_GEMMA_LORA.value)
        trainers_by_id = {t["id"]: t for t in catalog["trainers"]}
        assert "lora" in trainers_by_id
        assert trainers_by_id["lora"]["supported"] is True
        assert "sft" in trainers_by_id
        assert trainers_by_id["sft"]["supported"] is False

    def test_only_supported_trainers_in_supported_dict(self):
        catalog = build_training_catalog(ModelRunners.TRAINING_GEMMA_LORA.value)
        assert "lora" in catalog["supported"]["trainers"]
        assert "sft" not in catalog["supported"]["trainers"]

    def test_optimizers_include_adamw(self):
        catalog = build_training_catalog(ModelRunners.TRAINING_GEMMA_LORA.value)
        opt_ids = {o["id"] for o in catalog["optimizers"]}
        assert "adamw" in opt_ids

    def test_supported_optimizers(self):
        catalog = build_training_catalog(ModelRunners.TRAINING_GEMMA_LORA.value)
        assert "adamw" in catalog["supported"]["optimizers"]

    def test_invalid_runner_has_empty_models_and_clusters(self):
        catalog = build_training_catalog("nonexistent-runner")
        assert catalog["models"] == []
        assert catalog["clusters"] == []
        assert len(catalog["datasets"]) > 0
        assert len(catalog["trainers"]) > 0
        assert len(catalog["optimizers"]) > 0

    def test_models_populated_for_valid_runner(self):
        catalog = build_training_catalog(ModelRunners.TRAINING_GEMMA_LORA.value)
        assert len(catalog["models"]) > 0

    def test_clusters_populated_for_valid_runner(self):
        catalog = build_training_catalog(ModelRunners.TRAINING_GEMMA_LORA.value)
        assert len(catalog["clusters"]) > 0

    def test_llama_catalog_has_correct_model_and_cluster(self):
        catalog = build_training_catalog(ModelRunners.TRAINING_LLAMA_LORA.value)
        assert catalog["models"][0]["id"] == ModelNames.LLAMA_3_1_8B.value
        assert catalog["clusters"][0]["id"] == DeviceTypes.P300.value


class TestTrainingCatalogData:
    """Tests for the TRAINING_CATALOG_DATA constant."""

    def test_trainers_key_present(self):
        assert "trainers" in TRAINING_CATALOG_DATA

    def test_optimizers_key_present(self):
        assert "optimizers" in TRAINING_CATALOG_DATA

    def test_lora_trainer_is_supported(self):
        from config.constants import TrainingTrainers

        assert (
            TRAINING_CATALOG_DATA["trainers"][TrainingTrainers.LORA]["supported"]
            is True
        )

    def test_sft_trainer_is_not_supported(self):
        from config.constants import TrainingTrainers

        assert (
            TRAINING_CATALOG_DATA["trainers"][TrainingTrainers.SFT]["supported"]
            is False
        )

    def test_adamw_optimizer_is_supported(self):
        from config.constants import TrainingOptimizers

        assert (
            TRAINING_CATALOG_DATA["optimizers"][TrainingOptimizers.ADAMW]["supported"]
            is True
        )
