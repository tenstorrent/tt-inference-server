# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for test matrix expansion in suite_loader."""

import pytest

from server_tests.test_categorization_system.suite_loader import (
    expand_test_matrices,
    load_suite_files,
    load_suite_files_by_category,
)

SAMPLE_MODEL_CONFIGS = {
    "model_a": {
        "weights": ["model-a-weights-v1"],
        "category": "IMAGE",
        "compatible_devices": ["n150", "t3k", "galaxy"],
    },
    "model_b": {
        "weights": ["model-b-weights-v2"],
        "category": "IMAGE",
        "compatible_devices": ["t3k", "galaxy"],
    },
}


class TestExpandTestMatrices:
    def test_basic_expansion(self):
        matrices = [
            {
                "id_pattern": "{model}-{device}",
                "models": ["model_a"],
                "devices": ["n150", "t3k"],
                "test_cases": [
                    {"template": "LoadTest", "enabled": True, "description": "load"},
                ],
            }
        ]

        result = expand_test_matrices(matrices, SAMPLE_MODEL_CONFIGS)

        assert len(result) == 2
        assert result[0]["id"] == "model_a-n150"
        assert result[0]["weights"] == ["model-a-weights-v1"]
        assert result[0]["device"] == "n150"
        assert result[0]["model_marker"] == "model_a"
        assert len(result[0]["test_cases"]) == 1

        assert result[1]["id"] == "model_a-t3k"
        assert result[1]["device"] == "t3k"

    def test_multi_model_expansion(self):
        matrices = [
            {
                "models": ["model_a", "model_b"],
                "devices": ["t3k", "galaxy"],
                "test_cases": [
                    {"template": "LoadTest", "enabled": True},
                ],
            }
        ]

        result = expand_test_matrices(matrices, SAMPLE_MODEL_CONFIGS)

        assert len(result) == 4
        ids = {s["id"] for s in result}
        assert ids == {"model_a-t3k", "model_a-galaxy", "model_b-t3k", "model_b-galaxy"}

    def test_incompatible_device_skipped(self):
        """model_b is not compatible with n150 — that pair should be skipped."""
        matrices = [
            {
                "models": ["model_b"],
                "devices": ["n150", "t3k"],
                "test_cases": [
                    {"template": "LoadTest", "enabled": True},
                ],
            }
        ]

        result = expand_test_matrices(matrices, SAMPLE_MODEL_CONFIGS)

        assert len(result) == 1
        assert result[0]["id"] == "model_b-t3k"

    def test_num_of_devices_from_matrix(self):
        matrices = [
            {
                "models": ["model_a"],
                "devices": ["n150"],
                "num_of_devices": 2,
                "test_cases": [],
            }
        ]

        result = expand_test_matrices(matrices, SAMPLE_MODEL_CONFIGS)

        assert result[0]["num_of_devices"] == 2

    def test_num_of_devices_from_model_config(self):
        configs = {
            "model_x": {
                "weights": ["x"],
                "category": "TEST",
                "compatible_devices": ["n150"],
                "num_of_devices": 4,
            }
        }
        matrices = [
            {
                "models": ["model_x"],
                "devices": ["n150"],
                "test_cases": [],
            }
        ]

        result = expand_test_matrices(matrices, configs)

        assert result[0]["num_of_devices"] == 4

    def test_matrix_num_devices_overrides_model_config(self):
        configs = {
            "model_x": {
                "weights": ["x"],
                "category": "TEST",
                "compatible_devices": ["n150"],
                "num_of_devices": 4,
            }
        }
        matrices = [
            {
                "models": ["model_x"],
                "devices": ["n150"],
                "num_of_devices": 1,
                "test_cases": [],
            }
        ]

        result = expand_test_matrices(matrices, configs)

        assert result[0]["num_of_devices"] == 1

    def test_unknown_model_raises(self):
        matrices = [
            {
                "models": ["nonexistent"],
                "devices": ["n150"],
                "test_cases": [],
            }
        ]

        with pytest.raises(ValueError, match="nonexistent"):
            expand_test_matrices(matrices, SAMPLE_MODEL_CONFIGS)

    def test_empty_compatible_devices_allows_all(self):
        """If compatible_devices is empty/missing, all devices are allowed."""
        configs = {
            "open_model": {
                "weights": ["open"],
                "category": "TEST",
            }
        }
        matrices = [
            {
                "models": ["open_model"],
                "devices": ["n150", "t3k", "galaxy"],
                "test_cases": [{"template": "T", "enabled": True}],
            }
        ]

        result = expand_test_matrices(matrices, configs)

        assert len(result) == 3

    def test_test_cases_are_deep_copied(self):
        """Each suite should get independent copies of test_cases."""
        matrices = [
            {
                "models": ["model_a"],
                "devices": ["n150", "t3k"],
                "test_cases": [
                    {"template": "LoadTest", "enabled": True, "targets": {"time": 10}},
                ],
            }
        ]

        result = expand_test_matrices(matrices, SAMPLE_MODEL_CONFIGS)

        result[0]["test_cases"][0]["targets"]["time"] = 999
        assert result[1]["test_cases"][0]["targets"]["time"] == 10

    def test_custom_id_pattern(self):
        matrices = [
            {
                "id_pattern": "test-{model}-on-{device}",
                "models": ["model_a"],
                "devices": ["n150"],
                "test_cases": [],
            }
        ]

        result = expand_test_matrices(matrices, SAMPLE_MODEL_CONFIGS)

        assert result[0]["id"] == "test-model_a-on-n150"

    def test_weights_are_copied_not_shared(self):
        """Weights list should not be shared across suites."""
        matrices = [
            {
                "models": ["model_a"],
                "devices": ["n150", "t3k"],
                "test_cases": [],
            }
        ]

        result = expand_test_matrices(matrices, SAMPLE_MODEL_CONFIGS)

        result[0]["weights"].append("extra")
        assert "extra" not in result[1]["weights"]
        assert "extra" not in SAMPLE_MODEL_CONFIGS["model_a"]["weights"]

    def test_id_name_overrides_model_key_in_id(self):
        configs = {
            "my_model": {
                "id_name": "my-model",
                "weights": ["w"],
                "category": "TEST",
                "compatible_devices": ["n150"],
            }
        }
        matrices = [
            {
                "models": ["my_model"],
                "devices": ["n150"],
                "test_cases": [],
            }
        ]

        result = expand_test_matrices(matrices, configs)

        assert result[0]["id"] == "my-model-n150"
        assert result[0]["model_marker"] == "my_model"

    def test_model_targets_per_model(self):
        configs = {
            "fast": {
                "weights": ["fast-v1"],
                "category": "TEST",
                "compatible_devices": ["n150"],
            },
            "slow": {
                "weights": ["slow-v1"],
                "category": "TEST",
                "compatible_devices": ["n150"],
            },
        }
        matrices = [
            {
                "models": ["fast", "slow"],
                "devices": ["n150"],
                "test_cases": [
                    {
                        "template": "LoadTest",
                        "enabled": True,
                        "description": "run load",
                        "model_targets": {
                            "fast": {"time": 5},
                            "slow": {"time": 50},
                        },
                    },
                    {
                        "template": "ParamTest",
                        "enabled": True,
                        "description": "run params",
                    },
                ],
            }
        ]

        result = expand_test_matrices(matrices, configs)

        fast_suite = next(s for s in result if s["model_marker"] == "fast")
        slow_suite = next(s for s in result if s["model_marker"] == "slow")

        assert fast_suite["test_cases"][0]["targets"] == {"time": 5}
        assert slow_suite["test_cases"][0]["targets"] == {"time": 50}
        assert "targets" not in fast_suite["test_cases"][1]
        assert "targets" not in slow_suite["test_cases"][1]

    def test_model_targets_removed_from_output(self):
        """model_targets should not appear in expanded test cases."""
        configs = {
            "m": {"weights": ["w"], "category": "T", "compatible_devices": ["n150"]},
        }
        matrices = [
            {
                "models": ["m"],
                "devices": ["n150"],
                "test_cases": [
                    {
                        "template": "T",
                        "enabled": True,
                        "model_targets": {"m": {"time": 5}},
                    },
                ],
            }
        ]

        result = expand_test_matrices(matrices, configs)

        assert "model_targets" not in result[0]["test_cases"][0]
        assert result[0]["test_cases"][0]["targets"] == {"time": 5}

    def test_model_targets_model_device_specific(self):
        """model+device keys take priority over model-only keys."""
        configs = {
            "m": {
                "weights": ["w"],
                "category": "TEST",
                "compatible_devices": ["n150", "t3k"],
            }
        }
        matrices = [
            {
                "models": ["m"],
                "devices": ["n150", "t3k"],
                "test_cases": [
                    {
                        "template": "T",
                        "enabled": True,
                        "model_targets": {
                            "m": {"time": 10},
                            "m+t3k": {"time": 5},
                        },
                    },
                ],
            }
        ]

        result = expand_test_matrices(matrices, configs)
        suite_map = {s["device"]: s for s in result}

        assert suite_map["n150"]["test_cases"][0]["targets"] == {"time": 10}
        assert suite_map["t3k"]["test_cases"][0]["targets"] == {"time": 5}

    def test_model_targets_merge_with_base_targets(self):
        configs = {
            "m": {
                "weights": ["w"],
                "category": "TEST",
                "compatible_devices": ["n150"],
            }
        }
        matrices = [
            {
                "models": ["m"],
                "devices": ["n150"],
                "test_cases": [
                    {
                        "template": "T",
                        "enabled": True,
                        "targets": {"dataset": "60s"},
                        "model_targets": {"m": {"time": 7}},
                    },
                ],
            }
        ]

        result = expand_test_matrices(matrices, configs)

        assert result[0]["test_cases"][0]["targets"] == {"dataset": "60s", "time": 7}

    def test_model_targets_isolation_between_models(self):
        """model_targets for one model must not leak into another."""
        configs = {
            "a": {"weights": ["a"], "category": "T", "compatible_devices": ["n150"]},
            "b": {"weights": ["b"], "category": "T", "compatible_devices": ["n150"]},
        }
        matrices = [
            {
                "models": ["a", "b"],
                "devices": ["n150"],
                "test_cases": [
                    {
                        "template": "T",
                        "enabled": True,
                        "model_targets": {"a": {"val": 1}, "b": {"val": 2}},
                    },
                ],
            }
        ]

        result = expand_test_matrices(matrices, configs)

        a_suite = next(s for s in result if s["model_marker"] == "a")
        b_suite = next(s for s in result if s["model_marker"] == "b")
        assert a_suite["test_cases"][0]["targets"]["val"] == 1
        assert b_suite["test_cases"][0]["targets"]["val"] == 2

    def test_no_matching_model_target_keeps_base(self):
        """If model_targets exists but has no key for this model, base targets survive."""
        configs = {
            "a": {"weights": ["a"], "category": "T", "compatible_devices": ["n150"]},
            "b": {"weights": ["b"], "category": "T", "compatible_devices": ["n150"]},
        }
        matrices = [
            {
                "models": ["a", "b"],
                "devices": ["n150"],
                "test_cases": [
                    {
                        "template": "T",
                        "enabled": True,
                        "targets": {"shared": 99},
                        "model_targets": {"a": {"extra": 1}},
                    },
                ],
            }
        ]

        result = expand_test_matrices(matrices, configs)

        a_suite = next(s for s in result if s["model_marker"] == "a")
        b_suite = next(s for s in result if s["model_marker"] == "b")
        assert a_suite["test_cases"][0]["targets"] == {"shared": 99, "extra": 1}
        assert b_suite["test_cases"][0]["targets"] == {"shared": 99}


class TestVideoMatrixExpansion:
    """Validate that the migrated video.json produces the same suites as before."""

    ORIGINAL_VIDEO_SUITES = [
        {
            "id": "wan-t3k",
            "weights": ["Wan2.2-T2V-A14B-Diffusers"],
            "device": "t3k",
            "num_of_devices": 1,
            "model_marker": "wan",
        },
        {
            "id": "wan-galaxy",
            "weights": ["Wan2.2-T2V-A14B-Diffusers"],
            "device": "galaxy",
            "num_of_devices": 1,
            "model_marker": "wan",
        },
        {
            "id": "wan-p150x4",
            "weights": ["Wan2.2-T2V-A14B-Diffusers"],
            "device": "p150x4",
            "num_of_devices": 1,
            "model_marker": "wan",
        },
        {
            "id": "wan-p150x8",
            "weights": ["Wan2.2-T2V-A14B-Diffusers"],
            "device": "p150x8",
            "num_of_devices": 1,
            "model_marker": "wan",
        },
        {
            "id": "wan-p300x2",
            "weights": ["Wan2.2-T2V-A14B-Diffusers"],
            "device": "p300x2",
            "num_of_devices": 1,
            "model_marker": "wan",
        },
        {
            "id": "mochi-p150x4",
            "weights": ["mochi-1-preview"],
            "device": "p150x4",
            "num_of_devices": 1,
            "model_marker": "mochi",
        },
        {
            "id": "mochi-p300x2",
            "weights": ["mochi-1-preview"],
            "device": "p300x2",
            "num_of_devices": 1,
            "model_marker": "mochi",
        },
        {
            "id": "mochi-t3k",
            "weights": ["mochi-1-preview"],
            "device": "t3k",
            "num_of_devices": 1,
            "model_marker": "mochi",
        },
        {
            "id": "mochi-galaxy",
            "weights": ["mochi-1-preview"],
            "device": "galaxy",
            "num_of_devices": 1,
            "model_marker": "mochi",
        },
        {
            "id": "mochi-p150x8",
            "weights": ["mochi-1-preview"],
            "device": "p150x8",
            "num_of_devices": 1,
            "model_marker": "mochi",
        },
    ]

    WAN_LOAD_TARGETS = {
        "wan-t3k": {"video_generation_target_time": 1200, "poll_timeout": 1500},
        "wan-galaxy": {"video_generation_target_time": 250, "poll_timeout": 550},
        "wan-p150x4": {"video_generation_target_time": 370},
        "wan-p150x8": {"video_generation_target_time": 600, "poll_timeout": 900},
        "wan-p300x2": {"video_generation_target_time": 500, "poll_timeout": 800},
    }
    MOCHI_LOAD_TARGETS = {
        "mochi-p150x4": {
            "video_generation_target_time": 480,
            "num_inference_steps": 50,
        },
        "mochi-p300x2": {
            "video_generation_target_time": 900,
            "num_inference_steps": 50,
            "poll_timeout": 1100,
        },
        "mochi-t3k": {
            "video_generation_target_time": 600,
            "num_inference_steps": 50,
            "poll_timeout": 900,
        },
        "mochi-galaxy": {
            "video_generation_target_time": 650,
            "num_inference_steps": 50,
            "poll_timeout": 800,
        },
        "mochi-p150x8": {
            "video_generation_target_time": 900,
            "num_inference_steps": 50,
            "poll_timeout": 1000,
        },
    }

    def test_video_suite_count(self):
        suites = load_suite_files_by_category("video")
        assert len(suites) == 10

    def test_video_suite_ids_match(self):
        suites = load_suite_files_by_category("video")
        actual_ids = {s["id"] for s in suites}
        expected_ids = {s["id"] for s in self.ORIGINAL_VIDEO_SUITES}
        assert actual_ids == expected_ids

    def test_video_suite_properties(self):
        suites = load_suite_files_by_category("video")
        suite_map = {s["id"]: s for s in suites}

        for expected in self.ORIGINAL_VIDEO_SUITES:
            actual = suite_map[expected["id"]]
            assert actual["weights"] == expected["weights"], (
                f"weights mismatch for {expected['id']}"
            )
            assert actual["device"] == expected["device"], (
                f"device mismatch for {expected['id']}"
            )
            assert actual["model_marker"] == expected["model_marker"], (
                f"model_marker mismatch for {expected['id']}"
            )
            assert actual["num_of_devices"] == expected["num_of_devices"], (
                f"num_of_devices mismatch for {expected['id']}"
            )

    def test_video_wan_test_cases(self):
        suites = load_suite_files_by_category("video")
        wan_suites = [s for s in suites if s["model_marker"] == "wan"]

        for suite in wan_suites:
            assert len(suite["test_cases"]) == 2, (
                f"Expected 2 test cases for {suite['id']}"
            )

            load_test = suite["test_cases"][0]
            assert load_test["template"] == "VideoGenerationLoadTest"
            expected_targets = self.WAN_LOAD_TARGETS[suite["id"]]
            assert load_test["targets"] == expected_targets, (
                f"targets mismatch for {suite['id']}"
            )

            param_test = suite["test_cases"][1]
            assert param_test["template"] == "VideoGenerationParamTest"
            assert "targets" not in param_test

    def test_video_mochi_test_cases(self):
        suites = load_suite_files_by_category("video")
        mochi_suites = [s for s in suites if s["model_marker"] == "mochi"]

        for suite in mochi_suites:
            assert len(suite["test_cases"]) == 2, (
                f"Expected 2 test cases for {suite['id']}"
            )

            load_test = suite["test_cases"][0]
            assert load_test["template"] == "VideoGenerationLoadTest"
            expected_targets = self.MOCHI_LOAD_TARGETS[suite["id"]]
            assert load_test["targets"] == expected_targets, (
                f"targets mismatch for {suite['id']}"
            )

            param_test = suite["test_cases"][1]
            assert param_test["template"] == "VideoGenerationParamTest"
            assert "targets" not in param_test


class TestCnnMatrixExpansion:
    """Validate that the migrated cnn.json produces the same suites as before."""

    EXPECTED_IDS = {
        "mobilenetv2-n150",
        "mobilenetv2-n300",
        "resnet-50-n150",
        "resnet-50-n300",
        "efficientnet-n150",
        "segformer-n150",
        "unet-n150",
        "vit-n150",
        "vovnet-n150",
    }

    def test_cnn_suite_count(self):
        suites = load_suite_files_by_category("cnn")
        assert len(suites) == 9

    def test_cnn_suite_ids(self):
        suites = load_suite_files_by_category("cnn")
        ids = {s["id"] for s in suites}
        assert ids == self.EXPECTED_IDS

    def test_cnn_shared_test_cases(self):
        suites = load_suite_files_by_category("cnn")
        for suite in suites:
            assert len(suite["test_cases"]) == 2
            assert suite["test_cases"][0]["template"] == "CnnLoadTest"
            assert suite["test_cases"][0]["targets"] == {
                "cnn_time": 5,
                "response_format": "json",
                "top_k": 3,
                "min_confidence": 70.0,
            }
            assert suite["test_cases"][1]["template"] == "CnnParamTest"

    def test_cnn_num_of_devices(self):
        suites = load_suite_files_by_category("cnn")
        for suite in suites:
            assert suite["num_of_devices"] == 1

    def test_cnn_n300_only_compatible_models(self):
        """Only mobilenetv2 and resnet should have n300 suites."""
        suites = load_suite_files_by_category("cnn")
        n300_ids = {s["id"] for s in suites if s["device"] == "n300"}
        assert n300_ids == {"mobilenetv2-n300", "resnet-50-n300"}

    def test_cnn_resnet_id_name(self):
        """resnet model_marker should be 'resnet' but ID uses 'resnet-50'."""
        suites = load_suite_files_by_category("cnn")
        suite_map = {s["id"]: s for s in suites}
        assert suite_map["resnet-50-n150"]["model_marker"] == "resnet"
        assert suite_map["resnet-50-n150"]["weights"] == ["resnet-50"]


class TestAudioMatrixExpansion:
    """Validate that the migrated audio.json produces the same suites as before."""

    ORIGINAL_AUDIO_IDS = {
        "distil-whisper-n150",
        "distil-whisper-t3k",
        "distil-whisper-galaxy",
        "whisper-n150",
        "whisper-t3k",
        "whisper-galaxy",
    }

    def test_audio_suite_count(self):
        suites = load_suite_files_by_category("audio")
        assert len(suites) == 6

    def test_audio_suite_ids(self):
        suites = load_suite_files_by_category("audio")
        ids = {s["id"] for s in suites}
        assert ids == self.ORIGINAL_AUDIO_IDS

    def test_audio_n150_explicit_suites(self):
        """n150 suites are explicit (not matrix-expanded) due to unique test lists."""
        suites = load_suite_files_by_category("audio")
        suite_map = {s["id"]: s for s in suites}

        dw_n150 = suite_map["distil-whisper-n150"]
        assert len(dw_n150["test_cases"]) == 4
        templates = [tc["template"] for tc in dw_n150["test_cases"]]
        assert "DeviceStabilityTest" in templates

        w_n150 = suite_map["whisper-n150"]
        assert len(w_n150["test_cases"]) == 3
        templates = [tc["template"] for tc in w_n150["test_cases"]]
        assert "AudioTranscriptionParamTest" not in templates

    def test_audio_t3k_model_targets(self):
        """t3k suites should have model-specific timing targets from model_targets."""
        suites = load_suite_files_by_category("audio")
        suite_map = {s["id"]: s for s in suites}

        dw_t3k = suite_map["distil-whisper-t3k"]
        assert len(dw_t3k["test_cases"]) == 4
        load_30s = dw_t3k["test_cases"][0]
        assert load_30s["targets"]["audio_transcription_time"] == 4

        w_t3k = suite_map["whisper-t3k"]
        load_30s = w_t3k["test_cases"][0]
        assert load_30s["targets"]["audio_transcription_time"] == 5

    def test_audio_galaxy_dp2_tests(self):
        """Galaxy suites should include DP2 tests with shared targets."""
        suites = load_suite_files_by_category("audio")
        suite_map = {s["id"]: s for s in suites}

        for suite_id in ["distil-whisper-galaxy", "whisper-galaxy"]:
            suite = suite_map[suite_id]
            assert len(suite["test_cases"]) == 6
            dp2_5 = suite["test_cases"][0]
            assert dp2_5["template"] == "AudioTranscriptionLoadDp2Chunk5Test"
            assert dp2_5["targets"] == {"num_concurrent": 64, "dataset": "60s"}

    def test_audio_galaxy_model_specific_targets(self):
        suites = load_suite_files_by_category("audio")
        suite_map = {s["id"]: s for s in suites}

        dw_galaxy = suite_map["distil-whisper-galaxy"]
        load_30s = next(
            tc
            for tc in dw_galaxy["test_cases"]
            if tc["description"] == "Test audio 30s load"
        )
        assert load_30s["targets"]["audio_transcription_time"] == 2

        w_galaxy = suite_map["whisper-galaxy"]
        load_30s = next(
            tc
            for tc in w_galaxy["test_cases"]
            if tc["description"] == "Test audio 30s load"
        )
        assert load_30s["targets"]["audio_transcription_time"] == 4


class TestEmbeddingMatrixExpansion:
    """Validate that the migrated embedding.json produces the same suites as before."""

    ORIGINAL_IDS = {
        "bge-n150",
        "bge-t3k",
        "bge-galaxy",
        "qwen3-emb-8b-n150",
        "qwen3-emb-8b-t3k",
        "qwen3-emb-8b-galaxy",
    }

    def test_embedding_suite_count(self):
        suites = load_suite_files_by_category("embedding")
        assert len(suites) == 6

    def test_embedding_suite_ids(self):
        suites = load_suite_files_by_category("embedding")
        ids = {s["id"] for s in suites}
        assert ids == self.ORIGINAL_IDS

    def test_embedding_n150_targets(self):
        suites = load_suite_files_by_category("embedding")
        suite_map = {s["id"]: s for s in suites}

        for suite_id in ["bge-n150", "qwen3-emb-8b-n150"]:
            tc = suite_map[suite_id]["test_cases"][0]
            assert tc["targets"]["embedding_time"] == 0.1

    def test_embedding_other_device_targets(self):
        suites = load_suite_files_by_category("embedding")
        suite_map = {s["id"]: s for s in suites}

        for suite_id in [
            "bge-t3k",
            "bge-galaxy",
            "qwen3-emb-8b-t3k",
            "qwen3-emb-8b-galaxy",
        ]:
            tc = suite_map[suite_id]["test_cases"][0]
            assert tc["targets"]["embedding_time"] == 1

    def test_embedding_weights(self):
        suites = load_suite_files_by_category("embedding")
        suite_map = {s["id"]: s for s in suites}

        assert suite_map["bge-n150"]["weights"] == ["bge-large-en-v1.5"]
        assert suite_map["qwen3-emb-8b-t3k"]["weights"] == ["Qwen3-Embedding-8B"]


class TestTtsMatrixExpansion:
    """Validate that the migrated tts.json produces the same suites as before."""

    def test_tts_suite_count(self):
        suites = load_suite_files_by_category("text_to_speech")
        assert len(suites) == 2

    def test_tts_suite_ids(self):
        suites = load_suite_files_by_category("text_to_speech")
        ids = {s["id"] for s in suites}
        assert ids == {"speecht5-n150", "speecht5-p150"}

    def test_tts_test_cases_match(self):
        suites = load_suite_files_by_category("text_to_speech")
        for suite in suites:
            assert len(suite["test_cases"]) == 4
            templates = [tc["template"] for tc in suite["test_cases"]]
            assert templates == [
                "SpeechT5TTSTest",
                "TestTTSServerHealth",
                "TTSParamTest",
                "TTSIntegrationTest",
            ]

    def test_tts_health_test_targets(self):
        suites = load_suite_files_by_category("text_to_speech")
        for suite in suites:
            health_test = suite["test_cases"][1]
            assert health_test["targets"] == {
                "tts_generation_time": 10,
                "sample_count": 10,
                "compare_audio": True,
            }


class TestImageMatrixExpansion:
    """Validate that the migrated image.json produces the same suites as before."""

    ORIGINAL_IMAGE_IDS = {
        "sdxl-n150",
        "sdxl-t3k",
        "sdxl-galaxy",
        "sdxl-n300",
        "sdxl-p150x8",
        "sdxl-p300x2",
        "sdxl-img2img-n150",
        "sdxl-img2img-t3k",
        "sdxl-img2img-galaxy",
        "sdxl-inpaint-n150",
        "sdxl-inpaint-t3k",
        "sdxl-inpaint-galaxy",
        "sd35-t3k",
        "sd35-galaxy",
        "flux-dev-t3k",
        "flux-dev-galaxy",
        "flux-dev-p150x4",
        "flux-dev-p150x8",
        "flux-dev-p300",
        "flux-dev-p300x2",
        "flux-schnell-t3k",
        "flux-schnell-galaxy",
        "flux-schnell-p150x4",
        "flux-schnell-p150x8",
        "flux-schnell-p300",
        "flux-schnell-p300x2",
        "motif-t3k",
        "motif-galaxy",
        "motif-p150x4",
        "motif-p150x8",
        "motif-p300x2",
    }

    def test_image_suite_count(self):
        suites = load_suite_files_by_category("image")
        assert len(suites) == 31

    def test_image_suite_ids(self):
        suites = load_suite_files_by_category("image")
        ids = {s["id"] for s in suites}
        assert ids == self.ORIGINAL_IMAGE_IDS

    def test_sdxl_full_lora_suites(self):
        """n150, t3k, galaxy should have 7 test cases including LoRA tests."""
        suites = load_suite_files_by_category("image")
        suite_map = {s["id"]: s for s in suites}

        for suite_id in ["sdxl-n150", "sdxl-t3k", "sdxl-galaxy"]:
            suite = suite_map[suite_id]
            assert len(suite["test_cases"]) == 7, f"{suite_id}: expected 7 test cases"
            templates = [tc["template"] for tc in suite["test_cases"]]
            assert "ImageGenerationEvalsTest" in templates
            assert "ImageGenerationLoraLoadTest" in templates

    def test_sdxl_galaxy_timing_differs(self):
        """Galaxy should have different LoadTest timing (11/15/25 vs 10/14/23)."""
        suites = load_suite_files_by_category("image")
        suite_map = {s["id"]: s for s in suites}

        galaxy = suite_map["sdxl-galaxy"]
        load_tests = [
            tc
            for tc in galaxy["test_cases"]
            if tc["template"] == "ImageGenerationLoadTest"
        ]
        times = [lt["targets"]["image_generation_time"] for lt in load_tests]
        assert times == [11, 15, 25]

        n150 = suite_map["sdxl-n150"]
        load_tests = [
            tc
            for tc in n150["test_cases"]
            if tc["template"] == "ImageGenerationLoadTest"
        ]
        times = [lt["targets"]["image_generation_time"] for lt in load_tests]
        assert times == [10, 14, 23]

    def test_sdxl_reduced_suites(self):
        """n300, p150x8, p300x2 should have 4 test cases (no LoRA)."""
        suites = load_suite_files_by_category("image")
        suite_map = {s["id"]: s for s in suites}

        for suite_id in ["sdxl-n300", "sdxl-p150x8", "sdxl-p300x2"]:
            suite = suite_map[suite_id]
            assert len(suite["test_cases"]) == 4, f"{suite_id}: expected 4 test cases"

    def test_sdxl_reduced_num_devices(self):
        suites = load_suite_files_by_category("image")
        suite_map = {s["id"]: s for s in suites}

        assert suite_map["sdxl-n300"]["num_of_devices"] == 1
        assert suite_map["sdxl-p150x8"]["num_of_devices"] == 4
        assert suite_map["sdxl-p300x2"]["num_of_devices"] == 2

    def test_flux_dev_t3k_timing(self):
        """flux-dev on t3k has unique timing (16/23/36)."""
        suites = load_suite_files_by_category("image")
        suite_map = {s["id"]: s for s in suites}

        suite = suite_map["flux-dev-t3k"]
        load_tests = [
            tc
            for tc in suite["test_cases"]
            if tc["template"] == "ImageGenerationLoadTest"
        ]
        times = [lt["targets"]["image_generation_time"] for lt in load_tests]
        assert times == [16, 23, 36]

    def test_flux_dev_param_test_model_target(self):
        """ParamTest should have model-specific target from model_targets."""
        suites = load_suite_files_by_category("image")
        suite_map = {s["id"]: s for s in suites}

        suite = suite_map["flux-dev-t3k"]
        param_test = next(
            tc
            for tc in suite["test_cases"]
            if tc["template"] == "ImageGenerationParamTest"
        )
        assert param_test["targets"]["model"] == "FLUX.1-dev"

    def test_flux_schnell_param_test_model_target(self):
        suites = load_suite_files_by_category("image")
        suite_map = {s["id"]: s for s in suites}

        suite = suite_map["flux-schnell-galaxy"]
        param_test = next(
            tc
            for tc in suite["test_cases"]
            if tc["template"] == "ImageGenerationParamTest"
        )
        assert param_test["targets"]["model"] == "FLUX.1-schnell"

    def test_motif_excludes_p300(self):
        """motif should not have a p300 suite (not in compatible_devices)."""
        suites = load_suite_files_by_category("image")
        ids = {s["id"] for s in suites}
        assert "motif-p300" not in ids

    def test_motif_galaxy_timing(self):
        suites = load_suite_files_by_category("image")
        suite_map = {s["id"]: s for s in suites}

        suite = suite_map["motif-galaxy"]
        load_tests = [
            tc
            for tc in suite["test_cases"]
            if tc["template"] == "ImageGenerationLoadTest"
        ]
        times = [lt["targets"]["image_generation_time"] for lt in load_tests]
        assert times == [11, 15, 25]

    def test_sd35_per_device_timing(self):
        suites = load_suite_files_by_category("image")
        suite_map = {s["id"]: s for s in suites}

        t3k = suite_map["sd35-t3k"]
        load_tests = [
            tc
            for tc in t3k["test_cases"]
            if tc["template"] == "ImageGenerationLoadTest"
        ]
        times = [lt["targets"]["image_generation_time"] for lt in load_tests]
        assert times == [15, 20, 25]

        galaxy = suite_map["sd35-galaxy"]
        load_tests = [
            tc
            for tc in galaxy["test_cases"]
            if tc["template"] == "ImageGenerationLoadTest"
        ]
        times = [lt["targets"]["image_generation_time"] for lt in load_tests]
        assert times == [16, 22, 28]

    def test_no_model_targets_in_output(self):
        """model_targets should never leak into expanded suites."""
        suites = load_suite_files_by_category("image")
        for suite in suites:
            for tc in suite["test_cases"]:
                assert "model_targets" not in tc, (
                    f"model_targets leaked in {suite['id']}/{tc.get('template')}"
                )


class TestAllSuitesLoad:
    """Verify that all suite files load correctly together."""

    def test_total_suite_count(self):
        all_suites = load_suite_files()
        assert len(all_suites) == 64

    def test_no_duplicate_ids(self):
        all_suites = load_suite_files()
        ids = [s["id"] for s in all_suites]
        assert len(ids) == len(set(ids)), (
            f"Duplicate IDs found: {[x for x in ids if ids.count(x) > 1]}"
        )

    def test_all_suites_have_required_fields(self):
        all_suites = load_suite_files()
        for suite in all_suites:
            assert "id" in suite, f"Missing 'id' in suite: {suite}"
            assert "weights" in suite, f"Missing 'weights' in suite {suite['id']}"
            assert "device" in suite, f"Missing 'device' in suite {suite['id']}"
            assert "model_marker" in suite, (
                f"Missing 'model_marker' in suite {suite['id']}"
            )
            assert "test_cases" in suite, f"Missing 'test_cases' in suite {suite['id']}"
