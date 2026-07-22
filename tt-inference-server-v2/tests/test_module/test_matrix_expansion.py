# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Validate SDXL matrix expansion in v2's image.json and video.json."""

from __future__ import annotations

from test_module.test_categorization_system.suite_loader import (
    load_suite_files_by_category,
)


class TestImageMatrixExpansionSDXL:
    SDXL_SUITE_IDS = {
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
    }

    def test_sdxl_suite_ids_present(self):
        suites = load_suite_files_by_category("image")
        ids = {s["id"] for s in suites}
        missing = self.SDXL_SUITE_IDS - ids
        assert not missing, f"SDXL suites missing from v2 image.json: {missing}"

    def test_sdxl_full_lora_suites(self):
        """n150, t3k, galaxy should have 6 test cases including LoRA tests."""
        suites = load_suite_files_by_category("image")
        suite_map = {s["id"]: s for s in suites}

        for suite_id in ["sdxl-n150", "sdxl-t3k", "sdxl-galaxy"]:
            suite = suite_map[suite_id]
            assert len(suite["test_cases"]) == 6, f"{suite_id}: expected 6 test cases"
            templates = [tc["template"] for tc in suite["test_cases"]]
            assert "ImageGenerationEvalsTest" in templates
            assert "ImageGenerationLoraLoadTest" in templates

    def test_sdxl_galaxy_timing_differs(self):
        """Per-device LoadTest timing: galaxy 20/28/45, n150 12/16/23 (Forge
        path ~11-15s/img), t3k 10/14/23."""
        suites = load_suite_files_by_category("image")
        suite_map = {s["id"]: s for s in suites}

        galaxy = suite_map["sdxl-galaxy"]
        load_tests = [
            tc
            for tc in galaxy["test_cases"]
            if tc["template"] == "ImageGenerationLoadTest"
        ]
        times = [lt["targets"]["image_generation_time"] for lt in load_tests]
        assert times == [20, 28, 45]

        n150 = suite_map["sdxl-n150"]
        load_tests = [
            tc
            for tc in n150["test_cases"]
            if tc["template"] == "ImageGenerationLoadTest"
        ]
        times = [lt["targets"]["image_generation_time"] for lt in load_tests]
        assert times == [12, 16, 23]

        t3k = suite_map["sdxl-t3k"]
        load_tests = [
            tc
            for tc in t3k["test_cases"]
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

    def test_sdxl_img2img_suites_present(self):
        suites = load_suite_files_by_category("image")
        suite_map = {s["id"]: s for s in suites}

        for suite_id in [
            "sdxl-img2img-n150",
            "sdxl-img2img-t3k",
            "sdxl-img2img-galaxy",
        ]:
            suite = suite_map[suite_id]
            templates = [tc["template"] for tc in suite["test_cases"]]
            assert templates == ["Img2ImgGenerationParamTest"]

    def test_sdxl_inpaint_suites_present(self):
        suites = load_suite_files_by_category("image")
        suite_map = {s["id"]: s for s in suites}

        for suite_id in [
            "sdxl-inpaint-n150",
            "sdxl-inpaint-t3k",
            "sdxl-inpaint-galaxy",
        ]:
            suite = suite_map[suite_id]
            templates = [tc["template"] for tc in suite["test_cases"]]
            assert templates == ["InpaintingGenerationParamTest"]


class TestVideoMatrixExpansion:
    """Regression coverage for video.json matrix expansion + per-device targets.

    VIDEO is served exclusively by v2 (routed via workflows/v2_bridge), so this
    suite is the only expansion/target guard for live video config — the v1
    server_tests video.json and its TestVideoMatrixExpansion were removed.
    """

    # Expected expanded per-device suite ids (one per model x device).
    EXPECTED_SUITE_IDS = {
        "wan-t3k",
        "wan-galaxy",
        "wan-p150x4",
        "wan-p150x8",
        "wan-p300x2",
        "wan-i2v-t3k",
        "wan-i2v-galaxy",
        "wan-i2v-p150x4",
        "wan-i2v-p150x8",
        "wan-i2v-p300x2",
        "mochi-t3k",
        "mochi-galaxy",
        "mochi-p150x4",
        "mochi-p150x8",
        "mochi-p300x2",
    }

    # Expected VideoGenerationLoadTest targets per expanded suite: the base
    # template targets merged with per-device model_targets overrides. p150x4
    # has no override, so it keeps the bare base target (no poll_timeout).
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
    # Expected VideoGenerationI2VTest targets per expanded suite. p150x4 keeps
    # the base poll_timeout (no per-device override).
    WAN_I2V_TARGETS = {
        "wan-i2v-t3k": {
            "num_inference_steps": 40,
            "poll_timeout": 1500,
            "poll_interval": 5,
        },
        "wan-i2v-galaxy": {
            "num_inference_steps": 40,
            "poll_timeout": 550,
            "poll_interval": 5,
        },
        "wan-i2v-p150x4": {
            "num_inference_steps": 40,
            "poll_timeout": 1200,
            "poll_interval": 5,
        },
        "wan-i2v-p150x8": {
            "num_inference_steps": 40,
            "poll_timeout": 900,
            "poll_interval": 5,
        },
        "wan-i2v-p300x2": {
            "num_inference_steps": 40,
            "poll_timeout": 800,
            "poll_interval": 5,
        },
    }

    @staticmethod
    def _suite_map():
        suites = load_suite_files_by_category("video")
        return {s["id"]: s for s in suites}

    def _case_targets(self, suite_id, template):
        cases = [
            tc
            for tc in self._suite_map()[suite_id]["test_cases"]
            if tc["template"] == template
        ]
        assert len(cases) == 1, f"{suite_id}: expected exactly one {template}"
        return cases[0]["targets"]

    def test_video_suite_ids_match(self):
        assert set(self._suite_map()) == self.EXPECTED_SUITE_IDS

    def test_t2v_suites_have_load_and_param_cases(self):
        suite_map = self._suite_map()
        for suite_id in ["wan-t3k", "wan-p150x4", "mochi-t3k", "mochi-p300x2"]:
            templates = [tc["template"] for tc in suite_map[suite_id]["test_cases"]]
            assert templates == [
                "VideoGenerationLoadTest",
                "VideoGenerationParamTest",
            ], f"{suite_id}: unexpected templates {templates}"

    def test_i2v_suites_include_i2v_param_test(self):
        # The I2V param sweep is a live test case; guard that every I2V suite
        # still carries both the happy-path and the param sweep after expansion.
        suite_map = self._suite_map()
        for suite_id in self.WAN_I2V_TARGETS:
            templates = [tc["template"] for tc in suite_map[suite_id]["test_cases"]]
            assert templates == [
                "VideoGenerationI2VTest",
                "VideoGenerationI2VParamTest",
            ], f"{suite_id}: unexpected templates {templates}"

    def test_all_video_suites_single_device(self):
        for suite in self._suite_map().values():
            assert suite["num_of_devices"] == 1

    def test_wan_load_targets_merge_per_device(self):
        for suite_id, expected in self.WAN_LOAD_TARGETS.items():
            assert (
                self._case_targets(suite_id, "VideoGenerationLoadTest") == expected
            ), suite_id

    def test_mochi_load_targets_merge_per_device(self):
        for suite_id, expected in self.MOCHI_LOAD_TARGETS.items():
            assert (
                self._case_targets(suite_id, "VideoGenerationLoadTest") == expected
            ), suite_id

    def test_wan_i2v_targets_merge_per_device(self):
        for suite_id, expected in self.WAN_I2V_TARGETS.items():
            assert self._case_targets(suite_id, "VideoGenerationI2VTest") == expected, (
                suite_id
            )
