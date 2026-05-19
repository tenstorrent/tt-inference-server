# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Validate SDXL matrix expansion in v2's image.json.

Ported from v1's tests/test_matrix_expansion.py when SDXL ownership
moved to v2.
"""

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
