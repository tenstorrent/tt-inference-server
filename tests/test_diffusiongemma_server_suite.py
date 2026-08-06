# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from test_module.test_categorization_system.suite_loader import (
    load_server_tests_config,
    load_suite_files_by_category,
)


def test_diffusiongemma_server_suite_expands_on_p300x2():
    suites = load_suite_files_by_category("llm")
    suite = next(
        suite for suite in suites if suite["id"] == "diffusiongemma-26b-a4b-it-p300x2"
    )

    assert suite["weights"] == ["diffusiongemma-26B-A4B-it"]
    assert suite["device"] == "p300x2"
    assert suite["test_cases"] == [
        {
            "template": "VLLMDiffusionGemmaParamConformanceTest",
            "enabled": True,
            "description": ("DiffusionGemma block-serving and request-admission gates"),
        }
    ]


def test_diffusiongemma_server_template_points_to_model_specific_suite():
    config = load_server_tests_config()
    template = config["test_templates"]["VLLMDiffusionGemmaParamConformanceTest"]

    assert template["module"] == "test_module.llm_tests.vllm_param_conformance_test"
    assert {"param", "e2e", "slow", "heavy"} <= set(template["markers"])
