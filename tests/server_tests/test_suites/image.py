# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Test suites for IMAGE model category (SDXL, SD3.5, Img2Img, Inpainting).

Uses Python-native types with TestClasses references for:
- Cmd+Click to navigate to TestClasses definition
- Find Usages on TestClasses constants
- No import-time dependency loading
"""

from server_tests.test_suites.types import (
    Device,
    TestCase,
    TestClasses,
    TestConfig,
    TestSuite,
    suites_to_dicts,
)

# Shared configs
LOAD_TEST_CONFIG = TestConfig(test_timeout=3600, retry_attempts=1, retry_delay=60)
PARAM_TEST_CONFIG = TestConfig(test_timeout=3600, retry_attempts=1, retry_delay=60)


def _load_test(num_steps: int, gen_time: float) -> TestCase:
    """Create an ImageGenerationLoadTest case."""
    return TestCase(
        test_class=TestClasses.ImageGenerationLoadTest,  # ← Find Usages works!
        description=f"{num_steps} iterations",
        targets={"num_inference_steps": num_steps, "image_generation_time": gen_time},
        config=LOAD_TEST_CONFIG,
        markers=["load", "e2e", "slow", "heavy"],
    )


def _param_test(description: str = "Param validation") -> TestCase:
    """Create an ImageGenerationParamTest case."""
    return TestCase(
        test_class=TestClasses.ImageGenerationParamTest,  # ← Cmd+Click works!
        description=description,
        config=PARAM_TEST_CONFIG,
        markers=["param", "e2e", "slow"],
    )


def _standard_image_tests(time_20: float, time_30: float, time_50: float = None) -> list:
    """Create standard set of image generation tests (20, 30, optionally 50 iterations)."""
    tests = [
        _load_test(20, time_20),
        _load_test(30, time_30),
    ]
    if time_50 is not None:
        tests.append(_load_test(50, time_50))
    tests.append(_param_test())
    return tests


# =========================================================================
# SDXL Base Suites
# =========================================================================

_SDXL_N150 = TestSuite(
    id="sdxl-n150",
    weights=["stable-diffusion-xl-base-1.0"],
    device=Device.N150,
    model_marker="sdxl",
    test_cases=_standard_image_tests(time_20=10, time_30=14, time_50=23),
)

_SDXL_T3K = TestSuite(
    id="sdxl-t3k",
    weights=["stable-diffusion-xl-base-1.0"],
    device=Device.T3K,
    model_marker="sdxl",
    test_cases=_standard_image_tests(time_20=10, time_30=14, time_50=23),
)

_SDXL_GALAXY = TestSuite(
    id="sdxl-galaxy",
    weights=["stable-diffusion-xl-base-1.0"],
    device=Device.GALAXY,
    model_marker="sdxl",
    test_cases=_standard_image_tests(time_20=11, time_30=15, time_50=25),
)

# =========================================================================
# SDXL Img2Img Suites
# =========================================================================

_SDXL_IMG2IMG_N150 = TestSuite(
    id="sdxl-img2img-n150",
    weights=["stable-diffusion-xl-base-1.0-img-2-img"],
    device=Device.N150,
    model_marker="sdxl_img2img",
    test_cases=_standard_image_tests(time_20=12, time_30=16),
)

_SDXL_IMG2IMG_T3K = TestSuite(
    id="sdxl-img2img-t3k",
    weights=["stable-diffusion-xl-base-1.0-img-2-img"],
    device=Device.T3K,
    model_marker="sdxl_img2img",
    test_cases=_standard_image_tests(time_20=12, time_30=16),
)

_SDXL_IMG2IMG_GALAXY = TestSuite(
    id="sdxl-img2img-galaxy",
    weights=["stable-diffusion-xl-base-1.0-img-2-img"],
    device=Device.GALAXY,
    model_marker="sdxl_img2img",
    test_cases=_standard_image_tests(time_20=13, time_30=17),
)

# =========================================================================
# SDXL Inpainting Suites
# =========================================================================

_SDXL_INPAINT_N150 = TestSuite(
    id="sdxl-inpaint-n150",
    weights=["stable-diffusion-xl-1.0-inpainting-0.1"],
    device=Device.N150,
    model_marker="sdxl_inpaint",
    test_cases=_standard_image_tests(time_20=12, time_30=16),
)

_SDXL_INPAINT_T3K = TestSuite(
    id="sdxl-inpaint-t3k",
    weights=["stable-diffusion-xl-1.0-inpainting-0.1"],
    device=Device.T3K,
    model_marker="sdxl_inpaint",
    test_cases=_standard_image_tests(time_20=12, time_30=16),
)

_SDXL_INPAINT_GALAXY = TestSuite(
    id="sdxl-inpaint-galaxy",
    weights=["stable-diffusion-xl-1.0-inpainting-0.1"],
    device=Device.GALAXY,
    model_marker="sdxl_inpaint",
    test_cases=_standard_image_tests(time_20=13, time_30=17),
)

# =========================================================================
# Stable Diffusion 3.5 Suites
# =========================================================================

_SD35_T3K = TestSuite(
    id="sd35-t3k",
    weights=["stable-diffusion-3.5-large"],
    device=Device.T3K,
    model_marker="sd35",
    test_cases=_standard_image_tests(time_20=15, time_30=20),
)

_SD35_GALAXY = TestSuite(
    id="sd35-galaxy",
    weights=["stable-diffusion-3.5-large"],
    device=Device.GALAXY,
    model_marker="sd35",
    test_cases=_standard_image_tests(time_20=16, time_30=22),
)

# =========================================================================
# All Image Suites
# =========================================================================

_IMAGE_SUITE_OBJECTS = [
    _SDXL_N150,
    _SDXL_T3K,
    _SDXL_GALAXY,
    _SDXL_IMG2IMG_N150,
    _SDXL_IMG2IMG_T3K,
    _SDXL_IMG2IMG_GALAXY,
    _SDXL_INPAINT_N150,
    _SDXL_INPAINT_T3K,
    _SDXL_INPAINT_GALAXY,
    _SD35_T3K,
    _SD35_GALAXY,
]

# Export as dict format for backward compatibility with suite_loader
IMAGE_SUITES = suites_to_dicts(_IMAGE_SUITE_OBJECTS)
