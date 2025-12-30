# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Test suites for AUDIO model category (Whisper, Distil-Whisper).

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
STABILITY_TEST_CONFIG = TestConfig(test_timeout=3600, retry_attempts=1, retry_delay=60)


def _load_test(
    description: str,
    transcription_time: float,
    dataset: str = "30s",
    num_devices: int = None,
) -> TestCase:
    """Create an AudioTranscriptionLoadTest case."""
    targets = {"audio_transcription_time": transcription_time, "dataset": dataset}
    if num_devices is not None:
        targets["num_of_devices"] = num_devices
    return TestCase(
        test_class=TestClasses.AudioTranscriptionLoadTest,  # ← Find Usages works!
        description=description,
        targets=targets,
        config=LOAD_TEST_CONFIG,
        markers=["load", "e2e", "slow"],
    )


def _param_test(description: str = "Test audio transcription params") -> TestCase:
    """Create an AudioTranscriptionParamTest case."""
    return TestCase(
        test_class=TestClasses.AudioTranscriptionParamTest,  # ← Cmd+Click works!
        description=description,
        config=PARAM_TEST_CONFIG,
        markers=["param", "e2e", "slow"],
    )


def _stability_test(description: str = "Device stability test") -> TestCase:
    """Create a DeviceStabilityTest case."""
    return TestCase(
        test_class=TestClasses.DeviceStabilityTest,
        description=description,
        config=STABILITY_TEST_CONFIG,
        markers=["stability", "e2e", "slow", "heavy"],
    )


# =========================================================================
# Distil-Whisper Suites
# =========================================================================

_DISTIL_WHISPER_N150 = TestSuite(
    id="distil-whisper-n150",
    weights=["distil-large-v3"],
    device=Device.N150,
    model_marker="distil_whisper",
    test_cases=[
        _load_test("Test audio 30s load", transcription_time=0.7),
        _load_test("Test audio 60s load", transcription_time=4, dataset="60s"),
        _param_test(),
        _stability_test(),
    ],
)

_DISTIL_WHISPER_T3K = TestSuite(
    id="distil-whisper-t3k",
    weights=["distil-large-v3"],
    device=Device.T3K,
    model_marker="distil_whisper",
    test_cases=[
        _load_test("Test audio 30s load", transcription_time=4),
        _load_test("Test audio 60s load", transcription_time=5, dataset="60s"),
        _load_test(
            "Test single audio 60s transcription and expect chunking",
            transcription_time=2,
            dataset="60s",
            num_devices=1,
        ),
        _param_test(),
    ],
)

_DISTIL_WHISPER_GALAXY = TestSuite(
    id="distil-whisper-galaxy",
    weights=["distil-large-v3"],
    device=Device.GALAXY,
    model_marker="distil_whisper",
    test_cases=[
        _load_test("Test audio 30s load", transcription_time=2),
        _load_test("Test audio 60s load", transcription_time=2, dataset="60s"),
        _load_test(
            "Test single audio 60s transcription and expect chunking",
            transcription_time=2,
            dataset="60s",
            num_devices=1,
        ),
        _param_test(),
    ],
)

# =========================================================================
# Whisper Large V3 Suites
# =========================================================================

_WHISPER_N150 = TestSuite(
    id="whisper-n150",
    weights=["whisper-large-v3"],
    device=Device.N150,
    model_marker="whisper",
    test_cases=[
        _load_test("Test audio 30s load", transcription_time=3),
        _load_test("Test audio 60s load", transcription_time=5, dataset="60s"),
        _load_test(
            "Test single audio 60s transcription and expect chunking",
            transcription_time=3,
            dataset="60s",
            num_devices=1,
        ),
    ],
)

_WHISPER_T3K = TestSuite(
    id="whisper-t3k",
    weights=["whisper-large-v3"],
    device=Device.T3K,
    model_marker="whisper",
    test_cases=[
        _load_test("Test audio 30s load", transcription_time=4),
        _load_test("Test audio 60s load", transcription_time=6, dataset="60s"),
        _load_test(
            "Test single audio 60s transcription and expect chunking",
            transcription_time=3,
            dataset="60s",
            num_devices=1,
        ),
        _param_test(),
    ],
)

_WHISPER_GALAXY = TestSuite(
    id="whisper-galaxy",
    weights=["whisper-large-v3"],
    device=Device.GALAXY,
    model_marker="whisper",
    test_cases=[
        _load_test("Test audio 30s load", transcription_time=3),
        _load_test("Test audio 60s load", transcription_time=6, dataset="60s"),
        _load_test(
            "Test single audio 60s transcription and expect chunking",
            transcription_time=3,
            dataset="60s",
            num_devices=1,
        ),
        _param_test(),
    ],
)

# =========================================================================
# All Audio Suites
# =========================================================================

_AUDIO_SUITE_OBJECTS = [
    _DISTIL_WHISPER_N150,
    _DISTIL_WHISPER_T3K,
    _DISTIL_WHISPER_GALAXY,
    _WHISPER_N150,
    _WHISPER_T3K,
    _WHISPER_GALAXY,
]

# Export as dict format for backward compatibility with suite_loader
AUDIO_SUITES = suites_to_dicts(_AUDIO_SUITE_OBJECTS)
