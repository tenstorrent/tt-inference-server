# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Test suites for CNN model category (ResNet, VoVNet, MobileNet, etc.).

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

# Placeholder for CNN test suites
# Add test suites here following the pattern in audio.py and image.py
_CNN_SUITE_OBJECTS = []

# Export as dict format for backward compatibility with suite_loader
CNN_SUITES = suites_to_dicts(_CNN_SUITE_OBJECTS)
