# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Custom wheel builder for tt-vllm-plugin.

All package metadata and dependencies are defined in pyproject.toml.
This file only provides the custom BdistWheel class for platform-specific wheel building.
"""

from setuptools import setup
from wheel.bdist_wheel import bdist_wheel


class BdistWheel(bdist_wheel):
    """
    Custom wheel builder for a platform-specific Python package.

    - Marks the wheel as non-pure (`root_is_pure = False`) to ensure proper installation
      of native binaries.
     - Uses the actual Python version from the build environment (supports Python 3.10 and 3.11).
    """

    user_options = bdist_wheel.user_options + [
        ("code-coverage", None, "Enable code coverage for the build")
    ]

    def initialize_options(self):
        super().initialize_options()
        self.code_coverage = False  # Default value for code coverage

    def finalize_options(self):
        if self.code_coverage is None:
            self.code_coverage = False

        bdist_wheel.finalize_options(self)
        self.root_is_pure = False

    def get_tag(self):
        # Use the actual Python version from the environment instead of hardcoding
        return bdist_wheel.get_tag(self)


setup(
    cmdclass={
        "bdist_wheel": BdistWheel,
    },
)
