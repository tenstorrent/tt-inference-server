# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.


# Add server path to sys.path for imports
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.append(str(Path(__file__).parent.parent))

from tt_model_runners.forge_runners.forge_runner import ForgeRunner


class TestForgeRunner:
    """Test suite for ForgeRunner functionality"""

    @pytest.fixture
    def runner(self):
        """Create a ForgeRunner instance for testing"""
        with patch("tt_model_runners.forge_runners.forge_runner.ModelLoader"):
            return ForgeRunner(device_id="test_device_0")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
