# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Pytest configuration for memory_queue tests.
Handles special mocking requirements for numpy and other dependencies.
"""

import sys
from unittest.mock import MagicMock


def pytest_configure(config):
    """Configure pytest with proper mocks for memory_queue tests."""
    # Create a real numpy dtype mock that behaves like numpy

    # We need numpy for the memory_queue to work, so we'll mock it properly
    class MockNumpyDtype:
        def __init__(self, spec):
            # Calculate itemsize based on the spec
            self.itemsize = 100 * 4 + 4 + 450 * 4 + 4  # Approximate size

    class MockNumpy:
        def dtype(self, spec):
            return MockNumpyDtype(spec)

        def ndarray(self, shape, dtype, buffer, offset):
            # Return a mock array that can be indexed
            class MockArray:
                def __init__(self, shape, dtype, buffer, offset):
                    self.shape = shape
                    self.dtype = dtype
                    self.buffer = buffer
                    self.offset = offset
                    self._data = {}

                def __getitem__(self, idx):
                    if idx not in self._data:
                        # Create a mock row
                        class MockRow(dict):
                            def __init__(self):
                                super().__init__()
                                self["task_id"] = ""
                                self["is_final"] = 0
                                self["text"] = ""
                                self["item_available"] = 0

                            def copy(self):
                                row = MockRow()
                                row.update(self)
                                return row

                        self._data[idx] = MockRow()
                    return self._data[idx]

                def __setitem__(self, idx, value):
                    self._data[idx] = value

            return MockArray(shape, dtype, buffer, offset)

    # Only mock numpy if it's mocked in sys.modules
    if "numpy" in sys.modules and isinstance(sys.modules["numpy"], MagicMock):
        sys.modules["numpy"] = MockNumpy()
