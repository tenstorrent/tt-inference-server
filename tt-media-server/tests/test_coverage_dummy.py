# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Tests for coverage_test_dummy module.
Tests MOST functions to achieve >50% coverage.
Delete this file after verifying coverage works.
"""

import sys
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from coverage_test_dummy import (
    add_numbers,
    divide_numbers,
    multiply_numbers,
    uncovered_function,
)


class TestCoverageDummy:
    """Test class for coverage dummy functions."""

    def test_add_numbers_positive(self):
        """Test adding positive numbers."""
        assert add_numbers(2, 3) == 5

    def test_add_numbers_negative(self):
        """Test adding negative numbers."""
        assert add_numbers(-1, 1) == 0

    def test_add_numbers_zero(self):
        """Test adding zeros."""
        assert add_numbers(0, 0) == 0

    def test_multiply_numbers_positive(self):
        """Test multiplying positive numbers."""
        assert multiply_numbers(2, 3) == 6

    def test_multiply_numbers_negative(self):
        """Test multiplying with negative numbers."""
        assert multiply_numbers(-2, 3) == -6

    def test_multiply_numbers_zero(self):
        """Test multiplying by zero."""
        assert multiply_numbers(0, 5) == 0

    def test_divide_numbers(self):
        """Test dividing numbers."""
        assert divide_numbers(6, 2) == 3.0
        assert divide_numbers(5, 2) == 2.5

    def test_divide_by_zero(self):
        """Test divide by zero raises error."""
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            divide_numbers(5, 0)

    def test_uncovered_function(self):
        """Test the 'uncovered' function to get >50% coverage."""
        result = uncovered_function()
        assert result == 45  # sum of 0-9

    # NOTE: another_uncovered_function is intentionally NOT tested
    # to demonstrate partial coverage (should still pass >50%)
