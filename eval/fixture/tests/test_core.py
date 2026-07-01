import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from core import process


def test_process_none_returns_empty():
    assert process(None) == []


def test_process_empty_list():
    assert process([]) == []


def test_process_strings():
    assert process(["  hello  ", "  world  "]) == ["hello", "world"]


def test_process_mixed():
    assert process([1, "  test  "]) == [1, "test"]
