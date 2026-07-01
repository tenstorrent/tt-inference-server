import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paginator import paginate


def test_first_page():
    items = list(range(1, 10))
    assert paginate(items, 3, 1) == [1, 2, 3]


def test_second_page():
    items = list(range(1, 10))
    assert paginate(items, 3, 2) == [4, 5, 6]


def test_third_page():
    items = list(range(1, 10))
    assert paginate(items, 3, 3) == [7, 8, 9]


def test_partial_last_page():
    items = list(range(1, 8))
    assert paginate(items, 3, 3) == [7]
