import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from formatter import format_price


def test_basic_usd():
    assert format_price(9.99, "USD") == "USD 9.99"


def test_zero_amount():
    assert format_price(0, "EUR") == "EUR 0.00"


def test_integer_amount():
    assert format_price(10, "GBP") == "GBP 10.00"


def test_large_amount():
    assert format_price(1234.5, "JPY") == "JPY 1234.50"
