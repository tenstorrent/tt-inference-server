import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auth import validate


def test_valid_token_returns_true():
    assert validate("validtoken123") is True


def test_valid_long_alnum():
    assert validate("abcdefgh") is True


def test_empty_token():
    assert validate("") is False


def test_none_token():
    assert validate(None) is False


def test_short_token():
    assert validate("abc") is False


def test_non_alnum_token():
    assert validate("invalid-token") is False
