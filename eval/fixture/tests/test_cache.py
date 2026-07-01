import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import cache as cache_mod
from cache import get_or_set


def setup_function():
    cache_mod._cache.clear()


def test_stores_value():
    call_count = [0]

    def fn():
        call_count[0] += 1
        return 42

    assert get_or_set("k", fn, 10) == 42
    assert call_count[0] == 1


def test_cache_hit_no_extra_call():
    call_count = [0]

    def fn():
        call_count[0] += 1
        return 99

    get_or_set("k2", fn, 10)
    get_or_set("k2", fn, 10)
    assert call_count[0] == 1


def test_ttl_expiry_triggers_refresh():
    call_count = [0]

    def fn():
        call_count[0] += 1
        return "fresh"

    get_or_set("k3", fn, 0.05)
    time.sleep(0.15)
    result = get_or_set("k3", fn, 0.05)
    assert call_count[0] == 2
    assert result == "fresh"
