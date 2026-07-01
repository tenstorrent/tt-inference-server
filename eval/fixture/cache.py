import time

_cache = {}


def get_or_set(key, fn, ttl):
    if key in _cache:
        value, _ = _cache[key]
        return value
    value = fn()
    _cache[key] = (value, time.time())
    return value
