# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import time
import inspect
from functools import wraps

def log_execution_time(message=None):
    def decorator(func):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start
            print(f"[{func.__name__}] executed in {duration:.4f} seconds. {message or ''}")
            return result

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            result = await func(*args, **kwargs)
            duration = time.time() - start
            print(f"[{func.__name__}] async executed in {duration:.4f} seconds. {message or ''}")
            return result

        @wraps(func)
        async def async_generator_wrapper(*args, **kwargs):
            start = time.time()
            yielded_count = 0
            async for item in func(*args, **kwargs):
                yielded_count += 1
                yield item

            duration = time.time() - start
            print(f"[{func.__name__}] async generator completed in {duration:.4f} seconds. Yielded {yielded_count} items. {message or ''}")

        if inspect.isasyncgenfunction(func):
            return async_generator_wrapper
        elif inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    return decorator
