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

        return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper
    return decorator
