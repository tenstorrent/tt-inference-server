# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import time
import inspect
from functools import wraps

from telemetry.prometheus_metrics import TelmetryEvent, get_telemetry_client

def log_execution_time(message=None, telemetry_event_name: TelmetryEvent = None, device_id=None):
    def decorator(func):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            
            try:
                result = func(*args, **kwargs)  # Any error in func() will be caught here
                duration = time.time() - start
                
                # Record success telemetry
                print(f"[{func.__name__}] executed in {duration:.4f} seconds. {message or ''}")
                get_telemetry_client().record_telemetry_event_async(
                    event_name=telemetry_event_name,
                    device_id=device_id,
                    duration=duration,
                    status=True  # Success
                ) if telemetry_event_name else None
                
                return result
                
            except Exception as e:
                duration = time.time() - start
                
                # Record failure telemetry
                print(f"[{func.__name__}] failed after {duration:.4f} seconds. Error: {e}")
                get_telemetry_client().record_telemetry_event_async(
                    event_name=telemetry_event_name,
                    device_id=device_id,
                    duration=duration,
                    status=False  # Failure
                ) if telemetry_event_name else None

                raise

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            
            try:
                result = await func(*args, **kwargs)  # Any error in async func() will be caught here
                duration = time.time() - start
                
                # Record success telemetry
                print(f"[{func.__name__}] async executed in {duration:.4f} seconds. {message or ''}")
                get_telemetry_client().record_telemetry_event_async(
                    event_name=telemetry_event_name,
                    device_id=device_id,
                    duration=duration,
                    status=True  # Success
                ) if telemetry_event_name else None
                
                return result
                
            except Exception as e:
                duration = time.time() - start
                
                # Record failure telemetry
                print(f"[{func.__name__}] async failed after {duration:.4f} seconds. Error: {e}")
                get_telemetry_client().record_telemetry_event_async(
                    event_name=telemetry_event_name,
                    device_id=device_id,
                    duration=duration,
                    status=False  # Failure
                ) if telemetry_event_name else None

                raise

        @wraps(func)
        async def async_generator_wrapper(*args, **kwargs):
            start = time.time()
            yielded_count = 0
            
            try:
                async for item in func(*args, **kwargs):  # Any error in async generator will be caught here
                    yielded_count += 1
                    yield item

                duration = time.time() - start
                
                # Record success telemetry
                print(f"[{func.__name__}] async generator completed in {duration:.4f} seconds. Yielded {yielded_count} items. {message or ''}")
                get_telemetry_client().record_telemetry_event_async(
                    event_name=telemetry_event_name,
                    device_id=device_id,
                    duration=duration,
                    status=True  # Success
                ) if telemetry_event_name else None
                
            except Exception as e:
                duration = time.time() - start
                
                # Record failure telemetry
                print(f"[{func.__name__}] async generator failed after {duration:.4f} seconds. Yielded {yielded_count} items. Error: {e}")
                get_telemetry_client().record_telemetry_event_async(
                    event_name=telemetry_event_name,
                    device_id=device_id,
                    duration=duration,
                    status=False  # Failure
                ) if telemetry_event_name else None

                raise

        if inspect.isasyncgenfunction(func):
            return async_generator_wrapper
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator
