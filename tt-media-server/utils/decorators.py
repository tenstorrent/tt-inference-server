# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import inspect
import os
import time
from functools import wraps

from telemetry.telemetry_client import TelemetryEvent, get_telemetry_client

from utils.logger import TTLogger

logger = TTLogger()

log_time = os.getenv("LOG_LEVEL", "INFO").upper() in [
    "DEBUG",
    "INFO",
    "ERROR",
    "WARNING",
]


def debug_execution_time(
    message=None, telemetry_event_name: TelemetryEvent = None, device_id=None
):
    def decorator(func):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            duration = 0.0
            status = True

            try:
                result = func(
                    *args, **kwargs
                )  # Any error in func() will be caught here
                duration = time.time() - start

                # Record success telemetry
                logger.info(
                    f"[{func.__name__}] executed in {duration:.4f} seconds. {message or ''}"
                )
                return result

            except Exception as e:
                duration = time.time() - start

                # Record failure telemetry
                logger.error(
                    f"[{func.__name__}] failed after {duration:.4f} seconds. Error: {e}"
                )
                raise
            finally:
                get_telemetry_client().record_telemetry_event_async(
                    event_name=telemetry_event_name,
                    device_id=device_id,
                    duration=duration,
                    status=status,
                ) if telemetry_event_name else None

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            duration = 0.0
            status = True

            try:
                result = await func(
                    *args, **kwargs
                )  # Any error in async func() will be caught here
                duration = time.time() - start

                # Record success telemetry
                logger.info(
                    f"[{func.__name__}] async executed in {duration:.4f} seconds. {message or ''}"
                )
                return result

            except Exception as e:
                duration = time.time() - start

                # Record failure telemetry
                logger.error(
                    f"[{func.__name__}] async failed after {duration:.4f} seconds. Error: {e}"
                )
                raise
            finally:
                get_telemetry_client().record_telemetry_event_async(
                    event_name=telemetry_event_name,
                    device_id=device_id,
                    duration=duration,
                    status=status,
                ) if telemetry_event_name else None

        @wraps(func)
        async def async_generator_wrapper(*args, **kwargs):
            start = time.time()
            duration = 0.0
            yielded_count = 0
            status = True

            try:
                async for item in func(
                    *args, **kwargs
                ):  # Any error in async generator will be caught here
                    yielded_count += 1
                    yield item

                duration = time.time() - start

                # Record success telemetry
                logger.info(
                    f"[{func.__name__}] async generator completed in {duration:.4f} seconds. Yielded {yielded_count} items. {message or ''}"
                )

            except Exception as e:
                duration = time.time() - start

                # Record failure telemetry
                logger.error(
                    f"[{func.__name__}] async generator failed after {duration:.4f} seconds. Yielded {yielded_count} items. Error: {e}"
                )

                raise
            finally:
                get_telemetry_client().record_telemetry_event_async(
                    event_name=telemetry_event_name,
                    device_id=device_id,
                    duration=duration,
                    status=status,
                ) if telemetry_event_name else None

        if inspect.isasyncgenfunction(func):
            return async_generator_wrapper
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def log_execution_time(*args, **kwargs):
    """Only log execution time when in debug mode"""
    if log_time:
        return debug_execution_time(*args, **kwargs)
    else:
        # Return identity decorator (no-op)
        def identity(func):
            return func

        return identity
