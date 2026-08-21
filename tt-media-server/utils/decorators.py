# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import inspect
import time
from functools import wraps

from telemetry.telemetry_client import TelemetryEvent, get_telemetry_client

from utils.logger import TTLogger

logger = TTLogger()


def log_execution_time(
    message=None, telemetry_event_name: TelemetryEvent = None, device_id=None
):
    # device_id may be a zero-arg callable: decorators are evaluated at module import,
    # BEFORE the worker sets TT_VISIBLE_DEVICES, so a plain os.environ.get(...) argument
    # freezes as None. A callable is resolved at call time instead.
    def _resolve_device_id():
        return device_id() if callable(device_id) else device_id

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
                    device_id=_resolve_device_id(),
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
                    device_id=_resolve_device_id(),
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
                    device_id=_resolve_device_id(),
                    duration=duration,
                    status=status,
                ) if telemetry_event_name else None

        if inspect.isasyncgenfunction(func):
            return async_generator_wrapper
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
