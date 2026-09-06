# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Exception types that must survive the worker-process boundary.

``CpuWorkloadHandler`` runs service pre-processing in a separate process and
marshals failures back to the event loop through a ``multiprocessing.Queue``,
which cannot carry a live exception object. It therefore sends the message plus
the exception's class name, and the parent rebuilds the type from
:data:`RECONSTRUCTIBLE_ERRORS`.

Without that, every worker failure arrived as a bare ``Exception`` and route
handlers could only map it to 500 - including client mistakes like submitting
over-length audio, which should be a 4xx.

This module deliberately imports nothing from the package: ``cpu_workload_handler``
imports it, so anything heavier would risk an import cycle.
"""


class AudioTooLongError(ValueError):
    """Submitted audio exceeds the runner's maximum duration.

    Subclasses ValueError so existing ``except ValueError`` callers still catch
    it; the route layer maps it to 400 rather than the generic 500.
    """


# Only types listed here are rebuilt on the parent side. Anything else stays a
# generic Exception, preserving the previous behaviour for unknown failures.
RECONSTRUCTIBLE_ERRORS = {cls.__name__: cls for cls in (AudioTooLongError,)}


def reconstruct_worker_error(type_name, message):
    """Rebuild a worker exception from its class name, falling back to Exception.

    ``type_name`` is None for workers that predate the three-element error
    payload, so the fallback also covers a rolling upgrade.
    """
    error_class = RECONSTRUCTIBLE_ERRORS.get(type_name)
    if error_class is None:
        return Exception(message)
    return error_class(message)
