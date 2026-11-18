# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""TT weight-transfer backends for vLLM's native RL weight-sync API.

This package plugs a Tenstorrent device-to-device transport (tt-metal's
``WeightBridge`` over a ``ttnn.MeshSocket``) into vLLM's pluggable
``WeightTransferEngine`` abstraction so the co-located RL trainer can drive
weight updates through vLLM's *native* endpoints
(``/init_weight_transfer_engine``, ``/update_weights``, ``/pause``,
``/resume``) instead of the bespoke ``/v1/internal/weights/*`` control plane.

Importing this package is intentionally light: the engine module (which imports
``ttnn``) is only loaded on demand, so registering the backend or importing
this package from the API-server process does not drag in ttnn.
"""

from typing import TYPE_CHECKING, Any

# Module path of the engine, registered lazily with vLLM's factory so ttnn is
# only imported when a worker actually constructs the engine.
_ENGINE_MODULE = "tt_vllm_plugin.weight_transfer.tt_device_socket_engine"
_ENGINE_CLASS = "TTDeviceSocketWeightTransferEngine"
BACKEND_NAME = "device_socket"

if TYPE_CHECKING:
    from tt_vllm_plugin.weight_transfer.tt_device_socket_engine import (
        TTDeviceSocketWeightTransferEngine,
        TTWeightTransferInitInfo,
        TTWeightTransferUpdateInfo,
    )

__all__ = [
    "BACKEND_NAME",
    "TTDeviceSocketWeightTransferEngine",
    "TTWeightTransferInitInfo",
    "TTWeightTransferUpdateInfo",
    "register_tt_weight_transfer_engine",
]


def register_tt_weight_transfer_engine() -> None:
    """Register the TT ``device_socket`` backend with vLLM's factory.

    Idempotent and light: uses the factory's lazy (module-path) registration,
    so ttnn is not imported here -- only when ``create_engine`` is called on a
    worker. Only needed to drive the engine via
    ``--weight-transfer-config '{"backend": "device_socket"}'``; the TT worker
    can also construct the engine directly.
    """
    from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory

    try:
        WeightTransferEngineFactory.register_engine(
            BACKEND_NAME, _ENGINE_MODULE, _ENGINE_CLASS
        )
    except ValueError:
        # Already registered (idempotent re-import).
        pass


def __getattr__(name: str) -> Any:
    """Lazily expose the engine symbols without importing ttnn at package load."""
    if name in {
        "TTDeviceSocketWeightTransferEngine",
        "TTWeightTransferInitInfo",
        "TTWeightTransferUpdateInfo",
    }:
        import importlib

        module = importlib.import_module(_ENGINE_MODULE)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
