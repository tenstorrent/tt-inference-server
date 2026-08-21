# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""A ``WeightTransferEngine`` backed by tt-metal's device-socket ``WeightBridge``.

This is the TT backend for vLLM's native RL weight-sync API. It slots the
existing "co-locate via tt-run" transport (an MPI-rendezvoused
``ttnn.MeshSocket`` between the trainer mesh and the inference mesh) behind
vLLM's ``WeightTransferEngine`` contract so the trainer can use vLLM's own
``/init_weight_transfer_engine`` + ``/update_weights`` endpoints.

Division of labour (matches vLLM's design intent -- backends own *transport*,
workers own *apply*):

* This engine owns the ``WeightBridge`` lifecycle: import, connect (the MPI
  handshake + ``MeshSocket`` descriptor exchange), ``recv_state`` and the
  post-transfer ``barrier``.
* The TT worker owns the on-device *apply* (``Generator.update_weights`` ->
  in-place ``ttnn.copy``), the decode-trace release, and the weights-version
  counter -- it passes those in via the ``load_weights`` callback.

Impedance note: vLLM's ``receive_weights`` types ``load_weights`` as
``Callable[[list[tuple[str, torch.Tensor]]], None]`` (a torch, checkpoint-
oriented contract). TT weights are on-device ``ttnn.Tensor`` handles applied
via ``Generator.update_weights(hf_dict, ...)``, so we deliberately pass the
receiver's HF-keyed ``ttnn`` dict through that same callback slot. This works
at runtime (Python is duck-typed) but is the one seam that would benefit from
an upstream generalization of the tensor type on the base engine.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import ttnn

from vllm.config.parallel import ParallelConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)

logger = logging.getLogger(__name__)

# Backend name this engine registers under (WeightTransferConfig.backend).
BACKEND_NAME = "device_socket"

# Pinned filename of tt-metal's weight bridge; the directory is threaded in via
# additional_config['tt']['tt_weight_bridge_dir'] (survives the EngineCore's
# curated env) or the TT_WEIGHT_BRIDGE_DIR env var.
_BRIDGE_FILENAME = "inference_bridge.py"


@dataclass
class TTWeightTransferInitInfo(WeightTransferInitInfo):
    """Init payload sent by the trainer to ``/init_weight_transfer_engine``.

    ``sender_rank`` is the trainer's MPI rank in the shared ``tt-run`` world
    (the ``WeightBridge`` sender, ``TTML_RANK``, default 0). Everything else
    the rendezvous needs (submesh binding, ``FABRIC_2D``) is established at
    launch by ``TT_COLOCATED_INFERENCE=1`` and is not carried on the wire.
    """

    sender_rank: int = 0


@dataclass
class TTWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Per-update payload sent by the trainer to ``/update_weights``.

    ``is_checkpoint_format`` is forced ``False``: the weights arriving over the
    bridge are already on-device ``ttnn`` tensors in kernel format (replicated,
    DRAM-interleaved, TILE, bfloat16), not a HF checkpoint needing layerwise
    reload. ``hf_rope`` is forwarded to ``Generator.update_weights``.
    """

    is_checkpoint_format: bool = False
    hf_rope: bool = False


def _import_weight_bridge(bridge_dir: Optional[str]):
    """Import tt-metal's ``WeightBridge`` from ``inference_bridge.py`` by path.

    Loaded via ``spec_from_file_location`` (not an installed package) so it
    can't collide with the server's own ``utils`` package, and must resolve to
    the exact module the trainer (sender) imports so both ends speak the same
    wire protocol.
    """
    bridge_dir = bridge_dir or os.getenv("TT_WEIGHT_BRIDGE_DIR")
    if not bridge_dir:
        raise ImportError(
            "Weight-bridge directory not configured: set "
            "additional_config['tt']['tt_weight_bridge_dir'] or "
            f"TT_WEIGHT_BRIDGE_DIR to the directory containing {_BRIDGE_FILENAME}."
        )
    path = Path(bridge_dir) / _BRIDGE_FILENAME
    if not path.is_file():
        raise ImportError(
            f"tt-metal's WeightBridge not found at {path}: point "
            "additional_config['tt']['tt_weight_bridge_dir'] (or "
            f"TT_WEIGHT_BRIDGE_DIR) at the directory containing {_BRIDGE_FILENAME}."
        )
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TTDeviceSocketWeightTransferEngine(
    WeightTransferEngine[TTWeightTransferInitInfo, TTWeightTransferUpdateInfo]
):
    """Device-to-device weight transport over tt-metal's ``WeightBridge``."""

    init_info_cls = TTWeightTransferInitInfo
    update_info_cls = TTWeightTransferUpdateInfo

    def __init__(self, config: Any, parallel_config: ParallelConfig) -> None:
        super().__init__(config, parallel_config)
        # Runtime handles injected by the worker via bind_runtime() before
        # init_transfer_engine(); the engine cannot reach the mesh device
        # through the WeightTransferEngine(config, parallel_config) constructor.
        self._device: Optional["ttnn.MeshDevice"] = None
        self._bridge_dir: Optional[str] = None
        self._bridge = None
        self._peer_rank: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Runtime binding (worker-side, not part of the base contract)        #
    # ------------------------------------------------------------------ #

    def bind_runtime(
        self, *, device: "ttnn.MeshDevice", bridge_dir: Optional[str]
    ) -> None:
        """Give the engine the local mesh device + bridge module directory.

        Called by the TT worker once the mesh device is open (after
        ``load_model``) and before ``init_transfer_engine``.
        """
        self._device = device
        self._bridge_dir = bridge_dir

    # ------------------------------------------------------------------ #
    # WeightTransferEngine contract                                       #
    # ------------------------------------------------------------------ #

    def init_transfer_engine(self, init_info: TTWeightTransferInitInfo) -> None:
        """Construct + connect the receiver-side ``WeightBridge`` (role='ttt').

        ``connect()`` is the MPI handshake + ``MeshSocket`` descriptor exchange
        and blocks until the trainer (sender) also reaches ``connect()``. It is
        done once and reused across updates; only re-done if the peer changes.
        """
        if self._device is None:
            raise RuntimeError(
                "TTDeviceSocketWeightTransferEngine.init_transfer_engine called "
                "before bind_runtime(); the mesh device is not available yet."
            )
        sender_rank = int(init_info.sender_rank)
        if self._bridge is not None and self._peer_rank == sender_rank:
            return

        weight_bridge = _import_weight_bridge(self._bridge_dir)

        # The bridge requires an initialized ttnn distributed context.
        if not ttnn.distributed_context_is_initialized():
            ttnn.init_distributed_context()

        logger.info(
            "Creating receiver WeightBridge (role='ttt', peer/sender_rank=%s)",
            sender_rank,
        )
        bridge = weight_bridge.WeightBridge(
            role="ttt",
            peer_rank=sender_rank,
            device=self._device,
        )
        bridge.connect()  # blocks until the trainer also reaches connect()
        self._bridge = bridge
        self._peer_rank = sender_rank

    def receive_weights(
        self,
        update_info: TTWeightTransferUpdateInfo,
        load_weights: Callable[[Any], None],
    ) -> None:
        """Receive one full weight set over the bridge and hand it to the worker.

        ``load_weights`` is the worker-supplied apply callback. It receives the
        HF-keyed dict of on-device ``ttnn.Tensor`` handles produced by
        ``recv_state`` (NOT a torch ``list[(name, tensor)]`` -- see module
        docstring). The worker applies it in place, so we fence and run the
        post-transfer barrier only after ``load_weights`` returns.
        """
        if self._bridge is None:
            raise RuntimeError(
                "receive_weights called before init_transfer_engine(); no bridge."
            )
        logger.info("Receiving weight update via WeightBridge (peer=%s)", self._peer_rank)
        hf_dict = self._bridge.recv_state()
        try:
            # An empty dict is the plumbing-test payload (SIM_PAYLOAD=empty):
            # the worker no-ops the apply but still bumps the version.
            load_weights(hf_dict)
        finally:
            # Drop the received handles and fence so the sender can free its
            # source tensors before inference touches the model again.
            del hf_dict
            ttnn.synchronize_device(self._device)
            self._bridge.barrier()

    def shutdown(self) -> None:
        bridge = self._bridge
        self._bridge = None
        self._peer_rank = None
        if bridge is not None and hasattr(bridge, "release_recv_buffers"):
            try:
                bridge.release_recv_buffers()
            except Exception:  # noqa: BLE001 - best-effort teardown
                logger.exception("WeightBridge.release_recv_buffers failed")

    @staticmethod
    def trainer_send_weights(
        iterator: Iterator[tuple[str, "ttnn.Tensor"]],
        trainer_args: dict[str, Any] | Any,
    ) -> None:
        """Not implemented here: the sender lives in the trainer process.

        On TT the trainer (tt-training-service / tt-train) constructs its own
        ``WeightBridge`` with ``role='ttml'`` and calls ``send_state`` /
        ``transfer_weights`` directly against its autograd mesh. There is no
        vLLM worker in the trainer process, so this static entry point -- meant
        for CUDA setups where the trainer imports vLLM's engine -- does not
        apply. See tt-metal ``grpo/utils/inference_bridge.py``.
        """
        raise NotImplementedError(
            "TT weight sending is driven by tt-train's WeightBridge (role='ttml') "
            "in the trainer process, not through vLLM's trainer_send_weights."
        )
