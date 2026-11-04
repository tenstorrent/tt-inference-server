# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""What rank 0 sends to the other ranks, and how, over tt-metal's MPI context (no mpi4py, no SHM).

The messages and the request <-> plain-data conversion come first, then the
functions that carry them (``broadcast``, ``receive``, ``barrier``). Those are
blocking point-to-point calls from rank 0 to every other rank, usable only in a
process that has opened the mesh, which is what starts MPI in tt-metal. Messages
are pickled and preceded by a length header so the receiver knows how many
bytes to read.
"""

from __future__ import annotations

import importlib
import pickle
import struct
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

# MPI guarantees tags up to 32767. A dedicated tag keeps these messages apart
# from any other point-to-point traffic on the world communicator.
MESSAGE_TAG = 0x7E57
# The rank that sends every message: the one serving HTTP.
ROOT_RANK = 0
_LENGTH_HEADER = struct.Struct("!Q")


@dataclass(frozen=True)
class RequestData:
    """A request as plain data: its class, its fields and its private attributes."""

    module: str
    qualname: str
    fields: dict[str, Any]
    private: dict[str, Any]


@dataclass(frozen=True)
class RunMessage:
    requests: list[Any]


@dataclass(frozen=True)
class CanaryMessage:
    deep: bool


@dataclass(frozen=True)
class ShutdownMessage:
    pass


def to_request_data(request: BaseModel) -> RequestData:
    """Plain-data form of ``request`` that any rank can unpickle.

    Every field (prompts, ``image_prompts``, ``references``, ``aspect_ratio``,
    ...) and ``_task_id`` travel unchanged. ``_start_event`` is left out: it is
    a Manager Event proxy rank 0's job manager waits on, and another host can't
    unpickle it.
    """
    private = {
        name: value
        for name, value in (request.__pydantic_private__ or {}).items()
        if name != "_start_event"
    }
    cls = type(request)
    return RequestData(
        module=cls.__module__,
        qualname=cls.__qualname__,
        fields=dict(request.__dict__),
        private=private,
    )


def from_request_data(data: RequestData) -> BaseModel:
    """Rebuild the request :func:`to_request_data` produced."""
    cls: Any = importlib.import_module(data.module)
    for part in data.qualname.split("."):
        cls = getattr(cls, part)
    # model_construct: the fields were validated on rank 0 already, and warmup-style
    # requests deliberately hold values validation would reject.
    request = cls.model_construct(**data.fields)
    for name, value in data.private.items():
        setattr(request, name, value)
    return request


def broadcast(message: Any) -> None:
    """Rank 0 only: send ``message`` to every other rank."""
    import ttnn

    rank = int(ttnn.distributed_context_get_rank())
    if rank != ROOT_RANK:
        raise RuntimeError(f"broadcast called on rank {rank}")
    payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    header = _LENGTH_HEADER.pack(len(payload))
    for dest in range(ROOT_RANK + 1, int(ttnn.distributed_context_get_size())):
        ttnn.distributed_context_send_bytes(header, dest, MESSAGE_TAG)
        ttnn.distributed_context_send_bytes(payload, dest, MESSAGE_TAG)


def receive() -> Any:
    """Ranks 1..N-1: block until rank 0 broadcasts, and return the message."""
    import ttnn

    header = ttnn.distributed_context_recv_bytes(
        _LENGTH_HEADER.size, ROOT_RANK, MESSAGE_TAG
    )
    (length,) = _LENGTH_HEADER.unpack(header)
    payload = ttnn.distributed_context_recv_bytes(length, ROOT_RANK, MESSAGE_TAG)
    return pickle.loads(payload)


def barrier() -> None:
    import ttnn

    ttnn.distributed_context_barrier()
