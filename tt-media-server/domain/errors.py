# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Request-caused failures, told apart from device faults (#4811).

A per-request failure has two very different meanings depending on its cause:

* **The client sent something we cannot serve.** Retrying is pointless, the
  answer is 4xx, and the worker is perfectly healthy.
* **The device or the runner broke.** The answer is 5xx, and enough of these in
  a row mean the worker should be restarted.

``Scheduler.error_listener`` used to treat both identically, so six bad-input
requests walked a healthy worker past ``max_worker_restart_count`` into a
restart (see #4811). Everything here exists so that the distinction survives the
trip from wherever the failure happened — including inside a device worker
process, or on the far side of the SHM boundary in the multihost peer — all the
way to the code that decides whether a worker lives.

Three carriers, in order of preference:

1. :class:`ClientRequestError` raised directly by whatever validated the input.
2. The ``"NNN: "`` prefix the peer's stringified ``HTTPException`` leaves on
   ``VideoResponse.error_message`` (see :func:`parse_peer_status`).
3. :data:`CLIENT_INPUT_EXCEPTIONS` — a deliberately narrow set of stdlib
   exception types that only ever mean "the input was malformed". This is the
   backstop for a rogue parameter nobody has validated yet: it cannot produce
   the right HTTP status detail, but it does keep the failure off the worker's
   health record.
"""

from __future__ import annotations

import binascii
import re
import struct

from fastapi import HTTPException
from pydantic import ValidationError

__all__ = [
    "CLIENT_INPUT_EXCEPTIONS",
    "ClientRequestError",
    "classify_worker_error",
    "is_client_error",
    "parse_peer_status",
]


class ClientRequestError(HTTPException):
    """A failure caused by the request, not by the worker serving it.

    Subclasses ``HTTPException`` on purpose:

    * FastAPI already knows how to turn it into a response, so an endpoint's
      existing ``except HTTPException: raise`` hands the caller the right status
      with no extra plumbing.
    * ``str(exc)`` renders as ``"400: <detail>"``, which is exactly the wire
      format the multihost peer writes into SHM and :func:`parse_peer_status`
      reads back. Raising this inside a runner keeps that loop closed for free.

    ``HTTPException.__init__`` never calls ``super().__init__``, which leaves
    ``args`` empty and makes the default pickle protocol reconstruct the
    exception with no arguments — a ``TypeError`` on unpickle. These instances
    cross a ``multiprocessing.Queue`` on every worker error, so ``__reduce__``
    is defined explicitly rather than left to chance.
    """

    def __init__(self, detail: str, status_code: int = 400) -> None:
        super().__init__(status_code=status_code, detail=detail)

    def __reduce__(self):
        return (self.__class__, (self.detail, self.status_code))


# Narrow by design. These types mean "the bytes/values handed to us were the
# wrong shape" and nothing else:
#   * ``binascii.Error``      — base64 that does not decode
#   * ``struct.error``        — a value that does not fit its SHM field, e.g. a
#                               ``seed`` outside signed 64-bit
#   * ``UnicodeDecodeError``  — text that is not valid UTF-8
#   * ``ValidationError``     — a DTO rebuilt from a cross-process payload
# ``ValueError`` and ``OSError`` are deliberately absent: device and runner code
# raise those too, and misclassifying one of those would blind the watchdog.
CLIENT_INPUT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ClientRequestError,
    binascii.Error,
    struct.error,
    UnicodeDecodeError,
    ValidationError,
)

# ``str(HTTPException(400, "..."))`` -> ``"400: ..."``. The multihost peer
# stringifies its exception into ``VideoResponse.error_message``, so this prefix
# is the only channel a 4xx has across the SHM boundary.
_PEER_STATUS_RE = re.compile(r"^(4\d{2}): ")


def parse_peer_status(error_message: object) -> int | None:
    """Return the 4xx status the peer prefixed onto *error_message*, else None.

    Only 4xx is recognised. A 5xx prefix, or no prefix at all, means "treat as a
    worker fault" — the conservative direction, since a missed classification
    costs one false restart while a wrong one would stop the watchdog from ever
    restarting a genuinely sick worker.
    """
    if not isinstance(error_message, str):
        return None
    match = _PEER_STATUS_RE.match(error_message)
    return int(match.group(1)) if match else None


def is_client_error(exc: BaseException) -> bool:
    """True when *exc* was caused by the request rather than by the worker."""
    if isinstance(exc, ClientRequestError):
        return True
    # A 4xx HTTPException raised anywhere in the runner stack is a client error
    # by definition, whether or not it used our subclass.
    if isinstance(exc, HTTPException):
        return 400 <= exc.status_code < 500
    if isinstance(exc, CLIENT_INPUT_EXCEPTIONS):
        return True
    # Peer errors arrive wrapped: SPRunner raises
    # ``RuntimeError("Runner error for task X: 400: ...")``. SPRunner converts
    # these itself, but classify from the message too so an unconverted path
    # still cannot cost the worker its life.
    return _peer_status_in(str(exc)) is not None


def _peer_status_in(message: str) -> int | None:
    """Find a peer ``NNN: `` prefix anywhere in *message*, not just at the start.

    ``SPRunner`` prefixes the peer's message with its own context, so the status
    ends up mid-string.
    """
    for chunk in message.split(": "):
        status = parse_peer_status(f"{chunk}: ")
        if status is not None:
            return status
    return None


def classify_worker_error(
    worker_id: str,
    exc: BaseException,
    prefix: str = "",
) -> ClientRequestError | str:
    """Build the ``error_queue`` payload for *exc*.

    Returns a :class:`ClientRequestError` when the request is at fault — the
    scheduler keys off the type, so this must stay an instance and never be
    flattened into a string — and ``f"{prefix}{exc}"`` otherwise, which is the
    long-standing wire format for worker faults.
    """
    if isinstance(exc, ClientRequestError):
        return exc

    if is_client_error(exc):
        status = (
            exc.status_code
            if isinstance(exc, HTTPException)
            else _peer_status_in(str(exc)) or 400
        )
        detail = getattr(exc, "detail", None) or str(exc) or type(exc).__name__
        return ClientRequestError(str(detail), status_code=status)

    return f"{prefix}{exc}"
