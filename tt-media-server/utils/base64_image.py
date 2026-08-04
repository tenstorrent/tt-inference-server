# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Base64 image handling that request schemas can reach for.

Deliberately stdlib-only. ``utils.image_manager`` imports torch, numpy and PIL,
so a DTO validator cannot call into it without dragging the whole model stack
into request parsing — and into every process that merely imports the domain
layer. The decode half of that module lives here instead, and
``ImageManager.base64_to_pil_image`` calls it, so the API boundary and the device
runner agree byte-for-byte on what "decodable" means. If they disagreed we would
be back to accepting requests the runner then rejects on the device, which is
what #4811 was about.

The validation here is a header sniff, not a full decode. That is the right
depth for a request boundary: it costs nothing, needs no image library, and
catches the whole class of failure in the issue ("cannot identify image file" is
PIL failing to match a header). Payloads that pass the sniff but fail the real
decode later — a truncated PNG, say — are caught in ``ImageManager`` and raised
as :class:`~domain.errors.ClientRequestError` there, so they still never count
against worker health.
"""

from __future__ import annotations

import base64
import binascii

from domain.errors import ClientRequestError

__all__ = [
    "decode_base64_image",
    "normalize_base64_payload",
    "sniff_image_format",
    "validate_base64_image",
]

# Header signatures for the formats the video and image pipelines actually read.
# ``(offset, magic, name)`` — WebP needs the offset form because its magic sits
# after the RIFF chunk size.
_MAGIC_BYTES: tuple[tuple[int, bytes, str], ...] = (
    (0, b"\x89PNG\r\n\x1a\n", "png"),
    (0, b"\xff\xd8\xff", "jpeg"),
    (0, b"GIF87a", "gif"),
    (0, b"GIF89a", "gif"),
    (0, b"BM", "bmp"),
    (0, b"II*\x00", "tiff"),
    (0, b"MM\x00*", "tiff"),
    (8, b"WEBP", "webp"),
)

# Longest prefix any check above needs.
_SNIFF_BYTES = 12


def normalize_base64_payload(payload: str) -> str:
    """Strip a ``data:`` URL wrapper and whitespace, and restore lost padding.

    Three transport artefacts to undo, in order:

    * a ``data:image/png;base64,`` prefix,
    * embedded whitespace — MIME-style base64 (what the ``base64`` CLI emits) is
      wrapped at 76 columns, and ``b64decode`` only tolerates that with
      ``validate=False``, which also silently swallows genuine junk,
    * stripped ``=`` padding, which HTTP transports and many JSON serialisers
      drop.

    Doing this in one place is what keeps the API boundary check and the runner's
    decode from disagreeing about which payloads are valid.
    """
    if payload.startswith("data:"):
        _, _, payload = payload.partition(",")
    payload = "".join(payload.split())
    return payload + "=" * (-len(payload) % 4)


def decode_base64_image(payload: str, *, what: str = "image") -> bytes:
    """Decode *payload* to raw bytes, or raise ``ClientRequestError`` (400).

    ``validate=True`` so stray non-alphabet characters are an error instead of
    being silently discarded — otherwise ``"!!!not-base64!!!"`` decodes to a few
    junk bytes and the failure surfaces much later, as an undiagnosable image
    error inside the device worker.
    """
    normalized = normalize_base64_payload(payload)
    try:
        return base64.b64decode(normalized, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ClientRequestError(
            f"Could not decode {what}: {payload[:32]!r}... is not valid base64 ({e})"
        ) from e


def sniff_image_format(raw: bytes) -> str | None:
    """Return the image format *raw* starts with, or None if it isn't an image."""
    for offset, magic, name in _MAGIC_BYTES:
        if raw[offset : offset + len(magic)] == magic:
            return name
    return None


def validate_base64_image(payload: str, *, what: str = "image") -> str:
    """Return *payload* unchanged if it is a base64-encoded image, else raise.

    Raises :class:`~domain.errors.ClientRequestError` (400) so the same failure
    reads identically whether it is hit during request parsing or deeper in a
    runner. Pydantic wraps it into a 422 when used as a field validator, which is
    the correct status for a malformed request body.
    """
    raw = decode_base64_image(payload, what=what)
    if sniff_image_format(raw) is None:
        supported = sorted({name for _, _, name in _MAGIC_BYTES})
        raise ClientRequestError(
            f"Could not decode {what} ({len(raw)} bytes): the payload is valid "
            f"base64 but not a recognised image. Supported: {supported}."
        )
    return payload
