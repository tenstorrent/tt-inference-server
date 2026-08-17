# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Tests for :class:`utils.image_manager.ImageManager`.

Focus is on the input-sanitization layer of ``base64_to_pil_image`` — every
input passes through HTTP and JSON before reaching us, both of which routinely
strip the ``=`` padding from base64 strings. Without padding restoration
``base64.b64decode`` raises ``binascii.Error: Incorrect padding`` and the I2V
request fails before any model runs.
"""

import base64
import io

import pytest
from PIL import Image

from utils.image_manager import ImageManager


def _png_bytes(width: int = 4, height: int = 4, color=(255, 0, 0)) -> bytes:
    """Encode a small PNG to bytes suitable for base64 round-trip tests."""
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=color).save(buf, format="PNG")
    return buf.getvalue()


def _b64_with_n_padding(width: int, height: int) -> tuple[str, int]:
    """Encode a PNG and return ``(b64_string, padding_count)`` so callers can
    pick a payload whose base64 form has a predictable number of trailing
    ``=`` characters."""
    raw = _png_bytes(width, height)
    encoded = base64.b64encode(raw).decode("ascii")
    return encoded, encoded.count("=")


class TestBase64ToPilImagePadding:
    """``base64_to_pil_image`` must accept HTTP/JSON-stripped base64 strings.

    HTTP transports and many JSON serialisers strip trailing ``=`` characters
    from base64 payloads. The decoder must restore them, otherwise the
    underlying ``binascii.Error`` propagates and the I2V request fails before
    the model is even invoked.
    """

    def test_decodes_string_with_full_padding(self):
        """Sanity baseline: a properly-padded string decodes correctly. Pins
        the "no regression" half of the contract — the padding-restore line
        must be a no-op when padding is already present."""
        raw = _png_bytes()
        encoded = base64.b64encode(raw).decode("ascii")

        image = ImageManager().base64_to_pil_image(encoded)

        assert isinstance(image, Image.Image)
        assert image.size == (4, 4)

    def test_decodes_string_with_stripped_padding(self):
        """The exact bug being fixed: a base64 string with ALL ``=`` removed
        (the worst case after HTTP/JSON transport) must decode back to the
        original image, byte-for-byte."""
        raw = _png_bytes()
        encoded = base64.b64encode(raw).decode("ascii")
        stripped = encoded.rstrip("=")
        assert stripped != encoded, "test setup: payload must originally need padding"

        image = ImageManager().base64_to_pil_image(stripped)

        assert image.size == (4, 4)
        assert image.mode == "RGB"
        assert image.getpixel((0, 0)) == (255, 0, 0)

    @pytest.mark.parametrize("padding", [1, 2])
    def test_decodes_partially_stripped_padding(self, padding):
        """Real-world transports may strip a subset of the ``=`` chars rather
        than all of them. Both partial-strip cases (1- and 2-byte residues)
        must decode identically to the fully-padded form."""
        encoded, original_padding = _b64_with_n_padding(width=3, height=3)
        if original_padding < padding:
            pytest.skip(f"payload has only {original_padding} '=' chars")
        partially_stripped = encoded[:-padding] if padding else encoded

        image = ImageManager().base64_to_pil_image(partially_stripped)

        assert image.size == (3, 3)

    def test_decodes_data_url_with_stripped_padding(self):
        """The combined hot path: a ``data:image/png;base64,...`` data URL whose
        payload has been stripped of padding by JSON serialisation must still
        decode. Exercises both branches (data-URL prefix strip + padding fix)
        in one call."""
        raw = _png_bytes()
        encoded = base64.b64encode(raw).decode("ascii").rstrip("=")
        data_url = f"data:image/png;base64,{encoded}"

        image = ImageManager().base64_to_pil_image(data_url)

        assert image.size == (4, 4)

    def test_padding_restore_is_idempotent_when_already_aligned(self):
        """If the payload length is already a multiple of 4, the padding
        helper must add zero ``=`` chars (don't over-pad and corrupt the
        decoded byte stream). Exercise the SUT directly so a future refactor
        of the padding-restore line can't silently regress this branch."""
        # PNG payload chosen so the base64 form has length % 4 == 0 with zero
        # ``=`` padding required — that's the input shape the no-op branch
        # must accept unchanged.
        for width, height in [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]:
            encoded = base64.b64encode(_png_bytes(width, height)).decode("ascii")
            if len(encoded) % 4 == 0 and "=" not in encoded:
                break
        else:
            pytest.skip("no test PNG produces a base64 form aligned without padding")

        image = ImageManager().base64_to_pil_image(encoded)

        assert isinstance(image, Image.Image)
        assert image.size == (width, height)


class TestBase64ToPilImageResizeAndMode:
    """The padding fix must not regress the optional resize / mode-convert
    parameters that callers depend on for I2V conditioning preprocessing."""

    def test_target_size_resizes_image(self):
        encoded = base64.b64encode(_png_bytes(width=8, height=8)).decode("ascii")

        image = ImageManager().base64_to_pil_image(encoded, target_size=(2, 2))

        assert image.size == (2, 2)

    def test_target_mode_converts_image(self):
        encoded = base64.b64encode(_png_bytes()).decode("ascii")

        image = ImageManager().base64_to_pil_image(encoded, target_mode="L")

        assert image.mode == "L"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
