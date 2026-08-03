# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Shared helpers for the LibriTTS-R dataset used by the TTS tests.

The TTS quality eval and the TTS load test both stream samples from the
``blabble-io/libritts_r`` dataset and only read text fields, so the split
resolution and the audio-decode suppression logic live here to keep the two
callers from diverging.
"""

LIBRITTS_SPLIT_ALIASES = {
    "test": "test.clean",
    "test.clean": "test.clean",
    "test.other": "test.other",
    "dev": "dev.clean",
    "validation": "dev.clean",
    "dev.clean": "dev.clean",
    "dev.other": "dev.other",
    "train": "train.clean.100",
    "train.clean.100": "train.clean.100",
    "train.clean.360": "train.clean.360",
}


def resolve_split(split: str) -> str:
    """Map a requested split onto a valid LibriTTS-R split name."""
    return LIBRITTS_SPLIT_ALIASES.get(split, split)


def disable_audio_decode(dataset):
    """Return ``dataset`` with its audio column left as raw (undecoded) bytes.

    The TTS tests only read text fields, so decoding the audio column is
    wasted work and, with recent ``datasets`` releases, needs the optional
    ``torchcodec`` backend just to iterate. Disabling decode keeps the
    dependency footprint on ``datasets``/``librosa`` alone.
    """
    from datasets import Audio

    if "audio" in (getattr(dataset, "column_names", None) or []):
        return dataset.cast_column("audio", Audio(decode=False))
    return dataset


__all__ = ["LIBRITTS_SPLIT_ALIASES", "resolve_split", "disable_audio_decode"]
