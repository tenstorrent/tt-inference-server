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


def iter_speaker_clips(split: str, count: int):
    """Yield ``(speaker_id, wav_bytes, text)`` for ``count`` distinct speakers.

    Streams the dataset with audio decode disabled, so ``wav_bytes`` is the
    row's original (soundfile-readable) file content — ready to base64 into a
    TTS ``reference_audio`` field without any audio backend. One clip per
    speaker, first come first served, deterministic for a fixed split.
    """
    from datasets import load_dataset

    dataset = load_dataset(
        "blabble-io/libritts_r", "clean", split=resolve_split(split), streaming=True
    )
    dataset = disable_audio_decode(dataset)

    seen = set()
    for sample in dataset:
        speaker = str(sample.get("speaker_id"))
        if speaker in seen:
            continue
        audio = sample.get("audio") or {}
        wav_bytes = audio.get("bytes")
        if not wav_bytes:
            continue
        seen.add(speaker)
        yield (
            speaker,
            wav_bytes,
            sample.get("text_normalized") or sample.get("text_original", ""),
        )
        if len(seen) >= count:
            return


__all__ = [
    "LIBRITTS_SPLIT_ALIASES",
    "resolve_split",
    "disable_audio_decode",
    "iter_speaker_clips",
]
