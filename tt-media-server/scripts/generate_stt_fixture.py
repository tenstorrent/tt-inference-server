#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# One-time script to generate tests/fixtures/librispeech_6s.wav
# from HuggingFace LibriSpeech samples (streaming — no full dataset download).
#
# Usage:
#   pip install datasets soundfile numpy
#   python scripts/generate_stt_fixture.py

import wave
from pathlib import Path

import numpy as np

TARGET_DURATION_S = 6.0
TARGET_SR = 16000
OUTPUT_DIR = Path(__file__).parent.parent / "tests" / "fixtures"


def main():
    try:
        import datasets
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Install datasets first: pip install datasets soundfile numpy")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import io
        import soundfile as sf
    except ImportError:
        raise SystemExit("Install soundfile first: pip install soundfile")

    print("Streaming LibriSpeech test-clean from HuggingFace (no full download)...")
    # decode_audio=False avoids the torchcodec dependency — we decode with soundfile instead
    ds = load_dataset(
        "librispeech_asr",
        "clean",
        split="test",
        streaming=True,
    ).cast_column("audio", datasets.Audio(decode=False))

    accumulated: list[np.ndarray] = []
    texts: list[str] = []
    total_samples = 0

    for sample in ds:
        raw = sample["audio"]["bytes"]
        if raw is None:
            # Some entries store a path rather than inline bytes; skip
            continue
        pcm, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)

        if sr != TARGET_SR:
            try:
                import librosa

                pcm = librosa.resample(pcm, orig_sr=sr, target_sr=TARGET_SR)
            except ImportError:
                print(f"  skipping sample (sr={sr}, librosa not installed)")
                continue

        accumulated.append(pcm)
        texts.append(sample["text"])
        total_samples += len(pcm)
        print(f"  accumulated {total_samples / TARGET_SR:.1f}s")

        if total_samples / TARGET_SR >= TARGET_DURATION_S:
            break

    full = np.concatenate(accumulated)[: int(TARGET_DURATION_S * TARGET_SR)]
    pcm16 = (full * 32767).clip(-32768, 32767).astype(np.int16)

    wav_path = OUTPUT_DIR / "librispeech_6s.wav"
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_SR)
        wf.writeframes(pcm16.tobytes())
    print(f"\nWrote: {wav_path}  ({wav_path.stat().st_size / 1024:.0f} KB)")

    transcript = " ".join(texts)
    txt_path = OUTPUT_DIR / "librispeech_6s_transcript.txt"
    txt_path.write_text(transcript)
    print(f"Wrote: {txt_path}")
    print(f"Transcript preview: {transcript[:120]}...")


if __name__ == "__main__":
    main()
