# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""Turn an lmms-eval audio "visual" into an OpenAI chat ``input_audio`` part.

Audio tasks (fleurs, librispeech, ...) hand the client the HF ``Audio`` column:
a dict with a float ``array`` and its ``sampling_rate``, or a path to an audio
file. The server side (vLLM chat completions) accepts base64 audio under
``{"type": "input_audio", "input_audio": {"data": ..., "format": "wav"}}``,
which is also how the served Gemma 4 audio path was validated by hand.

No lmms-eval import here so the helpers are unit-testable outside the venv.
"""

import base64
import io
import wave
from pathlib import Path

AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus")


def is_audio_visual(visual) -> bool:
    """True for the HF Audio-column dict or an audio file path."""
    if isinstance(visual, dict):
        return "array" in visual and "sampling_rate" in visual
    return isinstance(visual, str) and visual.lower().endswith(AUDIO_SUFFIXES)


def wav_bytes_from_array(array, sampling_rate: int) -> bytes:
    """16-bit PCM mono WAV from a float waveform in [-1, 1] (stereo is averaged)."""
    import numpy as np

    samples = np.asarray(array, dtype=np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=0 if samples.shape[0] < samples.shape[1] else 1)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sampling_rate))
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def wav_bytes_from_visual(visual) -> bytes:
    if isinstance(visual, dict):
        return wav_bytes_from_array(visual["array"], visual["sampling_rate"])
    path = Path(visual)
    if path.suffix.lower() == ".wav":
        return path.read_bytes()
    # Other containers: decode with soundfile (lmms-eval[audio]) and re-encode.
    import soundfile as sf

    data, rate = sf.read(str(path), dtype="float32", always_2d=False)
    if getattr(data, "ndim", 1) == 2:
        data = data.T
    return wav_bytes_from_array(data, rate)


def audio_content_part(visual) -> dict:
    """The OpenAI chat content part carrying ``visual`` as base64 WAV."""
    data = base64.b64encode(wav_bytes_from_visual(visual)).decode("ascii")
    return {"type": "input_audio", "input_audio": {"data": data, "format": "wav"}}
