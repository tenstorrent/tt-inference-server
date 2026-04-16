#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""
Minimal WebSocket test client for /ws/stt-live.

Dev-only deps: pip install sounddevice numpy websockets soundfile

Usage:
    python3 live_stt_example.py                         # live mic
    python3 live_stt_example.py --wav sample.wav        # replay a WAV file
    STT_WS_URL=ws://host:7002/ws/stt-live python3 live_stt_example.py
"""

import argparse
import asyncio
import json
import os
import sys
import time

import numpy as np

SAMPLE_RATE = 16000
CHUNK_INTERVAL = 2.0


def float_to_pcm16(audio_float: np.ndarray) -> bytes:
    pcm = np.clip(audio_float, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    return pcm.tobytes()


async def stream_wav(ws_url: str, wav_path: str) -> None:
    import soundfile as sf
    import websockets

    data, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        raise SystemExit(f"WAV must be {SAMPLE_RATE} Hz, got {sr}")

    async with websockets.connect(ws_url) as ws:
        accumulated: list[np.ndarray] = []
        step = int(CHUNK_INTERVAL * SAMPLE_RATE)
        pos = 0
        while pos < len(data):
            end = min(pos + step, len(data))
            accumulated.append(data[pos:end])
            pos = end
            combined = np.concatenate(accumulated)
            await ws.send(float_to_pcm16(combined))
            is_final = pos >= len(data)
            await ws.send(json.dumps({"action": "final" if is_final else "transcribe"}))
            response = json.loads(await ws.recv())
            _print_response(response)
            if not is_final:
                await asyncio.sleep(CHUNK_INTERVAL)


async def stream_mic(ws_url: str) -> None:
    import sounddevice as sd
    import websockets

    buffer: list[np.ndarray] = []
    recording = True

    def callback(indata, frames, time_info, status):
        if recording:
            buffer.append(indata[:, 0].copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, callback=callback, blocksize=1024
    )
    stream.start()
    print("Recording. Ctrl-C to stop.", flush=True)

    try:
        async with websockets.connect(ws_url) as ws:
            try:
                while True:
                    await asyncio.sleep(CHUNK_INTERVAL)
                    if not buffer:
                        continue
                    combined = np.concatenate(buffer)
                    await ws.send(float_to_pcm16(combined))
                    await ws.send(json.dumps({"action": "transcribe"}))
                    response = json.loads(await ws.recv())
                    _print_response(response)
            except KeyboardInterrupt:
                pass
            finally:
                recording = False
                stream.stop()
                stream.close()
                if buffer:
                    combined = np.concatenate(buffer)
                    await ws.send(float_to_pcm16(combined))
                    await ws.send(json.dumps({"action": "final"}))
                    response = json.loads(await ws.recv())
                    _print_response(response, final=True)
    except KeyboardInterrupt:
        stream.stop()
        stream.close()


def _print_response(data: dict, final: bool = False) -> None:
    if data.get("type") == "transcript":
        tag = "FINAL" if (final or data.get("is_final")) else "partial"
        dur = data.get("duration", 0)
        ms = data.get("time_ms", 0)
        print(f"  [{tag}] {dur:.1f}s audio, {ms:.0f}ms inference: {data.get('text', '')!r}", flush=True)
    else:
        print(f"  [msg] {data}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=os.getenv("STT_WS_URL", "ws://localhost:7002/ws/stt-live"),
        help="WebSocket URL",
    )
    parser.add_argument("--wav", default=None, help="Replay this WAV file instead of mic")
    args = parser.parse_args()

    if args.wav:
        asyncio.run(stream_wav(args.url, args.wav))
    else:
        asyncio.run(stream_mic(args.url))


if __name__ == "__main__":
    main()
