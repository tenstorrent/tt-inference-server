#!/usr/bin/env python3
"""
Whisper Server - Persistent Speech-to-Text model server.

Keeps Whisper model loaded on TT device 2, accepts audio via Unix socket,
returns transcription.

Usage (inside container):
  TT_MESH_GRAPH_DESC_PATH=/home/container_app_user/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
  TT_VISIBLE_DEVICES='2' \
  python whisper_server.py

Protocol (Unix socket at /tmp/whisper_server.sock):
  REQ: {"audio_path": "/tmp/audio.wav"} or {"audio_base64": "..."}
  REP: {"status": "ok", "text": "transcribed text", "time_ms": 123.4}
"""

import os
import sys
import json
import time
import socket
import base64
import tempfile
import argparse
import numpy as np
from pathlib import Path

# Add tt-metal to path
sys.path.insert(0, "/home/container_app_user/tt-metal")


def main():
    parser = argparse.ArgumentParser(description="Whisper Server")
    parser.add_argument("--socket", type=str, default="/tmp/whisper_server.sock", help="Unix socket path")
    parser.add_argument("--model", type=str, default="distil-whisper/distil-large-v3", help="Model repo")
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("Whisper Server - Speech-to-Text (Persistent)", flush=True)
    print("=" * 60, flush=True)
    print(f"Socket: {args.socket}", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"TT_VISIBLE_DEVICES: {os.environ.get('TT_VISIBLE_DEVICES', 'not set')}", flush=True)

    import torch
    import ttnn
    from scipy.io import wavfile

    from models.demos.audio.whisper.demo.demo import (
        create_functional_whisper_for_conditional_generation_inference_pipeline,
    )

    # === Open device ===
    print("\n[1/3] Opening TT mesh device...", flush=True)
    t0 = time.time()
    
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 1),
        l1_small_size=32768,
        trace_region_size=100000000,
    )
    mesh_device.enable_program_cache()
    print(f"  Device opened in {time.time() - t0:.1f}s", flush=True)

    # === Create pipeline ===
    print("\n[2/3] Creating Whisper pipeline...", flush=True)
    t0 = time.time()
    
    pipeline = create_functional_whisper_for_conditional_generation_inference_pipeline(
        mesh_device=mesh_device,
        model_repo=args.model,
        language="en",
        task="transcribe",
        use_trace=True,
        batch_size_per_device=1,
    )
    print(f"  Pipeline created in {time.time() - t0:.1f}s", flush=True)

    # Extract generator from pipeline closure so we can call it
    # with a per-request prompt for wake word detection
    _freevars = pipeline.__code__.co_freevars
    _generator = pipeline.__closure__[_freevars.index('generator')].cell_contents
    WAKE_WORD_PROMPT = "Hey TT, hey tee tee."

    # === Warmup ===
    print("\n[3/3] Running warmup...", flush=True)
    t0 = time.time()
    
    # Create dummy audio (1 second of silence)
    dummy_audio = np.zeros(16000, dtype=np.float32)
    try:
        result = pipeline([(16000, dummy_audio)], stream=False)
        print(f"  Warmup result: {result}", flush=True)
    except Exception as e:
        print(f"  Warmup error (non-fatal): {e}", flush=True)
    
    print(f"  Warmup complete in {time.time() - t0:.1f}s", flush=True)

    # === Start server ===
    print("\n" + "=" * 60, flush=True)
    print("READY - Starting Unix socket server...", flush=True)
    print("=" * 60, flush=True)

    # Remove old socket if exists
    if os.path.exists(args.socket):
        os.unlink(args.socket)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(args.socket)
    server.listen(1)
    os.chmod(args.socket, 0o777)

    print(f"Listening on {args.socket}", flush=True)
    print("Send JSON: {\"audio_path\": \"/path/to/audio.wav\"}", flush=True)
    print("Commands: {\"cmd\": \"ping\"}, {\"cmd\": \"shutdown\"}\n", flush=True)

    try:
        while True:
            conn, addr = server.accept()
            try:
                data = conn.recv(1024 * 1024)  # 1MB max for base64 audio
                if not data:
                    continue

                request = json.loads(data.decode('utf-8'))

                # Handle commands
                if request.get("cmd") == "ping":
                    response = {"status": "ok", "model": "whisper", "ready": True}
                    conn.sendall(json.dumps(response).encode('utf-8'))
                    continue

                if request.get("cmd") == "shutdown":
                    response = {"status": "shutting_down"}
                    conn.sendall(json.dumps(response).encode('utf-8'))
                    break

                # Get audio data
                audio_path = request.get("audio_path")
                audio_base64 = request.get("audio_base64")
                
                temp_file = None
                if audio_base64:
                    # Decode base64 audio
                    audio_bytes = base64.b64decode(audio_base64)
                    
                    # Save raw audio (could be webm/opus from browser)
                    raw_file = tempfile.NamedTemporaryFile(suffix=".webm", delete=False)
                    raw_file.write(audio_bytes)
                    raw_file.close()
                    
                    # Convert to WAV using ffmpeg
                    wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    wav_file.close()
                    
                    import subprocess
                    try:
                        result = subprocess.run(
                            ['ffmpeg', '-y', '-i', raw_file.name, '-ar', '16000', '-ac', '1', '-f', 'wav', wav_file.name],
                            capture_output=True, timeout=10
                        )
                        if result.returncode == 0:
                            audio_path = wav_file.name
                        else:
                            # ffmpeg failed, try using raw file (might already be wav)
                            audio_path = raw_file.name
                    except Exception as e:
                        print(f"  ffmpeg conversion error: {e}", flush=True)
                        audio_path = raw_file.name
                    
                    # Cleanup raw file if we converted successfully
                    if audio_path != raw_file.name:
                        try:
                            os.unlink(raw_file.name)
                        except:
                            pass
                
                if not audio_path:
                    response = {"status": "error", "error": "missing 'audio_path' or 'audio_base64'"}
                    conn.sendall(json.dumps(response).encode('utf-8'))
                    continue

                print(f"[Whisper] Transcribing: {audio_path}", flush=True)
                t0 = time.time()

                try:
                    # Load audio
                    sr, audio = wavfile.read(audio_path)
                    
                    # Convert to float32 if needed
                    if audio.dtype == np.int16:
                        audio = audio.astype(np.float32) / 32768.0
                    elif audio.dtype == np.int32:
                        audio = audio.astype(np.float32) / 2147483648.0
                    
                    # Convert stereo to mono if needed
                    if len(audio.shape) > 1:
                        audio = audio.mean(axis=1)
                    
                    # Run transcription
                    is_wake = request.get("wake_word", False)
                    if is_wake:
                        result = _generator.generate(
                            current_batch=[(sr, audio)],
                            language="en",
                            task="transcribe",
                            prompt=WAKE_WORD_PROMPT,
                        )
                    else:
                        result = pipeline([(sr, audio)], stream=False)
                    text = result[0] if result else ""
                    
                    elapsed_ms = (time.time() - t0) * 1000
                    label = "[WakeWord]" if is_wake else "[Whisper]"
                    print(f"  {label} Result: \"{text[:50]}...\" in {elapsed_ms:.1f}ms", flush=True)

                    response = {
                        "status": "ok",
                        "text": text,
                        "time_ms": elapsed_ms,
                    }
                except Exception as e:
                    print(f"  Error: {e}", flush=True)
                    response = {"status": "error", "error": str(e)}
                finally:
                    # Clean up temp file
                    if temp_file and os.path.exists(temp_file.name):
                        os.unlink(temp_file.name)

                conn.sendall(json.dumps(response).encode('utf-8'))

            except Exception as e:
                print(f"[Error] {e}", flush=True)
                try:
                    conn.sendall(json.dumps({"status": "error", "error": str(e)}).encode('utf-8'))
                except:
                    pass
            finally:
                conn.close()

    except KeyboardInterrupt:
        print("\nShutting down...", flush=True)
    finally:
        server.close()
        if os.path.exists(args.socket):
            os.unlink(args.socket)
        ttnn.close_mesh_device(mesh_device)
        print("Device closed.", flush=True)


if __name__ == "__main__":
    main()
