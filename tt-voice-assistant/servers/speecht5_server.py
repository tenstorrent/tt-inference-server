#!/usr/bin/env python3
"""
SpeechT5 TTS Server - Fast Text-to-Speech (CPU-based).

Uses HuggingFace SpeechT5 for fast TTS (~200-500ms per phrase).
No TT device needed - runs on CPU.

Usage (inside container):
  python speecht5_server.py

Protocol (Unix socket at /tmp/tts_server.sock):
  REQ: {"text": "Hello world", "output_path": "/tmp/output.wav"}
  REP: {"status": "ok", "audio_path": "/tmp/output.wav", "time_ms": 234.5}
"""

import os
import sys
import json
import time
import socket
import argparse
import torch
import numpy as np
import soundfile as sf

# Add tt-metal to path (for any shared utilities)
sys.path.insert(0, "/home/container_app_user/tt-metal")


def main():
    parser = argparse.ArgumentParser(description="SpeechT5 TTS Server")
    parser.add_argument("--socket", type=str, default="/tmp/tts_server.sock", help="Unix socket path")
    parser.add_argument("--speaker-id", type=int, default=7306, help="Speaker embedding ID from CMU Arctic")
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("SpeechT5 TTS Server - Fast Text-to-Speech (CPU)", flush=True)
    print("=" * 60, flush=True)
    print(f"Socket: {args.socket}", flush=True)
    print(f"Speaker ID: {args.speaker_id}", flush=True)

    from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
    from datasets import load_dataset

    # === Load models ===
    print("\n[1/3] Loading SpeechT5 models...", flush=True)
    t0 = time.time()
    
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    model.eval()
    
    print(f"  Models loaded in {time.time() - t0:.1f}s", flush=True)

    # === Load speaker embeddings ===
    print("\n[2/3] Loading speaker embeddings...", flush=True)
    t0 = time.time()
    
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[args.speaker_id]["xvector"]).unsqueeze(0)
    
    print(f"  Speaker embeddings loaded in {time.time() - t0:.1f}s", flush=True)

    # === Warmup ===
    print("\n[3/3] Running warmup...", flush=True)
    t0 = time.time()
    
    inputs = processor(text="Hello", return_tensors="pt")
    with torch.no_grad():
        _ = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    
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
    print("Send JSON: {\"text\": \"...\", \"output_path\": \"...\"}", flush=True)
    print("Commands: {\"cmd\": \"ping\"}, {\"cmd\": \"shutdown\"}\n", flush=True)

    try:
        while True:
            conn, addr = server.accept()
            try:
                data = conn.recv(65536).decode('utf-8')
                if not data:
                    continue

                request = json.loads(data)

                # Handle commands
                if request.get("cmd") == "ping":
                    response = {"status": "ok", "model": "speecht5_tts", "ready": True}
                    conn.sendall(json.dumps(response).encode('utf-8'))
                    continue

                if request.get("cmd") == "shutdown":
                    response = {"status": "shutting_down"}
                    conn.sendall(json.dumps(response).encode('utf-8'))
                    break

                # Handle synthesis request
                text = request.get("text", "")
                output_path = request.get("output_path", "/tmp/tts_output.wav")

                if not text:
                    response = {"status": "error", "error": "missing 'text' field"}
                    conn.sendall(json.dumps(response).encode('utf-8'))
                    continue

                print(f"[TTS] Synthesizing: {text[:50]}...", flush=True)
                t0 = time.time()

                try:
                    # Split long text into sentences for chunked synthesis
                    import re
                    max_chars = 400  # Safe limit for SpeechT5
                    
                    if len(text) > max_chars:
                        # Split by sentences
                        sentences = re.split(r'(?<=[.!?])\s+', text)
                        chunks = []
                        current_chunk = ""
                        
                        for sentence in sentences:
                            if len(current_chunk) + len(sentence) < max_chars:
                                current_chunk += sentence + " "
                            else:
                                if current_chunk:
                                    chunks.append(current_chunk.strip())
                                current_chunk = sentence + " "
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        
                        print(f"  Split into {len(chunks)} chunks", flush=True)
                        
                        # Generate speech for each chunk
                        all_speech = []
                        for i, chunk in enumerate(chunks):
                            inputs = processor(text=chunk, return_tensors="pt")
                            with torch.no_grad():
                                chunk_speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
                            all_speech.append(chunk_speech.numpy())
                        
                        # Concatenate all chunks
                        speech = np.concatenate(all_speech)
                    else:
                        # Short text - process directly
                        inputs = processor(text=text, return_tensors="pt")
                        with torch.no_grad():
                            speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
                        speech = speech.numpy()
                    
                    # Save audio
                    sf.write(output_path, speech, 16000)
                    
                    elapsed_ms = (time.time() - t0) * 1000
                    audio_duration = len(speech) / 16000 if hasattr(speech, '__len__') else speech.shape[0] / 16000
                    
                    print(f"  Done in {elapsed_ms:.1f}ms (audio: {audio_duration:.2f}s)", flush=True)

                    response = {
                        "status": "ok",
                        "audio_path": output_path,
                        "time_ms": elapsed_ms,
                        "audio_duration": audio_duration,
                    }
                except Exception as e:
                    print(f"  Error: {e}", flush=True)
                    response = {"status": "error", "error": str(e)}

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
        print("Server stopped.", flush=True)


if __name__ == "__main__":
    main()
