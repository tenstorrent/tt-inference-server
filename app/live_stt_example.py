"""
Live Real-time STT with Whisper on TTNN — WebSocket Streaming
==============================================================
Streams audio input via WebSocket and gets real-time transcription updates.

How it works:
1. Connects to WebSocket endpoint /ws/stt-live
2. Records mic audio continuously
3. Every 2 seconds, sends all accumulated audio (PCM16) over WebSocket
4. Server runs Whisper distil-large-v3 on TTNN (~100ms)
5. Transcript updates in real-time as you speak

Architecture:
    This script ──WebSocket──→ FastAPI /ws/stt-live ──→ Whisper (TTNN)
        ↑                            ↓
        └──── transcript updates ←───┘

Usage:
    python live_stt_example.py

    # Or specify a different server:
    python live_stt_example.py --url ws://your-server:8080/ws/stt-live

Requirements:
    pip install sounddevice numpy websockets
"""

import asyncio
import argparse
import numpy as np
import sounddevice as sd
import json
import time


SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_INTERVAL = 2.0  # Send for transcription every 2 seconds


def float_to_pcm16(audio_float: np.ndarray) -> bytes:
    """Convert float32 audio to PCM16 bytes (what Whisper expects)."""
    pcm16 = (audio_float * 32767).astype(np.int16)
    return pcm16.tobytes()


async def live_stt(ws_url: str):
    import websockets
    
    print("=" * 60)
    print("  Live STT Demo — Whisper distil-large-v3 on TTNN")
    print("  WebSocket streaming to:", ws_url)
    print("=" * 60)
    print()
    print("Press Enter to start recording, Ctrl+C to stop.")
    print()
    input(">>> Press Enter to start...")
    
    # Audio state
    audio_buffer = []
    is_recording = True
    
    def audio_callback(indata, frames, time_info, status):
        if is_recording:
            audio_buffer.append(indata[:, 0].copy())
    
    # Start mic capture
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        callback=audio_callback,
        blocksize=1024
    )
    stream.start()
    print("🎤 Recording... speak now!\n")
    
    # Connect to WebSocket
    async with websockets.connect(ws_url) as ws:
        print(f"✅ WebSocket connected to {ws_url}\n")
        
        last_text = ""
        update_count = 0
        
        try:
            while True:
                # Wait for chunk interval
                await asyncio.sleep(CHUNK_INTERVAL)
                
                if not audio_buffer:
                    continue
                
                # Combine all audio accumulated so far
                combined = np.concatenate(audio_buffer)
                duration = len(combined) / SAMPLE_RATE
                
                # Convert to PCM16 and send via WebSocket
                pcm16_bytes = float_to_pcm16(combined)
                await ws.send(pcm16_bytes)
                
                # Request transcription
                await ws.send(json.dumps({"action": "transcribe"}))
                
                # Wait for response
                response = await ws.recv()
                data = json.loads(response)
                
                if data.get("type") == "transcript":
                    text = data.get("text", "")
                    whisper_ms = data.get("time_ms", 0)
                    update_count += 1
                    
                    if text and text != last_text:
                        print(f"\r\033[K  [{duration:.1f}s] (whisper: {whisper_ms:.0f}ms) {text}", end="", flush=True)
                        last_text = text
                
                elif data.get("type") == "error":
                    print(f"\n  ERROR: {data.get('message')}")
        
        except KeyboardInterrupt:
            pass
        
        # Final transcription
        print("\n\n⏹ Stopping...\n")
        is_recording = False
        stream.stop()
        stream.close()
        
        if audio_buffer:
            combined = np.concatenate(audio_buffer)
            pcm16_bytes = float_to_pcm16(combined)
            
            # Send final audio and request final transcription
            await ws.send(pcm16_bytes)
            await ws.send(json.dumps({"action": "final"}))
            
            response = await ws.recv()
            data = json.loads(response)
            
            text = data.get("text", "")
            whisper_ms = data.get("time_ms", 0)
            duration = len(combined) / SAMPLE_RATE
            
            print("=" * 60)
            print(f"  FINAL TRANSCRIPT:")
            print(f"  \"{text}\"")
            print(f"")
            print(f"  Audio duration:  {duration:.1f}s")
            print(f"  Whisper latency: {whisper_ms:.0f}ms")
            print(f"  Total updates:   {update_count}")
            print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live STT Demo with WebSocket streaming")
    parser.add_argument("--url", default="ws://localhost:8080/ws/stt-live",
                        help="WebSocket URL for STT endpoint")
    args = parser.parse_args()
    
    asyncio.run(live_stt(args.url))
