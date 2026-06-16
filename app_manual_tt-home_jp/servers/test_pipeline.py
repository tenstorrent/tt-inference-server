#!/usr/bin/env python3
"""
Test the voice assistant pipeline end-to-end.

Tests: Whisper (STT) -> [Llama would go here] -> SpeechT5 (TTS)
"""

import socket
import json
import time
import os


def send_request(sock_path: str, request: dict) -> dict:
    """Send request to server and get response."""
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(sock_path)
    client.sendall(json.dumps(request).encode('utf-8'))
    response = client.recv(1024 * 1024).decode('utf-8')
    client.close()
    return json.loads(response)


def test_whisper(audio_path: str) -> str:
    """Test Whisper transcription."""
    response = send_request("/tmp/whisper_server.sock", {"audio_path": audio_path})
    if response.get("status") == "ok":
        text = response.get("text", "")
        if isinstance(text, list):
            text = text[0] if text else ""
        return text, response.get("time_ms", 0)
    return "", 0


def test_tts(text: str, output_path: str) -> tuple:
    """Test TTS synthesis."""
    response = send_request("/tmp/tts_server.sock", {"text": text, "output_path": output_path})
    if response.get("status") == "ok":
        return response.get("audio_path", ""), response.get("time_ms", 0)
    return "", 0


def main():
    print("=" * 60)
    print("Voice Assistant Pipeline Test")
    print("=" * 60)
    
    # Check servers are running
    print("\n[1/4] Checking servers...")
    
    try:
        whisper_status = send_request("/tmp/whisper_server.sock", {"cmd": "ping"})
        print(f"  Whisper: {whisper_status.get('status', 'error')}")
    except Exception as e:
        print(f"  Whisper: ERROR - {e}")
        return
    
    try:
        tts_status = send_request("/tmp/tts_server.sock", {"cmd": "ping"})
        print(f"  TTS: {tts_status.get('status', 'error')}")
    except Exception as e:
        print(f"  TTS: ERROR - {e}")
        return
    
    # Create test audio with TTS
    print("\n[2/4] Creating test audio with TTS...")
    test_text = "Hello, I would like to know about Tenstorrent hardware."
    audio_path, tts_time = test_tts(test_text, "/tmp/pipeline_test_input.wav")
    print(f"  Generated audio in {tts_time:.1f}ms: {audio_path}")
    
    # Transcribe with Whisper
    print("\n[3/4] Transcribing with Whisper...")
    transcription, whisper_time = test_whisper(audio_path)
    print(f"  Transcription in {whisper_time:.1f}ms: \"{transcription}\"")
    
    # Generate response with TTS (simulating Llama response)
    print("\n[4/4] Generating response with TTS...")
    response_text = "Tenstorrent builds AI accelerator hardware using the Blackhole chip architecture."
    response_audio, response_tts_time = test_tts(response_text, "/tmp/pipeline_test_response.wav")
    print(f"  Response audio in {response_tts_time:.1f}ms: {response_audio}")
    
    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE TEST SUMMARY")
    print("=" * 60)
    print(f"  TTS (input):     {tts_time:.1f}ms")
    print(f"  Whisper (STT):   {whisper_time:.1f}ms")
    print(f"  TTS (response):  {response_tts_time:.1f}ms")
    print(f"  Total:           {tts_time + whisper_time + response_tts_time:.1f}ms")
    print("\n  Note: Llama inference would add ~100-500ms")
    print("=" * 60)


if __name__ == "__main__":
    main()
