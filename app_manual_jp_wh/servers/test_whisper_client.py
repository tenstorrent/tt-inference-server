#!/usr/bin/env python3
"""
Test client for Whisper server.

Usage:
  python test_whisper_client.py /path/to/audio.wav
  python test_whisper_client.py --ping
"""

import socket
import json
import argparse
import time


def send_request(sock_path: str, request: dict) -> dict:
    """Send request to Whisper server and get response."""
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(sock_path)
    
    client.sendall(json.dumps(request).encode('utf-8'))
    response = client.recv(65536).decode('utf-8')
    client.close()
    
    return json.loads(response)


def main():
    parser = argparse.ArgumentParser(description="Whisper Client")
    parser.add_argument("audio_path", nargs="?", default=None, help="Path to audio file")
    parser.add_argument("--socket", type=str, default="/tmp/whisper_server.sock", help="Socket path")
    parser.add_argument("--ping", action="store_true", help="Ping server")
    parser.add_argument("--shutdown", action="store_true", help="Shutdown server")
    args = parser.parse_args()

    if args.ping:
        print("Pinging Whisper server...")
        response = send_request(args.socket, {"cmd": "ping"})
        print(f"Response: {response}")
        return

    if args.shutdown:
        print("Shutting down Whisper server...")
        response = send_request(args.socket, {"cmd": "shutdown"})
        print(f"Response: {response}")
        return

    if not args.audio_path:
        print("Error: Please provide audio file path")
        print("Usage: python test_whisper_client.py /path/to/audio.wav")
        return

    print(f"Transcribing: {args.audio_path}")
    
    t0 = time.time()
    response = send_request(args.socket, {"audio_path": args.audio_path})
    total_time = (time.time() - t0) * 1000
    
    print(f"\nResponse: {response}")
    print(f"Total round-trip time: {total_time:.1f}ms")
    
    if response.get("status") == "ok":
        print(f"\n✅ Transcription: \"{response.get('text')}\"")
        print(f"   Server processing time: {response.get('time_ms', 0):.1f}ms")


if __name__ == "__main__":
    main()
