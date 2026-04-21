#!/usr/bin/env python3
"""
Test client for TTS server.

Usage:
  python test_tts_client.py "Hello from Tenstorrent"
  python test_tts_client.py --ping
"""

import socket
import json
import argparse
import time


def send_request(sock_path: str, request: dict) -> dict:
    """Send request to TTS server and get response."""
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(sock_path)
    
    client.sendall(json.dumps(request).encode('utf-8'))
    response = client.recv(65536).decode('utf-8')
    client.close()
    
    return json.loads(response)


def main():
    parser = argparse.ArgumentParser(description="TTS Client")
    parser.add_argument("text", nargs="?", default=None, help="Text to synthesize")
    parser.add_argument("--socket", type=str, default="/tmp/tts_server.sock", help="Socket path")
    parser.add_argument("--output", type=str, default="/tmp/tts_test_output.wav", help="Output path")
    parser.add_argument("--ping", action="store_true", help="Ping server")
    parser.add_argument("--shutdown", action="store_true", help="Shutdown server")
    args = parser.parse_args()

    if args.ping:
        print("Pinging TTS server...")
        response = send_request(args.socket, {"cmd": "ping"})
        print(f"Response: {response}")
        return

    if args.shutdown:
        print("Shutting down TTS server...")
        response = send_request(args.socket, {"cmd": "shutdown"})
        print(f"Response: {response}")
        return

    if not args.text:
        print("Error: Please provide text to synthesize")
        print("Usage: python test_tts_client.py \"Hello world\"")
        return

    print(f"Synthesizing: {args.text}")
    print(f"Output: {args.output}")
    
    t0 = time.time()
    response = send_request(args.socket, {
        "text": args.text,
        "output_path": args.output,
    })
    total_time = (time.time() - t0) * 1000
    
    print(f"\nResponse: {response}")
    print(f"Total round-trip time: {total_time:.1f}ms")
    
    if response.get("status") == "ok":
        print(f"\n✅ Audio saved to: {response.get('audio_path')}")
        print(f"   Server processing time: {response.get('time_ms', 0):.1f}ms")


if __name__ == "__main__":
    main()
