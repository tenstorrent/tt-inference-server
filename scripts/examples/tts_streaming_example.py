#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Streaming example for SpeechT5 TTS API.

This script demonstrates:
- Streaming TTS generation
- Real-time audio chunk receiving
- Progress tracking
- Comparing streaming vs non-streaming latency

Usage:
    python tts_streaming_example.py "Hello world"
    python tts_streaming_example.py "Your text" --compare
"""

import argparse
import requests
import sys
import time


# API Configuration
BASE_URL = "http://localhost:8000"
API_KEY = "your-secret-key"


def generate_speech_streaming(text: str, speaker_id: int = 0, output_file: str = "output_streaming.wav"):
    """
    Generate speech with streaming mode.
    
    Args:
        text: Text to convert to speech
        speaker_id: Speaker voice ID
        output_file: Output file path
    
    Returns:
        Tuple of (success, duration, file_size)
    """
    print(f"🔄 Streaming generation for: '{text}'")
    
    url = f"{BASE_URL}/tts/tts"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "stream": True,
        "speaker_id": speaker_id
    }
    
    try:
        start_time = time.time()
        first_chunk_time = None
        
        # Send streaming request
        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=60)
        
        if response.status_code != 200:
            print(f"✗ Error: Server returned status {response.status_code}")
            return False, 0, 0
        
        # Receive streaming chunks
        chunks = []
        chunk_count = 0
        
        print("Receiving audio chunks:")
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                    latency = first_chunk_time - start_time
                    print(f"  ⚡ First chunk received in {latency:.3f}s")
                
                chunks.append(chunk)
                chunk_count += 1
                print(f"  📦 Chunk {chunk_count}: {len(chunk)} bytes")
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Save combined audio
        audio_data = b''.join(chunks)
        with open(output_file, 'wb') as f:
            f.write(audio_data)
        
        file_size = len(audio_data)
        print("\n✓ Streaming complete!")
        print(f"  Total chunks: {chunk_count}")
        print(f"  Total size: {file_size} bytes")
        print(f"  First chunk latency: {latency:.3f}s")
        print(f"  Total time: {total_duration:.3f}s")
        print(f"  Saved to: {output_file}")
        
        return True, total_duration, file_size
        
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")
        return False, 0, 0


def generate_speech_non_streaming(text: str, speaker_id: int = 0, output_file: str = "output_non_streaming.wav"):
    """
    Generate speech without streaming (for comparison).
    
    Args:
        text: Text to convert to speech
        speaker_id: Speaker voice ID
        output_file: Output file path
    
    Returns:
        Tuple of (success, duration, file_size)
    """
    print(f"⏳ Non-streaming generation for: '{text}'")
    
    url = f"{BASE_URL}/tts/tts"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "stream": False,
        "speaker_id": speaker_id
    }
    
    try:
        start_time = time.time()
        
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        if response.status_code != 200:
            print(f"✗ Error: Server returned status {response.status_code}")
            return False, 0, 0
        
        # Save audio
        with open(output_file, 'wb') as f:
            f.write(response.content)
        
        file_size = len(response.content)
        print("✓ Non-streaming complete!")
        print(f"  Total size: {file_size} bytes")
        print(f"  Total time: {total_duration:.3f}s")
        print(f"  Saved to: {output_file}")
        
        return True, total_duration, file_size
        
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}: {e}")
        return False, 0, 0


def main():
    parser = argparse.ArgumentParser(
        description="Streaming TTS example with SpeechT5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tts_streaming_example.py "Hello world"
  python tts_streaming_example.py "Compare modes" --compare
  python tts_streaming_example.py "Test" --speaker-id 100
        """
    )
    
    parser.add_argument(
        "text",
        type=str,
        help="Text to convert to speech"
    )
    parser.add_argument(
        "-c", "--compare",
        action="store_true",
        help="Compare streaming vs non-streaming performance"
    )
    parser.add_argument(
        "-s", "--speaker-id",
        type=int,
        default=0,
        help="Speaker voice ID (default: 0)"
    )
    parser.add_argument(
        "--url",
        type=str,
        default=BASE_URL,
        help=f"TTS server URL (default: {BASE_URL})"
    )
    
    args = parser.parse_args()
    
    global BASE_URL
    BASE_URL = args.url
    
    print("=" * 60)
    print("  SpeechT5 TTS Streaming Example")
    print("=" * 60)
    print()
    
    # Streaming generation
    success_stream, time_stream, size_stream = generate_speech_streaming(
        args.text, args.speaker_id, "output_streaming.wav"
    )
    
    if not success_stream:
        sys.exit(1)
    
    # Comparison mode
    if args.compare:
        print()
        print("=" * 60)
        print("  Running non-streaming for comparison...")
        print("=" * 60)
        print()
        
        success_non_stream, time_non_stream, size_non_stream = generate_speech_non_streaming(
            args.text, args.speaker_id, "output_non_streaming.wav"
        )
        
        if success_non_stream:
            print()
            print("=" * 60)
            print("  Performance Comparison")
            print("=" * 60)
            print(f"  Streaming time:     {time_stream:.3f}s")
            print(f"  Non-streaming time: {time_non_stream:.3f}s")
            
            if time_stream < time_non_stream:
                improvement = ((time_non_stream - time_stream) / time_non_stream) * 100
                print(f"  ⚡ Streaming is {improvement:.1f}% faster!")
            else:
                diff = time_stream - time_non_stream
                print(f"  Note: Streaming took {diff:.3f}s longer (still useful for real-time applications)")
            
            print()
            print(f"  Both files size: {size_stream} bytes")
            print("  ✓ Files should be identical in content")
    
    print()
    print("🎵 Done! You can play the audio with:")
    print("   aplay output_streaming.wav")
    if args.compare:
        print("   aplay output_non_streaming.wav")
    print()
    
    sys.exit(0)


if __name__ == "__main__":
    main()




