#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Basic example of using the SpeechT5 TTS API for text-to-speech generation.

This script demonstrates:
- Non-streaming TTS generation
- Saving audio to WAV file
- Error handling

Usage:
    python tts_basic_example.py "Hello world"
    python tts_basic_example.py "Your text here" --output my_speech.wav
    python tts_basic_example.py "Custom speaker" --speaker-id 1234
"""

import argparse
import requests
import sys

from utils.logger import TTLogger


logger = TTLogger(__name__)


# API Configuration
BASE_URL = "http://localhost:8000"
API_KEY = "your-secret-key"


def generate_speech(
    text: str,
    speaker_id: int = 0,
    output_file: str = "output.wav",
    base_url: str = BASE_URL,
    api_key: str = API_KEY,
):
    """
    Generate speech from text using the TTS API.

    Args:
        text: Text to convert to speech
        speaker_id: Speaker voice ID (0-7456)
        output_file: Output WAV file path
        base_url: TTS server URL
        api_key: API authentication key

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Generating speech for: '{text}'")
    logger.info(f"Speaker ID: {speaker_id}")
    logger.info(f"Output file: {output_file}")

    # Prepare request
    url = f"{base_url}/tts/tts"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"text": text, "stream": False, "speaker_id": speaker_id}

    try:
        # Send request
        logger.info("Sending request to TTS server...")
        response = requests.post(url, headers=headers, json=payload, timeout=60)

        # Check response
        if response.status_code == 200:
            # Save audio
            with open(output_file, "wb") as f:
                f.write(response.content)

            file_size = len(response.content)
            logger.info(f"✅ Success! Generated {file_size} bytes of audio")
            logger.info(f"✅ Saved to: {output_file}")

            # Calculate approximate duration (16kHz, 16-bit, mono)
            duration = file_size / (16000 * 2)
            logger.info(f"✅ Approximate duration: {duration:.2f} seconds")

            return True
        else:
            logger.error(f"❌ Error: Server returned status code {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        logger.error("❌ Error: Could not connect to server")
        logger.error(f"Make sure the server is running at {base_url}")
        return False
    except requests.exceptions.Timeout:
        logger.error("❌ Error: Request timed out")
        logger.error("The server may be busy or the text is too long")
        return False
    except Exception as e:
        logger.error(f"❌ Error: {type(e).__name__}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate speech from text using SpeechT5 TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tts_basic_example.py "Hello world"
  python tts_basic_example.py "Good morning" --output morning.wav
  python tts_basic_example.py "Different voice" --speaker-id 500
  
Notes:
  - Make sure the TTS server is running on port 8000
  - Speaker IDs range from 0 to 7456 (CMU ARCTIC dataset)
  - Output is always 16kHz, mono, 16-bit WAV format
        """,
    )

    parser.add_argument("text", type=str, help="Text to convert to speech")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output.wav",
        help="Output WAV file path (default: output.wav)",
    )
    parser.add_argument(
        "-s",
        "--speaker-id",
        type=int,
        default=0,
        help="Speaker voice ID, 0-7456 (default: 0)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=BASE_URL,
        help=f"TTS server URL (default: {BASE_URL})",
    )
    parser.add_argument(
        "--api-key", type=str, default=API_KEY, help="API authentication key"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.text.strip():
        logger.error("Error: Text cannot be empty")
        sys.exit(1)

    if args.speaker_id < 0 or args.speaker_id > 7456:
        logger.warning("Warning: Speaker ID should be between 0 and 7456")

    # Generate speech
    success = generate_speech(
        args.text, args.speaker_id, args.output, args.url, args.api_key
    )

    if success:
        logger.info(f"🎵 You can play the audio with: aplay {args.output}")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
