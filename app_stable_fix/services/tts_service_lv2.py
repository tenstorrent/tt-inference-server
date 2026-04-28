"""
TTS Service - Lightning V2 (HTTP client).
Connects to the external lv2-tts Docker container via HTTP.
"""

import asyncio
import logging
import os
import time
import urllib.request
import urllib.error
import json
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

TTS_SERVER_URL = os.getenv("TTS_SERVER_URL", "http://172.17.0.1:2082")


class TTSService:
    """TTS via Lightning V2 HTTP server."""

    def __init__(self, device_id: int = 3):
        self.device_id = device_id
        self.service_name = "TTS"
        self.is_warmed_up = False
        self.warmup_time = 0
        self.server_url = TTS_SERVER_URL
        self.output_dir = "/home/container_app_user/voice-assistant/output"
        self.speakers: List[str] = []
        self.default_speaker = os.getenv("TTS_DEFAULT_SPEAKER", "MyVoice")

        logger.info(f"TTS service initialized (server: {self.server_url})")

    def _http_get(self, path: str, timeout: float = 30.0) -> Any:
        req = urllib.request.Request(f"{self.server_url}{path}", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _http_post_wav(self, path: str, body: dict, output_path: str, timeout: float = 120.0) -> float:
        """POST JSON, save the returned WAV bytes to disk. Returns elapsed seconds."""
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{self.server_url}{path}",
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            wav_bytes = resp.read()
        elapsed = time.time() - t0

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(wav_bytes)

        return elapsed

    async def warmup(self):
        """Check if Lightning V2 server is ready by fetching speakers list."""
        logger.info("Checking Lightning V2 TTS server...")

        try:
            os.makedirs(self.output_dir, exist_ok=True)

            loop = asyncio.get_event_loop()
            self.speakers = await loop.run_in_executor(
                None, lambda: self._http_get("/speakers")
            )

            self.is_warmed_up = True
            logger.info(f"TTS server ready! Speakers: {self.speakers}")

            if self.default_speaker not in self.speakers and self.speakers:
                self.default_speaker = self.speakers[0]
                logger.info(f"Default speaker set to: {self.default_speaker}")

        except Exception as e:
            logger.error(f"TTS server check failed: {e}")
            self.is_warmed_up = False

    async def synthesize(
        self,
        text: str,
        fast: bool = True,
        speaker: Optional[str] = None,
        speaker_id: int = None,
        speed: float = 1.0,
        language: str = "en",
        consistency: float = 1.0,
    ) -> Dict[str, Any]:
        """Synthesize speech from text via Lightning V2.

        `speaker` is the Lightning V2 speaker name (e.g. "Emma").
        `speaker_id` is accepted for backward compat but ignored;
        use `speaker` for Lightning V2.
        """
        if not speaker:
            speaker = self.default_speaker

        logger.info(f"TTS [{speaker}]: {text[:50]}...")

        try:
            output_path = os.path.join(
                self.output_dir, f"tts_{int(time.time() * 1000)}.wav"
            )

            body = {
                "text": text,
                "speaker": speaker,
                "language": language,
                "speed": speed,
                "consistency": consistency,
            }

            loop = asyncio.get_event_loop()
            elapsed = await loop.run_in_executor(
                None,
                lambda: self._http_post_wav("/tts", body, output_path),
            )

            logger.info(f"TTS completed in {elapsed * 1000:.1f}ms")
            return {
                "audio_path": output_path,
                "processing_time": elapsed,
            }

        except Exception as e:
            logger.error(f"TTS error: {e}")
            return {
                "audio_path": self._create_beep(),
                "processing_time": 0.1,
            }

    def _create_beep(self) -> str:
        """Create a simple beep sound as fallback."""
        try:
            from scipy.io import wavfile
            import numpy as np

            duration = 0.3
            sample_rate = 16000
            frequency = 800

            t = np.linspace(0, duration, int(sample_rate * duration))
            beep = np.sin(2 * np.pi * frequency * t) * 0.3
            beep = (beep * 32767).astype(np.int16)

            audio_path = os.path.join(self.output_dir, "beep.wav")
            wavfile.write(audio_path, sample_rate, beep)

            return audio_path
        except Exception:
            return ""

    def get_status(self) -> Dict[str, Any]:
        return {
            "service": self.service_name,
            "device_id": self.device_id,
            "is_warmed_up": self.is_warmed_up,
            "warmup_time": self.warmup_time,
            "server_url": self.server_url,
            "model": "Lightning V2",
            "speakers": self.speakers,
            "default_speaker": self.default_speaker,
        }

    async def shutdown(self):
        logger.info("Shutting down TTS service...")
        self.is_warmed_up = False
