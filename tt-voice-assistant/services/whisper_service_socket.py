"""
Whisper Service - Socket-based client for Whisper server.
Connects to the persistent Whisper server via Unix socket.
"""

import asyncio
import logging
import socket
import json
import tempfile
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)

WHISPER_SOCKET = "/tmp/whisper_server.sock"


class WhisperService:
    """Whisper STT via persistent socket server."""
    
    def __init__(self, device_id: int = 2):
        """Initialize Whisper service."""
        self.device_id = device_id
        self.service_name = "Whisper"
        self.is_warmed_up = False
        self.warmup_time = 0
        self.socket_path = WHISPER_SOCKET
        
        logger.info(f"Whisper service initialized (socket: {self.socket_path})")
    
    def _send_request(self, request: dict) -> dict:
        """Send request to Whisper server."""
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(60)  # 60 second timeout
        client.connect(self.socket_path)
        client.sendall(json.dumps(request).encode('utf-8'))
        response = client.recv(1024 * 1024).decode('utf-8')
        client.close()
        return json.loads(response)
    
    async def warmup(self):
        """Check if Whisper server is ready."""
        logger.info(f"🔥 Checking Whisper server...")
        
        try:
            # Ping the server
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._send_request({"cmd": "ping"})
            )
            
            if response.get("status") == "ok":
                self.is_warmed_up = True
                logger.info(f"✅ Whisper server is ready!")
            else:
                raise RuntimeError(f"Whisper server not ready: {response}")
                
        except Exception as e:
            logger.error(f"❌ Whisper server check failed: {e}")
            # Don't raise - allow pipeline to continue
            self.is_warmed_up = False
    
    async def transcribe(self, audio_data: bytes) -> Dict[str, Any]:
        """Transcribe audio to text."""
        logger.info("🎤 Transcribing audio...")
        
        try:
            import subprocess
            
            # Save raw audio to temp file (browser sends webm/opus)
            raw_file = tempfile.NamedTemporaryFile(suffix=".webm", delete=False)
            raw_file.write(audio_data)
            raw_file.close()
            
            # Convert to WAV using ffmpeg
            wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            wav_file.close()
            
            try:
                result = subprocess.run(
                    ['ffmpeg', '-y', '-i', raw_file.name, '-ar', '16000', '-ac', '1', '-f', 'wav', wav_file.name],
                    capture_output=True,
                    timeout=30
                )
                if result.returncode != 0:
                    logger.warning(f"ffmpeg conversion warning: {result.stderr[:200] if result.stderr else 'none'}")
            except Exception as e:
                logger.warning(f"ffmpeg conversion failed: {e}, trying direct read")
            
            # Clean up raw file
            os.unlink(raw_file.name)
            
            # Send to server
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._send_request({"audio_path": wav_file.name})
            )
            
            # Clean up wav file
            os.unlink(wav_file.name)
            
            if response.get("status") == "ok":
                text = response.get("text", "")
                if isinstance(text, list):
                    text = text[0] if text else ""
                
                logger.info(f"✅ Transcription: '{text[:50]}...'")
                return {
                    "text": text,
                    "confidence": 0.95,
                    "processing_time": response.get("time_ms", 0) / 1000
                }
            else:
                raise RuntimeError(response.get("error", "Unknown error"))
                
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return {
                "text": "",
                "confidence": 0,
                "processing_time": 0
            }
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status."""
        return {
            "service": self.service_name,
            "device_id": self.device_id,
            "is_warmed_up": self.is_warmed_up,
            "warmup_time": self.warmup_time,
            "socket": self.socket_path
        }
    
    async def shutdown(self):
        """Shutdown Whisper service."""
        logger.info("🛑 Shutting down Whisper service...")
        self.is_warmed_up = False
