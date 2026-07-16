"""
TTS Service - Socket-based client for TTS server (SpeechT5).
Connects to the persistent TTS server via Unix socket.

This uses SpeechT5 for fast TTS (~200-500ms per phrase).
Supports dual-device for podcast (HOST on device 4, GUEST on device 6).
"""

import asyncio
import logging
import socket
import json
import os
import re
from typing import Dict, Any


def _clean_for_tts(text: str) -> str:
    """Strip markdown/formatting, keep only spoken words for TTS."""
    t = text
    t = re.sub(r'\*\*([^*]*)\*\*', r'\1', t)
    t = re.sub(r'\*([^*]*)\*', r'\1', t)
    t = re.sub(r'\*+', '', t)
    t = re.sub(r'#{1,6}\s*', '', t)
    t = re.sub(r'^\s*[-•]\s+', '', t, flags=re.MULTILINE)
    t = re.sub(r'[-•]\s+', ' ', t)
    t = re.sub(r'^\s*\d+\.\s+', '', t, flags=re.MULTILINE)
    t = re.sub(r'`([^`]*)`', r'\1', t)
    t = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', t)
    t = re.sub(r'[_~]', '', t)
    t = re.sub(r'---+', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


logger = logging.getLogger(__name__)


def _detect_language(text: str) -> str:
    """Auto-detect language from text characters."""
    cjk_ja = 0
    cjk_zh = 0
    korean = 0
    cyrillic = 0
    for ch in text:
        cp = ord(ch)
        if 0x3040 <= cp <= 0x309F or 0x30A0 <= cp <= 0x30FF:  # Hiragana/Katakana
            cjk_ja += 1
        elif 0x4E00 <= cp <= 0x9FFF:  # CJK Unified
            cjk_zh += 1
        elif 0xAC00 <= cp <= 0xD7AF:  # Korean Hangul
            korean += 1
        elif 0x0400 <= cp <= 0x04FF:  # Cyrillic
            cyrillic += 1
    total = len(text)
    if cjk_ja > 0:
        return "japanese"
    if korean > total * 0.1:
        return "korean"
    if cjk_zh > total * 0.15:
        return "chinese"
    if cyrillic > total * 0.2:
        return "russian"
    return "english"

TTS_SOCKET = "/tmp/tts_server.sock"
TTS_GUEST_SOCKET = "/tmp/tts_server_guest.sock"


class TTSService:
    """TTS via persistent socket server (SpeechT5). Supports dual-device for podcast."""
    
    def __init__(self, device_id: int = 3):
        """Initialize TTS service."""
        self.device_id = device_id
        self.service_name = "TTS"
        self.is_warmed_up = False
        self.warmup_time = 0
        self.socket_path = TTS_SOCKET
        self.guest_socket_path = TTS_GUEST_SOCKET
        self.output_dir = "/home/container_app_user/voice-assistant/output"
        self.restarting = False
        self._consecutive_failures = 0
        self._restart_threshold = 3
        logger.info(f"TTS service initialized (host: {self.socket_path}, guest: {self.guest_socket_path})")
    
    def _send_request_to(self, sock_path: str, request: dict) -> dict:
        """Send a request to a specific TTS server socket and get response (raw JSON protocol).
        Retries on connection refused/busy (server processing another request)."""
        import time
        max_retries = 30
        for attempt in range(max_retries):
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(120)
            try:
                sock.connect(sock_path)
                data = json.dumps(request).encode('utf-8')
                sock.sendall(data)

                response_data = b""
                while True:
                    chunk = sock.recv(65536)
                    if not chunk:
                        break
                    response_data += chunk

                return json.loads(response_data.decode('utf-8'))
            except (ConnectionRefusedError, OSError) as e:
                sock.close()
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                raise
            finally:
                try:
                    sock.close()
                except:
                    pass
    
    def _send_request(self, request: dict) -> dict:
        """Send request to the host TTS server."""
        return self._send_request_to(self.socket_path, request)

    async def warmup(self):
        """Warmup / health check the TTS server."""
        await self.check_server()

    async def check_server(self):
        """Check if TTS server is ready."""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._send_request({"cmd": "ping"})
            )
            if response.get("status") == "ok":
                self.is_warmed_up = True
                logger.info(f"✅ TTS server is ready! (model: {response.get('model', 'unknown')})")
            else:
                raise RuntimeError(f"TTS server not ready: {response}")
                
        except Exception as e:
            logger.error(f"❌ TTS server check failed: {e}")
            self.is_warmed_up = False
    
    async def synthesize(self, text: str, fast: bool = True, speaker_id: int = None) -> Dict[str, Any]:
        """Synthesize speech from text. Optionally specify speaker_id for multi-voice (CPU fallback)."""
        if self.restarting:
            raise RuntimeError("TTS server is restarting, please wait")

        clean_text = _clean_for_tts(text)
        if not clean_text:
            return {"audio_path": None, "processing_time": 0}
        logger.info(f"🔊 TTS: {clean_text[:50]}...")
        
        try:
            import time
            output_path = os.path.join(self.output_dir, f"tts_{int(time.time() * 1000)}.wav")
            
            language = _detect_language(clean_text)
            req = {"text": clean_text, "output_path": output_path, "language": language}
            if language != "english":
                logger.info(f"    Language detected: {language}")
            if speaker_id is not None:
                req["speaker_id"] = speaker_id
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._send_request(req)
            )
            
            if response.get("status") == "ok":
                self._consecutive_failures = 0
                logger.info(f"✅ TTS completed in {response.get('time_ms', 0):.1f}ms")
                return {
                    "audio_path": response.get("audio_path", output_path),
                    "processing_time": response.get("time_ms", 0) / 1000
                }
            else:
                raise RuntimeError(response.get("error", "Unknown error"))
                
        except Exception as e:
            self._consecutive_failures += 1
            logger.error(f"TTS error ({self._consecutive_failures}/{self._restart_threshold}): {e}")
            if self._consecutive_failures >= self._restart_threshold and not self.restarting:
                loop = asyncio.get_event_loop()
                loop.run_in_executor(None, self._restart_server)
            raise RuntimeError(f"TTS synthesis failed: {e}")

    async def synthesize_guest(self, text: str) -> Dict[str, Any]:
        """Synthesize speech using the GUEST TTS server (second device, different speaker)."""
        clean_text = _clean_for_tts(text)
        if not clean_text:
            return {"audio_path": None, "processing_time": 0}
        logger.info(f"🔊 TTS (guest): {clean_text[:50]}...")
        try:
            import time
            output_path = os.path.join(self.output_dir, f"tts_guest_{int(time.time() * 1000)}.wav")
            language = _detect_language(clean_text)
            req = {"text": clean_text, "output_path": output_path, "language": language}
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._send_request_to(self.guest_socket_path, req)
            )
            if response.get("status") == "ok":
                logger.info(f"✅ TTS (guest) completed in {response.get('time_ms', 0):.1f}ms")
                return {
                    "audio_path": response.get("audio_path", output_path),
                    "processing_time": response.get("time_ms", 0) / 1000
                }
            else:
                raise RuntimeError(response.get("error", "Unknown error"))
        except Exception as e:
            logger.error(f"TTS guest error: {e}")
            raise RuntimeError(f"TTS guest synthesis failed: {e}")

    def _restart_server(self):
        """Restart TTS server after consecutive failures."""
        import subprocess
        self.restarting = True
        logger.warning("🔄 Restarting TTS server due to consecutive failures...")
        try:
            subprocess.run(['pkill', '-f', 'qwen3_tts_server.py'], timeout=5)
            import time
            time.sleep(2)
            subprocess.Popen([
                'bash', '-c',
                'source /home/container_app_user/tt-metal/python_env/bin/activate && '
                'cd /home/container_app_user/tt-metal && '
                'python /home/container_app_user/voice-assistant/servers/qwen3_tts_server.py '
                '--ref-audio /home/container_app_user/tt-metal/models/demos/qwen3_tts/demo/jim_reference.wav '
                '--ref-text "So basically you put up the high level overview slides." '
                '--socket /tmp/tts_server.sock --device-id 0 >> /tmp/tts_server.log 2>&1'
            ])
            time.sleep(60)
            self._consecutive_failures = 0
            self.restarting = False
            logger.info("✅ TTS server restarted successfully")
        except Exception as e:
            logger.error(f"Failed to restart TTS server: {e}")
            self.restarting = False

    async def wait_for_ready(self, timeout: int = 120):
        """Wait for TTS server to become ready."""
        import time
        start = time.time()
        while time.time() - start < timeout:
            try:
                await self.check_server()
                if self.is_warmed_up:
                    return True
            except Exception:
                pass
            await asyncio.sleep(5)
        return False
