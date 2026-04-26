"""
TTS Service - Socket-based client for TTS server (SpeechT5).
Connects to the persistent TTS server via Unix socket.

Note: Qwen3 TTS code is kept as backup in tts_service.py
This uses SpeechT5 for fast TTS (~200-500ms per phrase).
"""

import asyncio
import logging
import socket
import json
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)

TTS_SOCKET = "/tmp/tts_server.sock"


class TTSService:
    """TTS via persistent socket server (SpeechT5)."""
    
    def __init__(self, device_id: int = 3):
        """Initialize TTS service."""
        self.device_id = device_id  # Not used - SpeechT5 runs on CPU
        self.service_name = "TTS"
        self.is_warmed_up = False
        self.warmup_time = 0
        self.socket_path = TTS_SOCKET
        self.output_dir = "/home/container_app_user/voice-assistant/output"
        self.restarting = False
        self._consecutive_failures = 0
        self._restart_threshold = 2
        
        logger.info(f"TTS service initialized (socket: {self.socket_path})")
    
    def _send_request(self, request: dict) -> dict:
        """Send request to TTS server with retry on busy socket."""
        import time as _time
        last_err = None
        for attempt in range(5):
            try:
                client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                client.settimeout(120)
                client.connect(self.socket_path)
                client.sendall(json.dumps(request).encode('utf-8'))
                response = client.recv(65536).decode('utf-8')
                client.close()
                return json.loads(response)
            except OSError as e:
                last_err = e
                client.close()
                if attempt < 4:
                    _time.sleep(1.0 * (attempt + 1))
                    logger.warning(f"TTS socket busy, retry {attempt + 1}/5 in {attempt + 1}s...")
        raise last_err
    
    async def warmup(self):
        """Check if TTS server is ready."""
        logger.info(f"🔥 Checking TTS server...")
        
        try:
            # Create output directory
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Ping the server
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
            # Don't raise - allow pipeline to continue
            self.is_warmed_up = False
    
    async def synthesize(self, text: str, fast: bool = True, speaker_id: int = None) -> Dict[str, Any]:
        """Synthesize speech from text. Optionally specify speaker_id for multi-voice."""
        if self.restarting:
            raise RuntimeError("TTS server is restarting, please wait")

        logger.info(f"🔊 TTS: {text[:50]}...")
        
        try:
            import time
            output_path = os.path.join(self.output_dir, f"tts_{int(time.time() * 1000)}.wav")
            
            req = {"text": text, "output_path": output_path}
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
    
    def _restart_server(self):
        """Kill and restart the TTS server process when it's stuck."""
        import subprocess
        import time
        self.restarting = True
        logger.warning("🔄 TTS server appears stuck, restarting...")
        try:
            subprocess.run(["pkill", "-f", "speecht5_ttnn_server"], timeout=5)
        except Exception:
            pass
        time.sleep(3)
        try:
            os.remove(self.socket_path)
        except OSError:
            pass
        try:
            subprocess.Popen(
                ["bash", "-c",
                 "source /home/container_app_user/tt-metal/python_env/bin/activate && "
                 'export PYTHONPATH="/usr/local/lib/python3.10/dist-packages:/home/container_app_user/tt-metal:$PYTHONPATH" && '
                 "export TT_MESH_GRAPH_DESC_PATH=/home/container_app_user/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto && "
                 "export TT_VISIBLE_DEVICES=2 && "
                 "cd /home/container_app_user/tt-metal && "
                 "python /home/container_app_user/voice-assistant/servers/speecht5_ttnn_server.py --speaker-id 7306 "
                 "> /tmp/tts_server.log 2>&1"],
                start_new_session=True
            )
            logger.info("🔄 TTS server restart initiated, waiting for warmup...")
            for _ in range(24):
                time.sleep(5)
                if os.path.exists(self.socket_path):
                    try:
                        self._send_request({"cmd": "ping"})
                        self._consecutive_failures = 0
                        self.restarting = False
                        logger.info("✅ TTS server restarted successfully!")
                        return True
                    except Exception:
                        pass
            self.restarting = False
            logger.error("❌ TTS server failed to restart within 2 minutes")
        except Exception as e:
            logger.error(f"❌ TTS server restart failed: {e}")
        return False

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
        except:
            return ""
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status."""
        return {
            "service": self.service_name,
            "device_id": self.device_id,
            "is_warmed_up": self.is_warmed_up,
            "warmup_time": self.warmup_time,
            "socket": self.socket_path,
            "model": "SpeechT5 (fast)"
        }
    
    async def shutdown(self):
        """Shutdown TTS service."""
        logger.info("🛑 Shutting down TTS service...")
        self.is_warmed_up = False
