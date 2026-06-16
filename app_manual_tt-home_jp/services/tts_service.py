"""
TTS Service - Qwen3 TTS with Jim's voice
Device: 3 (from NOTES.md)

Currently uses subprocess per request. 
TODO: Implement persistent server for fast inference.
"""

import asyncio
import logging
import os
import subprocess
import tempfile
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Jim's voice reference
JIM_REFERENCE_AUDIO = "/home/container_app_user/tt-metal/models/demos/qwen3_tts/demo/jim_reference.wav"
JIM_REFERENCE_TEXT = "Let me also go over the review slides."

# Thread pool for non-blocking subprocess calls
_executor = ThreadPoolExecutor(max_workers=2)


class TTSService:
    """TTS Service using Qwen3 TTS with Jim's voice."""
    
    def __init__(self, device_id: int = 3):
        """Initialize TTS service."""
        self.device_id = device_id
        self.service_name = "TTS"
        self.is_warmed_up = False
        self.warmup_time = 0
        self.output_dir = "/home/container_app_user/voice-assistant/output"
        
        logger.info(f"TTS service initialized for device {device_id}")
    
    async def warmup(self):
        """Warm up TTS by running one synthesis."""
        logger.info(f"🔥 Warming up Qwen3 TTS (Jim's voice) on device {self.device_id}...")
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Create output directory
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Set environment
            env = os.environ.copy()
            env['TT_MESH_GRAPH_DESC_PATH'] = '/home/container_app_user/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto'
            env['TT_VISIBLE_DEVICES'] = str(self.device_id)
            
            warmup_output = os.path.join(self.output_dir, "warmup.wav")
            
            cmd = [
                "python",
                "models/demos/qwen3_tts/demo/demo_full_ttnn_tts.py",
                "--text", "Hello from Tenstorrent",
                "--ref-audio", JIM_REFERENCE_AUDIO,
                "--ref-text", JIM_REFERENCE_TEXT,
                "--output", warmup_output
            ]
            
            logger.info("Running Qwen3 TTS warmup...")
            
            # Run in thread pool to not block event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                _executor,
                lambda: subprocess.run(
                    cmd,
                    cwd="/home/container_app_user/tt-metal",
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=120
                )
            )
            
            if result.returncode != 0:
                logger.warning(f"TTS warmup stderr: {result.stderr[:500] if result.stderr else 'none'}")
            
            # Clean up warmup file
            if os.path.exists(warmup_output):
                os.unlink(warmup_output)
            
            self.warmup_time = asyncio.get_event_loop().time() - start_time
            self.is_warmed_up = True
            logger.info(f"✅ TTS warmed up in {self.warmup_time:.1f}s")
            
        except Exception as e:
            logger.error(f"❌ TTS warmup failed: {e}")
            # Mark as warmed up anyway so pipeline continues
            self.is_warmed_up = True
            self.warmup_time = 0
    
    async def synthesize(self, text: str, fast: bool = False) -> Dict[str, Any]:
        """Synthesize speech from text using Qwen3 TTS with Jim's voice."""
        logger.info(f"🔊 TTS: {text[:50]}...")
        
        try:
            start_time = asyncio.get_event_loop().time()
            output_path = os.path.join(self.output_dir, f"tts_{int(start_time * 1000)}.wav")
            
            env = os.environ.copy()
            env['TT_MESH_GRAPH_DESC_PATH'] = '/home/container_app_user/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto'
            env['TT_VISIBLE_DEVICES'] = str(self.device_id)
            
            cmd = [
                "python",
                "models/demos/qwen3_tts/demo/demo_full_ttnn_tts.py",
                "--text", text,
                "--ref-audio", JIM_REFERENCE_AUDIO,
                "--ref-text", JIM_REFERENCE_TEXT,
                "--output", output_path
            ]
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                _executor,
                lambda: subprocess.run(
                    cmd,
                    cwd="/home/container_app_user/tt-metal",
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
            )
            
            if result.returncode != 0:
                logger.warning(f"TTS stderr: {result.stderr[:200] if result.stderr else 'none'}")
            
            if os.path.exists(output_path):
                processing_time = asyncio.get_event_loop().time() - start_time
                logger.info(f"✅ TTS completed in {processing_time:.2f}s")
                return {
                    "audio_path": output_path,
                    "processing_time": processing_time
                }
            else:
                raise RuntimeError("TTS output file not created")
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return {
                "audio_path": self._create_beep(),
                "processing_time": 0.1
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
        except:
            return ""
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status."""
        return {
            "service": self.service_name,
            "device_id": self.device_id,
            "is_warmed_up": self.is_warmed_up,
            "warmup_time": self.warmup_time,
            "voice": "Jim"
        }
    
    async def shutdown(self):
        """Shutdown TTS service."""
        logger.info("🛑 Shutting down TTS service...")
        self.is_warmed_up = False
