"""
Whisper Service - Speech-to-Text on TT Metal
Device: 2 (from NOTES.md)
Uses subprocess to avoid mesh device conflicts
"""

import asyncio
import logging
import tempfile
import time
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any
import numpy as np

# Thread pool for running blocking subprocess calls
_executor = ThreadPoolExecutor(max_workers=4)

logger = logging.getLogger(__name__)


class WhisperService:
    """Whisper STT on TT Metal Device 2 (via subprocess)."""
    
    def __init__(self, device_id: int = 2):
        """Initialize Whisper service."""
        self.device_id = device_id
        self.service_name = "Whisper"
        self.is_warmed_up = False
        self.warmup_time = 0
        self.model_repo = "distil-whisper/distil-large-v3"
        
        logger.info(f"Whisper service initialized for device {device_id}")
    
    async def warmup(self):
        """Warm up Whisper model via subprocess."""
        logger.info(f"🔥 Warming up Whisper on device {self.device_id}...")
        
        try:
            start_time = time.time()
            
            # Create warmup script
            warmup_script = self._create_warmup_script()
            
            # Set environment
            env = os.environ.copy()
            env['TT_MESH_GRAPH_DESC_PATH'] = '/home/container_app_user/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto'
            env['TT_VISIBLE_DEVICES'] = str(self.device_id)
            env['HF_MODEL'] = self.model_repo
            
            # Run warmup in thread pool (non-blocking)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                _executor,
                lambda: subprocess.run(
                    ['python', warmup_script],
                    cwd='/home/container_app_user/tt-metal',
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=180
                )
            )
            
            os.unlink(warmup_script)
            
            if result.returncode == 0:
                self.warmup_time = time.time() - start_time
                self.is_warmed_up = True
                logger.info(f"✅ Whisper warmed up in {self.warmup_time:.1f}s")
            else:
                raise RuntimeError(f"Whisper warmup failed: {result.stderr[:500]}")
            
        except Exception as e:
            logger.error(f"❌ Whisper warmup failed: {e}")
            raise
    
    def _create_warmup_script(self) -> str:
        """Create warmup script."""
        script = '''
import sys
import numpy as np
sys.path.insert(0, "/home/container_app_user/tt-metal")

import ttnn
from models.demos.audio.whisper.demo.demo import create_functional_whisper_for_conditional_generation_inference_pipeline

print("Opening mesh device...")
mesh_device = ttnn.open_mesh_device(
    mesh_shape=ttnn.MeshShape(1, 1),
    l1_small_size=32768,
    trace_region_size=100000000
)
mesh_device.enable_program_cache()

print("Creating Whisper pipeline...")
pipeline = create_functional_whisper_for_conditional_generation_inference_pipeline(
    mesh_device=mesh_device,
    model_repo="distil-whisper/distil-large-v3",
    language="en",
    task="transcribe",
    use_trace=True,
    batch_size_per_device=1
)

print("Running warmup inference...")
dummy_audio = np.random.randn(16000).astype(np.float32) * 0.1
_ = pipeline([(16000, dummy_audio)], stream=False)

ttnn.close_mesh_device(mesh_device)
print("WARMUP_SUCCESS")
'''
        path = tempfile.mktemp(suffix='.py')
        with open(path, 'w') as f:
            f.write(script)
        return path
    
    async def transcribe(self, audio_data: bytes) -> Dict[str, Any]:
        """Transcribe audio to text."""
        logger.info("🎤 Transcribing audio...")
        
        try:
            start_time = time.time()
            
            # Save raw audio to temp file (browser sends webm/opus)
            raw_audio_path = tempfile.mktemp(suffix='.webm')
            with open(raw_audio_path, 'wb') as f:
                f.write(audio_data)
            
            # Convert to WAV using ffmpeg
            audio_path = tempfile.mktemp(suffix='.wav')
            convert_result = subprocess.run(
                ['ffmpeg', '-y', '-i', raw_audio_path, '-ar', '16000', '-ac', '1', '-f', 'wav', audio_path],
                capture_output=True,
                timeout=30
            )
            os.unlink(raw_audio_path)
            
            if convert_result.returncode != 0:
                logger.warning(f"ffmpeg conversion failed, trying direct read")
                # Fallback: try to read directly
                audio_path = raw_audio_path.replace('.webm', '.wav')
                with open(audio_path, 'wb') as f:
                    f.write(audio_data)
            
            # Create transcription script
            script = self._create_transcription_script(audio_path)
            
            # Set environment
            env = os.environ.copy()
            env['TT_MESH_GRAPH_DESC_PATH'] = '/home/container_app_user/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto'
            env['TT_VISIBLE_DEVICES'] = str(self.device_id)
            
            # Run transcription in thread pool (non-blocking)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                _executor,
                lambda: subprocess.run(
                    ['python', script],
                    cwd='/home/container_app_user/tt-metal',
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
            )
            
            # Cleanup
            os.unlink(audio_path)
            os.unlink(script)
            
            processing_time = time.time() - start_time
            
            if result.returncode == 0:
                # Extract transcription
                output = result.stdout
                text = ""
                if "TRANSCRIPTION:" in output:
                    text = output.split("TRANSCRIPTION:")[-1].strip()
                    # Clean up log lines
                    clean_lines = []
                    for line in text.split('\n'):
                        line = line.strip()
                        if not line:
                            continue
                        if '|' in line and any(x in line.lower() for x in ['info', 'debug', 'warning']):
                            break
                        if line.startswith('202'):
                            break
                        clean_lines.append(line)
                    text = ' '.join(clean_lines).strip('[]"\' ')
                
                logger.info(f"✅ Transcription: '{text[:50]}...' in {processing_time:.2f}s")
                return {
                    "text": text,
                    "confidence": 0.95,
                    "processing_time": processing_time
                }
            else:
                raise RuntimeError(f"Transcription failed: {result.stderr[:200]}")
                
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return {
                "text": "",
                "confidence": 0,
                "processing_time": 0
            }
    
    def _create_transcription_script(self, audio_path: str) -> str:
        """Create transcription script."""
        script = f'''
import sys
import numpy as np
from scipy.io import wavfile
sys.path.insert(0, "/home/container_app_user/tt-metal")

import ttnn
from models.demos.audio.whisper.demo.demo import create_functional_whisper_for_conditional_generation_inference_pipeline

mesh_device = ttnn.open_mesh_device(
    mesh_shape=ttnn.MeshShape(1, 1),
    l1_small_size=32768,
    trace_region_size=100000000
)
mesh_device.enable_program_cache()

pipeline = create_functional_whisper_for_conditional_generation_inference_pipeline(
    mesh_device=mesh_device,
    model_repo="distil-whisper/distil-large-v3",
    language="en",
    task="transcribe",
    use_trace=True,
    batch_size_per_device=1
)

# Load audio
try:
    sr, audio = wavfile.read("{audio_path}")
    print(f"Audio loaded: sr={{sr}}, shape={{audio.shape}}, dtype={{audio.dtype}}")
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    print(f"Audio processed: length={{len(audio)}} samples, {{len(audio)/sr:.2f}}s")
except Exception as e:
    print(f"ERROR loading audio: {{e}}")
    raise RuntimeError(f"Failed to load audio: {{e}}")

result = pipeline([(sr, audio)], stream=False)
text = result[0] if result else ""

ttnn.close_mesh_device(mesh_device)
print(f"TRANSCRIPTION: {{text}}")
'''
        path = tempfile.mktemp(suffix='.py')
        with open(path, 'w') as f:
            f.write(script)
        return path
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status."""
        return {
            "service": self.service_name,
            "device_id": self.device_id,
            "is_warmed_up": self.is_warmed_up,
            "warmup_time": self.warmup_time
        }
    
    async def shutdown(self):
        """Shutdown Whisper service."""
        logger.info("🛑 Shutting down Whisper service...")
        self.is_warmed_up = False