"""
Model Manager - Coordinates all AI model services
Device allocation from NOTES.md:
  - Device 0: Face Auth (YuNet + SFace)
  - Device 1: Llama 3.1 8B
  - Device 2: Whisper
  - Device 3: Qwen3 TTS
"""

import asyncio
import logging
import time
from typing import Dict, Any

from .face_auth_service import FaceAuthService
from .whisper_service import WhisperService
from .llama_service import LlamaService
from .tts_service import TTSService

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages all AI model services for the voice assistant."""
    
    def __init__(self, device_config: Dict[str, int]):
        """Initialize model manager with device configuration."""
        self.device_config = device_config
        self.start_time = time.time()
        
        # Initialize services with their assigned devices
        self.face_auth_service = FaceAuthService(device_config.get('face_auth_device', 0))
        self.llama_service = LlamaService(device_config.get('llama_device', 1))
        self.whisper_service = WhisperService(device_config.get('whisper_device', 2))
        self.tts_service = TTSService(device_config.get('tts_device', 3))
        
        self._ready = False
        logger.info(f"Model Manager initialized with devices: {device_config}")
    
    async def warmup_all_models(self):
        """Warm up all models in parallel using thread pool for subprocess-based services."""
        logger.info("🔥 Starting PARALLEL model warmup...")
        logger.info("📋 Device allocation:")
        logger.info("   Device 0: Face Auth (YuNet + SFace)")
        logger.info("   Device 1: Llama 3.1 8B")
        logger.info("   Device 2: Whisper")
        logger.info("   Device 3: Qwen3 TTS")
        
        # Create warmup tasks for all models
        async def warmup_with_logging(name, service):
            logger.info(f"🚀 Starting {name} warmup...")
            try:
                await service.warmup()
                logger.info(f"✅ {name} ready!")
                return True
            except Exception as e:
                logger.error(f"❌ {name} warmup failed: {e}")
                return False
        
        # Run ALL warmups in parallel
        # Llama blocks event loop but others use ThreadPoolExecutor
        results = await asyncio.gather(
            warmup_with_logging("Llama (Device 1)", self.llama_service),
            warmup_with_logging("Whisper (Device 2)", self.whisper_service),
            warmup_with_logging("TTS (Device 3)", self.tts_service),
            warmup_with_logging("Face Auth (Device 0)", self.face_auth_service),
            return_exceptions=True
        )
        
        # Check results
        service_names = ['Llama', 'Whisper', 'TTS', 'Face Auth']
        critical_services = ['Llama', 'Whisper']
        failed_critical = False
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"❌ {service_names[i]} exception: {result}")
                if service_names[i] in critical_services:
                    failed_critical = True
            elif result is False:
                if service_names[i] in critical_services:
                    failed_critical = True
        
        if failed_critical:
            raise RuntimeError("Critical model warmup failed (Llama or Whisper)")
        
        self._ready = True
        warmup_time = time.time() - self.start_time
        logger.info(f"\n🎉 All models ready! Total warmup time: {warmup_time:.1f}s (parallel)")
    
    def is_ready(self) -> bool:
        """Check if all models are ready."""
        return self._ready
    
    def get_uptime(self) -> float:
        """Get uptime in seconds."""
        return time.time() - self.start_time
    
    async def shutdown(self):
        """Shutdown all model services."""
        logger.info("🛑 Shutting down model services...")
        
        await self.face_auth_service.shutdown()
        await self.whisper_service.shutdown()
        await self.llama_service.shutdown()
        await self.tts_service.shutdown()
        
        logger.info("✅ All model services shut down")
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all services."""
        return {
            "ready": self._ready,
            "uptime": self.get_uptime(),
            "device_config": self.device_config,
            "services": {
                "face_auth": self.face_auth_service.get_status(),
                "llama": self.llama_service.get_status(),
                "whisper": self.whisper_service.get_status(),
                "tts": self.tts_service.get_status()
            }
        }