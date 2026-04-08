"""
Llama Service - Socket-based client for Llama server.
Connects to the persistent Llama server via Unix socket.
"""

import asyncio
import logging
import socket
import json
from typing import Dict, Any

logger = logging.getLogger(__name__)

LLAMA_SOCKET = "/tmp/llama_server.sock"


class LlamaService:
    """Llama LLM via persistent socket server."""
    
    def __init__(self, device_id: int = 1):
        """Initialize Llama service."""
        self.device_id = device_id
        self.service_name = "Llama"
        self.is_warmed_up = False
        self.warmup_time = 0
        self.socket_path = LLAMA_SOCKET
        
        logger.info(f"Llama service initialized (socket: {self.socket_path})")
    
    def _send_request(self, request: dict) -> dict:
        """Send request to Llama server."""
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(120)  # 2 minute timeout for generation
        client.connect(self.socket_path)
        client.sendall(json.dumps(request).encode('utf-8'))
        
        # Receive response (may be large)
        chunks = []
        while True:
            chunk = client.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
        
        client.close()
        response_data = b''.join(chunks).decode('utf-8')
        return json.loads(response_data)
    
    async def warmup(self):
        """Check if Llama server is ready."""
        logger.info(f"🔥 Checking Llama server...")
        
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._send_request({"cmd": "ping"})
            )
            
            if response.get("status") == "ok":
                self.is_warmed_up = True
                logger.info(f"✅ Llama server is ready!")
            else:
                raise RuntimeError(f"Llama server not ready: {response}")
                
        except Exception as e:
            logger.error(f"❌ Llama server check failed: {e}")
            self.is_warmed_up = False
    
    def is_ready(self) -> bool:
        """Check if service is ready."""
        return self.is_warmed_up
    
    def generate(self, prompt: str, max_tokens: int = 150) -> str:
        """Generate text from prompt (sync version for compatibility)."""
        try:
            response = self._send_request({
                "prompt": prompt,
                "max_tokens": max_tokens
            })
            
            if response.get("status") == "ok":
                return response.get("text", "")
            else:
                return f"Error: {response.get('error', 'Unknown error')}"
                
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Error generating response: {e}"
    
    async def generate_async(self, prompt: str, max_tokens: int = 150) -> str:
        """Generate text from prompt (async version)."""
        logger.info(f"💬 Generating response for: {prompt[:50]}...")
        
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._send_request({
                    "prompt": prompt,
                    "max_tokens": max_tokens
                })
            )
            
            if response.get("status") == "ok":
                text = response.get("text", "")
                logger.info(f"✅ Generated {len(text)} chars in {response.get('time_ms', 0):.1f}ms")
                return text
            else:
                raise RuntimeError(response.get("error", "Unknown error"))
                
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Error generating response: {e}"
    
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
        """Shutdown Llama service."""
        logger.info("🛑 Shutting down Llama service...")
        self.is_warmed_up = False
