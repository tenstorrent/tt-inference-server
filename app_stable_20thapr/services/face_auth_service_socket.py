"""
Face Auth Service - Socket-based client for Face Auth server.
Connects to the persistent Face Auth server via Unix socket.
"""

import asyncio
import logging
import socket
import json
import base64
import os
from typing import Dict, Any
from pathlib import Path
import numpy as np
import cv2

logger = logging.getLogger(__name__)

FACE_AUTH_SOCKET = "/tmp/face_auth_server.sock"
FACES_DIR = Path("/home/container_app_user/voice-assistant/registered_faces")


class FaceAuthService:
    """Face Auth via persistent socket server (YuNet + SFace on TT Metal)."""
    
    def __init__(self, device_id: int = 0):
        """Initialize face auth service."""
        self.device_id = device_id
        self.service_name = "FaceAuth"
        self.is_warmed_up = False
        self.warmup_time = 0
        self.socket_path = FACE_AUTH_SOCKET
        self.face_database = {}
        
        logger.info(f"Face Auth service initialized (socket: {self.socket_path})")
    
    def _send_request(self, request: dict) -> dict:
        """Send request to Face Auth server."""
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(60)
        client.connect(self.socket_path)
        client.sendall(json.dumps(request).encode('utf-8'))
        response = client.recv(10 * 1024 * 1024).decode('utf-8')
        client.close()
        return json.loads(response)
    
    async def warmup(self):
        """Check if Face Auth server is ready."""
        logger.info(f"🔥 Checking Face Auth server...")
        
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._send_request({"cmd": "ping"})
            )
            
            if response.get("status") == "ok":
                self.is_warmed_up = True
                self._load_faces()
                logger.info(f"✅ Face Auth server is ready!")
            else:
                raise RuntimeError(f"Face Auth server not ready: {response}")
                
        except Exception as e:
            logger.error(f"❌ Face Auth server check failed: {e}")
            self.is_warmed_up = False
    
    def _load_faces(self):
        """Load registered faces from disk (for local reference)."""
        FACES_DIR.mkdir(parents=True, exist_ok=True)
        
        for person_dir in FACES_DIR.iterdir():
            if person_dir.is_dir():
                name = person_dir.name
                embedding_path = person_dir / "embedding.npy"
                if embedding_path.exists():
                    self.face_database[name] = True  # Just track names
                    logger.info(f"Found registered face: {name}")
        
        logger.info(f"Loaded {len(self.face_database)} registered faces")
    
    async def authenticate(self, image_data: bytes) -> Dict[str, Any]:
        """Authenticate user via face."""
        logger.info("🔍 Running face authentication...")
        
        try:
            # Encode image as base64
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Send to server
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._send_request({"image_base64": image_b64})
            )
            
            if response.get("status") == "ok":
                faces = response.get("faces", [])
                
                if faces:
                    # Find best match
                    best_face = max(faces, key=lambda f: f.get("score", 0))
                    identity = best_face.get("identity", "Unknown")
                    score = best_face.get("score", 0)
                    
                    if identity != "Unknown" and score > 0.5:
                        logger.info(f"✅ Authenticated: {identity} (score: {score:.2f})")
                        return {
                            "authenticated": True,
                            "user_id": identity,
                            "confidence": score,
                            "message": f"Welcome {identity}!"
                        }
                    else:
                        logger.info(f"Face detected but not recognized (score: {score:.2f})")
                        return {
                            "authenticated": True,
                            "user_id": "Guest",
                            "confidence": score,
                            "message": "Face detected but not recognized"
                        }
                else:
                    logger.info("No face detected")
                    return {
                        "authenticated": False,
                        "user_id": None,
                        "confidence": 0,
                        "message": "No face detected"
                    }
            else:
                raise RuntimeError(response.get("error", "Unknown error"))
                
        except Exception as e:
            logger.error(f"Face auth error: {e}")
            # Fallback - allow guest access
            return {
                "authenticated": True,
                "user_id": "Guest",
                "confidence": 0.5,
                "message": f"Face auth error: {e}"
            }
    
    async def register_face(self, name: str, image_data: bytes) -> Dict[str, Any]:
        """Register a new face."""
        logger.info(f"📝 Registering face: {name}")
        
        try:
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._send_request({
                    "cmd": "register",
                    "name": name,
                    "image_base64": image_b64
                })
            )
            
            if response.get("status") == "ok":
                self.face_database[name] = True
                logger.info(f"✅ Registered face: {name}")
                return {"success": True, "message": f"Registered {name}"}
            else:
                return {"success": False, "message": response.get("error", "Unknown error")}
                
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return {"success": False, "message": str(e)}
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status."""
        return {
            "service": self.service_name,
            "device_id": self.device_id,
            "is_warmed_up": self.is_warmed_up,
            "warmup_time": self.warmup_time,
            "socket": self.socket_path,
            "registered_faces": list(self.face_database.keys())
        }
    
    async def shutdown(self):
        """Shutdown Face Auth service."""
        logger.info("🛑 Shutting down Face Auth service...")
        self.is_warmed_up = False
