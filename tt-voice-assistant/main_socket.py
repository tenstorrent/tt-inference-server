#!/usr/bin/env python3
"""
TT Voice Assistant - Main API Server (Socket-based)
Connects to persistent model servers via Unix sockets.

Servers must be running:
- Face Auth: /tmp/face_auth_server.sock (Device 0)
- Whisper: /tmp/whisper_server.sock (Device 2)  
- TTS: /tmp/tts_server.sock (CPU)
- Llama: Loaded directly in this app (Device 1)
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware

# Socket-based services
from services.face_auth_service_socket import FaceAuthService
from services.whisper_service_socket import WhisperService
from services.tts_service_socket import TTSService

# Llama runs directly (not via socket for now)
from services.llama_service import LlamaService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/container_app_user/voice-assistant/logs/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="TT Voice Assistant", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="/home/container_app_user/voice-assistant/static"), name="static")

# Initialize socket-based services
face_auth = FaceAuthService()
whisper = WhisperService()
tts = TTSService()
llama = LlamaService()


@app.on_event("startup")
async def startup():
    """Check that all servers are ready."""
    logger.info("🚀 Starting TT Voice Assistant (Socket Mode)...")
    
    # Check socket servers
    await face_auth.warmup()
    await whisper.warmup()
    await tts.warmup()
    
    # Llama loads directly
    await llama.warmup()
    
    logger.info("✅ All services ready!")


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "face_auth": face_auth.is_warmed_up,
        "whisper": whisper.is_warmed_up,
        "tts": tts.is_warmed_up,
        "llama": llama.is_warmed_up
    }


@app.get("/", response_class=HTMLResponse)
async def get_main_page():
    return FileResponse("/home/container_app_user/voice-assistant/templates/index.html")


@app.get("/chat", response_class=HTMLResponse)
async def get_chat_page():
    return FileResponse("/home/container_app_user/voice-assistant/templates/index_chat.html")


@app.post("/api/face-auth")
async def face_authentication(image: UploadFile = File(...)):
    try:
        image_data = await image.read()
        result = await face_auth.authenticate(image_data)
        return result
    except Exception as e:
        logger.error(f"Face auth error: {e}")
        return {"authenticated": False, "user_id": "Guest", "error": str(e)}


@app.post("/api/register-face")
async def register_face(name: str, image: UploadFile = File(...)):
    try:
        image_data = await image.read()
        result = await face_auth.register_face(name, image_data)
        return result
    except Exception as e:
        logger.error(f"Register face error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/speech-to-text")
async def speech_to_text(audio: UploadFile = File(...)):
    try:
        audio_data = await audio.read()
        result = await whisper.transcribe(audio_data)
        return result
    except Exception as e:
        logger.error(f"STT error: {e}")
        return {"text": "", "error": str(e)}


@app.post("/api/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        message = data.get("message", "")
        max_tokens = data.get("max_tokens", 150)
        
        response = await llama.generate_async(message, max_tokens=max_tokens)
        return {"response": response}
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return {"response": f"Error: {str(e)}"}


@app.post("/api/text-to-speech")
async def text_to_speech(request: Request):
    try:
        data = await request.json()
        text = data.get("text", "")
        
        result = await tts.synthesize(text)
        audio_path = result.get("audio_path")
        
        if audio_path and os.path.exists(audio_path):
            with open(audio_path, "rb") as f:
                audio_data = f.read()
            return Response(
                content=audio_data,
                media_type="audio/wav",
                headers={"Content-Disposition": "inline; filename=speech.wav"}
            )
        else:
            raise HTTPException(status_code=500, detail="No audio generated")
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8080"))
    
    logger.info(f"🎤 Starting on {host}:{port}")
    uvicorn.run("main_socket:app", host=host, port=port, log_level="info")
