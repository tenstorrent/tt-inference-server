#!/usr/bin/env python3
"""
TT Voice Assistant - Main API Server
Uses socket-based services that connect to persistent model servers.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, WebSocket, HTTPException, UploadFile, File, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware

# Import SOCKET-based services (connect to running servers)
from services.face_auth_service_socket import FaceAuthService
from services.whisper_service_socket import WhisperService
from services.tts_service_socket import TTSService

# Llama uses direct loading (not socket)
from services.llama_service import LlamaService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/container_app_user/voice-assistant/logs/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="TT Voice Assistant",
    description="AI Voice Assistant powered by Tenstorrent hardware",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="/home/container_app_user/voice-assistant/static"), name="static")

# Initialize socket-based services
face_auth_service = FaceAuthService()
whisper_service = WhisperService()
tts_service = TTSService()
llama_service = LlamaService()


@app.on_event("startup")
async def startup_event():
    """Initialize all model services on startup."""
    logger.info("🚀 Starting TT Voice Assistant (Socket Mode)...")
    
    try:
        # Check socket servers are ready
        await face_auth_service.warmup()
        await whisper_service.warmup()
        await tts_service.warmup()
        
        # Llama loads directly
        await llama_service.warmup()
        
        logger.info("✅ All services ready!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize: {e}")
        # Continue anyway - some services may work


@app.on_event("shutdown") 
async def shutdown_event():
    """Clean shutdown."""
    logger.info("🛑 TT Voice Assistant stopped")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "face_auth": face_auth_service.is_warmed_up,
        "whisper": whisper_service.is_warmed_up,
        "tts": tts_service.is_warmed_up,
        "llama": llama_service.is_warmed_up
    }


@app.get("/", response_class=HTMLResponse)
async def get_main_page():
    """Serve the main voice assistant web interface."""
    return FileResponse("/home/container_app_user/voice-assistant/templates/index.html")


@app.get("/chat", response_class=HTMLResponse)
async def get_chat_page():
    """Serve the new chat-style interface."""
    return FileResponse("/home/container_app_user/voice-assistant/templates/index_chat.html")


@app.post("/api/face-auth")
async def face_authentication(image: UploadFile = File(...)):
    """Authenticate user via face recognition."""
    try:
        image_data = await image.read()
        result = await face_auth_service.authenticate(image_data)
        return result
    except Exception as e:
        logger.error(f"Face auth error: {e}")
        return {
            "authenticated": False,
            "user_id": "Guest",
            "confidence": 0,
            "message": str(e)
        }


@app.post("/api/register-face")
async def register_face(name: str, image: UploadFile = File(...)):
    """Register a new face."""
    try:
        image_data = await image.read()
        result = await face_auth_service.register_face(name, image_data)
        return result
    except Exception as e:
        logger.error(f"Register face error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/speech-to-text")
async def speech_to_text(audio: UploadFile = File(...)):
    """Convert speech audio to text using Whisper."""
    try:
        audio_data = await audio.read()
        result = await whisper_service.transcribe(audio_data)
        return result
    except Exception as e:
        logger.error(f"Speech-to-text error: {e}")
        return {"text": "", "error": str(e)}


@app.post("/api/chat")
async def chat(request: Request):
    """Generate chat response using Llama."""
    try:
        data = await request.json()
        user_message = data.get("message", "")
        max_tokens = data.get("max_tokens", 150)
        
        if not user_message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        result = await llama_service.generate_response(user_message, max_tokens=max_tokens)
        return {
            "response": result.get("response", ""),
            "processing_time": result.get("processing_time", 0)
        }
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return {"response": f"Error: {str(e)}"}


@app.post("/api/text-to-speech")
async def text_to_speech(request: Request):
    """Convert text to speech using TTS."""
    try:
        data = await request.json()
        text = data.get("text", "")
        
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        result = await tts_service.synthesize(text)
        audio_path = result.get("audio_path")
        
        if audio_path and os.path.exists(audio_path):
            with open(audio_path, "rb") as f:
                audio_data = f.read()
            return Response(
                content=audio_data,
                media_type="audio/wav",
                headers={"Content-Disposition": "inline; filename=response.wav"}
            )
        else:
            raise HTTPException(status_code=500, detail="No audio generated")
            
    except Exception as e:
        logger.error(f"Text-to-speech error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8080"))
    
    logger.info(f"🎤 Starting TT Voice Assistant API on {host}:{port}")
    uvicorn.run("main:app", host=host, port=port, log_level="info", reload=False)
