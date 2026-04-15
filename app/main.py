#!/usr/bin/env python3
"""
TT Voice Assistant - Main API Server
Provides REST and WebSocket endpoints for voice assistant functionality.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, WebSocket, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Import our model services
from services.face_auth_service import FaceAuthService
from services.whisper_service import WhisperService  
from services.llama_service import LlamaService
from services.tts_service import TTSService
from services.model_manager import ModelManager

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
templates = Jinja2Templates(directory="/home/container_app_user/voice-assistant/templates")

# Global model manager
model_manager = None

@app.on_event("startup")
async def startup_event():
    """Initialize all model services on startup."""
    global model_manager
    logger.info("🚀 Starting TT Voice Assistant...")
    
    try:
        # Device allocation from NOTES.md:
        # Device 0: Face Auth (YuNet + SFace)
        # Device 1: Llama 3.1 8B Instruct
        # Device 2: Whisper (distil-large-v3)
        # Device 3: Qwen3 TTS
        model_manager = ModelManager({
            'face_auth_device': 0,
            'llama_device': 1,
            'whisper_device': 2,
            'tts_device': 3
        })
        
        # Warm up all models in parallel
        await model_manager.warmup_all_models()
        logger.info("✅ All models warmed up and ready!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize models: {e}")
        raise

@app.on_event("shutdown") 
async def shutdown_event():
    """Clean shutdown of model services."""
    global model_manager
    if model_manager:
        await model_manager.shutdown()
    logger.info("🛑 TT Voice Assistant stopped")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for container monitoring."""
    if not model_manager or not model_manager.is_ready():
        raise HTTPException(status_code=503, detail="Models not ready")
    
    return {
        "status": "healthy",
        "models_ready": model_manager.is_ready(),
        "uptime": model_manager.get_uptime()
    }

# Main web interface
@app.get("/", response_class=HTMLResponse)
async def get_main_page():
    """Serve the main voice assistant web interface."""
    return FileResponse("/home/container_app_user/voice-assistant/templates/index.html")

# Face authentication endpoint
@app.post("/api/face-auth")
async def face_authentication(image: UploadFile = File(...)):
    """Authenticate user via face recognition."""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Models not initialized")
    
    try:
        # Read uploaded image
        image_data = await image.read()
        
        # Run face authentication
        result = await model_manager.face_auth_service.authenticate(image_data)
        
        return {
            "authenticated": result["authenticated"],
            "user_id": result.get("user_id"),
            "confidence": result.get("confidence"),
            "message": result.get("message")
        }
        
    except Exception as e:
        logger.error(f"Face auth error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Speech-to-text endpoint
@app.post("/api/speech-to-text")
async def speech_to_text(audio: UploadFile = File(...)):
    """Convert speech audio to text using Whisper."""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Models not initialized")
    
    try:
        # Read uploaded audio
        audio_data = await audio.read()
        
        # Run speech recognition
        result = await model_manager.whisper_service.transcribe(audio_data)
        
        return {
            "text": result["text"],
            "confidence": result.get("confidence"),
            "processing_time": result.get("processing_time")
        }
        
    except Exception as e:
        logger.error(f"Speech-to-text error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Chat endpoint  
@app.post("/api/chat")
async def chat(request: Dict[str, Any]):
    """Generate chat response using Llama."""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Models not initialized")
    
    try:
        user_message = request.get("message", "")
        if not user_message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        # Generate response
        result = await model_manager.llama_service.generate_response(user_message)
        
        return {
            "response": result["response"],
            "processing_time": result.get("processing_time")
        }
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Text-to-speech endpoint
@app.post("/api/text-to-speech")
async def text_to_speech(request: Dict[str, Any]):
    """Convert text to speech using TTS (SpeechT5 for fast, Qwen3 for quality)."""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Models not initialized")
    
    try:
        text = request.get("text", "")
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        # Use fast=True for greetings (short text)
        fast = request.get("fast", False) or len(text) < 60
        
        # Generate speech
        result = await model_manager.tts_service.synthesize(text, fast=fast)
        
        # Return audio file
        return FileResponse(
            result["audio_path"],
            media_type="audio/wav",
            filename="response.wav"
        )
        
    except Exception as e:
        logger.error(f"Text-to-speech error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time interaction
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time voice assistant interaction."""
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            message_type = data.get("type")
            
            if message_type == "ping":
                await websocket.send_json({"type": "pong"})
                
            elif message_type == "chat":
                # Handle chat message
                user_message = data.get("message", "")
                result = await model_manager.llama_service.generate_response(user_message)
                
                await websocket.send_json({
                    "type": "chat_response",
                    "response": result["response"],
                    "processing_time": result.get("processing_time")
                })
                
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {message_type}"
                })
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info("WebSocket connection closed")

if __name__ == "__main__":
    # Get configuration from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8080"))
    
    logger.info(f"🎤 Starting TT Voice Assistant API on {host}:{port}")
    
    # Run the server
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        log_level="info",
        reload=False
    )
# New chat-style UI endpoint
@app.get("/chat", response_class=HTMLResponse)
async def get_chat_page():
    """Serve the new chat-style voice assistant interface."""
    return FileResponse("/home/container_app_user/voice-assistant/templates/index_chat.html")
