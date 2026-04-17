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
from fastapi.responses import HTMLResponse, FileResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# Import SOCKET-based services (connect to running servers)
from services.face_auth_service_socket import FaceAuthService
from services.whisper_service_socket import WhisperService
from services.tts_service_socket import TTSService

# Llama uses direct loading (not socket)
from services.llama_service import LlamaService
from services.mode_prompts import get_system_prompt, get_max_tokens, detect_mode_from_text, MODES
from services.document_service import DocumentService

# Conversation history for context (per session, simple in-memory store)
# Key: session_id, Value: list of {"role": "user"|"assistant", "content": str}
conversation_history: Dict[str, list] = {}
MAX_HISTORY_TURNS = 10  # Keep last 10 turns (20 messages) -- 4096 token context allows more history

# Track active mode per session
session_modes: Dict[str, str] = {}

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
document_service = DocumentService()


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


@app.get("/api/modes")
async def get_modes():
    """Return available modes for the frontend to render buttons."""
    return {
        "modes": {
            key: {"name": m["name"], "icon": m["icon"], "description": m["description"]}
            for key, m in MODES.items()
        },
        "default": "talk",
    }


@app.post("/api/upload-document")
async def upload_document(request: Request, file: UploadFile = File(None)):
    """Upload a document (PDF/text), paste text, or provide a URL for Docs Summary mode."""
    try:
        # Check if it's a file upload or JSON body (text/url)
        if file:
            file_data = await file.read()
            result = await document_service.extract_from_upload(file_data, file.filename)
            session_id = request.query_params.get("session_id", "default")
        else:
            data = await request.json()
            session_id = data.get("session_id", "default")
            url = data.get("url", "")
            text = data.get("text", "")

            if url:
                result = await document_service.extract_from_url(url)
            elif text:
                result = await document_service.extract_from_text(text)
            else:
                return {"error": "Provide a file, text, or url"}

        if "error" in result:
            return result

        document_service.store(session_id, result["text"])

        # Auto-switch to docs mode
        session_modes[session_id] = "docs"

        return {
            "status": "ok",
            "char_count": result.get("char_count", 0),
            "word_count": result.get("word_count", 0),
            "preview": result["text"][:200] + ("..." if len(result["text"]) > 200 else ""),
            "mode": "docs",
        }
    except Exception as e:
        logger.error(f"Document upload error: {e}")
        return {"error": str(e)}


@app.post("/api/chat")
async def chat(request: Request):
    """Generate chat response using Llama with mode-aware routing."""
    try:
        data = await request.json()
        user_message = data.get("message", "")
        session_id = data.get("session_id", "default")
        mode = data.get("mode")  # Explicit mode from button, or None

        if not user_message:
            raise HTTPException(status_code=400, detail="Message is required")

        # --- Mode resolution ---
        # Priority: explicit button mode > keyword detection > session's last mode > default
        if mode:
            session_modes[session_id] = mode
        else:
            detected = detect_mode_from_text(user_message)
            if detected:
                mode = detected
                session_modes[session_id] = mode
                logger.info(f"Auto-detected mode '{mode}' from keywords")
            else:
                mode = session_modes.get(session_id, "talk")

        system_prompt = get_system_prompt(mode)
        max_tokens = data.get("max_tokens", get_max_tokens(mode))

        # For docs mode, inject document text into the system prompt
        if mode == "docs":
            doc_text = document_service.get_context_for_prompt(session_id)
            if doc_text:
                system_prompt += f"\n\nHere is the document the user provided:\n---\n{doc_text}\n---"
            else:
                system_prompt += "\n\nNo document has been uploaded yet. Ask the user to upload or paste a document first."

        # For podcast mode, inject document if uploaded (otherwise Llama uses the topic)
        if mode == "podcast":
            doc_text = document_service.get_context_for_prompt(session_id)
            if doc_text:
                system_prompt += f"\n\nCreate the podcast discussion based on this source material. Cover the key points and make them conversational:\n---\n{doc_text}\n---"

        logger.info(f"Chat mode={mode}, session={session_id[:20]}...")

        # Get or create conversation history for this session
        if session_id not in conversation_history:
            conversation_history[session_id] = []

        history = conversation_history[session_id]

        # Podcast & Story: each request is independent -- no history needed
        # Talk & Docs: keep conversation history for multi-turn context
        if mode in ("podcast", "story"):
            chat_history = []
        else:
            history.append({"role": "user", "content": user_message[:200]})
            chat_history = history[:-1]

        # Generate response with mode-specific system prompt
        result = await llama_service.generate_response(
            user_message,
            max_tokens=max_tokens,
            conversation_history=chat_history,
            system_prompt=system_prompt,
        )

        assistant_response = result.get("response", "")

        # Save history (skip for podcast & story -- each is standalone)
        if mode not in ("podcast", "story"):
            history.append({"role": "assistant", "content": assistant_response[:200]})
            if len(history) > MAX_HISTORY_TURNS * 2:
                conversation_history[session_id] = history[-(MAX_HISTORY_TURNS * 2):]

        return {
            "response": assistant_response,
            "mode": mode,
            "processing_time": result.get("processing_time", 0),
        }
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return {"response": f"Error: {str(e)}"}


@app.post("/api/clear-history")
async def clear_history(request: Request):
    """Clear conversation history and mode for a session."""
    try:
        data = await request.json()
        session_id = data.get("session_id", "default")
        if session_id in conversation_history:
            conversation_history[session_id] = []
        if session_id in session_modes:
            del session_modes[session_id]
        document_service.clear(session_id)
        return {"status": "ok", "message": "History cleared"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/chat-stream")
async def chat_stream(request: Request):
    """Stream chat response with sentences, generating TTS audio for each.
    
    Returns Server-Sent Events (SSE) stream with events:
      - sentence: {text: "...", audio_url: "/api/tts-chunk/..."}
      - done: {full_response: "...", processing_time: ...}
      - error: {message: "..."}
    """
    import base64
    
    try:
        data = await request.json()
        user_message = data.get("message", "")
        session_id = data.get("session_id", "default")
        mode = data.get("mode")

        if not user_message:
            raise HTTPException(status_code=400, detail="Message is required")

        # Mode resolution (same as regular chat)
        if mode:
            session_modes[session_id] = mode
        else:
            detected = detect_mode_from_text(user_message)
            if detected:
                mode = detected
                session_modes[session_id] = mode
            else:
                mode = session_modes.get(session_id, "talk")

        system_prompt = get_system_prompt(mode)
        max_tokens = data.get("max_tokens", get_max_tokens(mode))

        if mode == "docs":
            doc_text = document_service.get_context_for_prompt(session_id)
            if doc_text:
                system_prompt += f"\n\nHere is the document the user provided:\n---\n{doc_text}\n---"

        if mode == "podcast":
            doc_text = document_service.get_context_for_prompt(session_id)
            if doc_text:
                system_prompt += f"\n\nCreate the podcast discussion based on this source material:\n---\n{doc_text}\n---"

        logger.info(f"Streaming chat mode={mode}, session={session_id[:20]}...")

        if session_id not in conversation_history:
            conversation_history[session_id] = []
        history = conversation_history[session_id]
        
        if mode in ("podcast", "story"):
            chat_history = []
        else:
            history.append({"role": "user", "content": user_message[:200]})
            chat_history = history[:-1]

        async def generate_stream():
            """SSE generator that streams sentences with inline audio."""
            full_response = ""
            sentence_count = 0
            
            # Speaker IDs for podcast multi-voice
            PODCAST_SPEAKERS = {"host": 7306, "guest": 1138}
            current_speaker = None  # Track current speaker for podcast mode
            
            try:
                async for chunk in llama_service.generate_response_streaming(
                    user_message,
                    max_tokens=max_tokens,
                    conversation_history=chat_history,
                    system_prompt=system_prompt,
                ):
                    if chunk["type"] == "sentence":
                        sentence_text = chunk["text"]
                        sentence_count += 1
                        
                        # For podcast mode: detect and handle [HOST]: / [GUEST]: tags
                        speaker_id = None
                        display_text = sentence_text
                        
                        if mode == "podcast":
                            # Check for speaker tags at start of sentence
                            import re
                            speaker_match = re.match(r'^\s*\[?(HOST|GUEST)\]?\s*:?\s*', sentence_text, re.IGNORECASE)
                            if speaker_match:
                                role = speaker_match.group(1).lower()
                                current_speaker = role
                                # Strip the tag from text for TTS
                                clean_text = sentence_text[speaker_match.end():].strip()
                                if clean_text:
                                    sentence_text = clean_text
                                else:
                                    # Tag only, no content - skip TTS
                                    yield f"data: {json_lib.dumps({'type': 'sentence', 'text': display_text, 'audio_b64': '', 'sentence_num': sentence_count, 'role': current_speaker})}\n\n"
                                    continue
                            
                            # Use current speaker's voice
                            if current_speaker:
                                speaker_id = PODCAST_SPEAKERS.get(current_speaker, 7306)
                        
                        # Generate TTS for this sentence
                        try:
                            tts_result = await tts_service.synthesize(sentence_text, speaker_id=speaker_id)
                            audio_path = tts_result.get("audio_path")
                            
                            # Read audio file and encode as base64
                            audio_b64 = ""
                            if audio_path and os.path.exists(audio_path):
                                with open(audio_path, "rb") as f:
                                    audio_b64 = base64.b64encode(f.read()).decode('utf-8')
                            
                            yield f"data: {json_lib.dumps({'type': 'sentence', 'text': display_text, 'audio_b64': audio_b64, 'sentence_num': sentence_count, 'role': current_speaker})}\n\n"
                        except Exception as tts_err:
                            logger.error(f"TTS error for sentence {sentence_count}: {tts_err}")
                            yield f"data: {json_lib.dumps({'type': 'sentence', 'text': display_text, 'audio_b64': '', 'sentence_num': sentence_count, 'role': current_speaker})}\n\n"
                    
                    elif chunk["type"] == "done":
                        full_response = chunk["full_response"]
                        
                        # Save to history
                        if mode not in ("podcast", "story"):
                            history.append({"role": "assistant", "content": full_response[:200]})
                            if len(history) > MAX_HISTORY_TURNS * 2:
                                conversation_history[session_id] = history[-(MAX_HISTORY_TURNS * 2):]
                        
                        yield f"data: {json_lib.dumps({'type': 'done', 'full_response': full_response, 'processing_time': chunk.get('processing_time', 0), 'mode': mode})}\n\n"
                    
                    elif chunk["type"] == "error":
                        yield f"data: {json_lib.dumps({'type': 'error', 'message': chunk.get('error', 'Unknown error')})}\n\n"
                
            except Exception as e:
                logger.error(f"Stream error: {e}")
                yield f"data: {json_lib.dumps({'type': 'error', 'message': str(e)})}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
        
    except Exception as e:
        logger.error(f"Chat stream error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/text-to-speech")
async def text_to_speech(request: Request):
    """Convert text to speech using TTS. Supports podcast multi-speaker mode."""
    try:
        data = await request.json()
        text = data.get("text", "")
        mode = data.get("mode", "")
        
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        # Podcast mode: detect HOST/GUEST tags in various formats
        if mode == "podcast" and re.search(r'(?:\*\*)?[\[\(]?HOST[\]\)]?(?:\*\*)?:', text, re.IGNORECASE):
            result = await _synthesize_podcast(text)
        else:
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


import re
import json as json_lib
import socket as socket_lib
import time as time_mod

TTS_SOCKET = "/tmp/tts_server.sock"

async def _synthesize_podcast(script_text: str) -> dict:
    """Parse a podcast script with HOST/GUEST tags and generate multi-speaker audio."""
    logger.info("Generating multi-speaker podcast audio...")
    
    # Normalize various tag formats to a standard form:
    #   [HOST]: / [GUEST]: / **HOST:** / **[HOST]:** / HOST: / (HOST): etc.
    normalized = script_text
    normalized = re.sub(r'\*\*\[?(HOST)\]?\*\*\s*:', r'[HOST]:', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\*\*\[?(GUEST)\]?\*\*\s*:', r'[GUEST]:', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'(?<!\[)\b(HOST)\b(?!\]):', r'[HOST]:', normalized)
    normalized = re.sub(r'(?<!\[)\b(GUEST)\b(?!\]):', r'[GUEST]:', normalized)
    
    # Also strip out non-spoken stage directions like [INTRO MUSIC], [OUTRO MUSIC], etc.
    normalized = re.sub(r'\*?\*?\[(?!HOST|GUEST)[^\]]*\]\*?\*?', '', normalized)
    
    # Parse [HOST]: and [GUEST]: segments
    segments = []
    pattern = r'\[(HOST|GUEST)\]:\s*(.*?)(?=\[(?:HOST|GUEST)\]:|$)'
    matches = re.findall(pattern, normalized, re.DOTALL)
    
    for role, text in matches:
        text = text.strip()
        if text:
            segments.append({"role": role.lower(), "text": text})
    
    if not segments:
        logger.warning("No HOST/GUEST segments found after normalization, falling back to regular TTS")
        return await tts_service.synthesize(script_text)
    
    logger.info(f"Parsed {len(segments)} podcast segments")
    
    output_path = f"/home/container_app_user/voice-assistant/output/podcast_{int(time_mod.time() * 1000)}.wav"
    
    request_data = {
        "cmd": "podcast",
        "segments": segments,
        "output_path": output_path,
    }
    
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: _send_tts_request(request_data)
    )
    
    if response.get("status") == "ok":
        logger.info(f"Podcast audio generated in {response.get('time_ms', 0):.0f}ms")
        return {
            "audio_path": response.get("audio_path", output_path),
            "processing_time": response.get("time_ms", 0) / 1000,
        }
    else:
        raise RuntimeError(response.get("error", "Podcast TTS failed"))


def _send_tts_request(request_data: dict) -> dict:
    """Send request to TTS socket server."""
    client = socket_lib.socket(socket_lib.AF_UNIX, socket_lib.SOCK_STREAM)
    client.settimeout(300)
    client.connect(TTS_SOCKET)
    client.sendall(json_lib.dumps(request_data).encode('utf-8'))
    response = client.recv(65536).decode('utf-8')
    client.close()
    return json_lib.loads(response)


if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8080"))
    
    logger.info(f"🎤 Starting TT Voice Assistant API on {host}:{port}")
    uvicorn.run("main:app", host=host, port=port, log_level="info", reload=False)
