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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import struct
import tempfile
import socket
import json

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


# ============================================================================
# WebSocket Streaming STT
# ============================================================================

WHISPER_SOCKET = "/tmp/whisper_server.sock"
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # 16-bit PCM
MIN_AUDIO_DURATION = 0.5  # Minimum audio duration in seconds before processing
MAX_AUDIO_DURATION = 30.0  # Maximum audio buffer (Whisper's window)


def _send_whisper_streaming_request(audio_path: str):
    """
    Send a streaming transcription request to Whisper server.
    Yields partial results as they arrive (newline-delimited JSON).
    """
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(120)
    client.connect(WHISPER_SOCKET)
    
    request = {"audio_path": audio_path, "stream": True}
    client.sendall(json.dumps(request).encode('utf-8'))
    
    buffer = b""
    while True:
        try:
            chunk = client.recv(4096)
            if not chunk:
                break
            buffer += chunk
            
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if line:
                    yield json.loads(line.decode('utf-8'))
        except socket.timeout:
            break
        except Exception as e:
            logger.error(f"Whisper streaming error: {e}")
            break
    
    client.close()


def _save_pcm_as_wav(pcm_data: bytes, sample_rate: int = 16000) -> str:
    """Save raw PCM16 data as a WAV file. Returns the file path."""
    import wave
    
    wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(wav_file.name, 'wb') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    
    return wav_file.name


@app.websocket("/ws/stt")
async def websocket_stt_streaming(websocket: WebSocket):
    """
    WebSocket endpoint for streaming Speech-to-Text.
    
    Protocol:
    - Client sends binary audio chunks (PCM16, 16kHz, mono)
    - Client sends JSON control messages: {"action": "start"}, {"action": "stop"}, {"action": "finalize"}
    - Server sends JSON responses: {"type": "partial", "text": "..."} or {"type": "final", "text": "..."}
    
    Flow:
    1. Client connects and sends {"action": "start"}
    2. Client streams PCM16 audio chunks (binary messages)
    3. When client wants transcription, sends {"action": "transcribe"}
    4. Server processes buffered audio and streams partial results
    5. Client can continue streaming more audio
    6. Client sends {"action": "stop"} to end session
    """
    await websocket.accept()
    logger.info("WebSocket STT connection established")
    
    audio_buffer = bytearray()
    session_active = True
    
    try:
        while session_active:
            try:
                message = await websocket.receive()
                
                if message["type"] == "websocket.disconnect":
                    break
                
                if "bytes" in message:
                    # Binary audio data (PCM16)
                    audio_buffer.extend(message["bytes"])
                    
                    # Calculate buffer duration
                    buffer_duration = len(audio_buffer) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
                    
                    # Auto-transcribe if buffer exceeds max duration
                    if buffer_duration >= MAX_AUDIO_DURATION:
                        logger.info(f"Auto-transcribing: buffer at {buffer_duration:.1f}s")
                        await _process_and_stream_transcription(websocket, audio_buffer)
                        audio_buffer = bytearray()
                
                elif "text" in message:
                    # JSON control message
                    try:
                        data = json.loads(message["text"])
                        action = data.get("action", "")
                        
                        if action == "start":
                            # Reset buffer for new session
                            audio_buffer = bytearray()
                            await websocket.send_json({"type": "status", "message": "ready"})
                            logger.info("STT session started")
                        
                        elif action == "transcribe":
                            # Process current buffer
                            buffer_duration = len(audio_buffer) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
                            
                            if buffer_duration < MIN_AUDIO_DURATION:
                                await websocket.send_json({
                                    "type": "error",
                                    "message": f"Audio too short ({buffer_duration:.1f}s < {MIN_AUDIO_DURATION}s)"
                                })
                            else:
                                logger.info(f"Transcribing {buffer_duration:.1f}s of audio")
                                await _process_and_stream_transcription(websocket, audio_buffer)
                                audio_buffer = bytearray()
                        
                        elif action == "stop":
                            # End session, transcribe any remaining audio
                            buffer_duration = len(audio_buffer) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
                            if buffer_duration >= MIN_AUDIO_DURATION:
                                logger.info(f"Final transcription: {buffer_duration:.1f}s")
                                await _process_and_stream_transcription(websocket, audio_buffer, is_final=True)
                            
                            await websocket.send_json({"type": "status", "message": "stopped"})
                            session_active = False
                            logger.info("STT session stopped")
                        
                        elif action == "clear":
                            # Clear buffer without transcribing
                            audio_buffer = bytearray()
                            await websocket.send_json({"type": "status", "message": "cleared"})
                        
                        else:
                            await websocket.send_json({
                                "type": "error",
                                "message": f"Unknown action: {action}"
                            })
                    
                    except json.JSONDecodeError:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Invalid JSON"
                        })
            
            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected")
                break
            except Exception as e:
                logger.error(f"WebSocket STT error: {e}")
                try:
                    await websocket.send_json({"type": "error", "message": str(e)})
                except:
                    pass
                break
    
    finally:
        logger.info("WebSocket STT connection closed")


def _send_whisper_normal_request(audio_path: str) -> dict:
    """Send a normal (non-streaming) transcription request to Whisper server."""
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(30)
    client.connect(WHISPER_SOCKET)
    request = {"audio_path": audio_path, "stream": False}
    client.sendall(json.dumps(request).encode('utf-8'))
    response = client.recv(1024 * 1024).decode('utf-8')
    client.close()
    return json.loads(response)


@app.websocket("/ws/stt-live")
async def websocket_stt_live(websocket: WebSocket):
    """
    Live real-time STT endpoint.
    
    Client sends full accumulated audio as PCM16 binary, then {"action": "transcribe"}.
    Server transcribes using normal Whisper (no stream_generation) and returns result.
    Client keeps sending updated audio as user speaks.
    
    When done, client sends {"action": "final"} with last audio for final transcription,
    which triggers LLM processing.
    """
    await websocket.accept()
    logger.info("WebSocket STT-Live connection established")
    
    audio_buffer = bytearray()
    session_active = True
    
    try:
        while session_active:
            try:
                message = await websocket.receive()
                
                if message["type"] == "websocket.disconnect":
                    break
                
                if "bytes" in message:
                    audio_buffer = bytearray(message["bytes"])
                
                elif "text" in message:
                    try:
                        data = json.loads(message["text"])
                        action = data.get("action", "")
                        
                        if action == "transcribe" or action == "final":
                            buffer_duration = len(audio_buffer) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
                            
                            if buffer_duration < 0.3:
                                await websocket.send_json({
                                    "type": "transcript",
                                    "text": "",
                                    "is_final": action == "final",
                                    "duration": buffer_duration
                                })
                            else:
                                wav_path = None
                                try:
                                    wav_path = _save_pcm_as_wav(bytes(audio_buffer))
                                    loop = asyncio.get_event_loop()
                                    response = await loop.run_in_executor(
                                        None,
                                        lambda: _send_whisper_normal_request(wav_path)
                                    )
                                    
                                    raw_text = response.get("text", "")
                                    if isinstance(raw_text, list):
                                        text = " ".join(raw_text).strip()
                                    else:
                                        text = str(raw_text).strip()
                                    time_ms = response.get("time_ms", 0)
                                    
                                    logger.info(f"STT-Live: {buffer_duration:.1f}s audio → \"{text[:50]}\" in {time_ms:.0f}ms")
                                    
                                    await websocket.send_json({
                                        "type": "transcript",
                                        "text": text,
                                        "is_final": action == "final",
                                        "time_ms": time_ms,
                                        "duration": buffer_duration
                                    })
                                finally:
                                    if wav_path and os.path.exists(wav_path):
                                        try:
                                            os.unlink(wav_path)
                                        except:
                                            pass
                            
                            if action == "final":
                                session_active = False
                        
                        elif action == "stop":
                            session_active = False
                            await websocket.send_json({"type": "status", "message": "stopped"})
                    
                    except json.JSONDecodeError:
                        await websocket.send_json({"type": "error", "message": "Invalid JSON"})
            
            except WebSocketDisconnect:
                logger.info("WebSocket STT-Live client disconnected")
                break
            except Exception as e:
                logger.error(f"WebSocket STT-Live error: {e}")
                try:
                    await websocket.send_json({"type": "error", "message": str(e)})
                except:
                    pass
                break
    finally:
        logger.info("WebSocket STT-Live connection closed")


async def _process_and_stream_transcription(
    websocket: WebSocket,
    audio_buffer: bytearray,
    is_final: bool = False
):
    """
    Process audio buffer and stream transcription results via WebSocket.
    """
    if not audio_buffer:
        return
    
    # Save audio to temp WAV file
    wav_path = None
    try:
        wav_path = _save_pcm_as_wav(bytes(audio_buffer))
        
        # Send streaming request to Whisper server
        loop = asyncio.get_event_loop()
        
        def stream_whisper():
            results = []
            for response in _send_whisper_streaming_request(wav_path):
                results.append(response)
            return results
        
        # Run in executor to avoid blocking
        results = await loop.run_in_executor(None, stream_whisper)
        
        # Stream results to WebSocket
        for response in results:
            if response.get("status") == "partial":
                await websocket.send_json({
                    "type": "partial",
                    "text": response.get("text", ""),
                    "is_final": False
                })
            elif response.get("status") == "ok":
                await websocket.send_json({
                    "type": "final" if is_final else "transcript",
                    "text": response.get("text", ""),
                    "is_final": response.get("is_final", True),
                    "time_ms": response.get("time_ms", 0)
                })
            elif response.get("status") == "error":
                await websocket.send_json({
                    "type": "error",
                    "message": response.get("error", "Transcription failed")
                })
    
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
    
    finally:
        # Clean up temp file
        if wav_path and os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
            except:
                pass


@app.get("/", response_class=HTMLResponse)
async def get_main_page():
    """Serve the main voice assistant web interface."""
    return FileResponse("/home/container_app_user/voice-assistant/templates/index.html")


@app.get("/chat", response_class=HTMLResponse)
async def get_chat_page():
    """Serve the new chat-style interface."""
    return FileResponse("/home/container_app_user/voice-assistant/templates/index_chat.html")


@app.get("/live-stt", response_class=HTMLResponse)
async def get_live_stt_demo():
    """Standalone live STT demo page."""
    return FileResponse("/home/container_app_user/voice-assistant/templates/live_stt_demo.html")


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


def _split_into_sentences(text: str) -> list:
    """Split text into small chunks for streaming TTS with low latency."""
    parts = re.split(r'(?<=[.!?;])\s+|(?<=,)\s+(?=\w{4})', text.strip())
    sentences = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if sentences and len(sentences[-1]) < 20:
            sentences[-1] += ' ' + p
        else:
            sentences.append(p)
    return sentences if sentences else [text]


@app.post("/api/text-to-speech-stream")
async def text_to_speech_stream(request: Request):
    """Stream TTS audio sentence-by-sentence for faster first-audio."""
    data = await request.json()
    text = data.get("text", "")
    mode = data.get("mode", "")

    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    PODCAST_SPEAKERS = {"host": 7306, "guest": 1138}

    is_podcast = mode == "podcast" and (
        re.search(r'(?:\*\*)?[\[\(]?(?:HOST|GUEST)[\]\)]?(?:\*\*)?:', text, re.IGNORECASE)
        or re.search(r'\[[A-Za-z .]+\]:', text)
    )

    if is_podcast:
        normalized = text
        normalized = re.sub(r'\*\*\[?(HOST)\]?\*\*\s*:', r'[HOST]:', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'\*\*\[?(GUEST)\]?\*\*\s*:', r'[GUEST]:', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'(?<!\[)\b(HOST)\b(?!\]):', r'[HOST]:', normalized)
        normalized = re.sub(r'(?<!\[)\b(GUEST)\b(?!\]):', r'[GUEST]:', normalized)

        # If Llama used character names instead of HOST/GUEST, map the first
        # unique name to HOST and the second to GUEST
        if not re.search(r'\[(?:HOST|GUEST)\]:', normalized):
            name_tags = re.findall(r'\[([A-Za-z .]+)\]:', normalized)
            seen = []
            for n in name_tags:
                if n not in seen:
                    seen.append(n)
            if len(seen) >= 2:
                normalized = normalized.replace(f'[{seen[0]}]:', '[HOST]:')
                normalized = normalized.replace(f'[{seen[1]}]:', '[GUEST]:')
                logger.info(f"Mapped speaker names: '{seen[0]}'->HOST, '{seen[1]}'->GUEST")
            elif len(seen) == 1:
                normalized = normalized.replace(f'[{seen[0]}]:', '[HOST]:')
                logger.info(f"Mapped single speaker '{seen[0]}'->HOST")

        normalized = re.sub(r'\*?\*?\[(?!HOST|GUEST)[^\]]*\]\*?\*?', '', normalized)
        pattern = r'\[(HOST|GUEST)\]:\s*(.*?)(?=\[(?:HOST|GUEST)\]:|$)'
        matches = re.findall(pattern, normalized, re.DOTALL)
        segments = [{"role": r.lower(), "text": t.strip()} for r, t in matches if t.strip()]
        logger.info(f"Streaming podcast TTS: {len(segments)} segments")
    else:
        segments = [{"role": "host", "text": s} for s in _split_into_sentences(text)]
        logger.info(f"Streaming TTS: {len(segments)} sentences")

    async def generate_chunks():
        for i, seg in enumerate(segments):
            try:
                speaker_id = PODCAST_SPEAKERS.get(seg["role"]) if seg["role"] else None
                result = await tts_service.synthesize(seg["text"], speaker_id=speaker_id)
                audio_path = result.get("audio_path")
                if audio_path and os.path.exists(audio_path):
                    with open(audio_path, "rb") as f:
                        chunk_data = f.read()
                    header = len(chunk_data).to_bytes(4, 'big')
                    yield header + chunk_data
                    role_tag = f" ({seg['role']})" if seg['role'] else ""
                    logger.info(f"  Streamed {i+1}/{len(segments)}{role_tag}: {seg['text'][:40]}...")
                    try:
                        os.remove(audio_path)
                    except:
                        pass
            except Exception as e:
                logger.error(f"TTS stream error on segment {i+1}: {e}")

    return StreamingResponse(
        generate_chunks(),
        media_type="application/octet-stream",
        headers={"X-Sentence-Count": str(len(segments))}
    )

TTS_SOCKET = "/tmp/tts_server.sock"

async def _synthesize_podcast(script_text: str) -> dict:
    """Parse a podcast script with HOST/GUEST tags and generate multi-speaker audio."""
    logger.info("Generating multi-speaker podcast audio...")
    
    normalized = script_text
    normalized = re.sub(r'\*\*\[?(HOST)\]?\*\*\s*:', r'[HOST]:', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'\*\*\[?(GUEST)\]?\*\*\s*:', r'[GUEST]:', normalized, flags=re.IGNORECASE)
    normalized = re.sub(r'(?<!\[)\b(HOST)\b(?!\]):', r'[HOST]:', normalized)
    normalized = re.sub(r'(?<!\[)\b(GUEST)\b(?!\]):', r'[GUEST]:', normalized)
    
    # If Llama used character names instead of HOST/GUEST, map them
    if not re.search(r'\[(?:HOST|GUEST)\]:', normalized):
        name_tags = re.findall(r'\[([A-Za-z .]+)\]:', normalized)
        seen = []
        for n in name_tags:
            if n not in seen:
                seen.append(n)
        if len(seen) >= 2:
            normalized = normalized.replace(f'[{seen[0]}]:', '[HOST]:')
            normalized = normalized.replace(f'[{seen[1]}]:', '[GUEST]:')
            logger.info(f"Mapped speaker names: '{seen[0]}'->HOST, '{seen[1]}'->GUEST")
        elif len(seen) == 1:
            normalized = normalized.replace(f'[{seen[0]}]:', '[HOST]:')
    
    normalized = re.sub(r'\*?\*?\[(?!HOST|GUEST)[^\]]*\]\*?\*?', '', normalized)
    
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
