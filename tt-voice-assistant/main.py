"""
TT Voice Assistant - Main FastAPI Application
Chat-style UI with Face Auth, Whisper STT, Llama LLM, SpeechT5 TTS

All services connect to persistent servers via Unix sockets.
"""

import os
import io
import logging
from pathlib import Path
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, Response
from fastapi.templating import Jinja2Templates

# Import socket-based services
from services.face_auth_service_socket import FaceAuthService
from services.whisper_service_socket import WhisperService
from services.tts_service_socket import TTSService

# Try to import llama service (may be in different location)
try:
    from services.llama_service import LlamaService
    llama = LlamaService()
except ImportError:
    llama = None
    logging.warning("LlamaService not found - chat will use fallback")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TT Voice Assistant")

# Templates
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Initialize services (socket clients)
face_auth = FaceAuthService()
whisper = WhisperService()
tts = TTSService()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the chat UI."""
    return templates.TemplateResponse("index_chat.html", {"request": request})


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "services": {
        "face_auth": face_auth.is_warmed_up,
        "whisper": whisper.is_warmed_up,
        "tts": tts.is_warmed_up,
        "llama": llama.is_ready() if llama else False
    }}


@app.post("/api/face-auth")
async def authenticate_face(image: UploadFile = File(...)):
    """Authenticate user via face recognition."""
    try:
        image_data = await image.read()
        result = await face_auth.authenticate(image_data)
        return result
    except Exception as e:
        logger.error(f"Face auth error: {e}")
        return {"authenticated": False, "user_id": "Guest", "error": str(e)}


@app.post("/api/register-face")
async def register_face(name: str, image: UploadFile = File(...)):
    """Register a new face."""
    try:
        image_data = await image.read()
        result = await face_auth.register_face(name, image_data)
        return result
    except Exception as e:
        logger.error(f"Register face error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/registered-faces")
async def get_registered_faces():
    """Get list of registered faces."""
    try:
        return {"faces": list(face_auth.face_database.keys())}
    except Exception as e:
        return {"faces": [], "error": str(e)}


@app.post("/api/speech-to-text")
async def speech_to_text(audio: UploadFile = File(...)):
    """Convert speech to text using Whisper."""
    try:
        audio_data = await audio.read()
        result = await whisper.transcribe(audio_data)
        return result
    except Exception as e:
        logger.error(f"STT error: {e}")
        return {"text": "", "error": str(e)}


@app.post("/api/chat")
async def chat(request: Request):
    """Generate chat response using Llama."""
    try:
        data = await request.json()
        message = data.get("message", "")
        max_tokens = data.get("max_tokens", 150)
        
        if llama:
            response = llama.generate(message, max_tokens=max_tokens)
        else:
            response = f"Echo: {message}"
        
        return {"response": response}
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return {"response": f"Error: {str(e)}"}


@app.post("/api/text-to-speech")
async def text_to_speech(request: Request):
    """Convert text to speech using SpeechT5."""
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


@app.on_event("startup")
async def startup():
    """Warmup services on startup."""
    logger.info("Starting TT Voice Assistant...")
    await face_auth.warmup()
    await whisper.warmup()
    await tts.warmup()
    logger.info("All services initialized!")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
