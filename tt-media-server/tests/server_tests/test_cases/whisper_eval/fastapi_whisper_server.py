
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
from server_whisper_evaluator import WhisperServer

app = FastAPI(title="Whisper Audio Transcription API")

# Initialize Whisper server (using copied lmms-eval code)
whisper_server = WhisperServer(
    model_name="openai/whisper-base",
    device="cuda"
)

# Request/Response models
class AudioData(BaseModel):
    array: List[float]
    sampling_rate: int

class TranscribeRequest(BaseModel):
    audio: AudioData
    generation_kwargs: Optional[Dict[str, Any]] = {}

class BatchTranscribeRequest(BaseModel):
    audio_batch: List[AudioData]
    generation_kwargs: Optional[Dict[str, Any]] = {}

class TranscribeResponse(BaseModel):
    success: bool
    transcription: str
    error: Optional[str] = None

class BatchTranscribeResponse(BaseModel):
    success: bool
    transcriptions: List[str]
    count: int
    error: Optional[str] = None

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": "whisper"}

@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_audio(request: TranscribeRequest):
    """Transcribe single audio sample"""
    try:
        # Convert to numpy array and create lmms-eval format
        audio_dict = {
            "array": np.array(request.audio.array, dtype=np.float32),
            "sampling_rate": request.audio.sampling_rate
        }
        
        # Call the copied lmms-eval evaluation code
        result = whisper_server.transcribe_audio(audio_dict, **request.generation_kwargs)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return TranscribeResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe_batch", response_model=BatchTranscribeResponse)
async def transcribe_audio_batch(request: BatchTranscribeRequest):
    """Transcribe multiple audio samples"""
    try:
        # Convert to numpy arrays and create lmms-eval format
        audio_dicts = []
        for audio in request.audio_batch:
            audio_dict = {
                "array": np.array(audio.array, dtype=np.float32),
                "sampling_rate": audio.sampling_rate
            }
            audio_dicts.append(audio_dict)
        
        # Call the copied lmms-eval evaluation code
        result = whisper_server.transcribe_audio_batch(audio_dicts, **request.generation_kwargs)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return BatchTranscribeResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    