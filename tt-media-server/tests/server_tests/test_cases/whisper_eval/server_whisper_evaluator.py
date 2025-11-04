"""
Server-Compatible Whisper Evaluator

This module extracts the core Whisper evaluation code from lmms-eval
and adapts it for server deployment. It maintains the original logic
while removing framework-specific dependencies.

Based on: lmms_eval/models/whisper.py
"""

from typing import List, Optional, Tuple, Union, Dict, Any
import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration
from tqdm import tqdm
import numpy as np


# Copied from lmms_eval.models.model_utils.audio_processing
def downsample_audio(audio_array: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    """Downsample audio to target sampling rate using librosa"""
    from librosa import resample
    audio_resample_array = resample(audio_array, orig_sr=original_sr, target_sr=target_sr)
    return audio_resample_array


class ServerWhisperEvaluator:
    """
    Server-compatible Whisper evaluator extracted from lmms-eval
    
    This class copies the essential evaluation logic from the original
    lmms-eval Whisper model while removing framework dependencies.
    """
    
    def __init__(
        self,
        pretrained: str = "openai/whisper-tiny",
        device: Optional[str] = "cuda",
        batch_size: int = 1,
        use_cache: bool = True,
        language: str = "en",
        task: str = "transcribe",
    ):
        """
        Initialize the Whisper evaluator for server use
        
        Args:
            pretrained: Pretrained model name/path (same as original)
            device: Device to use ('cuda', 'cpu', or 'auto')
            batch_size: Batch size for processing
            use_cache: Whether to use model cache
            language: Language for transcription
            task: Task type ('transcribe' or 'translate')
        """
        # Device handling (simplified from original)
        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)
        
        # Load model (copied from original __init__)
        self._model = WhisperForConditionalGeneration.from_pretrained(
            pretrained,
            torch_dtype="auto",
        ).eval().to(self._device)
        
        # Load processor (copied from original __init__)
        self.processor = AutoProcessor.from_pretrained(pretrained)
        self.processor.tokenizer.set_prefix_tokens(language=language, task=task)
        self._tokenizer = self.processor.tokenizer
        
        # Store config (copied from original)
        self._config = self._model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        
    @property
    def model(self):
        """Get the model (copied from original)"""
        return self._model
    
    @property
    def tokenizer(self):
        """Get the tokenizer (copied from original)"""
        return self._tokenizer
    
    @property
    def device(self):
        """Get the device (copied from original)"""
        return self._device
    
    @property
    def eot_token_id(self):
        """Get end-of-text token ID (copied from original)"""
        return self.tokenizer.eos_token_id
    
    def flatten(self, input_list):
        """Flatten nested list (copied from original)"""
        new_list = []
        for i in input_list:
            for j in i:
                new_list.append(j)
        return new_list
    
    def evaluate_audio_batch(
        self,
        audio_dicts: List[Dict[str, Any]],
        generation_kwargs: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Evaluate a batch of audio data
        
        Args:
            audio_dicts: List of audio dictionaries with 'array' and 'sampling_rate' keys
                        (same format as used in lmms-eval)
            generation_kwargs: Optional generation parameters
            
        Returns:
            List of transcription strings
        """
        if generation_kwargs is None:
            generation_kwargs = {}
        
        # Process audio data (adapted from original generate_until)
        flattened_audios = audio_dicts  # Already flattened for server use
        
        # Set default values for until and max_new_tokens (copied from original)
        until = [self.tokenizer.decode(self.eot_token_id)]
        
        # Update values from gen_kwargs if present (copied from original)
        gen_kwargs = generation_kwargs.copy()
        if "until" in gen_kwargs:
            until = gen_kwargs.pop("until")
            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")
        
        # Process inputs (copied from original generate_until)
        sampling_rate = self.processor.feature_extractor.sampling_rate
        audios = [
            downsample_audio(audio["array"], audio["sampling_rate"], sampling_rate) 
            for audio in flattened_audios
        ]
        inputs = self.processor(audio=audios, return_tensors="pt", sampling_rate=sampling_rate)
        inputs = inputs.to(self.device)
        
        # Set generation defaults (copied from original)
        if "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = 256
        if "temperature" not in gen_kwargs:
            gen_kwargs["temperature"] = 0
        if "top_p" not in gen_kwargs:
            gen_kwargs["top_p"] = None
        if "num_beams" not in gen_kwargs:
            gen_kwargs["num_beams"] = 1
        
        try:
            # Generate transcriptions (copied from original)
            predicted_ids = self.model.generate(
                **inputs,
                do_sample=True if gen_kwargs["temperature"] > 0 else False,
                temperature=gen_kwargs["temperature"],
                top_p=gen_kwargs["top_p"],
                num_beams=gen_kwargs["num_beams"],
                max_new_tokens=gen_kwargs["max_new_tokens"],
                min_new_tokens=1,
                use_cache=self.use_cache,
            )
            
            # Decode and post-process (copied from original)
            transcriptions = self.processor.batch_decode(predicted_ids)
            answers = [self.tokenizer.normalize(transcription) for transcription in transcriptions]
            
            # Apply until termination (copied from original)
            for i, ans in enumerate(answers):
                for term in until:
                    if len(term) > 0:
                        ans = ans.split(term)[0]
                answers[i] = ans
            
            return answers
            
        except Exception as e:
            print(f"Error while generating: {e}")
            return [""] * len(flattened_audios)
    
    def evaluate_single_audio(
        self,
        audio_dict: Dict[str, Any],
        generation_kwargs: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Evaluate a single audio sample
        
        Args:
            audio_dict: Audio dictionary with 'array' and 'sampling_rate' keys
            generation_kwargs: Optional generation parameters
            
        Returns:
            Transcription string
        """
        results = self.evaluate_audio_batch([audio_dict], generation_kwargs)
        return results[0]
    
    def evaluate_audio_from_arrays(
        self,
        audio_arrays: List[np.ndarray],
        sampling_rates: List[int],
        generation_kwargs: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Evaluate audio from numpy arrays
        
        Args:
            audio_arrays: List of audio arrays
            sampling_rates: List of corresponding sampling rates
            generation_kwargs: Optional generation parameters
            
        Returns:
            List of transcription strings
        """
        # Convert to lmms-eval format
        audio_dicts = [
            {"array": audio, "sampling_rate": sr}
            for audio, sr in zip(audio_arrays, sampling_rates)
        ]
        
        return self.evaluate_audio_batch(audio_dicts, generation_kwargs)


# Server integration utilities
class WhisperServer:
    """
    Example server wrapper for the Whisper evaluator
    """
    
    def __init__(self, model_name: str = "openai/whisper-base", device: str = "cuda"):
        """Initialize the server with a Whisper model"""
        self.evaluator = ServerWhisperEvaluator(
            pretrained=model_name,
            device=device,
            batch_size=4,  # Adjust based on your server's GPU memory
            use_cache=True
        )
        print(f"Whisper server initialized with model: {model_name} on device: {device}")
    
    def transcribe_audio(self, audio_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Transcribe audio data (single sample)
        
        Args:
            audio_data: Dictionary with audio array and sampling rate
            **kwargs: Generation parameters
            
        Returns:
            Dictionary with transcription result
        """
        try:
            transcription = self.evaluator.evaluate_single_audio(audio_data, kwargs)
            return {
                "success": True,
                "transcription": transcription,
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "transcription": "",
                "error": str(e)
            }
    
    def transcribe_audio_batch(
        self, 
        audio_batch: List[Dict[str, Any]], 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Transcribe multiple audio samples
        
        Args:
            audio_batch: List of audio dictionaries
            **kwargs: Generation parameters
            
        Returns:
            Dictionary with batch transcription results
        """
        try:
            transcriptions = self.evaluator.evaluate_audio_batch(audio_batch, kwargs)
            return {
                "success": True,
                "transcriptions": transcriptions,
                "count": len(transcriptions),
                "error": None
            }
        except Exception as e:
            return {
                "success": False,
                "transcriptions": [],
                "count": 0,
                "error": str(e)
            }


# Flask/FastAPI integration examples
def create_flask_endpoints(app, whisper_server: WhisperServer):
    """
    Add Whisper endpoints to a Flask app
    
    Example usage:
        from flask import Flask
        app = Flask(__name__)
        whisper_server = WhisperServer()
        create_flask_endpoints(app, whisper_server)
    """
    from flask import request, jsonify
    
    @app.route('/transcribe', methods=['POST'])
    def transcribe():
        """Single audio transcription endpoint"""
        data = request.json
        
        # Expect format: {"audio": {"array": [...], "sampling_rate": 16000}}
        audio_data = data.get('audio')
        gen_kwargs = data.get('generation_kwargs', {})
        
        if not audio_data:
            return jsonify({"error": "No audio data provided"}), 400
        
        result = whisper_server.transcribe_audio(audio_data, **gen_kwargs)
        return jsonify(result)
    
    @app.route('/transcribe_batch', methods=['POST'])
    def transcribe_batch():
        """Batch audio transcription endpoint"""
        data = request.json
        
        # Expect format: {"audio_batch": [{"array": [...], "sampling_rate": 16000}, ...]}
        audio_batch = data.get('audio_batch', [])
        gen_kwargs = data.get('generation_kwargs', {})
        
        if not audio_batch:
            return jsonify({"error": "No audio data provided"}), 400
        
        result = whisper_server.transcribe_audio_batch(audio_batch, **gen_kwargs)
        return jsonify(result)


def create_fastapi_endpoints(app, whisper_server: WhisperServer):
    """
    Add Whisper endpoints to a FastAPI app
    
    Example usage:
        from fastapi import FastAPI
        app = FastAPI()
        whisper_server = WhisperServer()
        create_fastapi_endpoints(app, whisper_server)
    """
    from fastapi import HTTPException
    from pydantic import BaseModel
    from typing import List, Dict, Any, Optional
    
    class AudioData(BaseModel):
        array: List[float]
        sampling_rate: int
    
    class TranscribeRequest(BaseModel):
        audio: AudioData
        generation_kwargs: Optional[Dict[str, Any]] = {}
    
    class BatchTranscribeRequest(BaseModel):
        audio_batch: List[AudioData]
        generation_kwargs: Optional[Dict[str, Any]] = {}
    
    @app.post("/transcribe")
    async def transcribe(request: TranscribeRequest):
        """Single audio transcription endpoint"""
        audio_dict = {
            "array": np.array(request.audio.array),
            "sampling_rate": request.audio.sampling_rate
        }
        
        result = whisper_server.transcribe_audio(audio_dict, **request.generation_kwargs)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
    
    @app.post("/transcribe_batch")
    async def transcribe_batch(request: BatchTranscribeRequest):
        """Batch audio transcription endpoint"""
        audio_dicts = [
            {
                "array": np.array(audio.array),
                "sampling_rate": audio.sampling_rate
            }
            for audio in request.audio_batch
        ]
        
        result = whisper_server.transcribe_audio_batch(audio_dicts, **request.generation_kwargs)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result


if __name__ == "__main__":
    # Example usage
    print("Server Whisper Evaluator Example")
    
    # Initialize the evaluator (same as original lmms-eval logic)
    evaluator = ServerWhisperEvaluator(
        pretrained="openai/whisper-tiny",  # Use tiny for demo
        device="cpu",  # Use CPU for demo
        batch_size=2
    )
    
    # Example audio data (same format as lmms-eval expects)
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_array = 0.3 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    audio_dict = {
        "array": audio_array,
        "sampling_rate": sample_rate
    }
    
    # Single audio evaluation
    print("\\nTesting single audio evaluation:")
    result = evaluator.evaluate_single_audio(audio_dict)
    print(f"Result: '{result}'")
    
    # Batch evaluation
    print("\\nTesting batch evaluation:")
    audio_batch = [audio_dict, audio_dict]  # Same audio twice for demo
    results = evaluator.evaluate_audio_batch(audio_batch)
    print(f"Batch results: {results}")
    
    # Server wrapper example
    print("\\nTesting server wrapper:")
    server = WhisperServer(model_name="openai/whisper-tiny", device="cpu")
    server_result = server.transcribe_audio(audio_dict)
    print(f"Server result: {server_result}")
    
    print("\\n✅ Server-compatible Whisper evaluator ready for deployment!")