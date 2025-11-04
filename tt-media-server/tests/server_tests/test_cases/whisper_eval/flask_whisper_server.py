
from flask import Flask, request, jsonify
import numpy as np
from server_whisper_evaluator import WhisperServer

app = Flask(__name__)

# Initialize Whisper server (using copied lmms-eval code)
whisper_server = WhisperServer(
    model_name="openai/whisper-base",  # Choose your model size
    device="cuda"  # or "cpu"
)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model": "whisper"})

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """
    Transcribe single audio sample
    
    Expected input format:
    {
        "audio": {
            "array": [0.1, 0.2, ...],  # Audio samples as list
            "sampling_rate": 16000
        },
        "generation_kwargs": {  # Optional
            "max_new_tokens": 256,
            "temperature": 0.0
        }
    }
    """
    try:
        data = request.json
        
        # Extract audio data
        audio_data = data.get('audio')
        if not audio_data:
            return jsonify({"error": "No audio data provided"}), 400
        
        # Convert list to numpy array
        audio_dict = {
            "array": np.array(audio_data["array"], dtype=np.float32),
            "sampling_rate": audio_data["sampling_rate"]
        }
        
        # Generation parameters
        gen_kwargs = data.get('generation_kwargs', {})
        
        # Call the copied lmms-eval evaluation code
        result = whisper_server.transcribe_audio(audio_dict, **gen_kwargs)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/transcribe_batch', methods=['POST'])
def transcribe_audio_batch():
    """
    Transcribe multiple audio samples
    
    Expected input format:
    {
        "audio_batch": [
            {
                "array": [0.1, 0.2, ...],
                "sampling_rate": 16000
            },
            {
                "array": [0.3, 0.4, ...],
                "sampling_rate": 16000
            }
        ],
        "generation_kwargs": {  # Optional
            "max_new_tokens": 256
        }
    }
    """
    try:
        data = request.json
        
        # Extract batch data
        audio_batch = data.get('audio_batch', [])
        if not audio_batch:
            return jsonify({"error": "No audio data provided"}), 400
        
        # Convert to numpy arrays
        audio_dicts = []
        for audio_data in audio_batch:
            audio_dict = {
                "array": np.array(audio_data["array"], dtype=np.float32),
                "sampling_rate": audio_data["sampling_rate"]
            }
            audio_dicts.append(audio_dict)
        
        # Generation parameters
        gen_kwargs = data.get('generation_kwargs', {})
        
        # Call the copied lmms-eval evaluation code
        result = whisper_server.transcribe_audio_batch(audio_dicts, **gen_kwargs)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
    