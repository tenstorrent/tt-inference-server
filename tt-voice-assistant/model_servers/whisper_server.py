#!/usr/bin/env python3
"""
Whisper STT Server - Persistent model server using ZMQ.

Keeps Whisper model loaded on TT device, accepts audio via ZMQ, returns transcription.

Usage:
  TT_MESH_GRAPH_DESC_PATH=.../p150_mesh_graph_descriptor.textproto \
  TT_VISIBLE_DEVICES='0' \
  python whisper_server.py --port 5555

Protocol:
  REQ: {"audio": base64_encoded_wav, "sample_rate": 16000}
  REP: {"text": "transcribed text", "time_ms": 123.4}
"""

import os
import sys
import json
import time
import base64
import argparse
import numpy as np
import zmq

def main():
    parser = argparse.ArgumentParser(description="Whisper STT Server")
    parser.add_argument("--port", type=int, default=5555, help="ZMQ port")
    parser.add_argument("--model", type=str, default="distil-whisper/distil-large-v3")
    args = parser.parse_args()

    print("=" * 60)
    print("Whisper STT Server")
    print("=" * 60)
    print(f"Port: {args.port}")
    print(f"Model: {args.model}")
    print(f"TT_VISIBLE_DEVICES: {os.environ.get('TT_VISIBLE_DEVICES', 'not set')}")

    import torch
    import ttnn
    from transformers import AutoProcessor

    print("\n[1/4] Opening TT device...")
    t0 = time.time()
    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    device.enable_program_cache()
    print(f"  Device opened in {time.time() - t0:.1f}s")

    print("\n[2/4] Loading Whisper model...")
    t0 = time.time()
    
    from models.demos.whisper.tt.ttnn_optimized_functional_whisper import (
        ttnn_optimized_whisper_for_conditional_generation,
        preprocess_encoder_inputs,
        preprocess_decoder_inputs,
    )
    from transformers import WhisperForConditionalGeneration
    
    processor = AutoProcessor.from_pretrained(args.model)
    hf_model = WhisperForConditionalGeneration.from_pretrained(args.model)
    config = hf_model.config
    
    parameters = ttnn_optimized_whisper_for_conditional_generation.preprocess_model_parameters(
        hf_model, device
    )
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    print("\n[3/4] Warming up model...")
    t0 = time.time()
    dummy_audio = np.zeros(16000, dtype=np.float32)
    inputs = processor(dummy_audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features
    
    encoder_hidden_states = ttnn_optimized_whisper_for_conditional_generation.encoder(
        config, input_features, parameters=parameters.encoder, device=device
    )
    
    decoder_input_ids = torch.tensor([[config.decoder_start_token_id]])
    _ = ttnn_optimized_whisper_for_conditional_generation.whisper(
        config,
        encoder_hidden_states,
        decoder_input_ids,
        parameters=parameters,
        device=device,
    )
    print(f"  Warmup complete in {time.time() - t0:.1f}s")

    print("\n[4/4] Starting ZMQ server...")
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://*:{args.port}")
    print(f"  Listening on tcp://*:{args.port}")
    print("\n" + "=" * 60)
    print("READY - Waiting for requests...")
    print("=" * 60 + "\n")

    def transcribe(audio_data: np.ndarray, sample_rate: int) -> tuple[str, float]:
        t0 = time.time()
        
        inputs = processor(audio_data, sampling_rate=sample_rate, return_tensors="pt")
        input_features = inputs.input_features
        
        encoder_hidden_states = ttnn_optimized_whisper_for_conditional_generation.encoder(
            config, input_features, parameters=parameters.encoder, device=device
        )
        
        generated_ids = [config.decoder_start_token_id]
        for _ in range(448):
            decoder_input_ids = torch.tensor([generated_ids])
            logits = ttnn_optimized_whisper_for_conditional_generation.whisper(
                config,
                encoder_hidden_states,
                decoder_input_ids,
                parameters=parameters,
                device=device,
            )
            next_token = logits[0, -1].argmax().item()
            if next_token == config.eos_token_id:
                break
            generated_ids.append(next_token)
        
        text = processor.decode(generated_ids, skip_special_tokens=True)
        elapsed_ms = (time.time() - t0) * 1000
        return text, elapsed_ms

    while True:
        try:
            message = socket.recv_json()
            
            if message.get("cmd") == "ping":
                socket.send_json({"status": "ok", "model": "whisper"})
                continue
            
            if message.get("cmd") == "shutdown":
                socket.send_json({"status": "shutting_down"})
                break
            
            audio_b64 = message.get("audio")
            sample_rate = message.get("sample_rate", 16000)
            
            if not audio_b64:
                socket.send_json({"error": "missing 'audio' field"})
                continue
            
            audio_bytes = base64.b64decode(audio_b64)
            audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
            
            text, elapsed_ms = transcribe(audio_data, sample_rate)
            
            print(f"[Transcribed] {text[:50]}... ({elapsed_ms:.1f}ms)")
            socket.send_json({"text": text, "time_ms": elapsed_ms})
            
        except Exception as e:
            print(f"[Error] {e}")
            try:
                socket.send_json({"error": str(e)})
            except:
                pass

    print("\nShutting down...")
    ttnn.close_device(device)
    socket.close()
    context.term()

if __name__ == "__main__":
    main()
