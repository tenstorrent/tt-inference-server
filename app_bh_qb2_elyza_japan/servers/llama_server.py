#!/usr/bin/env python3
"""
Llama Server - Persistent Llama 3.1 8B Instruct server.
Runs on TT Device 1 (P150), listens on Unix socket.

Usage:
    TT_MESH_GRAPH_DESC_PATH=/home/container_app_user/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
    TT_VISIBLE_DEVICES='1' \
    python servers/llama_server.py

Socket: /tmp/llama_server.sock
"""

import os
import sys
import json
import socket
import time
import logging
import argparse
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LlamaServer")

SOCKET_PATH = "/tmp/llama_server.sock"


def load_llama_model():
    """Load Llama 3.1 8B Instruct model on TT Metal."""
    logger.info("Loading Llama 3.1 8B Instruct...")
    
    import ttnn
    from models.tt_transformers.llama3.tt.llama_model import TtLlamaModel
    from models.tt_transformers.llama3.tt.llama_common import (
        get_model_config,
        get_tokenizer,
    )
    
    # Model config
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    
    # Open device
    logger.info("Opening TT device...")
    device = ttnn.open_device(device_id=0)  # Will be remapped by TT_VISIBLE_DEVICES
    device.enable_program_cache()
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = get_tokenizer(model_name)
    
    # Get model config
    logger.info("Getting model config...")
    model_config = get_model_config(model_name, device)
    
    # Load model
    logger.info("Loading model weights...")
    start = time.time()
    model = TtLlamaModel(device, model_config)
    load_time = time.time() - start
    logger.info(f"Model loaded in {load_time:.1f}s")
    
    # Warmup
    logger.info("Running warmup inference...")
    warmup_start = time.time()
    _ = generate_text(model, tokenizer, device, "Hello", max_tokens=10)
    warmup_time = time.time() - warmup_start
    logger.info(f"Warmup completed in {warmup_time:.1f}s")
    
    return model, tokenizer, device


def generate_text(model, tokenizer, device, prompt: str, max_tokens: int = 150) -> str:
    """Generate text from prompt."""
    import torch
    
    # Format as chat
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Keep responses concise."},
        {"role": "user", "content": prompt}
    ]
    
    # Apply chat template
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Tokenize
    input_ids = tokenizer.encode(formatted, return_tensors="pt")
    
    # Generate
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    
    # Decode
    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    
    return response.strip()


def handle_client(conn, model, tokenizer, device):
    """Handle a client connection."""
    try:
        data = conn.recv(65536).decode('utf-8')
        if not data:
            return
        
        request = json.loads(data)
        
        # Handle ping
        if request.get("cmd") == "ping":
            response = {"status": "ok", "model": "Llama-3.1-8B-Instruct"}
            conn.sendall(json.dumps(response).encode('utf-8'))
            return
        
        # Handle generation
        prompt = request.get("prompt", "")
        max_tokens = request.get("max_tokens", 150)
        
        if not prompt:
            response = {"status": "error", "error": "No prompt provided"}
            conn.sendall(json.dumps(response).encode('utf-8'))
            return
        
        logger.info(f"Generating for: {prompt[:50]}... (max_tokens={max_tokens})")
        
        start = time.time()
        text = generate_text(model, tokenizer, device, prompt, max_tokens)
        elapsed = (time.time() - start) * 1000
        
        logger.info(f"Generated {len(text)} chars in {elapsed:.1f}ms")
        
        response = {
            "status": "ok",
            "text": text,
            "time_ms": elapsed
        }
        conn.sendall(json.dumps(response).encode('utf-8'))
        
    except Exception as e:
        logger.error(f"Error handling request: {e}")
        try:
            response = {"status": "error", "error": str(e)}
            conn.sendall(json.dumps(response).encode('utf-8'))
        except:
            pass


def main():
    parser = argparse.ArgumentParser(description="Llama Server")
    parser.add_argument("--socket", default=SOCKET_PATH, help="Socket path")
    args = parser.parse_args()
    
    # Remove old socket
    if os.path.exists(args.socket):
        os.unlink(args.socket)
    
    # Load model
    logger.info("=" * 60)
    logger.info("Starting Llama Server")
    logger.info("=" * 60)
    
    model, tokenizer, device = load_llama_model()
    
    # Create socket
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(args.socket)
    server.listen(5)
    os.chmod(args.socket, 0o777)
    
    logger.info(f"✅ Llama Server ready on {args.socket}")
    logger.info("Waiting for connections...")
    
    try:
        while True:
            conn, _ = server.accept()
            try:
                handle_client(conn, model, tokenizer, device)
            finally:
                conn.close()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        server.close()
        if os.path.exists(args.socket):
            os.unlink(args.socket)


if __name__ == "__main__":
    main()
