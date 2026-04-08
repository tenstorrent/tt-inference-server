`#!/usr/bin/env python3
"""
TTS Server - Persistent Qwen3 TTS model server with proper state management.

Uses the server-mode infrastructure (init_server_context + run_inference) for:
- Pre-captured traces at startup (no per-request trace capture overhead)
- Proper KV cache management between requests
- Repetition penalty to avoid "let let let let" loops

Usage (inside container):
  TT_MESH_GRAPH_DESC_PATH=/home/container_app_user/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto \
  TT_VISIBLE_DEVICES='3' \
  python tts_server.py

Protocol (Unix socket at /tmp/tts_server.sock):
  REQ: {"text": "Hello world", "output_path": "/tmp/output.wav"}
  REP: {"status": "ok", "audio_path": "/tmp/output.wav", "time_ms": 1234.5}
"""

import os
import sys
import json
import time
import socket
import argparse
from pathlib import Path

# Add tt-metal to path
sys.path.insert(0, "/home/container_app_user/tt-metal")


def main():
    parser = argparse.ArgumentParser(description="TTS Server")
    parser.add_argument("--socket", type=str, default="/tmp/tts_server.sock", help="Unix socket path")
    parser.add_argument("--device-id", type=int, default=0, help="Logical device ID (after TT_VISIBLE_DEVICES)")
    args = parser.parse_args()

    print("=" * 60)
    print("TTS Server - Qwen3 TTS (Persistent, Server Mode)")
    print("=" * 60)
    print(f"Socket: {args.socket}")
    print(f"TT_VISIBLE_DEVICES: {os.environ.get('TT_VISIBLE_DEVICES', 'not set')}")

    # Import TT Metal
    import torch
    import numpy as np
    import soundfile as sf
    import ttnn

    # Import TTS components
    from models.demos.qwen3_tts.tt.qwen3_tts import Qwen3TTS
    from models.demos.qwen3_tts.demo.demo_full_ttnn_tts import (
        load_weights,
        encode_reference_audio,
        create_icl_embedding_ttnn,
        decode_audio,
        TTSConfig,
        init_server_context,
        run_inference,
    )
    from models.demos.qwen3_tts.demo.reference_icl_utils import trim_reference_for_icl_conditioning
    from transformers import AutoTokenizer

    # === Load model ===
    print("\n[1/6] Opening TT device...")
    t0 = time.time()
    # Use trace_region_size for server mode (pre-captured traces)
    device = ttnn.open_device(device_id=args.device_id, l1_small_size=32768, trace_region_size=100000000)
    device.enable_program_cache()
    print(f"  Device opened in {time.time() - t0:.1f}s")

    print("\n[2/6] Loading weights...")
    t0 = time.time()
    main_weights, decoder_weights = load_weights()
    # Convert to float for model init
    main_weights = {k: v.float() for k, v in main_weights.items()}
    print(f"  Weights loaded in {time.time() - t0:.1f}s")

    print("\n[3/6] Initializing TTNN model...")
    t0 = time.time()
    model = Qwen3TTS(device=device, state_dict=main_weights)
    print(f"  Model initialized in {time.time() - t0:.1f}s")

    print("\n[4/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", trust_remote_code=True)

    print("\n[5/6] Loading reference audio (Jim's voice)...")
    ref_audio_path = "/home/container_app_user/tt-metal/models/demos/qwen3_tts/demo/jim_reference.wav"
    ref_text = "Let me also go over the review slides."
    ref_codes_orig, ref_audio_data_orig = encode_reference_audio(ref_audio_path, main_weights)
    print(f"  Original reference: {ref_codes_orig.shape[0]} codec frames")

    # Extract speaker embedding once (from original full reference)
    print("  Extracting speaker embedding...")
    speaker_embedding = model.extract_speaker_embedding(ref_audio_data_orig)
    print(f"  Speaker embedding shape: {speaker_embedding.shape}")

    # Config with anti-repetition settings
    config = TTSConfig(
        max_new_tokens=256,
        trim_codec_frames=8,  # Trim reference bleed from output
        greedy=False,  # Use sampling to avoid repetition loops
        temperature=0.7,  # Lower temperature for more focused output
        top_k=30,  # Limit sampling pool
        repetition_penalty=1.3,  # Strong penalty for repeated tokens
    )

    # === Initialize server context (pre-captures all traces) ===
    print("\n[6/6] Initializing server context (pre-capturing traces)...")
    t0 = time.time()
    server_ctx = init_server_context(device, model, config, main_weights)
    print(f"  Server context initialized in {time.time() - t0:.1f}s")

    # Warmup with a real synthesis
    print("\n[Warmup] Running warmup synthesis...")
    t0 = time.time()
    try:
        _synthesize_server_mode(
            server_ctx=server_ctx,
            model=model,
            device=device,
            tokenizer=tokenizer,
            main_weights=main_weights,
            decoder_weights=decoder_weights,
            ref_codes_orig=ref_codes_orig,
            ref_audio_data_orig=ref_audio_data_orig,
            speaker_embedding=speaker_embedding,
            config=config,
            ref_text=ref_text,
            text="Hello from Tenstorrent",
            output_path="/tmp/tts_warmup.wav",
        )
        print(f"  Warmup complete in {time.time() - t0:.1f}s")
    except Exception as e:
        import traceback
        print(f"  Warmup failed (non-fatal): {e}")
        traceback.print_exc()

    # === Start server ===
    print("\n" + "=" * 60)
    print("READY - Starting Unix socket server...")
    print("=" * 60)

    # Remove old socket if exists
    if os.path.exists(args.socket):
        os.unlink(args.socket)

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(args.socket)
    server.listen(1)
    os.chmod(args.socket, 0o777)  # Allow other processes to connect

    print(f"Listening on {args.socket}")
    print("Send JSON: {\"text\": \"...\", \"output_path\": \"...\"}")
    print("Commands: {\"cmd\": \"ping\"}, {\"cmd\": \"shutdown\"}\n")

    try:
        while True:
            conn, addr = server.accept()
            try:
                data = conn.recv(65536).decode('utf-8')
                if not data:
                    continue

                request = json.loads(data)

                # Handle commands
                if request.get("cmd") == "ping":
                    response = {"status": "ok", "model": "tts", "ready": True}
                    conn.sendall(json.dumps(response).encode('utf-8'))
                    continue

                if request.get("cmd") == "shutdown":
                    response = {"status": "shutting_down"}
                    conn.sendall(json.dumps(response).encode('utf-8'))
                    break

                # Handle synthesis request
                text = request.get("text", "")
                output_path = request.get("output_path", "/tmp/tts_output.wav")

                if not text:
                    response = {"status": "error", "error": "missing 'text' field"}
                    conn.sendall(json.dumps(response).encode('utf-8'))
                    continue

                print(f"[TTS] Synthesizing: {text[:50]}...")
                t0 = time.time()

                try:
                    _synthesize_server_mode(
                        server_ctx=server_ctx,
                        model=model,
                        device=device,
                        tokenizer=tokenizer,
                        main_weights=main_weights,
                        decoder_weights=decoder_weights,
                        ref_codes_orig=ref_codes_orig,
                        ref_audio_data_orig=ref_audio_data_orig,
                        speaker_embedding=speaker_embedding,
                        config=config,
                        ref_text=ref_text,
                        text=text,
                        output_path=output_path,
                    )
                    elapsed_ms = (time.time() - t0) * 1000
                    print(f"  Done in {elapsed_ms:.1f}ms")

                    response = {
                        "status": "ok",
                        "audio_path": output_path,
                        "time_ms": elapsed_ms,
                    }
                except Exception as e:
                    import traceback
                    print(f"  Error: {e}")
                    traceback.print_exc()
                    response = {"status": "error", "error": str(e)}

                conn.sendall(json.dumps(response).encode('utf-8'))

            except Exception as e:
                import traceback
                print(f"[Error] {e}")
                traceback.print_exc()
                try:
                    conn.sendall(json.dumps({"status": "error", "error": str(e)}).encode('utf-8'))
                except:
                    pass
            finally:
                conn.close()

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.close()
        if os.path.exists(args.socket):
            os.unlink(args.socket)
        ttnn.close_device(device)
        print("Device closed.")


def _synthesize_server_mode(
    server_ctx,
    model,
    device,
    tokenizer,
    main_weights,
    decoder_weights,
    ref_codes_orig,
    ref_audio_data_orig,
    speaker_embedding,
    config,
    ref_text: str,
    text: str,
    output_path: str,
):
    """Run TTS synthesis using server mode (pre-captured traces, proper KV management)."""
    import torch
    import numpy as np
    import soundfile as sf
    import ttnn

    from models.demos.qwen3_tts.demo.demo_full_ttnn_tts import (
        create_icl_embedding_ttnn,
        decode_audio,
        run_inference,
    )
    from models.demos.qwen3_tts.demo.reference_icl_utils import trim_reference_for_icl_conditioning

    # Trim reference to satisfy ICL constraint: text_lens > codec_lens
    ref_codes, ref_audio_data = trim_reference_for_icl_conditioning(
        ref_codes_orig, ref_audio_data_orig, tokenizer, ref_text, text
    )
    
    # Verify ICL constraint
    ref_ids = tokenizer.encode(ref_text, add_special_tokens=False)
    tgt_ids = tokenizer.encode(text, add_special_tokens=False)
    text_lens = len(ref_ids) + len(tgt_ids) + 1
    codec_lens = 1 + ref_codes.shape[0]
    print(f"  ICL check: text_lens={text_lens} > codec_lens={codec_lens} -> {text_lens > codec_lens}")

    # Estimate max tokens based on text length
    word_count = len(text.split())
    estimated_frames = max(48, word_count * 5 + 20)
    estimated_frames += config.trim_codec_frames
    config.max_new_tokens = min(estimated_frames, 256)
    print(f"  Text: {word_count} words -> max_new_tokens={config.max_new_tokens}")

    # Create ICL embeddings
    inputs_embeds_tt, trailing_text_hidden, tts_pad_embed, _ = create_icl_embedding_ttnn(
        target_text=text,
        ref_text=ref_text,
        ref_codes=ref_codes,
        speaker_embedding=speaker_embedding,
        tokenizer=tokenizer,
        model=model,
        device=device,
        config=config,
        main_weights=main_weights,
        language="english",
    )

    # Generate codes using server mode (uses pre-captured traces)
    codes, timings, perf_text = run_inference(
        ctx=server_ctx,
        model=model,
        device=device,
        inputs_embeds_tt=inputs_embeds_tt,
        trailing_text_hidden=trailing_text_hidden,
        tts_pad_embed=tts_pad_embed,
        config=config,
    )

    if codes is None:
        raise RuntimeError("Failed to generate codes")

    # Trim reference echo
    if config.trim_codec_frames > 0 and len(codes) > config.trim_codec_frames:
        codes = codes[config.trim_codec_frames:]

    # Decode to audio
    audio = decode_audio(codes, decoder_weights)
    audio_np = audio.squeeze().detach().cpu().float().numpy()

    # Save
    sf.write(output_path, audio_np, 24000)


if __name__ == "__main__":
    main()
