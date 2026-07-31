#!/usr/bin/env python3
"""
TTS Server - Qwen3 TTS on Tenstorrent hardware.

Key insight: The model echoes part of the reference audio before speaking the target.
We trim the output based on reference length used to remove this echo.
"""

import os
import sys
import json
import time
import socket
import argparse

sys.path.insert(0, "/home/container_app_user/tt-metal")


class TTSServer:
    def __init__(self, device_id=0):
        import torch
        import ttnn
        from models.demos.qwen3_tts.tt.qwen3_tts import Qwen3TTS
        from models.demos.qwen3_tts.demo.demo_full_ttnn_tts import (
            load_weights,
            encode_reference_audio,
            TTSConfig,
        )
        from transformers import AutoTokenizer

        print("=" * 60)
        print("TTS Server - Qwen3 TTS")
        print("=" * 60)

        self.ttnn = ttnn
        self.torch = torch

        # Open device with trace region (like Whisper server)
        print("\n[1/5] Opening device...")
        self.device = ttnn.open_device(
            device_id=device_id,
            l1_small_size=32768,
            trace_region_size=100_000_000,  # 100MB for traces
        )
        self.device.enable_program_cache()

        # Load weights
        print("[2/5] Loading weights...")
        self.main_weights, self.decoder_weights = load_weights()
        self.main_weights = {k: v.float() for k, v in self.main_weights.items()}

        # Initialize model
        print("[3/5] Initializing model...")
        self.model = Qwen3TTS(device=self.device, state_dict=self.main_weights)

        # Load tokenizer
        print("[4/5] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base", trust_remote_code=True
        )

        # Load reference audio
        print("[5/5] Loading reference audio...")
        ref_path = "/home/container_app_user/tt-metal/models/demos/qwen3_tts/demo/jim_reference.wav"
        self.ref_text = "Let me also go over the review slides."
        self.ref_codes, self.ref_audio = encode_reference_audio(ref_path, self.main_weights)
        
        # Extract speaker embedding from FULL reference
        self.speaker_embedding = self.model.extract_speaker_embedding(self.ref_audio)
        print(f"  Reference: {self.ref_codes.shape[0]} frames ({self.ref_codes.shape[0]/12.5:.1f}s)")

        # TTS config
        self.config = TTSConfig(
            max_new_tokens=300,
            trim_codec_frames=0,
            greedy=False,
            temperature=0.7,
            top_k=50,
            repetition_penalty=1.15,
        )

        # Request counter for debugging
        self.request_count = 0

        # Warmup
        print("\n[Warmup] Running warmup...")
        self._synthesize("Hello world", "/tmp/warmup.wav")
        print("  Warmup complete!")

    def _synthesize(self, text: str, output_path: str):
        """Synthesize speech from text."""
        import soundfile as sf
        from models.demos.qwen3_tts.demo.demo_full_ttnn_tts import (
            create_icl_embedding_ttnn,
            generate_codes_ttnn,
            decode_audio,
        )

        self.request_count += 1
        print(f"  [Request #{self.request_count}]")

        # Calculate reference length
        ref_tokens = len(self.tokenizer.encode(self.ref_text, add_special_tokens=False))
        tgt_tokens = len(self.tokenizer.encode(text, add_special_tokens=False))
        text_len = ref_tokens + tgt_tokens + 1
        
        max_ref_for_icl = text_len - 2
        MIN_REF_FRAMES = 12
        orig_ref_frames = self.ref_codes.shape[0]
        
        ref_frames_to_use = max(MIN_REF_FRAMES, min(orig_ref_frames, max_ref_for_icl))
        ref_codes = self.ref_codes[:ref_frames_to_use]
        
        print(f"  Text: {len(text.split())} words, ref: {ref_frames_to_use} frames")

        # Create embeddings
        inputs_embeds, trailing_text, tts_pad, code_embeds = create_icl_embedding_ttnn(
            target_text=text,
            ref_text=self.ref_text,
            ref_codes=ref_codes,
            speaker_embedding=self.speaker_embedding,
            tokenizer=self.tokenizer,
            model=self.model,
            device=self.device,
            config=self.config,
            main_weights=self.main_weights,
            language="english",
        )

        # Set max tokens based on text length
        # ~12.5 frames/sec speech, ~2.5 words/sec = ~5 frames/word, add buffer
        word_count = len(text.split())
        self.config.max_new_tokens = min(300, max(60, word_count * 8 + 40))
        
        # Stronger repetition penalty for longer texts to prevent loops
        if word_count > 15:
            self.config.repetition_penalty = 1.3
        else:
            self.config.repetition_penalty = 1.15

        # Generate codes
        result = generate_codes_ttnn(
            model=self.model,
            device=self.device,
            inputs_embeds_tt=inputs_embeds,
            trailing_text_hidden=trailing_text,
            tts_pad_embed=tts_pad,
            code_pred_embeds=code_embeds,
            config=self.config,
            use_kv_cache=True,
            use_trace=False,
        )

        if result is None or result[0] is None:
            raise RuntimeError("Generation failed")

        codes = result[0]
        print(f"  Generated: {len(codes)} frames")

        # Trim reference echo - proportional to reference used + buffer
        trim_frames = ref_frames_to_use + 10
        
        if len(codes) > trim_frames + 10:
            codes = codes[trim_frames:]
            print(f"  Trimmed: {trim_frames} frames, remaining: {len(codes)}")

        # Decode to audio
        audio = decode_audio(codes, self.decoder_weights)
        audio_np = audio.squeeze().cpu().float().numpy()
        sf.write(output_path, audio_np, 24000)
        print(f"  Saved: {len(audio_np)/24000:.2f}s")

        # Sync device after each request
        self.ttnn.synchronize_device(self.device)

    def serve(self, socket_path: str):
        """Start the server."""
        if os.path.exists(socket_path):
            os.unlink(socket_path)

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(socket_path)
        server.listen(1)
        os.chmod(socket_path, 0o777)

        print(f"\n{'='*60}")
        print("READY - Listening on", socket_path)
        print("=" * 60 + "\n")

        try:
            while True:
                conn, _ = server.accept()
                try:
                    data = conn.recv(65536).decode()
                    if not data:
                        continue

                    req = json.loads(data)

                    if req.get("cmd") == "ping":
                        conn.sendall(json.dumps({"status": "ok"}).encode())
                        continue

                    if req.get("cmd") == "shutdown":
                        conn.sendall(json.dumps({"status": "bye"}).encode())
                        break

                    text = req.get("text", "")
                    output = req.get("output_path", "/tmp/tts_out.wav")

                    if not text:
                        conn.sendall(json.dumps({"status": "error", "error": "no text"}).encode())
                        continue

                    print(f"\n[TTS] '{text[:50]}...'")
                    t0 = time.time()

                    try:
                        self._synthesize(text, output)
                        ms = (time.time() - t0) * 1000
                        print(f"  Total: {ms:.0f}ms")
                        conn.sendall(json.dumps({
                            "status": "ok",
                            "audio_path": output,
                            "time_ms": ms
                        }).encode())
                    except Exception as e:
                        import traceback
                        print(f"  Error: {e}")
                        traceback.print_exc()
                        conn.sendall(json.dumps({"status": "error", "error": str(e)}).encode())

                except Exception as e:
                    import traceback
                    print(f"Error: {e}")
                    traceback.print_exc()
                finally:
                    conn.close()

        finally:
            server.close()
            if os.path.exists(socket_path):
                os.unlink(socket_path)
            self.ttnn.close_device(self.device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", default="/tmp/tts_server.sock")
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args()

    server = TTSServer(device_id=args.device_id)
    server.serve(args.socket)


if __name__ == "__main__":
    main()
