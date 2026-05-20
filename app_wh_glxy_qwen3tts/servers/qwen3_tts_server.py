#!/usr/bin/env python3
"""
Qwen3-TTS TTNN Socket Server.

Persistent server that loads the Qwen3-TTS model once, captures all traces,
then serves synthesis requests over a Unix socket.

Usage:
    python qwen3_tts_server.py --ref-audio /path/to/reference.wav \
        --ref-text "transcript of reference audio" \
        --socket /tmp/tts_server.sock \
        --device-id 0
"""

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

import torch
import soundfile as sf
import numpy as np


class Qwen3TTSServer:
    def __init__(self, device_id=0, ref_audio=None, ref_text=None, use_2cq=True, max_new_tokens=256):
        print("=" * 60)
        print("  Qwen3-TTS TTNN Server")
        print("=" * 60)

        self.device_id = device_id
        self.use_2cq = use_2cq
        self.max_new_tokens = max_new_tokens

        import ttnn
        from models.demos.qwen3_tts.tt.server import (
            TTSConfig,
            init_server_context,
            load_weights,
            encode_reference_audio,
            create_icl_embedding_ttnn,
            run_inference,
            decode_audio,
        )
        from models.demos.qwen3_tts.tt.qwen3_tts import Qwen3TTS
        from models.demos.qwen3_tts.reference.functional import (
            SpeechTokenizerDecoderConfig,
            speech_tokenizer_decoder_forward,
        )
        from models.demos.qwen3_tts.tt.generator import StreamingAudioDecoder
        from transformers import AutoTokenizer

        self.ttnn = ttnn
        self.run_inference = run_inference
        self.create_icl_embedding_ttnn = create_icl_embedding_ttnn
        self.decode_audio = decode_audio
        self.encode_reference_audio = encode_reference_audio
        self.StreamingAudioDecoder = StreamingAudioDecoder
        self.speech_tokenizer_decoder_forward = speech_tokenizer_decoder_forward
        self._decoder_cfg = SpeechTokenizerDecoderConfig()

        # Load weights
        print("\n[1/6] Loading weights...")
        t0 = time.time()
        self.main_weights, self.decoder_weights = load_weights()
        print(f"  Done in {time.time()-t0:.1f}s")

        # Load tokenizer
        print("[2/6] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", trust_remote_code=True)

        # Open device
        print(f"[3/6] Opening TT device {device_id}...")
        _ncq = 2 if use_2cq else 1
        self.device = ttnn.open_device(
            device_id=device_id,
            l1_small_size=32768,
            trace_region_size=200000000,
            num_command_queues=_ncq,
        )
        self.device.enable_program_cache()

        # Initialize model
        print("[4/6] Initializing TTNN model...")
        t0 = time.time()
        self.model = Qwen3TTS(device=self.device, state_dict=self.main_weights)
        print(f"  Done in {time.time()-t0:.1f}s")

        # Config
        self.config = TTSConfig()
        self.config.max_new_tokens = max_new_tokens

        # Encode reference audio
        print("[5/6] Encoding reference audio...")
        if ref_audio is None:
            ref_audio = os.path.join(os.path.dirname(__file__), 
                "../../tt-metal/models/demos/qwen3_tts/demo/jim_reference.wav")
        self.ref_audio_path = ref_audio
        self.ref_text = ref_text

        ref_cache = str(Path(ref_audio).with_suffix("")) + ".refcache.pt"
        self.ref_codes, self.audio_data = encode_reference_audio(
            ref_audio, self.main_weights, cache_path=ref_cache
        )
        self.speaker_embedding = self.model.extract_speaker_embedding(self.audio_data)
        print(f"  Speaker embedding: {self.speaker_embedding.shape}")

        # Init server context (captures all traces)
        print("[6/6] Capturing traces (this takes a while)...")
        t0 = time.time()
        self.ctx = init_server_context(self.device, self.model, self.config, self.main_weights)
        print(f"  Traces captured in {time.time()-t0:.1f}s")

        self._request_count = 0
        self._successful = 0
        self._failed = 0

        print("\n  Warmup complete!")

    def synthesize(self, text: str, output_path: str, language: str = "english") -> dict:
        """Synthesize speech for given text. Returns dict with status, audio_path, time_ms."""
        t0 = time.time()
        self._request_count += 1

        try:
            # Build ICL embedding for this text
            inputs_embeds_tt, trailing_text_hidden, tts_pad_embed, code_pred_embeds = \
                self.create_icl_embedding_ttnn(
                    target_text=text,
                    ref_text=self.ref_text,
                    ref_codes=self.ref_codes,
                    speaker_embedding=self.speaker_embedding,
                    tokenizer=self.tokenizer,
                    model=self.model,
                    device=self.device,
                    config=self.config,
                    main_weights=self.main_weights,
                    language=language,
                )

            # Set up streaming decoder (decodes audio in parallel with inference)
            def _streaming_decoder_fn(codes_input):
                codes_filtered = codes_input.clone().clamp(max=2047)
                return self.speech_tokenizer_decoder_forward(
                    codes_filtered, self.decoder_weights, self._decoder_cfg
                )

            streaming_decoder = self.StreamingAudioDecoder(
                _streaming_decoder_fn, chunk_size=50, sample_rate=24000
            )
            streaming_decoder.start()

            # Feed reference codes first (will be trimmed from final audio)
            for _ref_frame in self.ref_codes:
                streaming_decoder.add_tokens(_ref_frame.long())

            # Run inference with streaming decoder (audio decodes in parallel!)
            codes, timings, _perf_text = self.run_inference(
                ctx=self.ctx,
                model=self.model,
                device=self.device,
                inputs_embeds_tt=inputs_embeds_tt,
                trailing_text_hidden=trailing_text_hidden,
                tts_pad_embed=tts_pad_embed,
                config=self.config,
                use_2cq=self.use_2cq,
                streaming_decoder=streaming_decoder,
            )

            if codes is None:
                streaming_decoder.stop()
                self._failed += 1
                return {"status": "error", "error": "Failed to generate codes"}

            # Drain remaining audio from streaming decoder
            _drain_t0 = time.time()
            while not streaming_decoder.token_queue.empty() and time.time() - _drain_t0 < 10.0:
                time.sleep(0.01)
            audio = streaming_decoder.get_all_audio()
            streaming_decoder.stop()

            # Trim reference portion using exact codec frame rate (12 frames/sec → 2000 samples/frame)
            ref_codes_len = self.ref_codes.shape[0]
            audio_np = audio.squeeze().detach().cpu().float().numpy()
            samples_per_frame = 24000 // 12  # 2000 samples per codec frame
            cut_samples = ref_codes_len * samples_per_frame
            if cut_samples < len(audio_np):
                audio_np = audio_np[cut_samples:]
            else:
                total_codes_len = ref_codes_len + len(codes)
                cut_samples = int(ref_codes_len / total_codes_len * len(audio_np))
                audio_np = audio_np[cut_samples:]

            # Save
            sf.write(output_path, audio_np, 24000)

            elapsed_ms = (time.time() - t0) * 1000
            self._successful += 1
            print(f"  Done in {elapsed_ms:.0f}ms ({len(codes)} frames, {len(audio_np)/24000:.2f}s audio)")

            return {
                "status": "ok",
                "audio_path": output_path,
                "time_ms": elapsed_ms,
                "num_frames": len(codes),
                "audio_duration_s": len(audio_np) / 24000,
            }

        except Exception as e:
            self._failed += 1
            import traceback
            traceback.print_exc()
            return {"status": "error", "error": str(e)}

    def serve(self, socket_path: str):
        """Start Unix socket server."""
        if os.path.exists(socket_path):
            os.unlink(socket_path)

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(socket_path)
        server.listen(1)

        print(f"\n{'='*60}")
        print(f"READY - Listening on {socket_path}")
        print(f"{'='*60}\n")

        while True:
            conn, _ = server.accept()
            try:
                data = conn.recv(65536).decode()
                if not data:
                    continue

                req = json.loads(data)

                if req.get("cmd") == "ping":
                    conn.sendall(json.dumps({
                        "status": "ok",
                        "model": "qwen3-tts",
                        "requests_total": self._request_count,
                        "requests_successful": self._successful,
                    }).encode())
                    continue

                if req.get("cmd") == "shutdown":
                    conn.sendall(json.dumps({"status": "bye"}).encode())
                    break

                # Synthesis request
                text = req.get("text", "")
                output_path = req.get("output_path", f"/tmp/qwen3_tts_{int(time.time()*1000)}.wav")
                language = req.get("language", "english")

                if not text.strip():
                    conn.sendall(json.dumps({"status": "error", "error": "empty text"}).encode())
                    continue

                print(f"  🔊 Synthesizing: {text[:60]}...")
                result = self.synthesize(text, output_path, language)
                if result["status"] == "ok":
                    print(f"  ✅ Done in {result['time_ms']:.0f}ms ({result['num_frames']} frames)")
                else:
                    print(f"  ❌ Error: {result.get('error', 'unknown')}")

                conn.sendall(json.dumps(result).encode())

            except json.JSONDecodeError as e:
                print(f"  ⚠️ JSON error: {e}")
                try:
                    conn.sendall(json.dumps({"status": "error", "error": f"JSON parse error: {e}"}).encode())
                except:
                    pass
            except Exception as e:
                print(f"  ⚠️ Error: {e}")
                try:
                    conn.sendall(json.dumps({"status": "error", "error": str(e)}).encode())
                except:
                    pass
            finally:
                conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-TTS TTNN Socket Server")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--ref-audio", type=str, required=True, help="Path to reference audio WAV")
    parser.add_argument("--ref-text", type=str, required=True, help="Transcript of reference audio")
    parser.add_argument("--socket", type=str, default="/tmp/tts_server.sock")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--no-2cq", action="store_true", help="Disable dual command queues")
    args = parser.parse_args()

    server = Qwen3TTSServer(
        device_id=args.device_id,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        use_2cq=not args.no_2cq,
        max_new_tokens=args.max_tokens,
    )
    server.serve(args.socket)
