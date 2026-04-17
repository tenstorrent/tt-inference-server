#!/usr/bin/env python3
"""
SpeechT5 TTS Server - TTNN accelerated version.

Runs SpeechT5 on Tenstorrent hardware for high-performance TTS.
Based on models/experimental/speecht5_tts/demo_ttnn.py
"""

import os
import sys
import json
import time
import socket
import argparse

sys.path.insert(0, "/home/container_app_user/tt-metal")


class SpeechT5TTNNServer:
    def __init__(self, device_id=0, speaker_id=0, max_steps=800):
        import torch
        import ttnn
        import soundfile as sf
        from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
        from datasets import load_dataset

        from models.experimental.speecht5_tts.tt.ttnn_speecht5_encoder import (
            TTNNSpeechT5Encoder,
            TTNNEncoderConfig,
            preprocess_encoder_parameters,
        )
        from models.experimental.speecht5_tts.tt.ttnn_speecht5_decoder import (
            TTNNSpeechT5Decoder,
            TTNNDecoderConfig,
            preprocess_decoder_parameters,
            init_kv_cache,
        )
        from models.experimental.speecht5_tts.tt.ttnn_speecht5_postnet import (
            TTNNSpeechT5SpeechDecoderPostnet,
            TTNNPostNetConfig,
            preprocess_postnet_parameters,
        )
        from models.experimental.speecht5_tts.tt.ttnn_speecht5_generator import SpeechT5Generator

        print("=" * 60)
        print("SpeechT5 TTS Server - TTNN Accelerated")
        print("=" * 60)

        self.torch = torch
        self.ttnn = ttnn
        self.sf = sf
        self.max_steps = max_steps

        # Open device
        print("\n[1/6] Opening device...")
        self.device = ttnn.open_device(
            device_id=device_id,
            l1_small_size=300000,
            trace_region_size=100000000,  # 100MB like Whisper (was 20MB - caused trace overflow)
        )
        self.device.enable_program_cache()

        # Load HuggingFace models
        print("[2/6] Loading HuggingFace models...")
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        hf_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

        # Load speaker embeddings
        print("[3/6] Loading speaker embeddings...")
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        self.speaker_embeddings = torch.tensor(embeddings_dataset[speaker_id]["xvector"]).unsqueeze(0)
        
        # Pre-load speakers for podcast multi-voice support
        self.PODCAST_SPEAKERS = {"host": 7306, "guest": 1138}
        self.speaker_cache = {speaker_id: self.speaker_embeddings}
        for role, sid in self.PODCAST_SPEAKERS.items():
            self.speaker_cache[sid] = torch.tensor(embeddings_dataset[sid]["xvector"]).unsqueeze(0)
            print(f"  Loaded speaker {sid} ({role})")

        # Create TTNN models
        print("[4/6] Creating TTNN encoder...")
        encoder_config = TTNNEncoderConfig()
        encoder_params = preprocess_encoder_parameters(
            hf_model.speecht5.encoder, encoder_config, self.device
        )
        self.ttnn_encoder = TTNNSpeechT5Encoder(self.device, encoder_params, encoder_config)

        print("[5/6] Creating TTNN decoder and postnet...")
        decoder_config = TTNNDecoderConfig()
        decoder_params = preprocess_decoder_parameters(
            hf_model.speecht5.decoder, decoder_config, self.device, self.speaker_embeddings
        )
        self.ttnn_decoder = TTNNSpeechT5Decoder(
            self.device, decoder_params, decoder_config, max_sequence_length=max_steps
        )
        self.decoder_config = decoder_config

        postnet_config = TTNNPostNetConfig()
        postnet_params = preprocess_postnet_parameters(
            hf_model.speech_decoder_postnet, postnet_config, self.device
        )
        self.ttnn_postnet = TTNNSpeechT5SpeechDecoderPostnet(self.device, postnet_params, postnet_config)

        # CRITICAL: Pre-compile postnet BEFORE any trace capture
        # This prevents conv2d kernel recompilation while trace is active (causes hangs)
        print("  Pre-compiling postnet kernels...")
        dummy_decoder_output = ttnn.from_torch(
            torch.randn(1, 1, 1, decoder_config.hidden_size),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        _ = self.ttnn_postnet(dummy_decoder_output)
        ttnn.deallocate(dummy_decoder_output)
        print("  Postnet kernels compiled!")

        # Create generator for trace support
        print(f"[6/6] Creating generator with max_steps={max_steps}...")
        self.generator = SpeechT5Generator(
            encoder=self.ttnn_encoder,
            decoder=self.ttnn_decoder,
            postnet=self.ttnn_postnet,
            device=self.device,
            decoder_config=decoder_config,
            max_steps=max_steps,
        )

        # Warmup - run with warmup_mode=True to compile kernels without vocoder
        print("  Running warmup (compiling TTNN kernels)...")
        self._warmup("Hello world")
        
        # Capture traces for all encoder sizes
        print("  Capturing traces for all encoder sizes...")
        self.generator.capture_all_traces(self.processor, batch_size=1)
        
        # CRITICAL: Reset KV caches after warmup to prevent stale values from corrupting inference
        print("  Resetting KV caches...")
        self.generator._reset_kv_caches()
        
        # Do a full warmup run (with vocoder) to compile all paths
        print("  Running full warmup (with vocoder)...")
        self._synthesize("Hello, welcome!", "/tmp/warmup_full.wav")
        
        # Reset KV caches again after full warmup
        self.generator._reset_kv_caches()
        
        print("  Warmup complete!")

    def _warmup(self, text: str):
        """Run warmup inference to compile kernels."""
        from models.experimental.speecht5_tts.demo_ttnn import generate_speech_ttnn

        generate_speech_ttnn(
            text=text,
            speaker_embeddings=self.speaker_embeddings,
            processor=self.processor,
            vocoder=self.vocoder,
            ttnn_encoder=self.ttnn_encoder,
            ttnn_decoder=self.ttnn_decoder,
            ttnn_postnet=self.ttnn_postnet,
            device=self.device,
            max_steps=50,  # Shorter for warmup
            return_stats=False,
            warmup_mode=True,  # Skip vocoder during warmup
            generator=self.generator,
            use_kv_cache=True,
            decoder_config=self.decoder_config,
        )

    def _synthesize_segment_with_speaker(self, text: str, speaker_id: int):
        """Synthesize text with a specific speaker using CPU path (for podcast multi-speaker).
        
        TTNN decoder is compiled with single speaker, so we use HuggingFace CPU inference
        for multi-speaker podcast segments.
        """
        from transformers import SpeechT5ForTextToSpeech
        
        # Lazy load CPU model for podcast
        if not hasattr(self, '_cpu_model'):
            print("  Loading CPU model for multi-speaker podcast...")
            self._cpu_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
            self._cpu_model.eval()
        
        speaker_emb = self.speaker_cache.get(speaker_id, self.speaker_embeddings)
        
        inputs = self.processor(text=text, return_tensors="pt")
        with self.torch.no_grad():
            speech = self._cpu_model.generate_speech(inputs["input_ids"], speaker_emb, vocoder=self.vocoder)
        
        return speech, {}

    def _split_into_sentences(self, text: str) -> list:
        """Split text into sentences for processing."""
        import re
        # Split on sentence boundaries (. ! ?) but keep the punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        # Filter out empty sentences
        return [s.strip() for s in sentences if s.strip()]

    def _synthesize_segment(self, text: str):
        """Synthesize a single text segment, returning audio tensor."""
        from models.experimental.speecht5_tts.demo_ttnn import generate_speech_ttnn

        # Auto-calculate max_steps based on text length
        # SpeechT5 needs ~2-3 steps per character for reliable full generation
        estimated_steps = max(100, min(len(text) * 3, self.max_steps))

        speech, stats = generate_speech_ttnn(
            text=text,
            speaker_embeddings=self.speaker_embeddings,
            processor=self.processor,
            vocoder=self.vocoder,
            ttnn_encoder=self.ttnn_encoder,
            ttnn_decoder=self.ttnn_decoder,
            ttnn_postnet=self.ttnn_postnet,
            device=self.device,
            max_steps=estimated_steps,
            return_stats=True,
            generator=self.generator,
            use_kv_cache=True,
            decoder_config=self.decoder_config,
        )
        
        return speech, stats

    def _synthesize(self, text: str, output_path: str):
        """Synthesize speech from text, splitting long text into sentences."""
        import numpy as np
        import gc
        
        # For short text, process directly
        if len(text) * 3 <= self.max_steps:
            speech, stats = self._synthesize_segment(text)
            audio_np = speech.squeeze().detach().numpy()
            # Explicitly delete speech tensor to free memory
            del speech
        else:
            # Split into sentences and process each
            sentences = self._split_into_sentences(text)
            audio_chunks = []
            total_stats = {'token_per_sec': 0, 'count': 0}
            
            for i, sentence in enumerate(sentences):
                if not sentence:
                    continue
                speech, stats = self._synthesize_segment(sentence)
                audio_chunks.append(speech.squeeze().detach().numpy())
                total_stats['token_per_sec'] += stats.get('token_per_sec', 0)
                total_stats['count'] += 1
                # Explicitly delete speech tensor to free memory
                del speech
                # Sync device after every 3 segments to prevent trace accumulation
                if (i + 1) % 3 == 0:
                    self.ttnn.synchronize_device(self.device)
                    gc.collect()
            
            # Concatenate all audio chunks with small pause between sentences
            pause = np.zeros(int(16000 * 0.15), dtype=np.float32)  # 150ms pause
            audio_parts = []
            for i, chunk in enumerate(audio_chunks):
                audio_parts.append(chunk)
                if i < len(audio_chunks) - 1:
                    audio_parts.append(pause)
            
            audio_np = np.concatenate(audio_parts) if audio_parts else np.zeros(1600)
            stats = {'token_per_sec': total_stats['token_per_sec'] / max(1, total_stats['count'])}
            
            # Clear audio chunks to free memory
            del audio_chunks
            del audio_parts

        # Add 200ms silence at end
        silence = np.zeros(int(16000 * 0.2), dtype=audio_np.dtype)
        audio_np = np.concatenate([audio_np, silence])
        self.sf.write(output_path, audio_np, samplerate=16000)
        
        # CRITICAL: Reset KV caches after each synthesis to ensure clean state for next request
        self.generator._reset_kv_caches()
        
        # Synchronize device to ensure all operations are complete
        self.ttnn.synchronize_device(self.device)
        
        # Force garbage collection to free any lingering memory
        gc.collect()
        
        return stats

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

        # Request tracking for debugging
        request_count = 0
        successful_requests = 0
        failed_requests = 0

        try:
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
                            "requests_total": request_count,
                            "requests_successful": successful_requests,
                            "requests_failed": failed_requests
                        }).encode())
                        continue

                    if req.get("cmd") == "status":
                        # Return detailed status
                        conn.sendall(json.dumps({
                            "status": "ok",
                            "requests_total": request_count,
                            "requests_successful": successful_requests,
                            "requests_failed": failed_requests,
                            "max_steps": self.max_steps
                        }).encode())
                        continue

                    if req.get("cmd") == "shutdown":
                        conn.sendall(json.dumps({"status": "bye"}).encode())
                        break

                    # Handle podcast multi-speaker synthesis
                    if req.get("cmd") == "podcast":
                        segments = req.get("segments", [])
                        output_path = req.get("output_path", "/tmp/podcast_output.wav")
                        
                        if not segments:
                            conn.sendall(json.dumps({"status": "error", "error": "no segments provided"}).encode())
                            continue
                        
                        print(f"\n[Podcast] Generating {len(segments)} segments...")
                        t0 = time.time()
                        request_count += 1
                        
                        try:
                            import numpy as np
                            all_speech = []
                            silence_gap = np.zeros(int(16000 * 0.3), dtype=np.float32)  # 0.3s gap between speakers
                            
                            for i, seg in enumerate(segments):
                                seg_text = seg.get("text", "").strip()
                                seg_role = seg.get("role", "host").lower()
                                if not seg_text:
                                    continue
                                
                                # Get speaker for this role
                                sid = self.PODCAST_SPEAKERS.get(seg_role, self.PODCAST_SPEAKERS["guest"])
                                
                                print(f"  [{i+1}/{len(segments)}] {seg_role.upper()}: {seg_text[:40]}...")
                                
                                # Synthesize with role's speaker (using CPU fallback for multi-speaker)
                                # TTNN decoder is compiled with single speaker, so use CPU vocoder path
                                speech, stats = self._synthesize_segment_with_speaker(seg_text, sid)
                                all_speech.append(speech.squeeze().detach().numpy() if hasattr(speech, 'detach') else speech)
                                all_speech.append(silence_gap)
                            
                            # Concatenate all segments
                            final_audio = np.concatenate(all_speech) if all_speech else np.zeros(1600)
                            self.sf.write(output_path, final_audio, samplerate=16000)
                            
                            elapsed_ms = (time.time() - t0) * 1000
                            audio_duration = len(final_audio) / 16000
                            successful_requests += 1
                            
                            print(f"  Podcast done in {elapsed_ms:.0f}ms (audio: {audio_duration:.2f}s)")
                            
                            conn.sendall(json.dumps({
                                "status": "ok",
                                "audio_path": output_path,
                                "time_ms": elapsed_ms,
                                "audio_duration": audio_duration,
                            }).encode())
                        except Exception as e:
                            import traceback
                            failed_requests += 1
                            print(f"  Podcast error: {e}")
                            traceback.print_exc()
                            conn.sendall(json.dumps({"status": "error", "error": str(e)}).encode())
                        continue

                    text = req.get("text", "")
                    output = req.get("output_path", "/tmp/tts_out.wav")
                    speaker_id = req.get("speaker_id")  # Optional: for multi-voice

                    if not text:
                        conn.sendall(json.dumps({"status": "error", "error": "no text"}).encode())
                        continue

                    request_count += 1
                    
                    # WORKAROUND: Refresh traces every 5 requests to prevent hang at ~17
                    # The TTNN trace infrastructure accumulates state that causes hangs
                    # Note: Long texts split into segments, so refresh more frequently
                    if request_count > 1 and request_count % 5 == 0:
                        print(f"\n[TTS #{request_count}] Refreshing traces (preventive maintenance)...")
                        try:
                            # Release all traces
                            self.generator.cleanup()
                            # Sync device to ensure cleanup is complete
                            self.ttnn.synchronize_device(self.device)
                            # Reset KV caches
                            self.generator._reset_kv_caches()
                            # Recapture traces for all encoder sizes
                            self.generator.capture_all_traces(self.processor, batch_size=1)
                            # Reset KV caches again after trace capture
                            self.generator._reset_kv_caches()
                            print(f"  Traces refreshed successfully!")
                        except Exception as e:
                            print(f"  Warning: Trace refresh failed: {e}")
                    
                    speaker_label = f" [speaker:{speaker_id}]" if speaker_id else ""
                    print(f"\n[TTS #{request_count}]{speaker_label} '{text[:50]}{'...' if len(text) > 50 else ''}'")
                    t0 = time.time()

                    try:
                        # If speaker_id is provided, use CPU path for multi-speaker
                        if speaker_id is not None and speaker_id in self.speaker_cache:
                            import numpy as np
                            speech, _ = self._synthesize_segment_with_speaker(text, speaker_id)
                            audio_np = speech.squeeze().detach().numpy() if hasattr(speech, 'detach') else speech
                            # Add silence at end
                            silence = np.zeros(int(16000 * 0.2), dtype=audio_np.dtype)
                            audio_np = np.concatenate([audio_np, silence])
                            self.sf.write(output, audio_np, samplerate=16000)
                            stats = {}
                        else:
                            stats = self._synthesize(text, output)
                        ms = (time.time() - t0) * 1000
                        successful_requests += 1
                        print(f"  Generated in {ms:.0f}ms, tokens/sec: {stats.get('token_per_sec', 0):.1f}")
                        print(f"  Stats: {successful_requests}/{request_count} successful")
                        conn.sendall(json.dumps({
                            "status": "ok",
                            "audio_path": output,
                            "time_ms": ms,
                            "request_number": request_count
                        }).encode())
                    except Exception as e:
                        import traceback
                        failed_requests += 1
                        print(f"  Error: {e}")
                        print(f"  Stats: {failed_requests} failures out of {request_count} requests")
                        traceback.print_exc()
                        
                        # Try to recover by resetting KV caches
                        try:
                            print("  Attempting recovery by resetting KV caches...")
                            self.generator._reset_kv_caches()
                            self.ttnn.synchronize_device(self.device)
                            print("  Recovery complete")
                        except Exception as recovery_error:
                            print(f"  Recovery failed: {recovery_error}")
                        
                        conn.sendall(json.dumps({
                            "status": "error", 
                            "error": str(e),
                            "request_number": request_count
                        }).encode())

                except Exception as e:
                    import traceback
                    print(f"Error: {e}")
                    traceback.print_exc()
                finally:
                    conn.close()

        finally:
            print(f"\nServer shutting down. Total requests: {request_count}, "
                  f"Successful: {successful_requests}, Failed: {failed_requests}")
            server.close()
            if os.path.exists(socket_path):
                os.unlink(socket_path)
            self.ttnn.close_device(self.device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", default="/tmp/tts_server.sock")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--speaker-id", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=800)
    args = parser.parse_args()

    server = SpeechT5TTNNServer(
        device_id=args.device_id,
        speaker_id=args.speaker_id,
        max_steps=args.max_steps,
    )
    server.serve(args.socket)


if __name__ == "__main__":
    main()
