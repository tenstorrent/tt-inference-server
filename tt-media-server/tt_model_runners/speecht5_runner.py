# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import base64
import io
import os
import re
from typing import Any, AsyncGenerator, Dict, List

import soundfile as sf
import torch
import ttnn
from config.constants import SupportedModels
from config.settings import settings
from domain.text_to_speech_request import TextToSpeechRequest
from domain.text_to_speech_response import (
    TextToSpeechResponse,
)
from models.experimental.speecht5_tts.tt.ttnn_speecht5_decoder import (
    TTNNDecoderConfig,
    TTNNSpeechT5Decoder,
    init_kv_cache,
    preprocess_decoder_parameters,
)
from models.experimental.speecht5_tts.tt.ttnn_speecht5_encoder import (
    TTNNEncoderConfig,
    TTNNSpeechT5Encoder,
    preprocess_encoder_parameters,
)
from models.experimental.speecht5_tts.tt.ttnn_speecht5_generator import (
    SpeechT5Generator,
    get_padded_encoder_seq_len,
)
from models.experimental.speecht5_tts.tt.ttnn_speecht5_postnet import (
    TTNNPostNetConfig,
    TTNNSpeechT5SpeechDecoderPostnet,
    preprocess_postnet_parameters,
)
from telemetry.telemetry_client import TelemetryEvent
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor
from tt_model_runners.base_metal_device_runner import BaseMetalDeviceRunner
from utils.decorators import log_execution_time
from utils.speaker_embeddings import SpeakerEmbeddingsManager


DEFAULT_CHUNK_SIZE = 300  # Maximum characters per text chunk (matches demo_ttnn.py)


def chunk_text(text: str, max_chunk_size: int = DEFAULT_CHUNK_SIZE) -> List[str]:
    """Split long text into smaller chunks at sentence/word boundaries (same as demo_ttnn.py)."""
    if len(text) <= max_chunk_size:
        return [text]

    chunks = []
    remaining = text.strip()

    while remaining:
        if len(remaining) <= max_chunk_size:
            chunks.append(remaining)
            break

        chunk_candidate = remaining[:max_chunk_size]
        sentence_end = -1
        for match in re.finditer(r"[.!?]\s+", chunk_candidate):
            sentence_end = match.end()

        if sentence_end > max_chunk_size // 3:
            chunk = remaining[:sentence_end].strip()
            remaining = remaining[sentence_end:].strip()
        else:
            clause_end = -1
            for match in re.finditer(r"[,;]\s+", chunk_candidate):
                clause_end = match.end()

            if clause_end > max_chunk_size // 2:
                chunk = remaining[:clause_end].strip()
                remaining = remaining[clause_end:].strip()
            else:
                last_space = chunk_candidate.rfind(" ")
                if last_space > max_chunk_size // 2:
                    chunk = remaining[:last_space].strip()
                    remaining = remaining[last_space:].strip()
                else:
                    chunk = remaining[:max_chunk_size].strip()
                    remaining = remaining[max_chunk_size:].strip()

        if chunk:
            chunks.append(chunk)

    return chunks


class SpeechT5Constants:
    MAX_STEPS = 768  # Current optimal value
    SAMPLE_RATE = 16000
    REDUCTION_FACTOR = 2
    HIFIGAN_VOCODER_REPO = (
        "microsoft/speecht5_hifigan"  # Standard vocoder for all SpeechT5 models
    )


class TTSpeechT5Runner(BaseMetalDeviceRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.processor = None
        self.model = None  # HuggingFace model reference
        self.vocoder = None
        self.ttnn_encoder = None
        self.ttnn_decoder = None
        self.ttnn_postnet = None
        self.generator = None  # For trace execution
        self.speaker_manager = None
        self.decoder_config = None
        self._baked_speaker_embedding = None  # Tracks which speaker is baked into decoder params

        # Explicitly disable fabric for non-galaxy devices
        if not settings.is_galaxy:
            os.environ["TT_METAL_FABRIC_DISABLE"] = "1"

    def get_pipeline_device_params(self):
        device_params = {"l1_small_size": 150000, "trace_region_size": 50000000}
        return device_params

    def load_weights(self):
        """Load HuggingFace model weights for download verification"""
        self._load_huggingface_models()
        return True

    def _load_huggingface_models(self):
        """Load HuggingFace models - used by both load_weights() and _initialize_models()"""
        try:
            model_weights_path = (
                self.settings.model_weights_path or SupportedModels.SPEECHT5_TTS.value
            )
            self.logger.info(
                f"Device {self.device_id}: Loading HuggingFace model: {model_weights_path}"
            )

            # Load all required components for inference
            self.processor = SpeechT5Processor.from_pretrained(model_weights_path)
            self.model = SpeechT5ForTextToSpeech.from_pretrained(model_weights_path)
            # Vocoder is always the same for all SpeechT5 models (standard HiFi-GAN vocoder)
            self.vocoder = SpeechT5HifiGan.from_pretrained(
                SpeechT5Constants.HIFIGAN_VOCODER_REPO
            )

            self.logger.info(
                f"Device {self.device_id}: Successfully loaded HuggingFace model components"
            )
        except Exception as e:
            self.logger.error(
                f"Device {self.device_id}: Failed to load HuggingFace model: {e}"
            )
            raise RuntimeError(f"Failed to load reference model: {str(e)}") from e

    def _initialize_models(self):
        """Initialize SpeechT5 models and components"""
        try:
            # Load HuggingFace models (same as in load_weights())
            self._load_huggingface_models()

            model = self.model

            # Initialize speaker embeddings manager
            self.speaker_manager = SpeakerEmbeddingsManager()
            self.speaker_manager.preload_embeddings()

            # Create model configurations
            encoder_config = TTNNEncoderConfig(
                vocab_size=model.config.vocab_size,
                hidden_size=model.config.hidden_size,
                num_layers=model.config.encoder_layers,
                num_heads=model.config.encoder_attention_heads,
                ffn_dim=model.config.encoder_ffn_dim,
                max_position_embeddings=model.config.max_length,
                layer_norm_eps=model.config.layer_norm_eps,
            )

            decoder_config = TTNNDecoderConfig(
                hidden_size=model.config.hidden_size,
                num_layers=model.config.decoder_layers,
                num_heads=model.config.decoder_attention_heads,
                ffn_dim=model.config.decoder_ffn_dim,
                max_position_embeddings=model.config.max_length,
                layer_norm_eps=model.config.layer_norm_eps,
                num_mel_bins=model.config.num_mel_bins,
                reduction_factor=model.config.reduction_factor,
                speech_decoder_prenet_units=model.config.speech_decoder_prenet_units,
                speech_decoder_prenet_layers=model.config.speech_decoder_prenet_layers,
                speech_decoder_prenet_dropout=model.config.speech_decoder_prenet_dropout,
                speaker_embedding_dim=model.config.speaker_embedding_dim,
            )

            postnet_config = TTNNPostNetConfig(
                hidden_size=model.config.hidden_size,
                num_mel_bins=model.config.num_mel_bins,
                reduction_factor=model.config.reduction_factor,
                postnet_layers=model.config.speech_decoder_postnet_layers,
                postnet_units=model.config.speech_decoder_postnet_units,
                postnet_kernel=model.config.speech_decoder_postnet_kernel,
            )

            # Get a default speaker embedding for model initialization
            available_speakers = self.speaker_manager.list_available_speakers()
            if not available_speakers:
                # Create a zero embedding if no speakers are available
                self.logger.warning(
                    "No speaker embeddings available, using zero embedding for initialization"
                )
                default_speaker_embedding = torch.zeros(
                    self.speaker_manager.SPEECHT5_EMBEDDING_DIM, dtype=torch.float32
                ).unsqueeze(0)
            else:
                default_speaker_embedding = self.speaker_manager.get_speaker_embedding(
                    available_speakers[0]
                )

            # Create TTNN models
            self.logger.info(f"Device {self.device_id}: Creating TTNN encoder")
            self.ttnn_encoder = TTNNSpeechT5Encoder(
                self.ttnn_device,
                preprocess_encoder_parameters(
                    model.speecht5.encoder, encoder_config, self.ttnn_device
                ),
                encoder_config,
            )

            self.decoder_config = decoder_config
            self._baked_speaker_embedding = default_speaker_embedding

            self.logger.info(f"Device {self.device_id}: Creating TTNN decoder")
            self.ttnn_decoder = TTNNSpeechT5Decoder(
                self.ttnn_device,
                preprocess_decoder_parameters(
                    model.speecht5.decoder,
                    decoder_config,
                    self.ttnn_device,
                    default_speaker_embedding,
                ),
                decoder_config,
            )

            self.logger.info(f"Device {self.device_id}: Creating TTNN postnet")
            self.ttnn_postnet = TTNNSpeechT5SpeechDecoderPostnet(
                self.ttnn_device,
                preprocess_postnet_parameters(
                    model.speech_decoder_postnet, postnet_config, self.ttnn_device
                ),
                postnet_config,
            )

            # Optional: Initialize trace generator for faster inference
            try:
                self.generator = SpeechT5Generator(
                    self.ttnn_encoder,
                    self.ttnn_decoder,
                    self.ttnn_postnet,
                    self.ttnn_device,
                    decoder_config,
                    max_steps=SpeechT5Constants.MAX_STEPS,
                    max_batch_size=1,
                )
                self.logger.info(
                    f"Device {self.device_id}: Trace generator initialized"
                )
            except Exception as e:
                self.logger.warning(
                    f"Device {self.device_id}: Failed to initialize trace generator: {e}"
                )
                self.generator = None

            self.logger.info(
                f"Device {self.device_id}: All SpeechT5 models initialized successfully"
            )

        except Exception as e:
            self.logger.error(
                f"Device {self.device_id}: Failed to initialize models: {e}"
            )
            raise RuntimeError(f"Model initialization failed: {str(e)}") from e

    @log_execution_time(
        "SpeechT5 model load",
        TelemetryEvent.DEVICE_WARMUP,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    async def warmup(self) -> bool:
        try:
            device_id_int = int(self.device_id) if self.device_id else 0
            self.logger.info(f"Device {device_id_int}: Loading SpeechT5 model...")

            if self.ttnn_device is None:
                raise ValueError("Device not initialized. Call set_device() first.")

            try:
                await asyncio.to_thread(self._initialize_models)
                self.logger.info(f"Device {device_id_int}: Model initialization completed")
            except RuntimeError as e:
                self.logger.error(f"Device {device_id_int}: Model initialization failed: {e}")
                raise

            await asyncio.to_thread(self._warmup_encoder_and_postnet)
            self.logger.info(f"Device {device_id_int}: Encoder + postnet kernels compiled")

            if self.generator is not None:
                try:
                    self.logger.info(f"Device {device_id_int}: Capturing traces for all encoder sizes...")
                    await asyncio.to_thread(self.generator.capture_all_traces, self.processor)
                    self.logger.info(f"Device {device_id_int}: Trace compilation completed")
                except Exception as e:
                    self.logger.warning(f"Device {device_id_int}: Trace compilation failed: {e}")
                    self.generator = None

            self.logger.info(f"Device {device_id_int}: Model loaded successfully")
            self.logger.info(f"Device {device_id_int}: Model warmup completed")
            return True

        except Exception as e:
            device_id_int = int(self.device_id) if self.device_id else 0
            self.logger.error(f"Device {device_id_int}: Model loading failed: {e}")
            raise RuntimeError(f"Device {device_id_int}: Model loading failed: {str(e)}") from e


    def _warmup_encoder_and_postnet(self):
        """Compile encoder and postnet TTNN kernels before the first real request."""
        # Encoder warmup: compile encoder with actual tokenized text so the program
        # cache is pre-populated for realistic input lengths (not zero-padded tokens).
        # Multiple texts cover short and medium length inputs.
        warmup_texts = [
            "Hello world",
            "Hello world, this is a test of the speech synthesis system today.",
        ]
        for warmup_text in warmup_texts:
            warmup_ids = self.processor(text=warmup_text, return_tensors="pt")["input_ids"]
            dummy_ttnn_ids = ttnn.from_torch(
                warmup_ids,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.ttnn_device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            _ = self.ttnn_encoder(dummy_ttnn_ids)
            ttnn.deallocate(dummy_ttnn_ids)

        # Postnet warmup: compile postnet kernels with a dummy decoder output
        dummy_decoder_output = ttnn.from_torch(
            torch.randn(1, 1, 1, self.decoder_config.hidden_size),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.ttnn_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        _ = self.ttnn_postnet(dummy_decoder_output)
        ttnn.deallocate(dummy_decoder_output)

    def _prepare_speaker_embedding(self, request) -> torch.Tensor:
        """Prepare speaker embedding for the request"""
        if hasattr(request, 'speaker_embedding') and request.speaker_embedding:
            return self.speaker_manager.process_user_embedding(request.speaker_embedding)
        elif hasattr(request, 'speaker_id') and request.speaker_id:
            return self.speaker_manager.get_speaker_embedding(request.speaker_id)
        else:
            available_speakers = self.speaker_manager.list_available_speakers()
            if available_speakers:
                return self.speaker_manager.get_speaker_embedding(available_speakers[0])
            else:
                self.logger.warning('No speaker embeddings available, using zero embedding')
                return torch.zeros(
                    self.speaker_manager.SPEECHT5_EMBEDDING_DIM, dtype=torch.float32
                ).unsqueeze(0)

    def _generate_mel_for_chunk(self, text: str) -> torch.Tensor:
        """
        Run encoder + autoregressive decoder loop for one text chunk.
        Returns mel spectrogram: [1, steps * REDUCTION_FACTOR, num_mel_bins]
        Matches the per-chunk mel generation in demo_ttnn.py generate_speech_long().
        """
        inputs = self.processor(text=text, return_tensors="pt")
        token_ids = inputs["input_ids"]

        batch_size = token_ids.shape[0]
        num_mel_bins = 80

        # Run encoder on ACTUAL tokens only (no pre-padding).
        # copy_encoder_output zero-pads the encoder output to the nearest supported
        # size and creates the correct cross-attention mask, so the decoder never
        # attends to padding positions. Pre-padding token_ids contaminates real-token
        # representations with padding context, degrading audio quality.
        ttnn_input_ids = ttnn.from_torch(
            token_ids,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.ttnn_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Run encoder once before the generation loop
        # TTNNSpeechT5Encoder.__call__ returns (hidden_states,) tuple
        encoder_hidden_states = self.ttnn_encoder(ttnn_input_ids)[0]

        use_trace = self.generator is not None

        import time as _time
        if use_trace:
            # Trace path: reset state, copy encoder output into pre-allocated
            # tensors at stable memory addresses (required for trace reuse)
            _t0 = _time.perf_counter()
            self.generator._reset_kv_caches()
            self.logger.info(f"_reset_kv_caches: {_time.perf_counter()-_t0:.3f}s")
            self.generator._reset_decode_pos(0, batch_size)
            self.generator.copy_encoder_output(encoder_hidden_states)
        else:
            # Non-trace path: initialize fresh KV cache per inference
            encoder_seq_len = token_ids.shape[1]
            kv_cache, cross_attn_cache = init_kv_cache(
                self.decoder_config,
                self.ttnn_device,
                max_batch_size=batch_size,
                max_seq_len=SpeechT5Constants.MAX_STEPS + 10,
                encoder_seq_len=encoder_seq_len,
            )

        # Initial decoder input: zeros [batch, 1, num_mel_bins]
        decoder_input = ttnn.from_torch(
            torch.zeros(batch_size, 1, num_mel_bins),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.ttnn_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        spectrogram_frames_cpu = []  # Accumulate on CPU — avoids device allocs during active trace
        steps_completed = 0
        _t_loop_start = _time.perf_counter()

        # Autoregressive generation loop — single-frame decode per step
        for step in range(SpeechT5Constants.MAX_STEPS):
            if use_trace:
                # Update position counter in-place (stable memory address for trace)
                self.generator._reset_decode_pos(step, batch_size)

                # Run prenet + positional encoding OUTSIDE trace
                # (these are position-dependent and cannot be traced)
                _tpre = _time.perf_counter()
                preprocessed = self.generator.decoder.preprocess_decoder_inputs(
                    decoder_input_values=decoder_input,
                    position_offset=step,
                )
                if step <= 3:
                    self.logger.info("Step %d: prenet=%.3fs" % (step, _time.perf_counter()-_tpre))

                if step == 0:
                    # Non-traced first step: computes and populates cross-attn cache
                    _ta = _time.perf_counter()
                    decoder_hidden_states = self.generator.decoder(
                        decoder_input_values=None,
                        encoder_hidden_states=self.generator.encoder_hidden_states,
                        kv_cache=self.generator.kv_cache,
                        cross_attn_cache=self.generator.cross_attn_cache,
                        cross_attn_cache_valid=False,
                        current_decode_pos=self.generator.current_decode_pos,
                        preprocessed_hidden_states=preprocessed,
                        encoder_attention_mask=self.generator.encoder_attention_mask,
                    )
                    if step <= 2:
                        self.logger.info("Step %d: decoder=%.3fs" % (step, _time.perf_counter()-_ta))
                else:
                    # Traced steps 1+: execute compiled trace
                    # (cross_attn_cache already populated from step 0)
                    _ta = _time.perf_counter()
                    decoder_hidden_states = self.generator._execute_decoder_trace(
                        preprocessed, blocking=False
                    )
                    # Copy trace output to a fresh L1 tensor (sync barrier, matches demo pattern)
                    decoder_hidden_states = ttnn.to_memory_config(
                        decoder_hidden_states, ttnn.L1_MEMORY_CONFIG
                    )
                    if step <= 2:
                        self.logger.info("Step %d: trace=%.3fs" % (step, _time.perf_counter()-_ta))
            else:
                # Non-trace fallback path
                current_pos = ttnn.from_torch(
                    torch.tensor([step], dtype=torch.int32),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=self.ttnn_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                decoder_hidden_states = self.ttnn_decoder(
                    decoder_input_values=decoder_input,
                    encoder_hidden_states=encoder_hidden_states,
                    kv_cache=kv_cache,
                    cross_attn_cache=cross_attn_cache,
                    cross_attn_cache_valid=(step > 0),
                    current_decode_pos=current_pos,
                    position_offset=step,
                )

            # Postnet inference — mel_after shape: [batch, REDUCTION_FACTOR, num_mel_bins]
            _tp0 = _time.perf_counter()
            mel_before, mel_after, stop_logits = self.ttnn_postnet(
                decoder_hidden_states
            )
            if step <= 2:
                self.logger.info(f"Step {step}: postnet={_time.perf_counter()-_tp0:.3f}s")

            # Transfer mel frame to CPU for spectrogram accumulation
            # mel_after shape: [batch, REDUCTION_FACTOR, num_mel_bins]
            mel_after_cpu = ttnn.to_torch(mel_after)
            spectrogram_frames_cpu.append(mel_after_cpu)

            # Next decoder input: same spec as capture_all_traces dummy input
            # (from_torch with bfloat16 + TILE_LAYOUT + L1 matches trace compile context)
            decoder_input = ttnn.from_torch(
                mel_after_cpu[:, -1:, :],
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.ttnn_device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

            # Check stopping condition — exactly as in demo_ttnn.py, all on device
            # Only transfer a single scalar to CPU for the branch decision
            if step <= 4 or step % 50 == 0:
                self.logger.info(
                    f"Step {step}: {_time.perf_counter() - _t_loop_start:.3f}s elapsed, "
                    f"{step+1} steps done"
                )
            if step >= 10:
                sigmoid_logits = ttnn.sigmoid(stop_logits, memory_config=ttnn.L1_MEMORY_CONFIG)
                sum_prob = ttnn.sum(sigmoid_logits, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
                should_stop = ttnn.ge(sum_prob, 0.5, memory_config=ttnn.L1_MEMORY_CONFIG)
                any_stop_scalar = ttnn.sum(should_stop)
                stop_val = ttnn.to_torch(any_stop_scalar).item()
                if step % 50 == 0:
                    stop_logits_cpu = ttnn.to_torch(stop_logits)
                    sum_prob_cpu = ttnn.to_torch(sum_prob)
                    self.logger.info(
                        f"Step {step}: stop_logits={stop_logits_cpu.tolist()}, "
                        f"sum_prob={sum_prob_cpu.tolist()}, stop_val={stop_val}"
                    )
                if stop_val > 0:
                    steps_completed += 1
                    break

            steps_completed += 1

        # Stack all mel frames on CPU into [1, steps * REDUCTION_FACTOR, num_mel_bins]
        if spectrogram_frames_cpu:
            return torch.cat(spectrogram_frames_cpu, dim=1)
        return torch.zeros(batch_size, 1, num_mel_bins)

    async def _generate_audio_sync(
        self,
        text: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate audio with automatic chunking for long texts (matches demo_ttnn.py)."""
        try:
            chunks = chunk_text(text)
            if len(chunks) > 1:
                self.logger.info(f"Long text ({len(text)} chars) split into {len(chunks)} chunks")

            mel_spectrograms = []
            for i, chunk in enumerate(chunks):
                if len(chunks) > 1:
                    self.logger.info(
                        f"Processing chunk {i+1}/{len(chunks)}: "
                        f"'{chunk[:60]}{'...' if len(chunk) > 60 else ''}'"
                    )
                mel = self._generate_mel_for_chunk(chunk)
                mel_spectrograms.append(mel)

            # Concatenate mels from all chunks along time axis, then run vocoder once
            combined_mel = torch.cat(mel_spectrograms, dim=1) if len(mel_spectrograms) > 1 else mel_spectrograms[0]
            final_audio = self.vocoder(combined_mel)

            audio_buffer = io.BytesIO()
            sf.write(
                audio_buffer,
                final_audio.squeeze().detach().numpy(),
                SpeechT5Constants.SAMPLE_RATE,
                format="WAV",
            )
            audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode("utf-8")
            duration = len(final_audio.squeeze()) / SpeechT5Constants.SAMPLE_RATE

            yield {
                "type": "final_result",
                "result": TextToSpeechResponse(
                    audio=audio_base64,
                    duration=duration,
                    sample_rate=SpeechT5Constants.SAMPLE_RATE,
                    format="wav",
                ),
                "task_id": None,
            }

        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Audio generation failed: {e}")
            raise RuntimeError(f"Audio generation failed: {str(e)}") from e

    async def _run_async(self, requests: list[TextToSpeechRequest]):
        """Main inference method"""
        try:
            if (
                self.ttnn_encoder is None
                or self.ttnn_decoder is None
                or self.ttnn_postnet is None
            ):
                raise RuntimeError("Model components not loaded. Call warmup() first.")
            if self.ttnn_device is None:
                raise ValueError("TTNN device not initialized")

            if len(requests) > 1:
                self.logger.warning(
                    f"Device {self.device_id}: Batch processing not implemented. "
                    f"Processing only first of {len(requests)} requests"
                )

            request = requests[0]
            if request is None:
                raise ValueError("Request cannot be None")
            if not request.text or not request.text.strip():
                raise ValueError("Text cannot be empty")

            speaker_embedding = self._prepare_speaker_embedding(request)
            request._speaker_embedding_array = speaker_embedding.detach().numpy()
            if not torch.equal(speaker_embedding, self._baked_speaker_embedding):
                self.logger.info(
                    f"Device {self.device_id}: Re-baking decoder parameters for new speaker"
                )
                self.ttnn_decoder = TTNNSpeechT5Decoder(
                    self.ttnn_device,
                    preprocess_decoder_parameters(
                        self.model.speecht5.decoder,
                        self.decoder_config,
                        self.ttnn_device,
                        speaker_embedding,
                    ),
                    self.decoder_config,
                )
                self._baked_speaker_embedding = speaker_embedding

            final_result = None
            async for result in self._generate_audio_sync(request.text):
                result["task_id"] = request._task_id
                final_result = result
            if final_result and "result" in final_result:
                return final_result["result"]
            return final_result

        except Exception as e:
            self.logger.error(f"Device {self.device_id}: Inference failed: {e}")
            raise RuntimeError(f"Inference failed: {str(e)}") from e

    @log_execution_time(
        "Run SpeechT5 inference",
        TelemetryEvent.MODEL_INFERENCE,
        os.environ.get("TT_VISIBLE_DEVICES"),
    )
    def run(self, requests: list[TextToSpeechRequest]):
        """Synchronous wrapper for async inference"""
        result = asyncio.run(self._run_async(requests))
        # Wrap result in list as expected by device_worker
        return [result] if result is not None else []
