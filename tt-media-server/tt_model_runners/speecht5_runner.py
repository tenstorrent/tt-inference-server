# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

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
from num2words import num2words
from telemetry.telemetry_client import TelemetryEvent
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor
from tt_model_runners.base_metal_device_runner import BaseMetalDeviceRunner
from utils.decorators import log_execution_time
from utils.speaker_embeddings import SpeakerEmbeddingsManager

DEFAULT_CHUNK_SIZE = 256  # Maximum characters per text chunk

# Maximum KV cache slots pre-allocated in the generator.
# ~3 mel frames per input token * 256 max tokens = 768, rounded up to 800.
MAX_KV_STEPS = 800

# Encoder sizes to warm-up and capture traces for.
# 32/64 cover short texts; 128/256 cover typical chunked inputs.
# 384 causes L1 OOM on N150 — max supported is 256.
WARMUP_ENCODER_SIZES = [32, 64, 128, 160, 192, 256]


def chunk_text(
    text: str, max_chunk_size: int = DEFAULT_CHUNK_SIZE, processor=None
) -> List[str]:
    """Split text into chunks that always end at sentence boundaries.

    Sentences are packed greedily into chunks until adding the next sentence
    would exceed max_chunk_size characters. A single sentence that exceeds
    max_chunk_size is kept as one chunk (never split mid-sentence).

    The only exception: if a sentence exceeds MAX_ENCODER_TOKENS tokens (hard
    device limit), it is split at the last clause boundary (,;) within that
    token budget to avoid L1 OOM.
    """
    MAX_ENCODER_TOKENS = 250  # conservative limit below the 256 padded size

    if len(text) <= max_chunk_size:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if not sentences:
        return [text]

    def split_oversized(sentence):
        if processor is None:
            return [sentence]
        n_tokens = processor(text=sentence, return_tensors="pt")["input_ids"].shape[1]
        if n_tokens <= MAX_ENCODER_TOKENS:
            return [sentence]
        clauses = re.split(r"(?<=[,;])\s+", sentence)
        parts = []
        current = ""
        for clause in clauses:
            candidate = (current + " " + clause).strip() if current else clause
            n = processor(text=candidate, return_tensors="pt")["input_ids"].shape[1]
            if n <= MAX_ENCODER_TOKENS:
                current = candidate
            else:
                if current:
                    parts.append(current)
                current = clause
        if current:
            parts.append(current)
        return parts if parts else [sentence]

    flat_sentences = []
    for s in sentences:
        s = s.strip()
        if s:
            flat_sentences.extend(split_oversized(s))

    chunks = []
    current = ""
    for sentence in flat_sentences:
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= max_chunk_size:
            current = current + " " + sentence
        else:
            chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    return chunks


def normalize_text_for_tts(text: str) -> str:
    """Normalize text for SpeechT5 by converting numbers to spoken-word form.

    SpeechT5 was not trained robustly on digit tokens, so bare numbers like
    "9" or "133.9" are often skipped or garbled. Converting them to words
    ("nine", "one hundred and thirty-three point nine") gives the model tokens
    it can pronounce reliably.

    Handles: ordinals (1st, 2nd), decimals (133.9), number+unit (250cc),
    and plain integers (9, 122).
    """

    # Ordinals first (e.g. "1st", "23rd") — before plain integers
    def _ordinal(m):
        return num2words(int(m.group(1)), to="ordinal")

    text = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", _ordinal, text, flags=re.IGNORECASE)

    # Decimals (e.g. "133.9") — before plain integers so the dot isn't stranded
    def _decimal(m):
        return num2words(float(m.group(0)))

    text = re.sub(r"\b\d+\.\d+\b", _decimal, text)

    # Number+unit (e.g. "250cc") — split into "two hundred and fifty cc"
    def _number_unit(m):
        return num2words(int(m.group(1))) + " " + m.group(2)

    text = re.sub(r"\b(\d+)([a-zA-Z]+)\b", _number_unit, text)

    # Plain integers (e.g. "9", "122")
    def _integer(m):
        return num2words(int(m.group(0)))

    text = re.sub(r"\b\d+\b", _integer, text)

    return text


class SpeechT5Constants:
    MAX_STEPS = 300  # Default per-chunk step cap (auto-scaled per chunk at runtime)
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
        self._baked_speaker_id = (
            None  # Tracks which speaker ID is baked into decoder params
        )

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
            self.model.eval()
            # Vocoder is always the same for all SpeechT5 models (standard HiFi-GAN vocoder)
            self.vocoder = SpeechT5HifiGan.from_pretrained(
                SpeechT5Constants.HIFIGAN_VOCODER_REPO
            )
            self.vocoder.eval()

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
                use_fp32=True,  # Enable FP32 for precision (matches demo_ttnn.py)
            )

            postnet_config = TTNNPostNetConfig(
                hidden_size=model.config.hidden_size,
                num_mel_bins=model.config.num_mel_bins,
                reduction_factor=model.config.reduction_factor,
                postnet_layers=model.config.speech_decoder_postnet_layers,
                postnet_units=model.config.speech_decoder_postnet_units,
                postnet_kernel=model.config.speech_decoder_postnet_kernel,
            )

            # Get a default speaker embedding for model initialization.
            # Use speaker_7306 (cmu_us_slt_arctic, female) to match demo_ttnn.py which uses
            # embeddings_dataset[7306] from Matthijs/cmu-arctic-xvectors.
            DEFAULT_SPEAKER_ID = "speaker_7306"
            available_speakers = self.speaker_manager.list_available_speakers()
            if not available_speakers:
                self.logger.warning(
                    "No speaker embeddings available, using zero embedding for initialization"
                )
                default_speaker_embedding = torch.zeros(
                    self.speaker_manager.SPEECHT5_EMBEDDING_DIM, dtype=torch.float32
                ).unsqueeze(0)
                self._baked_speaker_id = None
            elif DEFAULT_SPEAKER_ID in available_speakers:
                default_speaker_embedding = self.speaker_manager.get_speaker_embedding(
                    DEFAULT_SPEAKER_ID
                )
                self._baked_speaker_id = DEFAULT_SPEAKER_ID
            else:
                default_speaker_embedding = self.speaker_manager.get_speaker_embedding(
                    available_speakers[0]
                )
                self._baked_speaker_id = available_speakers[0]

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
                max_sequence_length=512,
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
                    max_steps=MAX_KV_STEPS,  # Pre-allocate KV cache for up to 800 steps
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
                self.logger.info(
                    f"Device {device_id_int}: Model initialization completed"
                )
            except RuntimeError as e:
                self.logger.error(
                    f"Device {device_id_int}: Model initialization failed: {e}"
                )
                raise

            # Whisper-style warm-up: call _generate_mel_for_chunk with synthetic texts for
            # each canonical encoder size, exactly as Whisper calls pipeline(dummy_audio).
            # This runs the SAME code path as real inference, ensuring the program cache,
            # trace, and device L2 cache are all hot before the first request arrives.
            await asyncio.to_thread(self._warmup_full_pipeline)
            self.logger.info(f"Device {device_id_int}: Model warmup completed")
            return True

        except Exception as e:
            device_id_int = int(self.device_id) if self.device_id else 0
            self.logger.error(f"Device {device_id_int}: Model loading failed: {e}")
            raise RuntimeError(
                f"Device {device_id_int}: Model loading failed: {str(e)}"
            ) from e

    def _warmup_full_pipeline(self):
        """Warm up by calling _generate_mel_for_chunk with synthetic texts.

        Mirrors the Whisper runner pattern: whisper calls pipeline(dummy_audio) which
        runs the exact same code path as real inference. Here we call _generate_mel_for_chunk
        with synthetic texts that pad to each canonical encoder size [128, 256].

        This ensures:
        1. All TTNN kernels compiled for each encoder shape
        2. Trace captured for each encoder size (128, 256)
        3. 300 traced steps executed per size → device L2 cache fully hot
        4. No idle gap between warm-up and first real request (warmup IS the last thing run)
        """
        # First compile encoder and postnet kernels with dummy inputs
        self._warmup_encoder_and_postnet()

        # Synthetic texts: token counts that pad to 32, 64, 128, 160, 192, 256 respectively.
        # These are the same texts used in demo_ttnn.py's warm-up.
        warmup_texts_per_size = {
            32: "A " * 15,  # ~31 tokens -> pads to 32
            64: "A " * 31,  # ~63 tokens -> pads to 64
            128: "A " * 63,  # ~127 tokens -> pads to 128
            160: "A " * 79,  # ~159 tokens -> pads to 160
            192: "A " * 95,  # ~191 tokens -> pads to 192
            256: "A " * 127,  # ~255 tokens -> pads to 256
        }

        for enc_size in WARMUP_ENCODER_SIZES:
            warmup_text = warmup_texts_per_size[enc_size]
            self.logger.info(
                f"Device {self.device_id}: Warm-up for encoder_size={enc_size} "
                f"via _generate_mel_for_chunk..."
            )
            # Call the exact same inference path as real requests.
            # This captures the trace on first call and runs 300 traced steps,
            # exactly as demo_ttnn.py does with max_steps=300 during warm-up.
            _ = self._generate_mel_for_chunk(warmup_text)
            self.logger.info(
                f"Device {self.device_id}: Warm-up done for encoder_size={enc_size}"
            )

    def _warmup_encoder_and_postnet(self):
        """Compile encoder and postnet TTNN kernels before the first real request.

        Uses the same input-independent warm-up strategy as demo_ttnn.py:
        - Synthetic texts chosen so their tokenized length lands in each canonical
          encoder bucket (128, 256) after PAD-token padding.
        - This pre-compiles encoder kernels for both canonical shapes so any real
          input reuses the cached kernel → consistent TTFT on cold and warm cache.
        """
        # Synthetic texts: produce token counts that pad to 32, 64, 128, 160, 192, 256 respectively
        warmup_texts_per_size = {
            32: "A " * 15,  # ~31 tokens -> pads to 32
            64: "A " * 31,  # ~63 tokens -> pads to 64
            128: "A " * 63,  # ~127 tokens -> pads to 128
            160: "A " * 79,  # ~159 tokens -> pads to 160
            192: "A " * 95,  # ~191 tokens -> pads to 192
            256: "A " * 127,  # ~255 tokens -> pads to 256
        }

        for enc_size in WARMUP_ENCODER_SIZES:
            warmup_text = warmup_texts_per_size[enc_size]
            warmup_ids = self.processor(text=warmup_text, return_tensors="pt")[
                "input_ids"
            ]
            real_seq_len = warmup_ids.shape[1]
            padded_seq_len = get_padded_encoder_seq_len(real_seq_len)

            # Pad with the model's <pad> token (id=1), not zero.
            # Token id 0 is a real vocabulary entry and corrupts encoder representations.
            if real_seq_len < padded_seq_len:
                pad = torch.full(
                    (1, padded_seq_len - real_seq_len), 1, dtype=warmup_ids.dtype
                )
                warmup_ids = torch.cat([warmup_ids, pad], dim=1)

            dummy_ttnn_ids = ttnn.from_torch(
                warmup_ids,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.ttnn_device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            # Pass encoder self-attention mask during warmup so the same code path
            # (with the ttnn.add for mask) is compiled as during real inference.
            warmup_mask = None
            if real_seq_len < padded_seq_len:
                m = torch.zeros(1, 1, padded_seq_len, dtype=torch.float32)
                m[:, :, real_seq_len:] = -1e9
                warmup_mask = ttnn.from_torch(
                    m,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.ttnn_device,
                )
            _ = self.ttnn_encoder(dummy_ttnn_ids, attention_mask=warmup_mask)
            ttnn.deallocate(dummy_ttnn_ids)
            self.logger.info(
                f"Device {self.device_id}: Encoder warm-up done for size {enc_size}"
            )

        # Postnet warmup: compile postnet kernels with a dummy FP32 decoder output
        dummy_decoder_output = ttnn.from_torch(
            torch.randn(1, 1, 1, self.decoder_config.hidden_size),
            dtype=ttnn.float32,  # FP32 matches decoder use_fp32=True
            layout=ttnn.TILE_LAYOUT,
            device=self.ttnn_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        _ = self.ttnn_postnet(dummy_decoder_output)
        ttnn.deallocate(dummy_decoder_output)
        self.logger.info(f"Device {self.device_id}: Postnet warm-up done")

    def _prepare_speaker_embedding(self, request):
        """Prepare speaker embedding for the request. Returns (embedding, speaker_id)."""
        DEFAULT_SPEAKER_ID = "speaker_7306"
        if hasattr(request, "speaker_embedding") and request.speaker_embedding:
            return self.speaker_manager.process_user_embedding(
                request.speaker_embedding
            ), None
        elif hasattr(request, "speaker_id") and request.speaker_id:
            return self.speaker_manager.get_speaker_embedding(
                request.speaker_id
            ), request.speaker_id
        else:
            available_speakers = self.speaker_manager.list_available_speakers()
            if DEFAULT_SPEAKER_ID in available_speakers:
                speaker_id = DEFAULT_SPEAKER_ID
            elif available_speakers:
                speaker_id = available_speakers[0]
            else:
                self.logger.warning(
                    "No speaker embeddings available, using zero embedding"
                )
                return torch.zeros(
                    self.speaker_manager.SPEECHT5_EMBEDDING_DIM, dtype=torch.float32
                ).unsqueeze(0), None
            return self.speaker_manager.get_speaker_embedding(speaker_id), speaker_id

    def _generate_mel_for_chunk(self, text: str) -> torch.Tensor:
        """
        Run encoder + autoregressive decoder loop for one text chunk.
        Returns mel spectrogram: [1, steps * REDUCTION_FACTOR, num_mel_bins]
        Matches the per-chunk mel generation in demo_ttnn.py generate_speech_fp32().
        """
        inputs = self.processor(text=text, return_tensors="pt")
        token_ids = inputs["input_ids"]
        real_seq_len = token_ids.shape[1]
        batch_size = token_ids.shape[0]
        num_mel_bins = 80

        # Pad with the model's <pad> token (id=1) to the canonical encoder size.
        # Token id 0 is a real vocabulary entry and would corrupt encoder representations.
        # Using pad_token_id=1 matches the model's training padding scheme, so the encoder
        # treats padded positions correctly. The cross-attention mask (-1e9 on padded
        # positions) prevents the decoder from attending to them.
        padded_seq_len = get_padded_encoder_seq_len(real_seq_len)
        if real_seq_len < padded_seq_len:
            pad = torch.full(
                (1, padded_seq_len - real_seq_len), 1, dtype=token_ids.dtype
            )
            token_ids = torch.cat([token_ids, pad], dim=1)

        ttnn_input_ids = ttnn.from_torch(
            token_ids,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.ttnn_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Build encoder self-attention mask so pad tokens cannot corrupt real token
        # representations. Shape [1, 1, padded_seq_len] broadcasts over
        # [batch*heads, seq_len, seq_len] inside each encoder attention layer.
        encoder_self_attn_mask = None
        if real_seq_len < padded_seq_len:
            mask = torch.zeros(1, 1, padded_seq_len, dtype=torch.float32)
            mask[:, :, real_seq_len:] = -1e9
            encoder_self_attn_mask = ttnn.from_torch(
                mask,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.ttnn_device,
            )

        encoder_hidden_states = self.ttnn_encoder(
            ttnn_input_ids, attention_mask=encoder_self_attn_mask
        )[0]

        use_trace = self.generator is not None

        # Always allow the full KV cache budget so the stop condition can fire naturally.
        # The stop logits stay near zero throughout speech generation then spike sharply
        # (e.g. sum_prob ~1e-5 during speech, then 0.99 at the natural end).
        # Capping below MAX_KV_STEPS risks truncating before that spike occurs.
        chunk_max_steps = MAX_KV_STEPS

        import time as _time

        if use_trace:
            _t0 = _time.perf_counter()
            self.generator._reset_kv_caches()
            self.logger.info(f"_reset_kv_caches: {_time.perf_counter() - _t0:.3f}s")
            self.generator._reset_decode_pos(0, batch_size)
            # Pass real_seq_len so copy_encoder_output sets the cross-attention mask
            # correctly — masking out the <pad>-token positions.
            self.generator.copy_encoder_output(
                encoder_hidden_states, real_seq_len=real_seq_len
            )
        else:
            kv_cache, cross_attn_cache = init_kv_cache(
                self.decoder_config,
                self.ttnn_device,
                max_batch_size=batch_size,
                max_seq_len=chunk_max_steps + 10,
                encoder_seq_len=padded_seq_len,
            )

        decoder_input = ttnn.from_torch(
            torch.zeros(batch_size, 1, num_mel_bins),
            dtype=ttnn.float32,  # FP32 matches decoder use_fp32=True
            layout=ttnn.TILE_LAYOUT,
            device=self.ttnn_device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        spectrogram_frames_cpu = []
        steps_completed = 0
        _t_loop_start = _time.perf_counter()

        for step in range(chunk_max_steps):
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
                    self.logger.info(
                        "Step %d: prenet=%.3fs" % (step, _time.perf_counter() - _tpre)
                    )

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
                    self.generator.cross_attn_cache_valid = True
                    if step <= 2:
                        self.logger.info(
                            "Step %d: decoder=%.3fs"
                            % (step, _time.perf_counter() - _ta)
                        )
                    # Capture trace after step 0 if not yet captured for this encoder size
                    # (mirrors demo_ttnn.py: capture happens on first run, reused on all subsequent)
                    if not self.generator.trace_compiled:
                        self.generator._capture_decoder_trace(preprocessed)
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
                        self.logger.info(
                            "Step %d: trace=%.3fs" % (step, _time.perf_counter() - _ta)
                        )
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
                self.logger.info(
                    f"Step {step}: postnet={_time.perf_counter() - _tp0:.3f}s"
                )

            # Transfer mel frame to CPU for spectrogram accumulation
            # mel_after shape: [batch, REDUCTION_FACTOR, num_mel_bins]
            mel_after_cpu = ttnn.to_torch(mel_after)
            spectrogram_frames_cpu.append(mel_after_cpu)

            # Next decoder input: last mel frame, FP32 to match decoder use_fp32=True
            decoder_input = ttnn.from_torch(
                mel_after_cpu[:, -1:, :],
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.ttnn_device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

            # Check stopping condition — exactly as in demo_ttnn.py, all on device
            # Only transfer a single scalar to CPU for the branch decision
            if step <= 4 or step % 50 == 0:
                self.logger.info(
                    f"Step {step}: {_time.perf_counter() - _t_loop_start:.3f}s elapsed, "
                    f"{step + 1} steps done"
                )
            if step >= 10:
                sigmoid_logits = ttnn.sigmoid(
                    stop_logits, memory_config=ttnn.L1_MEMORY_CONFIG
                )
                sum_prob = ttnn.sum(
                    sigmoid_logits, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG
                )
                should_stop = ttnn.ge(
                    sum_prob, 0.5, memory_config=ttnn.L1_MEMORY_CONFIG
                )
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
            text = normalize_text_for_tts(text)
            chunks = chunk_text(text, processor=self.processor)
            if len(chunks) > 1:
                self.logger.info(
                    f"Long text ({len(text)} chars) split into {len(chunks)} chunks"
                )

            mel_spectrograms = []
            for i, chunk in enumerate(chunks):
                if len(chunks) > 1:
                    self.logger.info(
                        f"Processing chunk {i + 1}/{len(chunks)}: "
                        f"'{chunk[:60]}{'...' if len(chunk) > 60 else ''}'"
                    )
                mel = self._generate_mel_for_chunk(chunk)
                mel_spectrograms.append(mel)

            # Concatenate mels from all chunks along time axis, then run vocoder once
            combined_mel = (
                torch.cat(mel_spectrograms, dim=1)
                if len(mel_spectrograms) > 1
                else mel_spectrograms[0]
            )
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

            speaker_embedding, speaker_id = self._prepare_speaker_embedding(request)
            request._speaker_embedding_array = speaker_embedding.detach().numpy()
            # Compare by ID when available (avoids float-equality mismatch for same speaker).
            # Fall back to True (always update) only for custom embeddings (speaker_id=None).
            speaker_changed = (
                speaker_id != self._baked_speaker_id if speaker_id is not None else True
            )
            if speaker_changed:
                self.logger.info(
                    f"Device {self.device_id}: Updating speaker embedding in-place (no trace re-capture needed)"
                )
                # Update the speaker embedding tensor in DRAM directly — this overwrites only the
                # speaker_embeddings_normalized tensor inside the prenet, leaving decoder weights,
                # layers, and all captured traces intact. The prenet runs OUTSIDE the trace
                # (it's position-dependent), so changing the speaker here takes effect immediately
                # without any trace invalidation or re-capture.
                self.ttnn_decoder.update_speaker_embedding(speaker_embedding)
                self._baked_speaker_id = speaker_id
                self.logger.info(
                    f"Device {self.device_id}: Speaker embedding updated successfully"
                )

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
