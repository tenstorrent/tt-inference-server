# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""TTS Quality Test using Word Error Rate (WER).

WER measures TTS quality by transcribing generated audio and comparing
with the original text. Lower WER indicates better speech intelligibility.
"""

import asyncio
import base64
import io
import json
import logging
import re
import shutil
import time
from pathlib import Path

import aiohttp
import numpy as np
from server_tests.base_test import BaseTest

logger = logging.getLogger(__name__)

DATASET_DIR = "tests/server_tests/datasets/libritts_subset"
METADATA_FILE = "metadata.json"

headers = {
    "accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": "Bearer your-secret-key",
}


class TTSQualityTest(BaseTest):
    """Quality test for TTS using Word Error Rate (WER).

    WER measures speech intelligibility by comparing transcribed audio
    with original text. Lower WER = better quality:
    - WER < 5%: Excellent
    - WER 5-10%: Good
    - WER 10-20%: Acceptable
    - WER > 20%: Poor
    """

    def __init__(self, config, targets=None, **kwargs):
        super().__init__(config, targets)
        self._whisper_model = None
        self._whisper_processor = None

    async def _run_specific_test_async(self):
        test_start_time = time.time()
        self.url = f"http://localhost:{self.service_port}/audio/speech"

        # Configuration
        sample_count = self.targets.get("sample_count", 10)
        dataset_split = self.targets.get("dataset_split", "test")
        wer_threshold = self.targets.get("wer_threshold", 0.20)  # 20% WER
        batch_size = self.targets.get("num_of_devices", 1)

        logger.info(
            f"TTS Quality Test: samples={sample_count}, wer_threshold={wer_threshold * 100:.0f}%"
        )

        # Download samples from LibriTTS-R
        self._download_samples(count=sample_count, split=dataset_split)

        # Load metadata
        metadata = self._load_metadata()
        if not metadata:
            return {"success": False, "error": "No metadata found"}

        # Load Whisper model for transcription
        self._load_whisper_model()

        # Generate audio and calculate WER
        results = await self._run_quality_test(metadata, batch_size)

        # Calculate overall WER
        wer_values = [r["wer"] for r in results if r.get("wer") is not None]
        avg_wer = np.mean(wer_values) if wer_values else 1.0
        min_wer = np.min(wer_values) if wer_values else 1.0
        max_wer = np.max(wer_values) if wer_values else 1.0

        success = avg_wer <= wer_threshold
        valid_count = len(wer_values)
        total_count = len(results)

        # Cleanup
        if self.targets.get("cleanup", True):
            self._cleanup_samples()

        # Unload model
        self._unload_whisper_model()

        total_duration = time.time() - test_start_time
        logger.info(
            f"TTS Quality Test completed: avg_wer={avg_wer * 100:.1f}%, "
            f"valid={valid_count}/{total_count}, success={success}, "
            f"duration={total_duration:.1f}s"
        )

        return {
            "success": success,
            "avg_wer": round(avg_wer, 4),
            "min_wer": round(min_wer, 4),
            "max_wer": round(max_wer, 4),
            "wer_threshold": wer_threshold,
            "valid_samples": valid_count,
            "total_samples": total_count,
            "accuracy": round(valid_count / total_count, 4) if total_count > 0 else 0,
            "duration": round(total_duration, 2),
        }

    def _load_whisper_model(self):
        """Load Whisper model for speech recognition."""
        try:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration

            model_name = "openai/whisper-base"
            logger.info(f"Loading Whisper model: {model_name}")
            self._whisper_processor = WhisperProcessor.from_pretrained(model_name)
            self._whisper_model = WhisperForConditionalGeneration.from_pretrained(
                model_name
            )
            logger.info("Whisper model loaded successfully")
        except ImportError:
            logger.warning("transformers not available, using fallback transcription")
            self._whisper_model = None
            self._whisper_processor = None

    def _unload_whisper_model(self):
        """Unload Whisper model to free memory."""
        self._whisper_model = None
        self._whisper_processor = None

    async def _run_quality_test(
        self, metadata: list[dict], batch_size: int
    ) -> list[dict]:
        """Generate audio and calculate WER."""
        results = []

        session_timeout = aiohttp.ClientTimeout(total=2000)
        async with aiohttp.ClientSession(
            headers=headers, timeout=session_timeout
        ) as session:
            num_batches = (len(metadata) + batch_size - 1) // batch_size

            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, len(metadata))
                batch_samples = metadata[batch_start:batch_end]

                tasks = [
                    self._process_sample(session, sample) for sample in batch_samples
                ]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                for i, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.warning(f"Sample {batch_start + i} failed: {result}")
                        results.append({"wer": None, "error": str(result)})
                    else:
                        results.append(result)

        return results

    async def _process_sample(self, session, sample: dict) -> dict:
        """Generate audio for a sample and calculate WER."""
        text = sample.get("text", "")
        sample_id = sample.get("id", "unknown")

        try:
            # Generate audio via TTS API
            payload = {"text": text, "response_format": "verbose_json"}
            async with session.post(
                self.url, json=payload, headers=headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Status {response.status}: {error_text[:200]}")

                content_type = response.headers.get("content-type", "")
                if "application/json" in content_type:
                    result = await response.json()
                    audio_base64 = result.get("audio", "")
                    audio_duration = result.get("duration", 0)
                    generated_audio = base64.b64decode(audio_base64)
                elif "audio/wav" in content_type:
                    generated_audio = await response.read()
                    audio_duration = self._get_wav_duration(generated_audio)
                else:
                    raise Exception(f"Unexpected content-type: {content_type}")

            transcribed_text = self._transcribe_audio(generated_audio)

            wer = self._calculate_wer(text, transcribed_text)
            logger.info(f"Sample {sample_id}: WER={wer * 100:.1f}%")
            logger.debug(f"  Original: {text[:100]}...")
            logger.debug(f"  Transcribed: {transcribed_text[:100]}...")

            return {
                "sample_id": sample_id,
                "wer": wer,
                "original_text": text,
                "transcribed_text": transcribed_text,
                "audio_duration": audio_duration,
            }

        except Exception as e:
            logger.error(f"Sample {sample_id} failed: {e}")
            raise

    def _transcribe_audio(self, audio_bytes: bytes) -> str:
        """Transcribe audio using Whisper model."""
        if self._whisper_model is None:
            return self._transcribe_audio_fallback(audio_bytes)

        try:
            import librosa
            import torch

            audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)

            input_features = self._whisper_processor(
                audio, sampling_rate=16000, return_tensors="pt"
            ).input_features

            # Generate transcription
            with torch.no_grad():
                predicted_ids = self._whisper_model.generate(input_features)

            transcription = self._whisper_processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]

            return transcription.strip()

        except Exception as e:
            logger.warning(f"Whisper transcription failed: {e}")
            return self._transcribe_audio_fallback(audio_bytes)

    def _transcribe_audio_fallback(self, audio_bytes: bytes) -> str:
        """Fallback transcription - returns empty string."""
        logger.warning("No transcription model available")
        return ""

    def _calculate_wer(self, reference: str, hypothesis: str) -> float:
        """
        Calculate Word Error Rate between reference and hypothesis.

        WER = (S + D + I) / N
        where:
        - S = substitutions
        - D = deletions
        - I = insertions
        - N = number of words in reference

        Returns:
            WER as float (0.0 = perfect, 1.0 = 100% error)
        """

        ref_words = self._normalize_text(reference).split()
        hyp_words = self._normalize_text(hypothesis).split()

        if not ref_words:
            return 0.0 if not hyp_words else 1.0

        # Levenshtein distance at word level
        n = len(ref_words)
        m = len(hyp_words)

        # DP table
        dp = [[0] * (m + 1) for _ in range(n + 1)]

        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if ref_words[i - 1] == hyp_words[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(
                        dp[i - 1][j] + 1,  # deletion
                        dp[i][j - 1] + 1,  # insertion
                        dp[i - 1][j - 1] + 1,  # substitution
                    )

        edit_distance = dp[n][m]
        wer = edit_distance / n

        return min(wer, 1.0)

    def _normalize_text(self, text: str) -> str:
        """Normalize text for WER comparison."""

        text = text.lower()

        text = re.sub(r"[^\w\s']", " ", text)

        text = " ".join(text.split())
        return text

    def _get_wav_duration(self, wav_bytes: bytes) -> float:
        """Get duration of WAV audio in seconds."""
        import wave

        with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
            return wav.getnframes() / wav.getframerate()

    def _array_to_wav(self, audio_array: np.ndarray, sample_rate: int) -> bytes:
        """Convert numpy audio array to WAV bytes."""
        import wave

        audio_array = audio_array / (np.max(np.abs(audio_array)) + 1e-8)
        audio_int16 = (audio_array * 32767).astype(np.int16)

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(audio_int16.tobytes())

        return buffer.getvalue()

    def _download_samples(self, count: int = 20, split: str = "test") -> None:
        """Download samples from LibriTTS-R dataset."""
        if count <= 0:
            raise ValueError("Sample count must be positive.")

        try:
            from datasets import load_dataset
            import itertools

            dataset = load_dataset(
                "blabble-io/libritts_r", "clean", split=split, streaming=True
            )

            dataset_subset = list(itertools.islice(dataset, count))

            output_path = Path(DATASET_DIR)
            if output_path.exists():
                shutil.rmtree(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

            metadata = []
            for idx, sample in enumerate(dataset_subset):
                text_original = sample.get("text_original", "")
                sample_id = sample.get("id", f"libritts_{idx:05d}")

                metadata.append(
                    {
                        "index": idx,
                        "id": sample_id,
                        "text": text_original,
                        "normalized_text": sample.get("text_normalized", text_original),
                        "split": split,
                    }
                )

            metadata_path = output_path / METADATA_FILE
            with metadata_path.open("w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Downloaded {len(metadata)} samples")

        except ImportError:
            logger.error("datasets library not available")
            raise

    def _load_metadata(self) -> list[dict]:
        """Load metadata from saved JSON file."""
        metadata_path = Path(DATASET_DIR) / METADATA_FILE
        if not metadata_path.exists():
            logger.warning(f"Metadata file not found: {metadata_path}")
            return []

        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        if not isinstance(metadata, list):
            raise ValueError("Metadata must be a list")

        logger.debug(f"Loaded {len(metadata)} samples from metadata")
        return metadata

    def _cleanup_samples(self) -> None:
        """Clean up downloaded samples directory."""
        dataset_path = Path(DATASET_DIR)
        if dataset_path.exists():
            try:
                shutil.rmtree(dataset_path)
                logger.debug(f"Cleaned up dataset directory: {dataset_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up dataset directory: {e}")

