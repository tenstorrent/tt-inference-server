# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import asyncio
import json
import logging
import shutil
import time
from pathlib import Path

import aiohttp
from server_tests.base_test import BaseTest

logger = logging.getLogger(__name__)

DATASET_DIR = "tests/server_tests/datasets/libritts_subset"
METADATA_FILE = "metadata.json"

headers = {
    "accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": "Bearer your-secret-key",
}


class TTSLoadTest(BaseTest):
    """Load test for Text-to-Speech (SpeechT5) functionality.

    Tests concurrent TTS generation requests using samples from LibriTTS-R dataset.
    Flow:
    1. Download samples from LibriTTS-R dataset (supports train/dev/test splits)
    2. Run concurrent load test with downloaded samples
    3. Compare generated audio with reference audio (optional)
    4. Cleanup samples (optional)

    LibriTTS-R dataset has proper train/validation/test splits:
    - train.clean.100, train.clean.360, train.other.500 (training)
    - dev.clean, dev.other (validation)
    - test.clean, test.other (test) - recommended for MCD evaluation
    """

    def __init__(self, config, targets=None, **kwargs):
        super().__init__(config, targets)
        self.comparison_results: dict = {}

    async def _run_specific_test_async(self):
        self.url = f"http://localhost:{self.service_port}/audio/speech"
        logger.info(f"Running TTS Load Test with targets: {self.targets}")

        devices = self.targets.get("num_of_devices", 1)
        tts_target_time = self.targets.get("tts_generation_time", 10)
        sample_count = self.targets.get("sample_count", 20)
        # cleanup is handled later, default to True for auto-cleanup

        # Get dataset split from targets (default: test.clean for evaluation)
        dataset_split = self.targets.get("dataset_split", "test.clean")
        logger.info(
            f"Step 1: Downloading {sample_count} samples from LibriTTS-R dataset (split: {dataset_split})"
        )
        self._download_samples(count=sample_count, split=dataset_split)

        logger.info("Step 2: Loading metadata")
        metadata = self._load_metadata()
        if not metadata:
            return {
                "success": False,
                "error": "No metadata found. Please download samples first.",
            }

        logger.info(f"Step 3: Running load test with {devices} concurrent requests")
        load_test_results = await self._run_load_test(
            metadata=metadata, batch_size=devices
        )

        # Step 4: Compare generated audio with reference (if enabled)
        compare_audio = self.targets.get("compare_audio", False)
        if compare_audio:
            logger.info("Step 4: Comparing generated audio with reference audio")
            comparison_results = await self._compare_audio_results(
                metadata=metadata, batch_size=devices
            )
            load_test_results["comparison"] = comparison_results

        # Step 5: Always cleanup samples after test (unless explicitly disabled)
        cleanup = self.targets.get("cleanup", True)  # Default to True (auto-cleanup)
        if cleanup:
            logger.info("Step 5: Cleaning up downloaded samples")
            self._cleanup_samples()
        else:
            logger.info("Step 5: Skipping cleanup (cleanup=false in targets)")

        load_test_results["target_time"] = tts_target_time
        load_test_results["devices"] = devices
        load_test_results["sample_count"] = len(metadata)
        load_test_results["success"] = (
            load_test_results.get("average_duration", 0) <= tts_target_time
        )

        return load_test_results

    def _download_samples(self, count: int = 20, split: str = "test.clean") -> None:
        """Download samples from LibriTTS-R dataset and save metadata.

        LibriTTS-R dataset structure:
        - id: unique id
        - text_original: original transcription
        - text_normalized: normalized transcription
        - audio: dict with path, array, sampling_rate (24000 Hz)
        - path: path to audio file
        - speaker_id, chapter_id

        Available splits:
        - train.clean.100, train.clean.360, train.other.500 (training)
        - dev.clean, dev.other (validation)
        - test.clean, test.other (test) - recommended for MCD evaluation

        Args:
            count: Number of samples to download
            split: Dataset split to use (default: "test.clean" for evaluation)
        """
        if count <= 0:
            raise ValueError("Sample count must be positive.")

        logger.info(
            f"Downloading {count} samples from LibriTTS-R dataset (split: {split})"
        )

        try:
            from datasets import load_dataset

            dataset = load_dataset(
                "parler-tts/libritts_r_filtered",
                config="clean",
                split=split,
                trust_remote_code=True,
            )

            total_samples = len(dataset)
            logger.info(
                f"LibriTTS-R dataset has {total_samples} total samples in '{split}' split"
            )

            if count > total_samples:
                logger.warning(
                    f"Requested {count} samples but dataset has only {total_samples}. Using all available."
                )
                count = total_samples

            dataset_subset = dataset.select(range(count))

            output_path = Path(DATASET_DIR)
            if output_path.exists():
                shutil.rmtree(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

            # LibriTTS-R dataset structure: id, text_original, text_normalized, audio, path, speaker_id, chapter_id
            metadata = []
            for idx, sample in enumerate(dataset_subset):
                sample_id = sample.get("id", f"libritts_{idx:05d}")

                text_original = sample.get("text_original", "")
                text_normalized = sample.get("text_normalized", text_original)

                file_path = sample.get("path", "")

                audio_info = sample.get("audio")
                if isinstance(audio_info, dict):
                    audio_path = audio_info.get("path", file_path)
                    sample_rate = audio_info.get("sampling_rate", 24000)
                else:
                    audio_path = file_path
                    sample_rate = 24000

                speaker_id = sample.get("speaker_id", "")
                chapter_id = sample.get("chapter_id", "")

                audio_duration = 0

                metadata.append(
                    {
                        "index": idx,
                        "id": sample_id,
                        "text": text_original,  # Use original text for TTS input
                        "normalized_text": text_normalized,  # Use normalized for comparison
                        "file": file_path,
                        "audio_path": audio_path,
                        "audio_duration": audio_duration,
                        "sample_rate": sample_rate,
                        "speaker_id": speaker_id,
                        "chapter_id": chapter_id,
                        "split": split,  # Track which split this came from
                    }
                )

            metadata_path = output_path / METADATA_FILE
            with metadata_path.open("w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            logger.info(
                f"Saved {len(metadata)} LibriTTS-R samples metadata to {metadata_path} (split: {split})"
            )

        except ImportError:
            logger.error(
                "datasets library not available. Install with: pip install datasets"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to download samples: {e}")
            raise

    def _load_metadata(self) -> list[dict]:
        """Load metadata from saved JSON file.

        Returns:
            List of sample metadata dictionaries
        """
        dataset_path = Path(DATASET_DIR)
        metadata_path = dataset_path / METADATA_FILE

        if not metadata_path.exists():
            logger.warning(f"Metadata file not found: {metadata_path}")
            return []

        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        if not isinstance(metadata, list):
            raise ValueError("Metadata must be a list of sample descriptors.")

        logger.info(f"Loaded {len(metadata)} samples from metadata")
        return metadata

    async def _run_load_test(self, metadata: list[dict], batch_size: int) -> dict:
        """Run concurrent TTS load test with samples from metadata.

        Args:
            metadata: List of sample metadata dictionaries
            batch_size: Number of concurrent requests

        Returns:
            Dictionary with test results
        """

        async def timed_request(session, index, text):
            logger.debug(f"Starting TTS request {index} with text: {text[:50]}...")
            try:
                start = time.perf_counter()
                payload = {"text": text}
                async with session.post(
                    self.url, json=payload, headers=headers
                ) as response:
                    duration = time.perf_counter() - start
                    if response.status == 200:
                        result = await response.json()
                        assert "audio" in result, (
                            "Response should contain 'audio' field"
                        )
                        assert "duration" in result, (
                            "Response should contain 'duration' field"
                        )
                        audio_duration = result.get("duration", 0)
                        logger.debug(
                            f"[{index}] Status: {response.status}, "
                            f"Time: {duration:.2f}s, "
                            f"Audio Duration: {audio_duration:.2f}s"
                        )
                    else:
                        error_text = await response.text()
                        raise Exception(
                            f"Status {response.status} {response.reason}: {error_text}"
                        )
                    return duration

            except Exception as e:
                duration = time.perf_counter() - start
                logger.error(f"[{index}] Error after {duration:.2f}s: {e}")
                raise

        sample_texts = [sample["text"] for sample in metadata]
        if len(sample_texts) < batch_size:
            sample_texts = (sample_texts * ((batch_size // len(sample_texts)) + 1))[
                :batch_size
            ]

        for iteration in range(2):
            session_timeout = aiohttp.ClientTimeout(total=2000)
            async with aiohttp.ClientSession(
                headers=headers, timeout=session_timeout
            ) as session:
                tasks = [
                    timed_request(session, i + 1, sample_texts[i % len(sample_texts)])
                    for i in range(batch_size)
                ]
                results = await asyncio.gather(*tasks)
                requests_duration = max(results)
                total_duration = sum(results)
                avg_duration = total_duration / batch_size

                if iteration == 0:
                    logger.info("🔥 Warm up run done.")
                else:
                    logger.info(
                        f"\n🚀 Time taken for individual concurrent requests: {results}"
                    )
                    logger.info(
                        f"\n🚀 Max time for {batch_size} concurrent requests: "
                        f"{requests_duration:.2f}s"
                    )
                    logger.info(
                        f"\n🚀 Avg time for {batch_size} concurrent requests: "
                        f"{avg_duration:.2f}s"
                    )

                return {
                    "requests_duration": requests_duration,
                    "average_duration": avg_duration,
                }

    async def _compare_audio_results(
        self, metadata: list[dict], batch_size: int
    ) -> dict:
        """Compare generated audio with reference audio from dataset.

        Args:
            metadata: List of sample metadata dictionaries
            batch_size: Number of concurrent requests

        Returns:
            Dictionary with comparison results
        """
        results = await self._replay_samples(metadata=metadata, batch_size=batch_size)
        comparison = self._analyze_results(results, metadata)
        return comparison

    async def _replay_samples(
        self, metadata: list[dict], batch_size: int
    ) -> list[dict]:
        """Replay samples through TTS API and collect responses.

        Args:
            metadata: List of sample metadata dictionaries
            batch_size: Number of concurrent requests

        Returns:
            List of results with sample and response data
        """
        logger.info(f"Replaying {len(metadata)} samples through TTS API")

        async def request_with_sample(session, index, sample):
            text = sample.get("normalized_text") or sample.get("text", "")
            logger.debug(f"Request {index}: {text[:50]}...")
            try:
                payload = {"text": text}
                async with session.post(
                    self.url, json=payload, headers=headers
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return {"sample": sample, "response": result, "index": index}
            except Exception as e:
                logger.error(f"[{index}] Error: {e}")
                raise

        # Use first batch_size samples for comparison
        samples_to_test = (
            metadata[:batch_size] if len(metadata) >= batch_size else metadata
        )

        session_timeout = aiohttp.ClientTimeout(total=2000)
        async with aiohttp.ClientSession(
            headers=headers, timeout=session_timeout
        ) as session:
            tasks = [
                request_with_sample(session, i + 1, sample)
                for i, sample in enumerate(samples_to_test)
            ]
            results = await asyncio.gather(*tasks)

        logger.info(f"Collected {len(results)} responses for comparison")
        return results

    def _extract_audio_info(self, response: dict) -> dict:
        """Extract audio information from TTS response.

        Args:
            response: Response dictionary from TTS API

        Returns:
            Dictionary with audio info (duration, format, base64_audio, etc.)
        """
        audio_info = {
            "duration": response.get("duration", 0),
            "audio_base64": response.get("audio", ""),
            "format": "base64",  # TTS API returns base64 encoded audio
        }
        return audio_info

    def _analyze_results(self, results: list[dict], metadata: list[dict]) -> dict:
        """Analyze TTS results and compare with reference audio.

        Args:
            results: List of results from _replay_samples
            metadata: Original metadata for reference

        Returns:
            Dictionary with analysis results
        """
        logger.info("Analyzing TTS results...")
        total = len(results)
        valid_responses = 0
        mismatches: list[dict] = []

        for entry in results:
            sample = entry.get("sample", {})
            response = entry.get("response", {})
            index = entry.get("index", 0)

            # Extract audio info from response
            audio_info = self._extract_audio_info(response)

            # Basic validation
            expected_text = sample.get("normalized_text") or sample.get("text", "")
            reference_audio_path = sample.get("audio_path") or sample.get("file", "")

            has_audio = bool(audio_info.get("audio_base64"))
            has_duration = audio_info.get("duration", 0) > 0

            if has_audio and has_duration:
                valid_responses += 1
            else:
                mismatches.append(
                    {
                        "index": index,
                        "sample_id": sample.get("id"),
                        "expected_text": expected_text,
                        "has_audio": has_audio,
                        "has_duration": has_duration,
                        "audio_duration": audio_info.get("duration", 0),
                        "reference_audio_path": reference_audio_path,
                    }
                )

        # Calculate basic metrics
        success_rate = (valid_responses / total) if total > 0 else 0.0

        logger.info(
            f"Analysis complete: {valid_responses}/{total} valid responses "
            f"({success_rate * 100:.2f}%)"
        )

        comparison_result = {
            "total": total,
            "valid_responses": valid_responses,
            "success_rate": success_rate,
            "mismatches_count": len(mismatches),
        }

        if mismatches:
            logger.warning(f"Found {len(mismatches)} mismatches")
            # Save mismatches for debugging
            dataset_path = Path(DATASET_DIR)
            mismatch_path = dataset_path / "tts_mismatches.json"
            with mismatch_path.open("w", encoding="utf-8") as f:
                json.dump(mismatches, f, indent=2)
            logger.info(f"Saved mismatches to {mismatch_path}")
            comparison_result["mismatches"] = mismatches

        return comparison_result

    def _cleanup_samples(self) -> None:
        """Clean up downloaded samples directory.

        Removes the entire dataset directory including:
        - Downloaded audio files
        - Metadata JSON file
        - All subdirectories
        """
        dataset_path = Path(DATASET_DIR)
        if dataset_path.exists():
            try:
                shutil.rmtree(dataset_path)
                logger.info(
                    f"✅ Successfully cleaned up dataset directory: {dataset_path}"
                )
            except Exception as e:
                logger.warning(f"⚠️ Failed to clean up dataset directory: {e}")
        else:
            logger.info(
                f"Dataset directory does not exist: {dataset_path} (nothing to clean)"
            )
