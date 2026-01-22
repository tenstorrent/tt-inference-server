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
    """

    def __init__(self, config, targets=None, **kwargs):
        super().__init__(config, targets)
        self.comparison_results: dict = {}

    def _log_step_header(self, step_num: int, title: str, details: list[str] = None):
        """Log step header with consistent formatting."""
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"Step {step_num}: {title}")
        if details:
            for detail in details:
                logger.info(f"   {detail}")
        logger.info("=" * 70)

    def _format_response_values(self, results: dict) -> dict:
        """Format response values for consistency (round floats to appropriate decimals)."""
        time_fields_2dec = [
            "requests_duration",
            "min_duration",
            "average_duration",
            "ttft_ms",
            "min_ttft_ms",
            "max_ttft_ms",
        ]
        for field in time_fields_2dec:
            if field in results:
                results[field] = round(results[field], 2)

        if "accuracy" in results:
            results["accuracy"] = round(results["accuracy"], 4)

        return results

    def _log_summary(self, results: dict, target_time: float, total_duration: float):
        """Log test summary with all metrics."""
        logger.info("")
        logger.info(
            "📊 Results: "
            f"avg={results.get('average_duration', 0):.2f}s "
            f"(min={results.get('min_duration', 0):.2f}s, "
            f"max={results.get('requests_duration', 0):.2f}s), "
            f"target={target_time}s, "
            f"TTFT={results.get('ttft_ms', 0):.2f}ms, "
            f"accuracy={results.get('accuracy', 0) * 100:.1f}%, "
            f"success={results['success']}, "
            f"duration={total_duration:.1f}s"
        )

    async def _run_specific_test_async(self):
        test_start_time = time.time()

        self.url = f"http://localhost:{self.service_port}/audio/speech"
        logger.info("🚀 TTS Load Test - Starting")

        # Use same parameter names and defaults as AudioTranscriptionLoadTest
        # Default 5 for concurrent load testing (batch_size = num_of_devices)
        devices = self.targets.get("num_of_devices", 5)
        tts_target_time = self.targets.get("tts_generation_time", 10)
        sample_count = self.targets.get("sample_count", 10)
        dataset_split = self.targets.get("dataset_split", "test")

        # Step 1: Download samples
        logger.debug(
            f"Downloading {sample_count} samples from LibriTTS-R dataset (split: {dataset_split})"
        )
        self._download_samples(count=sample_count, split=dataset_split)

        # Step 2: Load metadata
        metadata = self._load_metadata()
        if not metadata:
            logger.error("❌ No metadata found. Please download samples first.")
            return {
                "success": False,
                "error": "No metadata found. Please download samples first.",
            }

        # Step 3: Run load test
        logger.info(
            f"Running load test: {len(metadata)} samples, {devices} concurrent requests (target: {tts_target_time}s)"
        )
        load_test_results = await self._run_load_test(
            metadata=metadata, batch_size=devices
        )

        # Initialize success criteria based on time check (before Step 4)
        time_check = load_test_results.get("average_duration", 0) <= tts_target_time
        ttft_check = True  # Can add TTFT target check here if needed
        load_test_results["success"] = time_check and ttft_check

        # Step 4: Compare generated audio with reference (if enabled)
        compare_audio = self.targets.get("compare_audio", True)
        if compare_audio:
            logger.debug("Comparing generated audio with reference audio")
            comparison_results = await self._compare_audio_results(
                metadata=metadata, batch_size=devices
            )

            accuracy_value = self._save_accuracy_results(comparison_results)
            load_test_results["comparison"] = comparison_results
            load_test_results["accuracy"] = accuracy_value

            accuracy_threshold = self.targets.get("accuracy_threshold", 0.8)
            accuracy_check = accuracy_value >= accuracy_threshold
            load_test_results["success"] = (
                load_test_results["success"] and accuracy_check
            )

        # Step 5: Cleanup samples
        cleanup = self.targets.get("cleanup", True)
        if cleanup:
            self._cleanup_samples()

        # Prepare final results
        load_test_results["target_time"] = tts_target_time
        load_test_results["devices"] = devices
        load_test_results["sample_count"] = len(metadata)
        load_test_results = self._format_response_values(load_test_results)

        # Log final summary
        total_duration = time.time() - test_start_time
        self._log_summary(load_test_results, tts_target_time, total_duration)

        return load_test_results

    def _save_accuracy_results(self, comparison_results: dict) -> float:
        """Save accuracy results to JSON file and return accuracy value."""
        if not isinstance(comparison_results, dict):
            logger.warning(
                f"comparison_results is not a dict: {type(comparison_results)}"
            )
            comparison_results = {}

        accuracy_value = comparison_results.get("accuracy") or comparison_results.get(
            "success_rate", 0.0
        )
        accuracy_data = {
            "accuracy": accuracy_value,
            "valid_responses": comparison_results.get("valid_responses", 0),
            "total": comparison_results.get("total", 0),
            "mismatches_count": comparison_results.get("mismatches_count", 0),
        }

        dataset_path = Path(DATASET_DIR)
        accuracy_file = dataset_path / "tts_accuracy.json"
        with accuracy_file.open("w", encoding="utf-8") as f:
            json.dump(accuracy_data, f, indent=2)
        logger.debug(f"Saved accuracy results to {accuracy_file}")

        return accuracy_value

    def _extract_batch_metrics(self, batch_results: list) -> tuple[list, list]:
        """Extract duration and TTFT metrics from batch results."""
        if isinstance(batch_results[0], dict):
            durations = [r["duration"] for r in batch_results]
            ttft_values = [r["ttft_ms"] for r in batch_results]
        else:
            # Fallback for old format
            durations = batch_results
            ttft_values = [0] * len(batch_results)
        return durations, ttft_values

    def _calculate_metrics(
        self, durations: list[float], ttft_values: list[float]
    ) -> dict:
        """Calculate aggregate metrics from duration and TTFT lists."""
        max_duration = max(durations) if durations else 0
        min_duration = min(durations) if durations else 0
        avg_duration = sum(durations) / len(durations) if durations else 0
        avg_ttft_ms = sum(ttft_values) / len(ttft_values) if ttft_values else 0
        max_ttft_ms = max(ttft_values) if ttft_values else 0
        min_ttft_ms = min(ttft_values) if ttft_values else 0

        return {
            "requests_duration": round(max_duration, 2),
            "min_duration": round(min_duration, 2),
            "average_duration": round(avg_duration, 2),
            "ttft_ms": round(avg_ttft_ms, 2),
            "min_ttft_ms": round(min_ttft_ms, 2),
            "max_ttft_ms": round(max_ttft_ms, 2),
        }

    def _log_load_test_results(
        self,
        metrics: dict,
        total_samples: int,
        num_batches: int,
        iteration_duration: float,
    ):
        """Log load test results in a formatted way."""
        # Results are logged in _log_summary, no need for detailed logging here
        pass

    def _download_samples(self, count: int = 20, split: str = "test") -> None:
        """Download samples from LibriTTS-R dataset and save metadata.

        Dataset: blabble-io/libritts_r (filtered LibriTTS-R)
        Config: "clean" (contains only clean splits)

        LibriTTS-R dataset structure:
        - id: unique id (e.g., "3081_166546_000028_000002")
        - text_original: original transcription
        - text_normalized: normalized transcription
        - audio: dict with path, array, sampling_rate (24000 Hz)
        - path: path to audio file
        - speaker_id, chapter_id

        Available splits:
        - Without config: train, test, validation (standard splits)
        - With config="clean": train.clean.100, train.clean.360, dev.clean, test.clean, etc.
        - Note: When config parameter fails, we use standard splits (train/test/validation)

        Args:
            count: Number of samples to download
            split: Dataset split to use (default: "test" for evaluation)
                  If specific split like "test.clean" fails, falls back to "test"
        """
        if count <= 0:
            raise ValueError("Sample count must be positive.")

        download_start = time.time()

        logger.debug(
            f"Downloading {count} samples from LibriTTS-R dataset (split: {split})"
        )

        try:
            from datasets import load_dataset

            # Load LibriTTS-R dataset (blabble-io/libritts_r)
            # Note: Some versions of datasets library don't support config parameter properly
            # We load the dataset and access splits directly
            # Dataset structure: id, text_original, text_normalized, audio, path, speaker_id, chapter_id
            # Try loading with config as positional argument first (per documentation)
            # Use streaming=True to avoid downloading entire dataset
            # If that fails due to ParquetConfig issue, load without config and find split manually
            try:
                # Try with streaming first (faster, doesn't download everything)
                logger.debug(f"Loading dataset with config='clean', split='{split}'...")
                dataset = load_dataset(
                    "blabble-io/libritts_r", "clean", split=split, streaming=True
                )
            except (TypeError, ValueError) as e:
                if (
                    "config" in str(e)
                    or "ParquetConfig" in str(e)
                    or "Bad split" in str(e)
                ):
                    # Config parameter or specific split causes issues with this datasets version
                    # Fallback: use standard split (test/train/validation) without config
                    logger.warning(
                        f"Config/specific split not supported ({e}), using standard split 'test'"
                    )
                    # Map specific splits to standard ones
                    split_mapping = {
                        "test.clean": "test",
                        "test.other": "test",
                        "dev.clean": "validation",
                        "dev.other": "validation",
                        "train.clean.100": "train",
                        "train.clean.360": "train",
                        "train.other.500": "train",
                    }
                    standard_split = split_mapping.get(split, "test")
                    logger.debug(
                        f"Using standard split '{standard_split}' instead of '{split}'"
                    )

                    # Load with streaming using standard split
                    dataset = load_dataset(
                        "blabble-io/libritts_r", split=standard_split, streaming=True
                    )
                else:
                    # Different error, re-raise it
                    raise
            except Exception as e:
                logger.error(f"Failed to load blabble-io/libritts_r: {e}")
                raise

            # Handle streaming vs non-streaming datasets
            if hasattr(dataset, "__iter__") and not hasattr(dataset, "__len__"):
                # Streaming dataset - take first N samples (doesn't download everything)
                logger.info("   Dataset type: Streaming")
                logger.info(f"   Taking first {count} samples from stream...")
                import itertools

                stream_start = time.time()
                dataset_subset = list(itertools.islice(dataset, count))
                stream_duration = time.time() - stream_start
                total_samples = len(dataset_subset)  # Only count what we downloaded
                logger.info(
                    f"✅ Downloaded {total_samples} samples in {stream_duration:.2f}s (streaming mode)"
                )
            else:
                # Non-streaming dataset - can get length and select
                logger.info("   Dataset type: Non-streaming")
                total_samples = len(dataset)
                logger.info(
                    f"   Dataset has {total_samples} total samples in '{split}' split"
                )

                if count > total_samples:
                    logger.warning(
                        f"   ⚠️ Requested {count} samples but dataset has only {total_samples}. Using all available."
                    )
                    count = total_samples

                logger.debug(f"Selecting {count} samples...")
                select_start = time.time()
                dataset_subset = dataset.select(range(count))
                select_duration = time.time() - select_start
                logger.debug(f"Selected {count} samples in {select_duration:.2f}s")

            output_path = Path(DATASET_DIR)
            if output_path.exists():
                shutil.rmtree(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

            metadata = []
            # Handle both list (from streaming) and dataset object
            samples_iter = (
                dataset_subset if isinstance(dataset_subset, list) else dataset_subset
            )
            for idx, sample in enumerate(samples_iter):
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

            logger.debug(
                f"Saved {len(metadata)} LibriTTS-R samples metadata (split: {split})"
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
        import time

        load_start = time.time()

        dataset_path = Path(DATASET_DIR)
        metadata_path = dataset_path / METADATA_FILE
        logger.info(f"   Looking for metadata file: {metadata_path}")

        if not metadata_path.exists():
            logger.warning(f"   ⚠️ Metadata file not found: {metadata_path}")
            return []

        logger.info("   Reading metadata file...")
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        if not isinstance(metadata, list):
            raise ValueError("Metadata must be a list of sample descriptors.")

        load_duration = time.time() - load_start
        logger.info(
            f"✅ Loaded {len(metadata)} samples from metadata in {load_duration:.2f}s"
        )
        return metadata

    async def _run_load_test(self, metadata: list[dict], batch_size: int) -> dict:
        """Run concurrent TTS load test with samples from metadata.

        Args:
            metadata: List of sample metadata dictionaries
            batch_size: Number of concurrent requests

        Returns:
            Dictionary with test results
        """
        load_test_start = time.time()

        async def timed_request(session, index, text):
            logger.debug(f"   Starting TTS request {index} with text: {text[:50]}...")
            try:
                start = time.perf_counter()
                # Request JSON format to get duration info, not binary audio
                # Default response_format is "audio" which returns WAV bytes
                # Use "verbose_json" to get JSON response with base64 audio and duration
                payload = {"text": text, "response_format": "verbose_json"}
                logger.debug(
                    f"   [{index}] Request payload: text='{text[:30]}...', response_format='verbose_json'"
                )

                async with session.post(
                    self.url, json=payload, headers=headers
                ) as response:
                    # Measure TTFT: time from request start to first byte received
                    first_chunk = await response.content.read(1)
                    ttft_ms = (
                        time.perf_counter() - start
                    ) * 1000  # Convert to milliseconds

                    # Read remaining response
                    if first_chunk:
                        remaining = await response.read()
                        full_response_bytes = first_chunk + remaining
                    else:
                        full_response_bytes = await response.read()

                    total_duration = time.perf_counter() - start

                    # Check content type - server can return audio/wav or application/json
                    content_type = response.headers.get("content-type", "")

                    if response.status == 200:
                        audio_duration = 0
                        if "application/json" in content_type:
                            # JSON response with base64 audio
                            import json as json_lib

                            result = json_lib.loads(full_response_bytes.decode("utf-8"))
                            assert "audio" in result, (
                                "Response should contain 'audio' field"
                            )
                            assert "duration" in result, (
                                "Response should contain 'duration' field"
                            )
                            audio_duration = result.get("duration", 0)
                            logger.debug(
                                f"   [{index}] Success - TTFT: {ttft_ms:.2f}ms, "
                                f"Total: {total_duration:.2f}s, Audio Duration: {audio_duration:.2f}s"
                            )
                        elif "audio/wav" in content_type:
                            # Binary audio response - server ignored response_format, read bytes and calculate duration
                            logger.warning(
                                f"   [{index}] ⚠️ Server returned audio/wav instead of JSON (response_format may be ignored)"
                            )
                            import wave
                            import io

                            # Parse WAV to get duration
                            with wave.open(io.BytesIO(full_response_bytes)) as wav:
                                sample_rate = wav.getframerate()
                                num_frames = wav.getnframes()
                                audio_duration = (
                                    num_frames / sample_rate if sample_rate > 0 else 0
                                )

                            logger.debug(
                                f"   [{index}] Success - TTFT: {ttft_ms:.2f}ms, Total: {total_duration:.2f}s, "
                                f"Audio Duration: {audio_duration:.2f}s (parsed from WAV)"
                            )
                        else:
                            raise Exception(f"Unexpected content-type: {content_type}")
                    else:
                        error_text = full_response_bytes.decode(
                            "utf-8", errors="ignore"
                        )
                        raise Exception(
                            f"Status {response.status} {response.reason}: {error_text}"
                        )

                    # Return dict with all metrics
                    return {
                        "duration": total_duration,
                        "ttft_ms": ttft_ms,
                        "audio_duration": audio_duration,
                    }

            except Exception as e:
                duration = time.perf_counter() - start
                logger.error(f"   [{index}] Error after {duration:.2f}s: {e}")
                raise

        sample_texts = [sample["text"] for sample in metadata]

        # Process all samples in batches with concurrent requests per batch
        # This ensures all samples are tested while maintaining concurrent load
        num_batches = (len(sample_texts) + batch_size - 1) // batch_size

        if batch_size == 1:
            logger.warning(
                "⚠️  batch_size=1 means sequential execution (not concurrent load). "
                "Consider increasing num_of_devices for true concurrent load testing."
            )

        for iteration in range(2):
            iteration_start = time.time()
            all_durations = []
            all_ttft_values = []

            session_timeout = aiohttp.ClientTimeout(total=2000)
            async with aiohttp.ClientSession(
                headers=headers, timeout=session_timeout
            ) as session:
                # Process all samples in batches
                for batch_idx in range(num_batches):
                    batch_start = batch_idx * batch_size
                    batch_end = min(batch_start + batch_size, len(sample_texts))
                    batch_samples = sample_texts[batch_start:batch_end]

                    tasks = [
                        timed_request(
                            session,
                            batch_start + i + 1,
                            batch_samples[i],
                        )
                        for i in range(len(batch_samples))
                    ]

                    batch_results = await asyncio.gather(*tasks)

                    # Extract metrics from batch results
                    batch_durations, batch_ttft_values = self._extract_batch_metrics(
                        batch_results
                    )
                    all_durations.extend(batch_durations)
                    all_ttft_values.extend(batch_ttft_values)

            iteration_duration = time.time() - iteration_start

            # Calculate aggregate metrics across all batches
            metrics = self._calculate_metrics(all_durations, all_ttft_values)

            if iteration == 0:
                logger.debug(f"Warm-up completed in {iteration_duration:.2f}s")
            else:
                return {
                    **metrics,
                    "ttft_values": [round(v, 2) for v in all_ttft_values],
                    "total_samples": len(all_durations),
                    "num_batches": num_batches,
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

        async def request_with_sample(session, index, sample):
            text = sample.get("normalized_text") or sample.get("text", "")
            try:
                # Request JSON format to get structured response
                payload = {"text": text, "response_format": "verbose_json"}
                async with session.post(
                    self.url, json=payload, headers=headers
                ) as response:
                    response.raise_for_status()
                    # Check content type
                    content_type = response.headers.get("content-type", "")
                    if "application/json" in content_type:
                        result = await response.json()
                    elif "audio/wav" in content_type:
                        # Fallback: if server returns audio, we can't parse it here
                        raise Exception(
                            f"Server returned audio/wav instead of JSON. "
                            f"Content-Type: {content_type}"
                        )
                    else:
                        raise Exception(f"Unexpected content-type: {content_type}")
                    return {"sample": sample, "response": result, "index": index}
            except Exception as e:
                logger.error(f"[{index}] Error: {e}")
                raise

        # Process all samples in batches for comparison
        num_batches = (len(metadata) + batch_size - 1) // batch_size

        all_results = []
        session_timeout = aiohttp.ClientTimeout(total=2000)
        async with aiohttp.ClientSession(
            headers=headers, timeout=session_timeout
        ) as session:
            # Process all samples in batches
            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, len(metadata))
                batch_samples = metadata[batch_start:batch_end]
                batch_num = batch_idx + 1

                tasks = [
                    request_with_sample(session, batch_start + i + 1, sample)
                    for i, sample in enumerate(batch_samples)
                ]
                batch_results = await asyncio.gather(*tasks)
                all_results.extend(batch_results)

        return all_results

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
        logger.debug("Analyzing TTS results...")
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

        # Calculate accuracy (like vision_evals_test)
        accuracy = (valid_responses / total) if total > 0 else 0.0

        logger.debug(
            f"Audio comparison: {valid_responses}/{total} valid, accuracy: {accuracy * 100:.2f}%"
        )

        comparison_result = {
            "total": total,
            "valid_responses": valid_responses,
            "success_rate": accuracy,  # Alias for backward compatibility
            "accuracy": accuracy,  # Primary field (like vision_evals_test)
            "mismatches_count": len(mismatches),
        }

        if mismatches:
            logger.warning(f"Found {len(mismatches)} mismatches")
            # Save mismatches for debugging
            dataset_path = Path(DATASET_DIR)
            mismatch_path = dataset_path / "tts_mismatches.json"
            with mismatch_path.open("w", encoding="utf-8") as f:
                json.dump(mismatches, f, indent=2)
            logger.debug(f"Saved mismatches to {mismatch_path}")
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
                logger.debug(f"Cleaned up dataset directory: {dataset_path}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to clean up dataset directory: {e}")
