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
    """

    def __init__(self, config, targets=None, **kwargs):
        super().__init__(config, targets)

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
        return results

    async def _run_specific_test_async(self):
        test_start_time = time.time()
        self.url = f"http://localhost:{self.service_port}/audio/speech"

        devices = self.targets.get("num_of_devices", 1)
        tts_target_time = self.targets.get("tts_generation_time", 10)
        sample_count = self.targets.get("sample_count", 10)
        dataset_split = self.targets.get("dataset_split", "test")

        logger.info(
            f"TTS Load Test: devices={devices}, target={tts_target_time}s, samples={sample_count}"
        )

        # Download samples
        self._download_samples(count=sample_count, split=dataset_split)

        # Load metadata
        metadata = self._load_metadata()
        if not metadata:
            logger.error("No metadata found. Please download samples first.")
            return {"success": False, "error": "No metadata found"}

        # Run load test
        load_test_results = await self._run_load_test(
            metadata=metadata, batch_size=devices
        )

        # Check success criteria
        time_check = load_test_results.get("average_duration", 0) <= tts_target_time
        load_test_results["success"] = time_check

        # Cleanup samples
        cleanup = self.targets.get("cleanup", True)
        if cleanup:
            self._cleanup_samples()

        # Prepare final results
        load_test_results["target_time"] = tts_target_time
        load_test_results["devices"] = devices
        load_test_results["sample_count"] = len(metadata)
        load_test_results = self._format_response_values(load_test_results)

        total_duration = time.time() - test_start_time
        logger.info(
            f"TTS Load Test completed: avg={load_test_results.get('average_duration', 0):.2f}s, "
            f"target={tts_target_time}s, TTFT={load_test_results.get('ttft_ms', 0):.2f}ms, "
            f"success={load_test_results['success']}, duration={total_duration:.1f}s"
        )

        return load_test_results

    def _extract_batch_metrics(self, batch_results: list) -> tuple[list, list]:
        """Extract duration and TTFT metrics from batch results."""
        if isinstance(batch_results[0], dict):
            durations = [r["duration"] for r in batch_results]
            ttft_values = [r["ttft_ms"] for r in batch_results]
        else:
            durations = batch_results
            ttft_values = [0] * len(batch_results)
        return durations, ttft_values

    def _calculate_metrics(
        self, durations: list[float], ttft_values: list[float]
    ) -> dict:
        """Calculate aggregate metrics from duration and TTFT lists."""
        return {
            "requests_duration": round(max(durations), 2) if durations else 0,
            "min_duration": round(min(durations), 2) if durations else 0,
            "average_duration": round(sum(durations) / len(durations), 2)
            if durations
            else 0,
            "ttft_ms": round(sum(ttft_values) / len(ttft_values), 2)
            if ttft_values
            else 0,
            "min_ttft_ms": round(min(ttft_values), 2) if ttft_values else 0,
            "max_ttft_ms": round(max(ttft_values), 2) if ttft_values else 0,
        }

    def _download_samples(self, count: int = 20, split: str = "test") -> None:
        """Download samples from LibriTTS-R dataset and save metadata."""
        if count <= 0:
            raise ValueError("Sample count must be positive.")

        try:
            from datasets import load_dataset
            import itertools

            # Try with streaming first
            try:
                dataset = load_dataset(
                    "blabble-io/libritts_r", "clean", split=split, streaming=True
                )
            except (TypeError, ValueError) as e:
                if (
                    "config" in str(e)
                    or "ParquetConfig" in str(e)
                    or "Bad split" in str(e)
                ):
                    split_mapping = {
                        "test.clean": "test",
                        "test.other": "test",
                        "dev.clean": "validation",
                        "dev.other": "validation",
                        "train.clean.100": "train",
                        "train.clean.360": "train",
                    }
                    standard_split = split_mapping.get(split, "test")
                    dataset = load_dataset(
                        "blabble-io/libritts_r", split=standard_split, streaming=True
                    )
                else:
                    raise

            # Get samples from stream
            if hasattr(dataset, "__iter__") and not hasattr(dataset, "__len__"):
                dataset_subset = list(itertools.islice(dataset, count))
            else:
                if count > len(dataset):
                    count = len(dataset)
                dataset_subset = dataset.select(range(count))

            output_path = Path(DATASET_DIR)
            if output_path.exists():
                shutil.rmtree(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

            metadata = []
            samples_iter = (
                dataset_subset if isinstance(dataset_subset, list) else dataset_subset
            )
            for idx, sample in enumerate(samples_iter):
                text_original = sample.get("text_original", "")
                metadata.append(
                    {
                        "index": idx,
                        "id": sample.get("id", f"libritts_{idx:05d}"),
                        "text": text_original,
                        "normalized_text": sample.get("text_normalized", text_original),
                        "split": split,
                    }
                )

            metadata_path = output_path / METADATA_FILE
            with metadata_path.open("w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            logger.debug(f"Downloaded {len(metadata)} samples from LibriTTS-R")

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

    async def _run_load_test(self, metadata: list[dict], batch_size: int) -> dict:
        """Run concurrent TTS load test."""

        async def timed_request(session, index, text):
            try:
                start = time.perf_counter()
                payload = {"text": text, "response_format": "verbose_json"}

                async with session.post(
                    self.url, json=payload, headers=headers
                ) as response:
                    # Measure TTFT
                    first_chunk = await response.content.read(1)
                    ttft_ms = (time.perf_counter() - start) * 1000

                    # Read remaining response
                    if first_chunk:
                        remaining = await response.read()
                        full_response_bytes = first_chunk + remaining
                    else:
                        full_response_bytes = await response.read()

                    content_type = response.headers.get("content-type", "")

                    if response.status == 200:
                        audio_duration = 0
                        if "application/json" in content_type:
                            result = json.loads(full_response_bytes.decode("utf-8"))
                            audio_duration = result.get("duration", 0)
                        elif "audio/wav" in content_type:
                            import wave
                            import io

                            with wave.open(io.BytesIO(full_response_bytes)) as wav:
                                audio_duration = (
                                    wav.getnframes() / wav.getframerate()
                                    if wav.getframerate() > 0
                                    else 0
                                )
                        else:
                            raise Exception(f"Unexpected content-type: {content_type}")
                    else:
                        error_text = full_response_bytes.decode(
                            "utf-8", errors="ignore"
                        )
                        raise Exception(f"Status {response.status}: {error_text[:200]}")

                    return {
                        "duration": time.perf_counter() - start,
                        "ttft_ms": ttft_ms,
                        "audio_duration": audio_duration,
                    }

            except Exception as e:
                logger.error(f"Request {index} failed: {e}")
                raise

        sample_texts = [sample["text"] for sample in metadata]
        num_batches = (len(sample_texts) + batch_size - 1) // batch_size

        # Run warmup and actual test
        for iteration in range(2):
            all_durations = []
            all_ttft_values = []

            session_timeout = aiohttp.ClientTimeout(total=2000)
            async with aiohttp.ClientSession(
                headers=headers, timeout=session_timeout
            ) as session:
                for batch_idx in range(num_batches):
                    batch_start = batch_idx * batch_size
                    batch_end = min(batch_start + batch_size, len(sample_texts))
                    batch_samples = sample_texts[batch_start:batch_end]

                    tasks = [
                        timed_request(session, batch_start + i + 1, batch_samples[i])
                        for i in range(len(batch_samples))
                    ]
                    batch_results = await asyncio.gather(*tasks)

                    batch_durations, batch_ttft_values = self._extract_batch_metrics(
                        batch_results
                    )
                    all_durations.extend(batch_durations)
                    all_ttft_values.extend(batch_ttft_values)

            if iteration == 0:
                logger.debug("Warm-up completed")
            else:
                metrics = self._calculate_metrics(all_durations, all_ttft_values)
                return {
                    **metrics,
                    "ttft_values": [round(v, 2) for v in all_ttft_values],
                    "total_samples": len(all_durations),
                    "num_batches": num_batches,
                }

    def _cleanup_samples(self) -> None:
        """Clean up downloaded samples directory."""
        dataset_path = Path(DATASET_DIR)
        if dataset_path.exists():
            try:
                shutil.rmtree(dataset_path)
                logger.debug(f"Cleaned up dataset directory: {dataset_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up dataset directory: {e}")
