# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import logging
import time

import aiohttp
from server_tests.base_test import BaseTest

# Import the dataset from the Python file
from utils.test_payloads.audio_payload_30s import dataset

# Set up logging
logger = logging.getLogger(__name__)

# Base payload with default parameters
base_payload = {
    "file": dataset["file"],
    "stream": False,
    "response_format": "verbose_json",
    "is_preprocessing_enabled": True,
    "perform_diarization": False,
    "temperatures": None,
    "compression_ratio_threshold": None,
    "logprob_threshold": None,
    "no_speech_threshold": None,
    "return_timestamps": False,
    "prompt": None,
}

# Payload with different response_format
response_format_text_payload = {
    **base_payload,
    "response_format": "text",
}

# Payload with different response_format (json)
response_format_json_payload = {
    **base_payload,
    "response_format": "json",
}

# Payload with preprocessing disabled
no_preprocessing_payload = {
    **base_payload,
    "is_preprocessing_enabled": False,
}

# Payload with diarization enabled
diarization_payload = {
    **base_payload,
    "perform_diarization": True,
}

# Payload with custom temperatures
temperatures_payload = {
    **base_payload,
    "temperatures": "0.0,0.2,0.4,0.6,0.8",
}

# Payload with compression ratio threshold
compression_ratio_payload = {
    **base_payload,
    "compression_ratio_threshold": 2.4,
}

# Payload with logprob threshold
logprob_threshold_payload = {
    **base_payload,
    "logprob_threshold": -1.0,
}

# Payload with no speech threshold
no_speech_threshold_payload = {
    **base_payload,
    "no_speech_threshold": 0.6,
}

# Payload with timestamps enabled
timestamps_payload = {
    **base_payload,
    "return_timestamps": True,
}

# Payload with custom prompt
prompt_payload = {
    **base_payload,
    "prompt": "This is a conversation about technology and artificial intelligence.",
}

# Payload with multiple parameters changed
combined_params_payload = {
    **base_payload,
    "response_format": "verbose_json",
    "is_preprocessing_enabled": True,
    "return_timestamps": True,
    "temperatures": "0.0,0.2,0.4",
    "prompt": "Technology discussion",
}

headers = {
    "accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": "Bearer your-secret-key",
}


class AudioTranscriptionParamTest(BaseTest):
    async def _run_specific_test_async(self):
        self.url = f"http://localhost:{self.service_port}/audio/transcriptions"
        logger.info(f"Testing audio transcription parameters at {self.url}")

        # Create list of payloads to test different parameters
        payloads = [
            {"name": "base_default", "payload": base_payload},
            {
                "name": "base_duplicate",
                "payload": base_payload,
            },  # Duplicate to verify consistency
            {"name": "response_format_text", "payload": response_format_text_payload},
            {"name": "response_format_json", "payload": response_format_json_payload},
            {"name": "no_preprocessing", "payload": no_preprocessing_payload},
            {"name": "with_diarization", "payload": diarization_payload},
            {"name": "with_temperatures", "payload": temperatures_payload},
            {"name": "with_compression_ratio", "payload": compression_ratio_payload},
            {"name": "with_logprob_threshold", "payload": logprob_threshold_payload},
            {
                "name": "with_no_speech_threshold",
                "payload": no_speech_threshold_payload,
            },
            {"name": "with_timestamps", "payload": timestamps_payload},
            {"name": "with_prompt", "payload": prompt_payload},
            {"name": "combined_params", "payload": combined_params_payload},
        ]

        # Get response data from all requests
        response_data_list = await self.test_concurrent_audio_transcription(payloads)

        # Analyze results
        logger.info(f"\n📊 Received {len(response_data_list)} responses")

        results = {"num_responses": len(response_data_list), "tests": {}}

        # Check if same requests produce identical results
        base_match = response_data_list[0]["data"] == response_data_list[1]["data"]
        results["same_requests_match"] = base_match
        logger.info(f"✓ Same requests match: {base_match}")

        # Check if different parameters produce different results
        param_tests = []
        for i in range(2, len(response_data_list)):
            test_name = payloads[i]["name"]
            differs_from_base = (
                response_data_list[0]["data"] != response_data_list[i]["data"]
            )
            results["tests"][test_name] = {
                "differs_from_base": differs_from_base,
                "status": response_data_list[i]["status"],
                "duration": response_data_list[i]["duration"],
            }
            param_tests.append(differs_from_base)
            logger.info(
                f"  {test_name}: differs={differs_from_base}, status={response_data_list[i]['status']}"
            )

        # Success if base requests match and at least some parameter changes produce different results
        # (some params may not affect output, like preprocessing for same audio)
        success = base_match and any(param_tests)
        results["success"] = success

        logger.info(
            f"\n{'✅' if success else '❌'} Test {'PASSED' if success else 'FAILED'}"
        )

        return results

    async def test_concurrent_audio_transcription(self, payloads):
        """
        Test concurrent audio transcription with a list of payloads.

        Args:
            payloads: List of dictionaries with 'name' and 'payload' keys.

        Returns:
            List of response data dictionaries from each request.
        """

        async def timed_request(session, index, test_config):
            test_name = test_config["name"]
            request_payload = test_config["payload"]
            logger.info(f"Starting request {index}: {test_name}")
            try:
                start = time.perf_counter()
                async with session.post(
                    self.url, json=request_payload, headers=headers
                ) as response:
                    duration = time.perf_counter() - start
                    data = None
                    if response.status == 200:
                        # Handle different response formats
                        content_type = response.headers.get("content-type", "")
                        if "application/json" in content_type:
                            data = await response.json()
                        elif "text/plain" in content_type:
                            data = {"text": await response.text()}
                        else:
                            data = {"raw": await response.text()}
                    else:
                        logger.warning(
                            f"[{index}] {test_name} - Error: Status {response.status}"
                        )
                        data = {
                            "error": f"Status {response.status}",
                            "status": response.status,
                        }

                    logger.info(
                        f"[{index}] {test_name} - Status: {response.status}, Time: {duration:.2f}s"
                    )
                    return {
                        "index": index,
                        "name": test_name,
                        "duration": duration,
                        "data": data,
                        "status": response.status,
                    }

            except Exception as e:
                duration = time.perf_counter() - start
                logger.error(
                    f"[{index}] {test_name} - Error after {duration:.2f}s: {e}"
                )
                return {
                    "index": index,
                    "name": test_name,
                    "duration": duration,
                    "data": None,
                    "error": str(e),
                    "status": 0,
                }

        batch_size = len(payloads)
        response_data_list = []

        for iteration in range(2):
            session_timeout = aiohttp.ClientTimeout(total=2000)
            async with aiohttp.ClientSession(
                headers=headers, timeout=session_timeout
            ) as session:
                tasks = [
                    timed_request(session, i + 1, payloads[i])
                    for i in range(batch_size)
                ]
                results = await asyncio.gather(*tasks)

                if iteration == 0:
                    logger.info("🔥 Warm up run done.")
                else:
                    # Second iteration - collect the actual data
                    response_data_list = results
                    durations = [r["duration"] for r in results]
                    requests_duration = max(durations)
                    avg_duration = sum(durations) / batch_size

                    logger.info(
                        f"\n🚀 Time taken for individual concurrent requests: {durations}"
                    )
                    logger.info(
                        f"\n🚀 Max time for {batch_size} concurrent requests: {requests_duration:.2f}s"
                    )
                    logger.info(
                        f"\n🚀 Avg time for {batch_size} concurrent requests: {avg_duration:.2f}s"
                    )

        # Return list of response data in the same order as input payloads
        return sorted(response_data_list, key=lambda x: x["index"])
