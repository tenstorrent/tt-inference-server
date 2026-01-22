# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import asyncio
import json
import logging
import time

import aiohttp
from server_tests.base_test import BaseTest

logger = logging.getLogger(__name__)

# Constants
ACCURACY_REFERENCE_PATH = "evals/eval_targets/model_accuracy_reference.json"

# Base payload with default parameters
default_payload = {
    "prompt": "A beautiful sunset over a mountain landscape with vibrant colors, cinematic quality, smooth camera movement",
    "negative_prompt": "blurry, low quality, distorted, shaky",
    "num_inference_steps": 40,
    "seed": 42,
}

# Payload with different num_inference_steps
inference_steps_15_payload = {
    "prompt": "A beautiful sunset over a mountain landscape with vibrant colors, cinematic quality, smooth camera movement",
    "negative_prompt": "blurry, low quality, distorted, shaky",
    "num_inference_steps": 15,
    "seed": 42,
}

# Payload with different seed
seed_123_payload = {
    "prompt": "A beautiful sunset over a mountain landscape with vibrant colors, cinematic quality, smooth camera movement",
    "negative_prompt": "blurry, low quality, distorted, shaky",
    "num_inference_steps": 20,
    "seed": 123,
}

# Payload without negative_prompt
no_negative_prompt_payload = {
    "prompt": "A beautiful sunset over a mountain landscape with vibrant colors, cinematic quality, smooth camera movement",
    "num_inference_steps": 20,
    "seed": 42,
}

# Payload with different prompt
different_prompt_payload = {
    "prompt": "A serene ocean wave crashing on a tropical beach at dawn, crystal clear water, 4K quality",
    "negative_prompt": "blurry, low quality, distorted, shaky",
    "num_inference_steps": 20,
    "seed": 42,
}

headers = {
    "accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": "Bearer your-secret-key",
}


class VideoGenerationParamTest(BaseTest):
    async def _run_specific_test_async(self):
        self.url = f"http://localhost:{self.service_port}/video/generations"
        logger.info(f"Testing video generation parameters at {self.url}")

        # Determine model name and get appropriate num_inference_steps
        model_name = self.config.get("model", "test-model")
        default_steps = self._get_num_inference_steps_from_reference(model_name, 40)
        logger.info(f"Using num_inference_steps={default_steps} for model={model_name}")

        # Update payloads with model-specific default steps
        default_payload["num_inference_steps"] = default_steps
        seed_123_payload["num_inference_steps"] = default_steps
        no_negative_prompt_payload["num_inference_steps"] = default_steps
        different_prompt_payload["num_inference_steps"] = default_steps

        # Create list of payloads to test different parameters
        payloads = [
            {"name": "default_payload", "payload": default_payload},
            {
                "name": "duplicate_default",
                "payload": default_payload,
            },  # Duplicate to verify consistency
            {
                "name": "inference_steps_15_payload",
                "payload": inference_steps_15_payload,
            },
            {"name": "seed_123_payload", "payload": seed_123_payload},
            {
                "name": "no_negative_prompt_payload",
                "payload": no_negative_prompt_payload,
            },
            {"name": "different_prompt_payload", "payload": different_prompt_payload},
        ]

        # Get response data from all requests
        response_data_list = await self.test_concurrent_video_generation(payloads)

        # Analyze results
        logger.info(f"\n📊 Received {len(response_data_list)} responses")

        results = {"num_responses": len(response_data_list), "tests": {}}

        # Check if same requests produce identical results
        base_match = response_data_list[0]["job_id"] == response_data_list[1]["job_id"]
        results["same_requests_match"] = base_match
        logger.info(f"✅ Same requests produce same job behavior: {base_match}")

        # Check if different parameters produce different results
        param_tests = []
        for i in range(2, len(response_data_list)):
            test_name = payloads[i]["name"]
            differs_from_base = (
                response_data_list[0]["job_id"] != response_data_list[i]["job_id"]
            )
            results["tests"][test_name] = {
                "differs_from_base": differs_from_base,
                "status": response_data_list[i]["status"],
                "duration": response_data_list[i]["duration"],
            }
            param_tests.append(differs_from_base)
            logger.info(
                f"  {test_name}: differs={differs_from_base}, status={response_data_list[i]['status']}, duration={response_data_list[i]['duration']:.2f}s"
            )

        # Success if:
        # - Base requests are handled consistently (both succeed)
        # - Different parameters produce different job IDs (each request is unique)
        # - All requests succeed with status 202 (Accepted)
        all_succeeded = all(r["status"] == 202 for r in response_data_list)
        success = all_succeeded and all(param_tests)
        results["success"] = success

        logger.info(
            f"\n{'✅' if success else '❌'} Test {'PASSED' if success else 'FAILED'}"
        )

        return results

    async def test_concurrent_video_generation(self, payloads):
        """
        Test concurrent video generation requests with a list of payloads.

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
                    if response.status == 202:
                        data = await response.json()
                        job_id = data.get("id")
                    else:
                        logger.warning(
                            f"[{index}] {test_name} - Error: Status {response.status}"
                        )
                        data = {
                            "error": f"Status {response.status}",
                            "status": response.status,
                        }
                        job_id = None

                    logger.info(
                        f"[{index}] {test_name} - Status: {response.status}, Job ID: {job_id}, Time: {duration:.2f}s"
                    )
                    return {
                        "index": index,
                        "name": test_name,
                        "duration": duration,
                        "data": data,
                        "job_id": job_id,
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
                    "job_id": None,
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

    def _load_accuracy_reference(self) -> dict:
        """Load accuracy reference data from JSON file."""
        logger.info(f"Loading accuracy reference from: {ACCURACY_REFERENCE_PATH}")
        try:
            with open(ACCURACY_REFERENCE_PATH, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(
                f"Accuracy reference file not found: {ACCURACY_REFERENCE_PATH}, using defaults"
            )
            return {}
        except json.JSONDecodeError as e:
            logger.warning(
                f"Invalid JSON in accuracy reference file: {e}, using defaults"
            )
            return {}

    def _get_num_inference_steps_from_reference(
        self, model_name: str, default: int
    ) -> int:
        """Get num_inference_steps from reference data for a given model."""
        reference_data = self._load_accuracy_reference()
        if model_name in reference_data:
            num_steps = reference_data[model_name].get("num_inference_steps")
            if num_steps:
                return num_steps
        return default
