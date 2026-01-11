# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import logging
import time

import aiohttp
from server_tests.base_test import BaseTest

# Set up logging
logger = logging.getLogger(__name__)

# Base payload with default parameters
base_payload = {
    "input": "The quick brown fox jumps over the lazy dog",
    "model": "test-model",  # Model is required but not tested for variations
}

# Payload with same input (duplicate for consistency check)
duplicate_input_payload = {
    "input": "The quick brown fox jumps over the lazy dog",
    "model": "test-model",
}

# Payload with different input text
different_input_payload = {
    "input": "Artificial intelligence and machine learning are transforming technology",
    "model": "test-model",
}

# Payload with dimensions specified
dimensions_payload = {
    "input": "The quick brown fox jumps over the lazy dog",
    "model": "test-model",
    "dimensions": 512,
}

# Payload with different dimensions
different_dimensions_payload = {
    "input": "The quick brown fox jumps over the lazy dog",
    "model": "test-model",
    "dimensions": 256,
}

# Payload with dimensions and different input
dimensions_different_input_payload = {
    "input": "Machine learning models require large amounts of training data",
    "model": "test-model",
    "dimensions": 512,
}

# Payload with longer input text
long_input_payload = {
    "input": "The field of natural language processing has advanced significantly in recent years. "
    "Modern transformer models can understand context, generate coherent text, and perform "
    "complex reasoning tasks. These capabilities have enabled new applications in areas such as "
    "chatbots, translation, summarization, and question answering systems.",
    "model": "test-model",
}

# Payload with short input text
short_input_payload = {
    "input": "Hello world",
    "model": "test-model",
}

# Payload with empty-like input (single character)
single_char_input_payload = {
    "input": "A",
    "model": "test-model",
}

headers = {
    "accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": "Bearer your-secret-key",
}


class EmbeddingParamTest(BaseTest):
    async def _run_specific_test_async(self):
        self.url = f"http://localhost:{self.service_port}/v1/embeddings"
        logger.info(f"Testing embedding parameters at {self.url}")

        # Create list of payloads to test different parameters
        payloads = [
            {"name": "base_default", "payload": base_payload},
            {
                "name": "base_duplicate",
                "payload": duplicate_input_payload,
            },  # Duplicate to verify consistency
            {"name": "different_input", "payload": different_input_payload},
            {"name": "with_dimensions", "payload": dimensions_payload},
            {"name": "different_dimensions", "payload": different_dimensions_payload},
            {
                "name": "dimensions_different_input",
                "payload": dimensions_different_input_payload,
            },
            {"name": "long_input", "payload": long_input_payload},
            {"name": "short_input", "payload": short_input_payload},
            {"name": "single_char_input", "payload": single_char_input_payload},
        ]

        # Get response data from all requests
        response_data_list = await self.test_concurrent_embedding(payloads)

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
        # Different inputs should produce different embeddings
        # Different dimensions should produce embeddings of different sizes
        success = base_match and any(param_tests)
        results["success"] = success

        logger.info(
            f"\n{'✅' if success else '❌'} Test {'PASSED' if success else 'FAILED'}"
        )

        return results

    async def test_concurrent_embedding(self, payloads):
        """
        Test concurrent embedding requests with a list of payloads.

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
                        data = await response.json()
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
