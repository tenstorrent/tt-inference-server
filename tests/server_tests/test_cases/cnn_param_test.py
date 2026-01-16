# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import asyncio
import base64
import logging
import time

import aiohttp
from server_tests.base_test import BaseTest

logger = logging.getLogger(__name__)

# Load base64 image payload
with open("utils/test_payloads/image_client_image_payload", "r") as f:
    image_payload_base64 = f.read()

# Base payload with default parameters (JSON format with base64)
default_json_payload = {
    "prompt": image_payload_base64,
    "response_format": "json",
    "top_k": 3,
    "min_confidence": 70.0,
}

# Payload with verbose_json format
verbose_json_payload = {
    "prompt": image_payload_base64,
    "response_format": "verbose_json",
    "top_k": 3,
    "min_confidence": 70.0,
}

# Payload with different top_k
top_k_5_payload = {
    "prompt": image_payload_base64,
    "response_format": "json",
    "top_k": 5,
    "min_confidence": 70.0,
}

# Payload with different min_confidence
min_confidence_50_payload = {
    "prompt": image_payload_base64,
    "response_format": "json",
    "top_k": 3,
    "min_confidence": 50.0,
}

headers = {
    "accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": "Bearer your-secret-key",
}


class CnnParamTest(BaseTest):
    async def _run_specific_test_async(self):
        self.url = f"http://localhost:{self.service_port}/cnn/search-image"
        logger.info(f"Testing CNN parameters at {self.url}")

        # Create list of payloads to test different parameters
        payloads = [
            {
                "name": "default_json_payload",
                "payload": default_json_payload,
                "is_json": True,
            },
            {
                "name": "duplicate_default",
                "payload": default_json_payload,
                "is_json": True,
            },  # Duplicate to verify consistency
            {
                "name": "verbose_json_payload",
                "payload": verbose_json_payload,
                "is_json": True,
            },
            {"name": "top_k_5_payload", "payload": top_k_5_payload, "is_json": True},
            {
                "name": "min_confidence_50_payload",
                "payload": min_confidence_50_payload,
                "is_json": True,
            },
        ]

        # Get response data from all requests
        response_data_list = await self.test_concurrent_cnn(payloads)

        # Analyze results
        logger.info(f"\n📊 Received {len(response_data_list)} responses")

        results = {"num_responses": len(response_data_list), "tests": {}}

        # Check if same requests produce identical results
        base_match = response_data_list[0]["data"] == response_data_list[1]["data"]
        results["same_requests_match"] = base_match
        logger.info(f"✅ Same requests match: {base_match}")

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
        # Different response_formats should produce different outputs
        # Different top_k values should produce different number of results
        # Different min_confidence should potentially produce different results
        success = base_match and any(param_tests)
        results["success"] = success

        logger.info(
            f"\n{'✅' if success else '❌'} Test {'PASSED' if success else 'FAILED'}"
        )

        return results

    async def test_concurrent_cnn(self, payloads):
        """
        Test concurrent CNN requests with a list of payloads.

        Args:
            payloads: List of dictionaries with 'name', 'payload', and 'is_json' keys.

        Returns:
            List of response data dictionaries from each request.
        """

        async def timed_request(session, index, test_config):
            test_name = test_config["name"]
            request_payload = test_config["payload"]
            is_json = test_config["is_json"]
            logger.info(f"Starting request {index}: {test_name}")
            try:
                start = time.perf_counter()
                if is_json:
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
                else:
                    # Multipart form-data request
                    form_data = aiohttp.FormData()
                    # Decode base64 to bytes for file upload
                    image_bytes = base64.b64decode(request_payload["prompt"])
                    form_data.add_field(
                        "file",
                        image_bytes,
                        filename="test_image.jpg",
                        content_type="image/jpeg",
                    )
                    form_data.add_field(
                        "response_format", request_payload["response_format"]
                    )
                    form_data.add_field("top_k", str(request_payload["top_k"]))
                    form_data.add_field(
                        "min_confidence", str(request_payload["min_confidence"])
                    )

                    multipart_headers = {
                        "accept": "application/json",
                        "Authorization": "Bearer your-secret-key",
                    }

                    async with session.post(
                        self.url, data=form_data, headers=multipart_headers
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
