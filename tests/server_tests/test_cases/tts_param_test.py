# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import asyncio
import logging
import time

import aiohttp
from tests.server_tests.base_test import BaseTest

logger = logging.getLogger(__name__)

default_payload = {
    "text": "Hello, this is a test of the text to speech system.",
    "response_format": "verbose_json",
}

short_text_payload = {
    "text": "Hello world.",
    "response_format": "verbose_json",
}

response_format_audio_payload = {
    "text": "Hello, this is a test of the text to speech system.",
    "response_format": "audio",
}

response_format_wav_payload = {
    "text": "Hello, this is a test of the text to speech system.",
    "response_format": "wav",
}

response_format_mp3_payload = {
    "text": "Hello, this is a test of the text to speech system.",
    "response_format": "mp3",
}

response_format_ogg_payload = {
    "text": "Hello, this is a test of the text to speech system.",
    "response_format": "ogg",
}

headers = {
    "accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": "Bearer your-secret-key",
}


class TTSParamTest(BaseTest):
    async def _run_specific_test_async(self):
        self.url = f"http://localhost:{self.service_port}/audio/speech"

        payloads = [
            default_payload,
            default_payload,
            short_text_payload,
            response_format_audio_payload,
            response_format_wav_payload,
            response_format_mp3_payload,
            response_format_ogg_payload,
        ]

        response_data_list = await self.test_concurrent_tts(payloads)

        logger.info(f"\nReceived {len(response_data_list)} responses")
        same_requests = (
            True if response_data_list[0] == response_data_list[1] else False
        )
        text_length_differs = response_data_list[0] != response_data_list[2]

        return {
            "num_responses": len(response_data_list),
            "same_requests_match": same_requests,
            "text_length_differs": text_length_differs,
            "success": same_requests and text_length_differs,
        }

    async def test_concurrent_tts(self, payloads):
        """
        Test concurrent TTS with a list of payloads.

        Args:
            payloads: List of payload dictionaries to send.

        Returns:
            List of response data dictionaries from each request.
        """

        async def timed_request(session, index, request_payload):
            print(f"Starting request {index}")
            try:
                start = time.perf_counter()
                async with session.post(
                    self.url, json=request_payload, headers=headers
                ) as response:
                    duration = time.perf_counter() - start
                    data = None
                    if response.status == 200:
                        content_type = response.headers.get("content-type", "")
                        if "application/json" in content_type:
                            data = await response.json()
                        elif "audio/" in content_type:
                            audio_bytes = await response.read()
                            data = {"audio_length": len(audio_bytes)}
                        else:
                            data = {"raw": await response.text()}
                    else:
                        print(f"[{index}] Error: Status {response.status}")
                        data = {
                            "error": f"Status {response.status}",
                            "status": response.status,
                        }

                    print(
                        f"[{index}] Status: {response.status}, Time: {duration:.2f}s",
                    )
                    return {
                        "index": index,
                        "duration": duration,
                        "data": data,
                        "status": response.status,
                    }

            except Exception as e:
                duration = time.perf_counter() - start
                print(f"[{index}] Error after {duration:.2f}s: {e}")
                return {
                    "index": index,
                    "duration": duration,
                    "data": None,
                    "error": str(e),
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
                    print("Warm up run done.")
                else:
                    response_data_list = results
                    durations = [r["duration"] for r in results]
                    requests_duration = max(durations)
                    avg_duration = sum(durations) / batch_size

                    print(
                        f"\n🚀 Time taken for individual concurrent requests: {durations}"
                    )
                    print(
                        f"\n🚀 Max time for {batch_size} concurrent requests: {requests_duration:.2f}s"
                    )
                    print(
                        f"\n🚀 Avg time for {batch_size} concurrent requests: {avg_duration:.2f}s"
                    )

        return [
            result["data"]
            for result in sorted(response_data_list, key=lambda x: x["index"])
        ]
