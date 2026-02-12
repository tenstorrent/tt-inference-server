# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import pytest
import aiohttp
import asyncio
import time

API_URL = "http://localhost:9001/image/generations"

payload = {
    "prompt": "Porsche 911 from year 2001 in silver color with 22 inch wheels",
    "output_format": "FILE",
    "num_inference_steps": 20,
}


@pytest.mark.asyncio
async def test_concurrent_image_generation():
    start_total = time.perf_counter()

    async def timed_request(session, index):
        start = time.perf_counter()
        try:
            async with session.post(API_URL, json=payload) as response:
                data = await response.json()
                duration = time.perf_counter() - start
                print(f"[{index}] Status: {response.status}, Time: {duration:.2f}s")
                assert response.status == 200
                assert "image_url" in data or "file_path" in data
        except Exception as e:
            duration = time.perf_counter() - start
            print(f"[{index}] Error after {duration:.2f}s: {e}")
            assert False, f"Request {index} failed"

    async with aiohttp.ClientSession(headers={"accept": "application/json"}) as session:
        tasks = [timed_request(session, i + 1) for i in range(20)]
        await asyncio.gather(*tasks)

    total_duration = time.perf_counter() - start_total
    print(f"\n🚀 Total time for 20 requests: {total_duration:.2f}s")
