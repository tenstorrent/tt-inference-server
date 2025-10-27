import aiohttp

from tests.server_tests.base_test import BaseTest


class MediaServerLivenessTest(BaseTest):
    async def _run_specific_test_async(self, targets=None):
        url = f"http://localhost:{self.service_port}/tt-liveness"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                assert response.status == 200, f"Expected status 200, got {response.status}"
                data = await response.json()
                print(f"Liveness check response: {data}")
                return True