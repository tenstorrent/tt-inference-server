# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import os
import time
import logging
import requests
from utils.prompt_client_cli import (
    get_authorization,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_api_health_url():
    DEPLOY_URL = os.getenv("DEPLOY_URL", "http://127.0.0.1")
    health_url = f"{DEPLOY_URL}:{os.getenv('SERVICE_PORT', '8000')}/health"
    return health_url


def wait_for_healthy(base_url: str, timeout: int = 300, interval: int = 10) -> bool:
    """
    Check the health endpoint until the service is ready.
    """
    health_url = get_api_health_url()
    start_time = time.time()
    headers = {"Authorization": f"Bearer {get_authorization()}"}
    total_time_waited = 0
    while time.time() - start_time < timeout:
        try:
            response = requests.get(health_url, headers=headers, timeout=interval)
            if response.status_code == 200:
                startup_time = time.time() - start_time
                logger.info(
                    f"vLLM service is healthy. startup_time:= {startup_time} seconds"
                )
                return True
        except requests.exceptions.RequestException as e:
            logger.warning(f"Health check failed: {e}")

        total_time_waited += interval
        logger.info(
            f"Service not ready after {total_time_waited} seconds, waiting {interval} seconds before polling ..."
        )
        time.sleep(0.05)

    logger.error(f"Service did not become healthy within {timeout} seconds")
    return False
