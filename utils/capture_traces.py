# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import logging

from utils.prompt_configs import EnvironmentConfig
from utils.prompt_client import PromptClient

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def capture_input_sizes():
    env_config = EnvironmentConfig()
    prompt_client = PromptClient(env_config)
    prompt_client.capture_traces()


if __name__ == "__main__":
    capture_input_sizes()
