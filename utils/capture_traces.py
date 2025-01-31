# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import argparse
import logging

from utils.prompt_configs import EnvironmentConfig
from utils.prompt_client import PromptClient

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def add_cli_args(parser):
    parser.add_argument(
        "--include_images",
        action="store_true",
        help="Include randomly generated images with prompts",
    )
    parser.add_argument(
        "--image_width",
        type=int,
        default=256,
        help="Width of generated images",
    )
    parser.add_argument(
        "--image_height",
        type=int,
        default=256,
        help="Height of generated images",
    )
    return parser


def capture_input_sizes(arg):
    env_config = EnvironmentConfig()
    prompt_client = PromptClient(env_config)
    image_resolutions = []
    if args.include_images:
        image_resolutions = [(args.image_width, args.image_height)]
    prompt_client.capture_traces(image_resolutions=image_resolutions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_cli_args(parser)
    args = parser.parse_args()
    capture_input_sizes(args)
