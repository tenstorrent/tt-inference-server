# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import sys
from pathlib import Path
from typing import Set

# Add the script's directory to the Python path
# this for 0 setup python setup script
project_root = Path(__file__).resolve().parent.parent
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))

from workflows.model_config import config_templates, generate_docker_tag, VERSION
from workflows.workflow_types import DeviceTypes, ModelStatusTypes



# Mapping device type to hardware link text
DEVICE_HARDWARE_LINKS = {
    DeviceTypes.T3K: "[WH-QuietBox](https://tenstorrent.com/hardware/tt-quietbox)/[WH-LoudBox](https://tenstorrent.com/hardware/tt-loudbox) (T3K)",
    DeviceTypes.N150: "[n150](https://tenstorrent.com/hardware/wormhole)",
    DeviceTypes.N300: "[n300](https://tenstorrent.com/hardware/wormhole)",
    DeviceTypes.GALAXY: "[Galaxy](https://tenstorrent.com/hardware/galaxy)",
    DeviceTypes.P100: "[p100](https://tenstorrent.com/hardware/blackhole)",
    DeviceTypes.P150: "[p150](https://tenstorrent.com/hardware/blackhole)",
    DeviceTypes.P150X4: "[BH-QuietBox](https://tenstorrent.com/hardware/tt-quietbox) (P150X4)",
}


def get_hardware_str(devices: Set[DeviceTypes]) -> str:
    device_links = []
    for d in devices:
        link = DEVICE_HARDWARE_LINKS.get(d)
        if link:
            device_links.append(link)

    hardware_str = ", ".join(device_links)
    return hardware_str


def get_status_str(status):
    return ModelStatusTypes.to_display_string(status)


def generate_markdown_table() -> str:
    header = (
        "| Model Weights | Hardware | Status | tt-metal commit | vLLM commit | Docker Image |\n"
        "|---------------|----------|--------|-----------------|-------------|--------------|\n"
    )
    rows = []

    for config in config_templates:
        try:
            # Create a descriptive model architecture name
            default_hardware = {
                device 
                for device, dev_spec in config.device_model_spec_map.items()
                if dev_spec.default_impl
            }
            hardware = get_hardware_str(default_hardware)
            if not hardware:
                continue

            # Create multiple HF repo weight links from the weights list
            model_weights = []
            for weight in config.weights:
                model_weights.append(f"[{Path(weight).name}](https://huggingface.co/{weight})")
            model_weights_str = "<br/>".join(model_weights)

            status_str = get_status_str(config.status)
            # Generate code link directly since ModelConfigTemplate doesn't have code_link
            code_link = f"{config.impl.repo_url}/tree/{config.tt_metal_commit}/{config.impl.code_path}"
            tt_metal_commit = f"[{config.tt_metal_commit[:16]}]({code_link})"
            vllm_commit = f"[{config.vllm_commit[:8]}](https://github.com/tenstorrent/vllm/tree/{config.vllm_commit})"
            
            # Handle docker_image which might be None for templates
            if config.docker_image:
                _, ghcr_tag = config.docker_image.split(":")
            else:
                # Generate default docker image like ModelConfig does
                ghcr_tag = generate_docker_tag(VERSION, config.tt_metal_commit, config.vllm_commit)
            
            # NOTE: because %2F is used in package name it gets decoded by browser when clinking link
            # best is to link to package root with ghcr.io, cannot link directly to the tag
            docker_image = f"[{ghcr_tag}](https://ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64)"
            row = f"| {model_weights_str} | {hardware} | {status_str} | {tt_metal_commit} | {vllm_commit} | {docker_image} |"
            rows.append(row)
        except Exception as e:
            print(f"Error processing ModelConfigTemplate: {config}", file=sys.stderr)
            raise e

    markdown_str = header + "\n".join(rows)

    return markdown_str


def main():
    # Generate the markdown table
    markdown_table = generate_markdown_table()
    print(markdown_table)


if __name__ == "__main__":
    main()
