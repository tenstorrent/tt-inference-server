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

from workflows.model_config import MODEL_CONFIGS
from workflows.workflow_types import DeviceTypes


# Mapping device type to hardware link text
DEVICE_HARDWARE_LINKS = {
    DeviceTypes.T3K: "[TT-LoudBox/TT-QuietBox](https://tenstorrent.com/hardware/tt-quietbox)",
    DeviceTypes.N150: "[n150](https://tenstorrent.com/hardware/wormhole)",
    DeviceTypes.N300: "[n150](https://tenstorrent.com/hardware/wormhole)",
    DeviceTypes.GALAXY: "[Tenstorrent Galaxy](https://tenstorrent.com/hardware/galaxy)",
}


def get_hardware_column(devices: Set[DeviceTypes]) -> str:
    # list smallest available device
    if DeviceTypes.N150 in devices:
        device_link = DEVICE_HARDWARE_LINKS[DeviceTypes.N150]
    elif DeviceTypes.N300 in devices:
        device_link = DEVICE_HARDWARE_LINKS[DeviceTypes.N300]
    elif DeviceTypes.T3K in devices:
        device_link = DEVICE_HARDWARE_LINKS[DeviceTypes.T3K]
    elif DeviceTypes.GALAXY in devices:
        device_link = DEVICE_HARDWARE_LINKS[DeviceTypes.GALAXY]
    return device_link


def generate_markdown_table() -> str:
    header = (
        "| Model Name | Model URL | Hardware | Status | tt-metal commit | vLLM commit | Docker Image |\n"
        "|------------|-----------|----------|--------|-----------------|-------------|--------------|\n"
    )
    rows = []
    for model_name, config in MODEL_CONFIGS.items():
        model_readme_link = f"[{model_name}](vllm-tt-metal-llama3/README.md)"
        model_url = f"[HF Repo](https://huggingface.co/{config.hf_model_repo})"
        hardware = get_hardware_column(config.device_configurations)
        status_str = "✅ ready" if config.status == "ready" else "🔍 preview"
        tt_metal_commit = f"[{config.tt_metal_commit[:16]}]({config.code_link})"
        vllm_commit = f"[{config.vllm_commit[:8]}](https://github.com/tenstorrent/vllm/tree/{config.vllm_commit})"
        ghcr_package, ghcr_tag = config.docker_image.split(":")
        # NOTE: because %2F is used in package name it gets decoded by browser when clinking link
        # best is to link to package root with ghcr.io, cannot link directly to the tag
        docker_image = f"[{ghcr_tag}](https://ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-20.04-amd64)"
        row = f"| {model_readme_link} | {model_url} | {hardware} | {status_str} | {tt_metal_commit} | {vllm_commit} | {docker_image} |"
        rows.append(row)

    markdown_str = header + "\n".join(rows)

    return markdown_str


def main():
    # Generate the markdown table
    markdown_table = generate_markdown_table()
    print(markdown_table)


if __name__ == "__main__":
    main()
