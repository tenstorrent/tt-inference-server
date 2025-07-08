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

from workflows.model_config import config_list
from workflows.workflow_types import DeviceTypes, ModelStatusTypes



# Mapping device type to hardware link text
DEVICE_HARDWARE_LINKS = {
    DeviceTypes.T3K: "[TT-QuietBox](https://tenstorrent.com/hardware/tt-quietbox)/[TT-LoudBox](https://tenstorrent.com/hardware/tt-loudbox) (T3K)",
    DeviceTypes.N150: "[n150](https://tenstorrent.com/hardware/wormhole)",
    DeviceTypes.N300: "[n300](https://tenstorrent.com/hardware/wormhole)",
    DeviceTypes.GALAXY: "[Galaxy](https://tenstorrent.com/hardware/galaxy)",
    DeviceTypes.P100: "[p100](https://tenstorrent.com/hardware/blackhole)",
    DeviceTypes.P150: "[p150](https://tenstorrent.com/hardware/blackhole)",
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
<<<<<<< HEAD

    # Group configs by model name since each config now represents a single device
    model_groups = {}
    for model_id, config in MODEL_CONFIGS.items():
        model_name = config.model_name
        if model_name not in model_groups:
            model_groups[model_name] = []
        model_groups[model_name].append(config)

    for model_name, configs in model_groups.items():
        # Use the first config for shared properties
        config = configs[0]
        model_readme_link = f"[{model_name}](vllm-tt-metal-llama3/README.md)"
        model_url = f"[HF Repo](https://huggingface.co/{config.hf_model_repo})"

        # Collect all device types for this model
        device_types = {config.device_type for config in configs}
        hardware = get_hardware_column(device_types)

=======
    for config in config_list:
        # Create a descriptive model architecture name
        model_arch_name = config.weights[0].split("/")[-1]
        default_hardware = {device for device, is_default in config.default_impl_map.items() if is_default}
        hardware = get_hardware_str(default_hardware)
        if not hardware:
            continue

        # Create multiple HF repo weight links from the weights list
        model_weights = []
        for weight in config.weights:
            model_weights.append(f"[{Path(weight).name}](https://huggingface.co/{weight})")
        model_weights_str = "<br/>".join(model_weights)
        
>>>>>>> a923fd1 (update model statuses with new Enum class, only show default model implementations in README.md, one line per model implementation)
        status_str = get_status_str(config.status)
        tt_metal_commit = f"[{config.tt_metal_commit[:16]}]({config.code_link})"
        vllm_commit = f"[{config.vllm_commit[:8]}](https://github.com/tenstorrent/vllm/tree/{config.vllm_commit})"
        ghcr_package, ghcr_tag = config.docker_image.split(":")
        # NOTE: because %2F is used in package name it gets decoded by browser when clinking link
        # best is to link to package root with ghcr.io, cannot link directly to the tag
        docker_image = f"[{ghcr_tag}](https://ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-20.04-amd64)"
        row = f"| {model_weights_str} | {hardware} | {status_str} | {tt_metal_commit} | {vllm_commit} | {docker_image} |"
        rows.append(row)

    markdown_str = header + "\n".join(rows)

    return markdown_str


def main():
    # Generate the markdown table
    markdown_table = generate_markdown_table()
    print(markdown_table)


if __name__ == "__main__":
    main()
