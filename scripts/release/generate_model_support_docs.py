#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Generate Model Support documentation from MODEL_SPECS.

This script generates a documentation system for model support including:
- Model type table pages (llm/README.md, vlm/README.md, etc.)
- Models by hardware page (models_by_hardware.md)
- Combined model+page-group pages (e.g., Llama-3.1-8B_galaxy.md contains both GALAXY and GALAXY_T3K)

Pages are organized by HardwarePageGroup, which groups related devices together.
For example, GALAXY and GALAXY_T3K share a single page with sections for each device.

Usage:
    python scripts/release/generate_model_support_docs.py
    python scripts/release/generate_model_support_docs.py --dry-run
    python scripts/release/generate_model_support_docs.py --output-dir docs/model_support
"""

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add repo root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from workflows.model_spec import (
    VERSION,
    ModelSpecTemplate,
    generate_default_docker_link,
    model_weights_to_model_name,
    spec_templates,
)
from workflows.workflow_types import (
    DeviceTypes,
    InferenceEngine,
    ModelStatusTypes,
    ModelType,
)


@dataclass(frozen=True)
class HardwarePageGroup:
    """Configuration for a hardware documentation page group.

    Attributes:
        name: Display name used as header in hardware columns for tables
        device_ordering: Tuple of DeviceTypes that appear on this page, in display order
    """

    name: str
    device_ordering: Tuple[DeviceTypes, ...]

    @classmethod
    def from_device_type(cls, device_type: DeviceTypes) -> "HardwarePageGroup":
        return cls(name=device_type.to_product_str(), device_ordering=(device_type,))


# Mapping inference engine to documentation link (reused from update_model_spec.py)
INFERENCE_ENGINE_README_LINKS = {
    InferenceEngine.VLLM.value: "../../../vllm-tt-metal/README.md",
    InferenceEngine.MEDIA.value: "../../../tt-media-server/README.md",
    InferenceEngine.FORGE.value: "../../../tt-media-server/README.md",
}

# Mapping device type to hardware link text (reused from update_model_spec.py)
DEVICE_HARDWARE_LINKS = {
    DeviceTypes.T3K: "https://tenstorrent.com/hardware/tt-loudbox",
    DeviceTypes.N150: "https://tenstorrent.com/hardware/wormhole",
    DeviceTypes.N300: "https://tenstorrent.com/hardware/wormhole",
    DeviceTypes.GALAXY: "https://tenstorrent.com/hardware/galaxy",
    DeviceTypes.GALAXY_T3K: "https://tenstorrent.com/hardware/galaxy",
    DeviceTypes.DUAL_GALAXY: "https://tenstorrent.com/hardware/galaxy",
    DeviceTypes.QUAD_GALAXY: "https://tenstorrent.com/hardware/galaxy",
    DeviceTypes.P100: "https://tenstorrent.com/hardware/blackhole",
    DeviceTypes.P150: "https://tenstorrent.com/hardware/blackhole",
    DeviceTypes.P150X4: "https://tenstorrent.com/hardware/tt-quietbox",
    DeviceTypes.P150X8: "https://tenstorrent.com/hardware/tt-loudbox",
    DeviceTypes.BLACKHOLE_GALAXY: "https://tenstorrent.com/hardware/galaxy",
}

# Shared instances for devices that map to the same page
_GALAXY_PAGE_GROUP = HardwarePageGroup(
    name=DeviceTypes.GALAXY.to_product_str(),
    device_ordering=(DeviceTypes.GALAXY, DeviceTypes.GALAXY_T3K),
)
_WH_SINGLE_CARD_PAGE_GROUP = HardwarePageGroup(
    name="N150/N300",
    device_ordering=(DeviceTypes.N150, DeviceTypes.N300),
)
_BH_SINGLE_CARD_PAGE_GROUP = HardwarePageGroup(
    name="P100/P150",
    device_ordering=(DeviceTypes.P100, DeviceTypes.P150),
)
_DUAL_GALAXY_PAGE_GROUP = HardwarePageGroup.from_device_type(DeviceTypes.DUAL_GALAXY)
_QUAD_GALAXY_PAGE_GROUP = HardwarePageGroup.from_device_type(DeviceTypes.QUAD_GALAXY)

# Maps DeviceTypes to their hardware page group configuration
# - Key: DeviceType
# - Value: HardwarePageGroup with name (for headers) and device_ordering (page layout)
# Multiple DeviceTypes can map to the same HardwarePageGroup instance
DEVICE_HARDWARE_PAGE_GROUPS_MAPPING: Dict[DeviceTypes, HardwarePageGroup] = {
    DeviceTypes.DUAL_GALAXY: _DUAL_GALAXY_PAGE_GROUP,
    DeviceTypes.QUAD_GALAXY: _QUAD_GALAXY_PAGE_GROUP,
    DeviceTypes.GALAXY: _GALAXY_PAGE_GROUP,
    DeviceTypes.GALAXY_T3K: _GALAXY_PAGE_GROUP,
    DeviceTypes.BLACKHOLE_GALAXY: HardwarePageGroup.from_device_type(
        DeviceTypes.BLACKHOLE_GALAXY
    ),
    DeviceTypes.P150X8: HardwarePageGroup.from_device_type(DeviceTypes.P150X8),
    DeviceTypes.P150X4: HardwarePageGroup.from_device_type(DeviceTypes.P150X4),
    DeviceTypes.P150: _BH_SINGLE_CARD_PAGE_GROUP,
    DeviceTypes.P100: _BH_SINGLE_CARD_PAGE_GROUP,
    DeviceTypes.T3K: HardwarePageGroup.from_device_type(DeviceTypes.T3K),
    DeviceTypes.N300: _WH_SINGLE_CARD_PAGE_GROUP,
    DeviceTypes.N150: _WH_SINGLE_CARD_PAGE_GROUP,
}

UNIQUE_DEVICE_PAGE_GROUPS = {
    v.device_ordering[0] for v in DEVICE_HARDWARE_PAGE_GROUPS_MAPPING.values()
}

# Model type descriptions for directory page
MODEL_TYPE_DESCRIPTIONS = {
    ModelType.LLM: "Large Language Models",
    ModelType.VLM: "Vision-Language Models",
    ModelType.AUDIO: "Speech-to-text models",
    ModelType.IMAGE: "Image generation models",
    ModelType.CNN: "Convolutional Neural Networks",
    ModelType.EMBEDDING: "Text embedding models",
    ModelType.TEXT_TO_SPEECH: "Text-to-speech models",
    ModelType.VIDEO: "Video generation models",
}

# Status groups for supported models (excludes EXPERIMENTAL)
SUPPORTED_STATUS_GROUPS = (
    ModelStatusTypes.TOP_PERF,
    ModelStatusTypes.COMPLETE,
    ModelStatusTypes.FUNCTIONAL,
)

# Devices to exclude from model+device page generation
EXCLUDED_DEVICES = (DeviceTypes.CPU, DeviceTypes.GPU)

# Devices to exclude from the Models by Hardware page
MODELS_BY_HARDWARE_EXCLUDED_DEVICES = (DeviceTypes.GALAXY_T3K,)

# Highlighted models to pin at top of model type tables
# These models will appear first in their respective tables, in the order specified
MODEL_HIGHLIGHTS: Dict[ModelType, Tuple[str, ...]] = {
    ModelType.LLM: (
        "gpt-oss-120b",  # Maps to weights: openai/gpt-oss-120b
        "gpt-oss-20b",  # Maps to weights: openai/gpt-oss-20b
        "Llama-3.3-70B-Instruct",  # Maps to weights: meta-llama/Llama-3.3-70B-Instruct
        "Qwen3-32B",  # Maps to weights: Qwen/Qwen3-32B
        "Llama-3.1-8B",  # Maps to weights: meta-llama/Llama-3.1-8B
    ),
    # Add other model types as needed:
    # ModelType.VLM: (...),
    # ModelType.AUDIO: (...),
}


def get_unique_hardware_page_groups() -> List[HardwarePageGroup]:
    """Get unique HardwarePageGroup instances in consistent order."""
    seen = set()
    groups = []
    for device in DEVICE_HARDWARE_PAGE_GROUPS_MAPPING:
        group = DEVICE_HARDWARE_PAGE_GROUPS_MAPPING[device]
        if id(group) not in seen:
            seen.add(id(group))
            groups.append(group)
    return groups


def is_primary_device(device: DeviceTypes) -> bool:
    """Check if device is the primary (first) in its page group."""
    group = DEVICE_HARDWARE_PAGE_GROUPS_MAPPING.get(device)
    return group is not None and group.device_ordering[0] == device


def sanitize_filename(name: str) -> str:
    """Convert a model name to a safe filename."""
    # Replace characters that are problematic in filenames
    safe_name = re.sub(r'[<>:"/\\|?*]', "-", name)
    # Replace multiple dashes with single dash
    safe_name = re.sub(r"-+", "-", safe_name)
    return safe_name


def generate_section_anchor(section_title: str) -> str:
    """
    Generate a GitHub-flavored markdown anchor from a section title.
    GitHub converts headers to anchors by:
    - Converting to lowercase
    - Replacing spaces with hyphens
    - Removing special characters except hyphens
    """
    anchor = section_title.lower()
    # Replace spaces with hyphens
    anchor = anchor.replace(" ", "-")
    # Remove special characters (keep alphanumeric, hyphens, underscores)
    anchor = re.sub(r"[^a-z0-9\-_]", "", anchor)
    # Replace multiple hyphens with single hyphen
    anchor = re.sub(r"-+", "-", anchor)
    return anchor


def get_model_display_name(template: ModelSpecTemplate) -> str:
    """Get the display name for a model template."""
    return model_weights_to_model_name(template.weights[0])


def get_model_device_filename(model_name: str, device: DeviceTypes) -> str:
    """Get the markdown filename for a model+device page."""
    return f"{sanitize_filename(model_name)}_{device.name.lower()}.md"


def get_first_device_for_model(
    model_templates: List[ModelSpecTemplate],
) -> DeviceTypes:
    """
    Get the first device from a model's templates' DeviceSpec list.
    Used for generating model name links in tables.
    """
    for template in model_templates:
        if template.device_model_specs:
            return template.device_model_specs[0].device
    # Fallback (should not happen with valid templates)
    return DeviceTypes.N150


def get_devices_for_model_type(
    templates: List[ModelSpecTemplate], model_type: ModelType
) -> Set[DeviceTypes]:
    """Get all devices that have at least one model of the given type."""
    devices = set()
    for template in templates:
        if template.model_type != model_type:
            continue
        for dev_spec in template.device_model_specs:
            devices.add(dev_spec.device)
    return devices


def get_model_page_group_filename(model_name: str, group: HardwarePageGroup) -> str:
    """Get the markdown filename for a model's page group page.

    Uses the primary device (first in device_ordering) for the filename.
    """
    primary_device = group.device_ordering[0]
    return f"{sanitize_filename(model_name)}_{primary_device.name.lower()}.md"


def get_page_groups_for_model_type(
    templates: List[ModelSpecTemplate], model_type: ModelType
) -> List[HardwarePageGroup]:
    """Get page groups that have at least one model of the given type."""
    devices_set = get_devices_for_model_type(templates, model_type)
    groups = []
    for group in get_unique_hardware_page_groups():
        if any(d in devices_set for d in group.device_ordering):
            groups.append(group)
    return groups


def get_page_group_status_for_model(
    model_templates: List[ModelSpecTemplate], group: HardwarePageGroup
) -> Optional[ModelStatusTypes]:
    """Get the best status for a model across all devices in the page group.

    Returns max(ModelStatusTypes) or None if not supported on any device.
    """
    statuses = []
    for device in group.device_ordering:
        for template in model_templates:
            for dev_spec in template.device_model_specs:
                if dev_spec.device == device:
                    statuses.append(template.status)
    return max(statuses) if statuses else None


def get_page_group_status_link(
    model_name: str,
    model_templates: List[ModelSpecTemplate],
    group: HardwarePageGroup,
) -> str:
    """Get markdown link for model status on a page group.

    Uses max(ModelStatusTypes) for display. Links to the combined page group page
    (e.g., _galaxy.md), NOT individual device pages.
    """
    best_status = get_page_group_status_for_model(model_templates, group)
    if best_status is None:
        return "-"
    filename = get_model_page_group_filename(model_name, group)
    return f"[{best_status.display_string}]({filename})"


def group_templates_by_model(
    templates: List[ModelSpecTemplate],
) -> Dict[str, List[ModelSpecTemplate]]:
    """
    Group templates by their model display name.
    Multiple templates may exist for the same model targeting different devices.
    """
    groups = defaultdict(list)
    for template in templates:
        display_name = get_model_display_name(template)
        groups[display_name].append(template)
    return dict(groups)


def get_device_status_for_template(
    template: ModelSpecTemplate, device: DeviceTypes
) -> str:
    """Get the status string for a template on a specific device."""
    for dev_spec in template.device_model_specs:
        if dev_spec.device == device:
            return template.status.display_string
    return "-"


def get_device_status_for_model(
    model_templates: List[ModelSpecTemplate], device: DeviceTypes
) -> str:
    """Get the status string for a model (from multiple templates) on a specific device."""
    for template in model_templates:
        for dev_spec in template.device_model_specs:
            if dev_spec.device == device:
                return template.status.display_string
    return "-"


def get_device_status_enum_for_model(
    model_templates: List[ModelSpecTemplate], device: DeviceTypes
) -> Optional[ModelStatusTypes]:
    """Get the ModelStatusTypes enum for a model on a specific device, or None if not supported."""
    for template in model_templates:
        for dev_spec in template.device_model_specs:
            if dev_spec.device == device:
                return template.status
    return None


def get_best_status_for_model(
    model_templates: List[ModelSpecTemplate],
) -> ModelStatusTypes:
    """Get the best (highest) status across all templates for a model."""
    statuses = [t.status for t in model_templates]
    return max(statuses) if statuses else ModelStatusTypes.EXPERIMENTAL


def get_model_subdir(model_type: ModelType) -> str:
    """Get the subdirectory name for a model type."""
    return model_type.short_name.lower()


def get_device_status_link(
    model_name: str,
    model_templates: List[ModelSpecTemplate],
    device: DeviceTypes,
) -> str:
    """
    Get a markdown link for the status of a model on a specific device.
    Links to the model+device page directly.
    Returns "-" if the device is not supported.
    """
    status = get_device_status_for_model(model_templates, device)
    if status == "-":
        return "-"

    # Generate link to the model+device page (same directory as README.md)
    filename = get_model_device_filename(model_name, device)

    return f"[{status}]({filename})"


def get_model_type_for_templates(templates: List[ModelSpecTemplate]) -> ModelType:
    """Get the model type from a list of templates (assumes all have same type)."""
    for template in templates:
        if template.model_type:
            return template.model_type
    return ModelType.LLM  # Default fallback


def get_all_devices_for_model(
    templates: List[ModelSpecTemplate],
) -> List[DeviceTypes]:
    """
    Get all devices supported by a model from its templates.
    Returns devices ordered by page group.
    """
    seen_devices = set()
    for template in templates:
        for dev_spec in template.device_model_specs:
            seen_devices.add(dev_spec.device)

    # Return in page group order
    ordered_devices = []
    for group in get_unique_hardware_page_groups():
        for device in group.device_ordering:
            if device in seen_devices:
                ordered_devices.append(device)

    # Add any remaining devices (CPU, GPU, etc.)
    for device in seen_devices:
        if device not in ordered_devices:
            ordered_devices.append(device)

    return ordered_devices


def generate_model_page_group_page(
    model_name: str,
    templates: List[ModelSpecTemplate],
    group: HardwarePageGroup,
) -> str:
    """
    Generate markdown content for a model's page group page.

    Contains sections for each device in group.device_ordering that supports the model.
    For example, _galaxy.md may contain:
    - Main content for GALAXY
    - Appended section for GALAXY_T3K (if supported)
    """
    lines = []

    # Get model type for back link
    model_type = get_model_type_for_templates(templates)

    # Collect devices in this group that support the model
    supported_devices_in_group = []
    device_template_map = {}  # device -> (template, dev_spec)
    for device in group.device_ordering:
        for template in templates:
            for dev_spec in template.device_model_specs:
                if dev_spec.device == device:
                    supported_devices_in_group.append(device)
                    device_template_map[device] = (template, dev_spec)
                    break
            if device in device_template_map:
                break

    if not supported_devices_in_group:
        return f"# {model_name} - No devices found in group {group.name}\n"

    # Use first supported device for common sections
    first_device = supported_devices_in_group[0]
    first_template, first_dev_spec = device_template_map[first_device]

    # Page title with group name
    lines.append(f"# {model_name} Tenstorrent Support on {group.name}")
    lines.append("")

    # Supported weights (only show if multiple weights are supported)
    if first_template.weights and len(first_template.weights) > 1:
        default_weights = first_template.weights[0]
        default_weights_name = default_weights.split("/")[-1]

        lines.append("Supported weights variants for this model implementation are:")
        lines.append("")
        for idx, weight in enumerate(first_template.weights):
            weight_name = weight.split("/")[-1]
            if idx == 0:
                lines.append(
                    f"- `{weight_name}`: [{weight}](https://huggingface.co/{weight}) **(default)** "
                )
            else:
                lines.append(
                    f"- `{weight_name}`: [{weight}](https://huggingface.co/{weight})"
                )

        lines.append("")
        lines.append(
            f"To use non-default weights, replace `{default_weights_name}` in commands below."
        )
        lines.append("")

    # Useful links section
    lines.append("#### Useful links")
    lines.append("")
    lines.append(
        f"- [{group.name} details]({DEVICE_HARDWARE_LINKS[group.device_ordering[0]]})"
    )
    lines.append(
        f"- [Search other {model_type.short_name.lower()} models](./README.md)"
    )
    lines.append(
        "- [Search other models by model type](../../../README.md#models-by-model-type)"
    )
    lines.append("")

    # "Also supported on" section - links to OTHER page groups only
    model_devices = set()
    for template in templates:
        for dev_spec in template.device_model_specs:
            model_devices.add(dev_spec.device)

    other_page_groups = []
    for other_group in get_unique_hardware_page_groups():
        if id(other_group) == id(group):
            continue
        for device in other_group.device_ordering:
            if device in model_devices and device not in EXCLUDED_DEVICES:
                other_page_groups.append(other_group)
                break

    if other_page_groups:
        lines.append(f"`{model_name}` is also supported on hardware:")
        lines.append("")
        for other_group in other_page_groups:
            other_filename = get_model_page_group_filename(model_name, other_group)
            lines.append(f"- [{other_group.name}]({other_filename})")
        lines.append("")

    # Generate sections for each supported device in the group
    for idx, device in enumerate(supported_devices_in_group):
        target_template, target_dev_spec = device_template_map[device]
        product_name = device.to_product_str()

        # Section header for secondary devices
        if idx > 0:
            lines.append("---")
            lines.append("")
            lines.append(f"## {device.name} Configuration")
            lines.append("")

        # Quickstart section
        if idx == 0:
            lines.append(
                f"## Quickstart - Deploy {model_name} Inference Server on {device.to_product_str()}"
            )
        else:
            lines.append(f"### Quickstart - Deploy on {product_name}")
        lines.append("")

        # Prerequisites link (only for first device)
        if idx == 0:
            lines.append(
                "See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues."
            )
            lines.append("")

            # Inference Engine link
            inference_engine_display_name = InferenceEngine.from_string(
                target_template.inference_engine
            ).display_name
            lines.append(
                f"This model is supported by [{inference_engine_display_name}]({INFERENCE_ENGINE_README_LINKS[target_template.inference_engine]}) inference engine."
            )
            lines.append("")

        # docker run command (skip for multihost - requires special deployment)
        if target_template.inference_engine == InferenceEngine.VLLM.value:
            is_multihost = device.is_multihost()
            if is_multihost:
                # Multihost requires Controller/Worker architecture, link to guide
                lines.append(
                    f"**Note:** {product_name} requires multi-host deployment with Controller and Worker containers. "
                    "See the [Multi-Host Deployment Guide](../../multihost_deployment.md) for detailed setup instructions."
                )
                lines.append("")
            else:
                docker_image = (
                    target_template.docker_image
                    or generate_default_docker_link(
                        target_template.version,
                        target_template.tt_metal_commit,
                        target_template.vllm_commit,
                        multihost=is_multihost,
                    )
                )
                lines.append("**docker run command**")
                lines.append("")
                lines.append("```bash")
                lines.extend(
                    [
                        "docker run \\",
                        '  --env "HF_TOKEN=$HF_TOKEN" \\',
                        "  --ipc host \\",
                        "  --publish 8000:8000 \\",
                        "  --device /dev/tenstorrent \\",
                        "  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \\",
                        f"  --volume volume_id_{model_name}:/home/container_app_user/cache_root \\",
                        f"  {docker_image} \\",
                        f"  --model {model_name} \\",
                        f"  --tt-device {device.name.lower()}",
                    ]
                )
                lines.append("```")
                lines.append("")

        # run.py command
        lines.append("**via run.py command**")
        lines.append("")
        device_arg = device.name.lower()
        lines.append("```bash")
        lines.append(
            f"python3 run.py --model {model_name} --device {device_arg} --workflow server --docker-server"
        )
        lines.append("```")
        if idx == 0:
            lines.append(
                "For details on the run.py command, see the [run.py CLI Options](../../workflows_user_guide.md#runpy-cli-options) section of the User Guide."
            )
        lines.append("")

        # Get docker image
        if target_template.docker_image:
            docker_image = target_template.docker_image
        else:
            docker_image = generate_default_docker_link(
                VERSION,
                target_template.tt_metal_commit,
                target_template.vllm_commit,
                multihost=device.is_multihost(),
            )

        # Model Parameters table
        if idx == 0:
            lines.append("## Model Parameters")
        else:
            lines.append("### Model Parameters")
        lines.append("")
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")

        # Weights with HuggingFace repo links
        if target_template.weights:
            weights_links = []
            for weight in target_template.weights:
                hf_url = f"https://huggingface.co/{weight}"
                weights_links.append(f"[{weight}]({hf_url})")
            lines.append(f"| Weights | {', '.join(weights_links)} |")

        # Model status
        lines.append(f"| Model Status | {target_template.status.display_string} |")

        lines.append(f"| Max Batch Size | {target_dev_spec.max_concurrency} |")
        # Max Context Length is only relevant for LLM and VLM models
        if model_type in (ModelType.LLM, ModelType.VLM):
            lines.append(f"| Max Context Length | {target_dev_spec.max_context} |")

        # Code link
        code_link = f"{target_template.impl.repo_url}/tree/{target_template.tt_metal_commit}/{target_template.impl.code_path}"
        lines.append(
            f"| Implementation Code | [{target_template.impl.impl_name}]({code_link}) |"
        )
        lines.append(f"| tt-metal Commit | `{target_template.tt_metal_commit}` |")

        if target_template.vllm_commit:
            lines.append(f"| vLLM Commit | `{target_template.vllm_commit}` |")

        lines.append(f"| Docker Image | `{docker_image}` |")
        lines.append("")

    return "\n".join(lines)


def generate_model_type_table(
    templates: List[ModelSpecTemplate],
    model_type: ModelType,
    page_groups: List[HardwarePageGroup],
    is_experimental: bool = False,
) -> str:
    """
    Generate a markdown table for models of a given type.
    Uses HardwarePageGroup.name for column headers.
    Groups templates by model name to avoid duplicate rows.
    """
    # Filter templates by model type
    type_templates = [t for t in templates if t.model_type == model_type]

    # Group by model name
    model_groups = group_templates_by_model(type_templates)

    # Filter model groups by status
    # - Supported table: models that have at least one supported status (TOP_PERF, COMPLETE, FUNCTIONAL)
    # - Experimental table: models that ONLY have EXPERIMENTAL status (no supported statuses)
    filtered_groups = {}
    for model_name, model_templates in model_groups.items():
        has_supported = any(
            t.status in SUPPORTED_STATUS_GROUPS for t in model_templates
        )
        has_experimental = any(
            t.status == ModelStatusTypes.EXPERIMENTAL for t in model_templates
        )

        if is_experimental:
            # Only include if model has experimental AND no supported statuses
            if has_experimental and not has_supported:
                filtered_groups[model_name] = model_templates
        else:
            # Include if model has any supported status
            if has_supported:
                filtered_groups[model_name] = model_templates

    if not filtered_groups:
        return ""

    lines = []

    # Table header - use group.name with links to hardware pages (primary device)
    group_headers = [
        f"[{group.name}]({DEVICE_HARDWARE_LINKS[group.device_ordering[0]]})"
        for group in page_groups
    ]
    header_cols = ["Model Name"] + group_headers
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(header_cols)) + " |")

    # Get highlight list for this model type (empty tuple if not defined)
    highlights = MODEL_HIGHLIGHTS.get(model_type, ())

    def sort_key(model_name: str) -> Tuple:
        model_tmpls = filtered_groups[model_name]
        best_status = get_best_status_for_model(model_tmpls)

        # Highlighted models come first, in their specified order
        if model_name in highlights:
            highlight_order = highlights.index(model_name)
            return (0, highlight_order, 0, "")  # Highlighted: sort by highlight order
        else:
            # Non-highlighted: sort by status (descending), then name (alphabetical)
            return (1, 0, -best_status.value, model_name.lower())

    # Table rows - one per model (not per template)
    for model_name in sorted(filtered_groups.keys(), key=sort_key):
        model_templates = filtered_groups[model_name]

        # Model name links to first page group's combined page
        first_group = None
        for group in page_groups:
            if get_page_group_status_for_model(model_templates, group) is not None:
                first_group = group
                break
        if first_group is None:
            continue
        filename = get_model_page_group_filename(model_name, first_group)

        # Model name with link (same directory since README.md is in the model type subdir)
        row = [f"[{model_name}]({filename})"]

        # Page group status columns - link to combined page group page
        for group in page_groups:
            status_link = get_page_group_status_link(model_name, model_templates, group)
            row.append(status_link)

        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def generate_model_type_page(
    templates: List[ModelSpecTemplate], model_type: ModelType
) -> str:
    """Generate markdown content for a model type table page."""
    short_name = model_type.short_name
    full_name = model_type.display_name
    lines = []

    # Page title - use short name for brevity
    lines.append(f"# {short_name} Models")
    lines.append("")
    lines.append(
        f"This page lists all supported {full_name.lower()} models and their device compatibility."
    )
    lines.append("")

    # Back link to model types index in root README (from docs/model_support/{type}/README.md)
    lines.append(
        "[Search other models by model type](../../../README.md#models-by-model-type)"
    )
    lines.append("")

    # Get page groups that have models of this type
    page_groups = get_page_groups_for_model_type(templates, model_type)
    if not page_groups:
        lines.append("_No models available for this type._")
        return "\n".join(lines)

    # Supported models table
    supported_table = generate_model_type_table(
        templates, model_type, page_groups, is_experimental=False
    )
    if supported_table:
        lines.append("## Supported Models")
        lines.append("")
        lines.append("Models with status: TOP_PERF, COMPLETE, or FUNCTIONAL.")
        lines.append("")
        lines.append(supported_table)
        lines.append("")

    # Experimental models table
    experimental_table = generate_model_type_table(
        templates, model_type, page_groups, is_experimental=True
    )
    if experimental_table:
        lines.append("## Experimental Models")
        lines.append("")
        lines.append(
            "Models with EXPERIMENTAL status are under active development and may have stability or performance issues."
        )
        lines.append("")
        lines.append(experimental_table)
        lines.append("")

    return "\n".join(lines)


def get_devices_with_templates(templates: List[ModelSpecTemplate]) -> Set[DeviceTypes]:
    """Get all devices that have at least one template."""
    devices = set()
    for template in templates:
        for dev_spec in template.device_model_specs:
            devices.add(dev_spec.device)
    return devices


def get_devices_with_templates_ordered() -> List[DeviceTypes]:
    """Get all devices that have page group mappings, in consistent order.

    Excludes EXCLUDED_DEVICES and MODELS_BY_HARDWARE_EXCLUDED_DEVICES.
    """
    # Return devices in the order they appear in DEVICE_HARDWARE_PAGE_GROUPS_MAPPING
    excluded = set(EXCLUDED_DEVICES) | set(MODELS_BY_HARDWARE_EXCLUDED_DEVICES)
    return [d for d in DEVICE_HARDWARE_PAGE_GROUPS_MAPPING.keys() if d not in excluded]


def get_device_hardware_page_display_name(device: DeviceTypes) -> str:
    """Get a unique display name for a device on the hardware page.

    Uses product name with device enum name suffix when product names collide
    (e.g., GALAXY and GALAXY_T3K both have product name "WH Galaxy").
    """
    product_name = device.to_product_str()
    # Check if this product name is shared by multiple devices (excluding all excluded devices)
    excluded = set(EXCLUDED_DEVICES) | set(MODELS_BY_HARDWARE_EXCLUDED_DEVICES)
    devices_with_same_product = [
        d
        for d in DEVICE_HARDWARE_PAGE_GROUPS_MAPPING.keys()
        if d not in excluded and d.to_product_str() == product_name
    ]
    if len(devices_with_same_product) > 1:
        return f"{product_name} ({device.name})"
    return product_name


def generate_models_by_hardware_page(templates: List[ModelSpecTemplate]) -> str:
    """Generate the docs/model_support/models_by_hardware.md page."""
    lines = []

    lines.append("# Models by Hardware")
    lines.append("")
    lines.append("This page lists all supported models organized by hardware type.")
    lines.append("")

    # Back link to model types index in root README
    lines.append("[Search model by model type](../../README.md#models-by-model-type)")
    lines.append("")

    # Get devices that have templates
    devices_with_templates = get_devices_with_templates(templates)

    # Group templates by model name
    model_groups = group_templates_by_model(templates)

    # Iterate through all devices with page group mappings (excluding EXCLUDED_DEVICES)
    for device in get_devices_with_templates_ordered():
        if device not in devices_with_templates:
            continue

        # Section header using unique device display name with link to hardware page
        display_name = get_device_hardware_page_display_name(device)
        hardware_link = DEVICE_HARDWARE_LINKS.get(device)
        if hardware_link:
            lines.append(f"## [{display_name}]({hardware_link})")
        else:
            lines.append(f"## {display_name}")
        lines.append("")

        # Collect models that support this device
        device_models = []
        for model_name, model_templates in model_groups.items():
            status_enum = get_device_status_enum_for_model(model_templates, device)
            if status_enum is not None:
                model_type = get_model_type_for_templates(model_templates)
                device_models.append(
                    (model_name, status_enum, model_type, model_templates)
                )

        if not device_models:
            lines.append("_No models available for this hardware._")
            lines.append("")
            continue

        # Sort by Status (descending numerically), Type, Model
        # Status is ModelStatusTypes IntEnum: higher value = better status
        device_models.sort(
            key=lambda x: (
                -x[1].value,  # Status descending (higher value = better)
                x[2].short_name.lower(),  # Type
                x[0].lower(),  # Model
            )
        )

        # Table header
        lines.append("| Status | Type | Model |")
        lines.append("|--------|------|-------|")

        for model_name, status_enum, model_type, model_templates in device_models:
            subdir = get_model_subdir(model_type)
            filename = get_model_device_filename(model_name, device)

            model_link = f"[{model_name}]({subdir}/{filename})"
            type_short = model_type.short_name
            status_display = status_enum.display_string

            lines.append(f"| {status_display} | {type_short} | {model_link} |")

        lines.append("")

    return "\n".join(lines)


def generate_directory_readme(templates: List[ModelSpecTemplate]) -> str:
    """Generate the docs/model_support/README.md directory page."""
    lines = []

    lines.append("# Model Support")
    lines.append("")
    lines.append(
        "This directory contains documentation for all models supported on Tenstorrent hardware."
    )
    lines.append("")
    lines.append("### Models by Model Type")
    lines.append("")
    lines.append("Browse models by type:")
    lines.append("")

    # Get model types in the order they first appear in spec_templates
    seen_model_types = set()
    ordered_model_types = []
    for t in templates:
        if t.model_type and t.model_type not in seen_model_types:
            seen_model_types.add(t.model_type)
            ordered_model_types.append(t.model_type)

    for model_type in ordered_model_types:
        subdir = model_type.short_name.lower()
        description = MODEL_TYPE_DESCRIPTIONS.get(model_type, model_type.display_name)
        short_name = model_type.short_name

        lines.append(f"- [{short_name} Models]({subdir}/README.md) - {description}")

    lines.append("")

    # Models by Hardware section
    lines.append("### Models by Hardware Configuration")
    lines.append("")
    lines.append("Browse models by hardware:")
    lines.append("")

    # Get devices that have templates
    devices_with_templates = get_devices_with_templates(templates)

    # Iterate through all devices with page group mappings (excluding EXCLUDED_DEVICES)
    for device in get_devices_with_templates_ordered():
        if device not in devices_with_templates:
            continue

        display_name = get_device_hardware_page_display_name(device)
        anchor = generate_section_anchor(display_name)
        lines.append(f"- [{display_name}](models_by_hardware.md#{anchor})")

    lines.append("")

    return "\n".join(lines)


def write_file(path: Path, content: str, dry_run: bool = False) -> None:
    """Write content to a file, creating directories as needed."""
    if dry_run:
        print(f"[DRY RUN] Would write: {path}")
        print(f"  Content length: {len(content)} characters")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    print(f"Wrote: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Model Support documentation from MODEL_SPECS"
    )
    parser.add_argument(
        "--output-dir",
        default="docs/model_support",
        help="Output directory (default: docs/model_support)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be generated without writing files",
    )
    parser.add_argument(
        "--model-spec-path",
        default="workflows/model_spec.py",
        help="Path to model_spec.py (default: workflows/model_spec.py)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    templates = spec_templates

    print(f"Generating Model Support documentation from {len(templates)} templates")
    print(f"Output directory: {output_dir}")
    if args.dry_run:
        print("[DRY RUN MODE]")
    print()

    # Note: docs/model_support/README.md is no longer generated here.
    # The model support content is maintained directly in root README.md
    # via update_model_spec.py's update_readme_model_support() function.

    # Generate models by hardware page
    hardware_content = generate_models_by_hardware_page(templates)
    write_file(output_dir / "models_by_hardware.md", hardware_content, args.dry_run)

    # Generate model type table pages (in subdirectory as README.md)
    for model_type in ModelType:
        # Check if there are any templates of this type
        type_templates = [t for t in templates if t.model_type == model_type]
        if not type_templates:
            continue

        subdir = model_type.short_name.lower()
        page_content = generate_model_type_page(templates, model_type)
        write_file(output_dir / subdir / "README.md", page_content, args.dry_run)

    # Group templates by model name and generate per-page-group pages in subdirectories
    model_groups = group_templates_by_model(templates)

    for model_name, model_templates in model_groups.items():
        # Get model type subdirectory
        model_type = get_model_type_for_templates(model_templates)
        subdir = get_model_subdir(model_type)

        # Get all devices for this model
        model_devices = set()
        for template in model_templates:
            for dev_spec in template.device_model_specs:
                model_devices.add(dev_spec.device)

        # Generate one page per page group (not per device)
        generated_groups = set()
        for device in model_devices:
            if device in EXCLUDED_DEVICES:
                continue
            group = DEVICE_HARDWARE_PAGE_GROUPS_MAPPING.get(device)
            if group and id(group) not in generated_groups:
                generated_groups.add(id(group))
                filename = get_model_page_group_filename(model_name, group)
                page_content = generate_model_page_group_page(
                    model_name, model_templates, group
                )
                write_file(output_dir / subdir / filename, page_content, args.dry_run)

    print()
    print("Documentation generation complete!")
    if not args.dry_run:
        print(f"Output directory: {output_dir}")
        print()
        print(
            "NOTE: Old per-device pages (e.g., Llama-3.1-8B_galaxy_t3k.md) should be removed"
        )
        print("      manually, as they have been replaced by combined page group pages")
        print("      (e.g., Llama-3.1-8B_galaxy.md covers both GALAXY and GALAXY_T3K)")


if __name__ == "__main__":
    main()
