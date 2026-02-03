#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Generate Model Support documentation from MODEL_SPECS.

This script generates a documentation system for model support including:
- Directory README with links to model type tables and hardware tables
- Model type table pages (llm/README.md, vlm/README.md, etc.)
- Models by hardware page (models_by_hardware.md)
- Individual model pages with device-specific sections

Usage:
    python scripts/release/generate_model_support_docs.py
    python scripts/release/generate_model_support_docs.py --dry-run
    python scripts/release/generate_model_support_docs.py --output-dir docs/model_support
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

# Add repo root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from workflows.model_spec import (
    ModelSpecTemplate,
    spec_templates,
    model_weights_to_model_name,
    generate_default_docker_link,
    VERSION,
)
from workflows.workflow_types import (
    DeviceTypes,
    InferenceEngine,
    ModelStatusTypes,
    ModelType,
)

# Mapping inference engine to documentation link (reused from update_model_spec.py)
INFERENCE_ENGINE_README_LINKS = {
    InferenceEngine.VLLM.value: "../../vllm-tt-metal-llama3/README.md",
    InferenceEngine.MEDIA.value: "../../tt-media-server/README.md",
    InferenceEngine.FORGE.value: "../../tt-media-server/README.md",
}

# Mapping device type to hardware link text (reused from update_model_spec.py)
DEVICE_HARDWARE_LINKS = {
    DeviceTypes.T3K: "[TT-LoudBox](https://tenstorrent.com/hardware/tt-loudbox)",
    DeviceTypes.N150: "[n150](https://tenstorrent.com/hardware/wormhole)",
    DeviceTypes.N300: "[n300](https://tenstorrent.com/hardware/wormhole)",
    DeviceTypes.GALAXY: "[Galaxy](https://tenstorrent.com/hardware/galaxy)",
    DeviceTypes.P100: "[p100](https://tenstorrent.com/hardware/blackhole)",
    DeviceTypes.P150: "[p150](https://tenstorrent.com/hardware/blackhole)",
    DeviceTypes.P150X4: "[BH-QuietBox](https://tenstorrent.com/hardware/tt-quietbox)",
    DeviceTypes.P150X8: "[BH-LoudBox](https://tenstorrent.com/hardware/tt-loudbox)",
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


def get_device_section_title(model_name: str, device: DeviceTypes) -> str:
    """
    Generate the section title for a device section on a model page.
    Must match the titles generated in generate_model_page().
    """
    product_name = get_device_product_name(device)

    # Match the logic in generate_model_page for disambiguation
    if device in [DeviceTypes.GALAXY_T3K]:
        return f"Run {model_name} on {product_name} ({device.name})"
    else:
        return f"Run {model_name} on {product_name}"


def get_model_display_name(template: ModelSpecTemplate) -> str:
    """Get the display name for a model template."""
    if template.display_name:
        return template.display_name
    return model_weights_to_model_name(template.weights[0])


def get_model_filename(template: ModelSpecTemplate) -> str:
    """Get the markdown filename for a model page."""
    display_name = get_model_display_name(template)
    return f"{sanitize_filename(display_name)}.md"


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
    Links to the specific device section on the model page.
    Returns "-" if the device is not supported.
    """
    status = get_device_status_for_model(model_templates, device)
    if status == "-":
        return "-"

    # Generate the link to the model page with section anchor
    # (model pages are in the same directory as the model type README.md)
    filename = f"{sanitize_filename(model_name)}.md"
    section_title = get_device_section_title(model_name, device)
    anchor = generate_section_anchor(section_title)

    return f"[{status}]({filename}#{anchor})"


def get_device_product_name(device: DeviceTypes) -> str:
    """
    Get the product name for a device, with fallback for devices without mappings.
    Uses DeviceTypes.to_product_str() when available.
    """
    try:
        return device.to_product_str()
    except ValueError:
        # Fallback for devices without product name mapping (e.g., GPU, CPU)
        return device.name


def get_model_type_for_templates(templates: List[ModelSpecTemplate]) -> ModelType:
    """Get the model type from a list of templates (assumes all have same type)."""
    for template in templates:
        if template.model_type:
            return template.model_type
    return ModelType.LLM  # Default fallback


def generate_model_page(model_name: str, templates: List[ModelSpecTemplate]) -> str:
    """
    Generate markdown content for an individual model page.
    Consolidates device specs from multiple templates for the same model.
    Uses DeviceTypes.to_product_str() for section headers.
    """
    lines = []

    # Get model type for back link
    model_type = get_model_type_for_templates(templates)
    short_name = model_type.short_name

    # Page title
    lines.append(f"# {model_name} Tenstorrent Support")
    lines.append("")

    # Collect all device specs from all templates, avoiding duplicates
    seen_devices = set()
    device_sections = []

    for template in templates:
        for dev_spec in template.device_model_specs:
            if dev_spec.device in seen_devices:
                continue
            seen_devices.add(dev_spec.device)
            device_sections.append((dev_spec, template))

    # Sort by device order for consistent output
    device_order = [
        DeviceTypes.N150,
        DeviceTypes.N300,
        DeviceTypes.T3K,
        DeviceTypes.GALAXY,
        DeviceTypes.GALAXY_T3K,
        DeviceTypes.P100,
        DeviceTypes.P150,
        DeviceTypes.P150X4,
        DeviceTypes.P150X8,
        DeviceTypes.CPU,
        DeviceTypes.GPU,
    ]
    device_sections.sort(
        key=lambda x: device_order.index(x[0].device)
        if x[0].device in device_order
        else 999
    )

    # Generate one section per device
    for dev_spec, template in device_sections:
        device = dev_spec.device
        product_name = get_device_product_name(device)

        # Add device type suffix for disambiguation when product names are the same
        # (e.g., GALAXY vs GALAXY_T3K both map to "Tenstorrent Galaxy")
        if device in [DeviceTypes.GALAXY_T3K]:
            section_title = f"## Run {model_name} on {product_name} ({device.name})"
        else:
            section_title = f"## Run {model_name} on {product_name}"

        lines.append(section_title)
        lines.append("")

        # Back link to model type table (README.md in same directory)
        lines.append(f"[{short_name} Model Support Table](README.md)")
        lines.append("")

        # Quickstart section
        lines.append("### Quickstart - Deploy Inference Server")
        lines.append("")

        # Prerequisites link
        lines.append(
            "See [prerequisites](../../prerequisites.md) for system software setup, e.g. for first-run or when experiencing issues."
        )
        lines.append("")

        # Supported weights (only show if multiple weights are supported)
        if template.weights and len(template.weights) > 1:
            default_weights = template.weights[0]
            # Use model name without HF org prefix for display
            default_weights_name = default_weights.split("/")[-1]
            additional_weights = template.weights[1:]

            lines.append(
                f"The default model weights for this implementation is `{default_weights_name}`, the following weights are supported as well:"
            )
            lines.append("")
            for weight in additional_weights:
                weight_name = weight.split("/")[-1]
                lines.append(f"- `{weight_name}`")
            lines.append("")
            lines.append(
                f"To use these weights simply swap `{default_weights_name}` for your desired weights in commands below."
            )
            lines.append("")

        # Option 1: run.py
        # lines.append("**Option 1: via run.py command**")
        lines.append("**via run.py command**")
        lines.append("")
        device_arg = device.name.lower()
        lines.append("```bash")
        lines.append(
            f"python3 run.py --model {model_name} --device {device_arg} --workflow server --docker-server"
        )
        lines.append("```")
        lines.append("")

        # Get docker image
        if template.docker_image:
            docker_image = template.docker_image
        else:
            docker_image = generate_default_docker_link(
                VERSION, template.tt_metal_commit, template.vllm_commit
            )

        # TODO: add docker inferface post: https://github.com/tenstorrent/tt-inference-server/issues/1253
        # hf_model = template.weights[0] if template.weights else model_name
        # Option 2: Docker
        # lines.append("**Option 2: Direct via Docker image**")
        # lines.append("")
        # lines.append("```bash")
        # lines.append("docker run --rm \\")
        # lines.append("  --device /dev/tenstorrent \\")
        # lines.append("  -e HF_TOKEN=$HF_TOKEN \\")
        # lines.append(f"  -e HF_MODEL={hf_model} \\")
        # lines.append(f"  {docker_image}")
        # lines.append("```")
        # lines.append("")

        # Model Parameters table
        lines.append("### Model Parameters")
        lines.append("")
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")

        # Weights with HuggingFace repo links
        if template.weights:
            weights_links = []
            for weight in template.weights:
                hf_url = f"https://huggingface.co/{weight}"
                weights_links.append(f"[{weight}]({hf_url})")
            lines.append(f"| Weights | {', '.join(weights_links)} |")

        # Model status
        lines.append(f"| Model Status | {template.status.display_string} |")

        lines.append(f"| Max Batch Size | {dev_spec.max_concurrency} |")
        # Max Context Length is only relevant for LLM and VLM models
        if model_type in (ModelType.LLM, ModelType.VLM):
            lines.append(f"| Max Context Length | {dev_spec.max_context} |")

        # Code link
        code_link = f"{template.impl.repo_url}/tree/{template.tt_metal_commit}/{template.impl.code_path}"
        lines.append(
            f"| Implementation Code | [{template.impl.impl_name}]({code_link}) |"
        )
        lines.append(f"| tt-metal Commit | `{template.tt_metal_commit}` |")

        if template.vllm_commit:
            lines.append(f"| vLLM Commit | `{template.vllm_commit}` |")

        lines.append(f"| Docker Image | `{docker_image}` |")
        lines.append("")

    return "\n".join(lines)


def generate_model_type_table(
    templates: List[ModelSpecTemplate],
    model_type: ModelType,
    devices: List[DeviceTypes],
    is_experimental: bool = False,
) -> str:
    """
    Generate a markdown table for models of a given type.
    Uses DeviceTypes.name for column headers.
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

    # Table header - use DeviceTypes.name for column headers
    header_cols = ["Model Name"] + [d.name for d in devices]
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(header_cols)) + " |")

    # Table rows - one per model (not per template)
    for model_name in sorted(filtered_groups.keys(), key=str.lower):
        model_templates = filtered_groups[model_name]

        filename = f"{sanitize_filename(model_name)}.md"

        # Model name with link (same directory since README.md is in the model type subdir)
        row = [f"[{model_name}]({filename})"]

        # Device status columns - link to specific device section on model page
        for device in devices:
            status_link = get_device_status_link(model_name, model_templates, device)
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
        f"This page lists all supported {full_name.lower()}s and their device compatibility."
    )
    lines.append("")

    # Back link to model types index in root README (from docs/model_support/{type}/README.md)
    lines.append("[Search model by model type](../../../README.md#models-by-model-type)")
    lines.append("")

    # Get devices that have models of this type
    devices_set = get_devices_for_model_type(templates, model_type)
    if not devices_set:
        lines.append("_No models available for this type._")
        return "\n".join(lines)

    # Sort devices for consistent ordering
    device_order = [
        DeviceTypes.N150,
        DeviceTypes.N300,
        DeviceTypes.T3K,
        DeviceTypes.GALAXY,
        DeviceTypes.GALAXY_T3K,
        DeviceTypes.P100,
        DeviceTypes.P150,
        DeviceTypes.P150X4,
        DeviceTypes.P150X8,
    ]
    devices = [d for d in device_order if d in devices_set]

    # Supported models table
    supported_table = generate_model_type_table(
        templates, model_type, devices, is_experimental=False
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
        templates, model_type, devices, is_experimental=True
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

    # Device order for consistent output
    device_order = [
        DeviceTypes.N150,
        DeviceTypes.N300,
        DeviceTypes.T3K,
        DeviceTypes.GALAXY,
        DeviceTypes.GALAXY_T3K,
        DeviceTypes.P100,
        DeviceTypes.P150,
        DeviceTypes.P150X4,
        DeviceTypes.P150X8,
    ]

    # Group templates by model name
    model_groups = group_templates_by_model(templates)

    for device in device_order:
        if device not in devices_with_templates:
            continue

        product_name = get_device_product_name(device)

        # Section header
        lines.append(f"## {product_name} ({device.name})")
        lines.append("")

        # Collect models that support this device
        device_models = []
        for model_name, model_templates in model_groups.items():
            status = get_device_status_for_model(model_templates, device)
            if status != "-":
                model_type = get_model_type_for_templates(model_templates)
                device_models.append((model_name, status, model_type, model_templates))

        if not device_models:
            lines.append("_No models available for this hardware._")
            lines.append("")
            continue

        # Sort by model name
        device_models.sort(key=lambda x: x[0].lower())

        # Table header
        lines.append("| Model | Type | Status |")
        lines.append("|-------|------|--------|")

        for model_name, status, model_type, model_templates in device_models:
            subdir = get_model_subdir(model_type)
            filename = f"{sanitize_filename(model_name)}.md"
            section_title = get_device_section_title(model_name, device)
            anchor = generate_section_anchor(section_title)

            model_link = f"[{model_name}]({subdir}/{filename}#{anchor})"
            type_short = model_type.short_name

            lines.append(f"| {model_link} | {type_short} | {status} |")

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

    # Get model types that actually have templates
    model_types_with_templates = set(t.model_type for t in templates if t.model_type)

    for model_type in ModelType:
        if model_type not in model_types_with_templates:
            continue

        subdir = model_type.short_name.lower()
        description = MODEL_TYPE_DESCRIPTIONS.get(model_type, model_type.display_name)
        short_name = model_type.short_name

        lines.append(f"- [{short_name} Models]({subdir}/README.md) - {description}")

    lines.append("")

    # Models by Hardware section
    lines.append("### Models by Hardware")
    lines.append("")
    lines.append("Browse models by hardware type:")
    lines.append("")

    # Get devices that have templates
    devices_with_templates = get_devices_with_templates(templates)

    # Device order for consistent output
    device_order = [
        DeviceTypes.N150,
        DeviceTypes.N300,
        DeviceTypes.T3K,
        DeviceTypes.GALAXY,
        DeviceTypes.GALAXY_T3K,
        DeviceTypes.P100,
        DeviceTypes.P150,
        DeviceTypes.P150X4,
        DeviceTypes.P150X8,
    ]

    for device in device_order:
        if device not in devices_with_templates:
            continue

        product_name = get_device_product_name(device)
        anchor = generate_section_anchor(f"{product_name} ({device.name})")
        lines.append(
            f"- [{product_name} ({device.name})](models_by_hardware.md#{anchor})"
        )

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

    # Group templates by model name and generate consolidated pages in subdirectories
    model_groups = group_templates_by_model(templates)

    for model_name, model_templates in model_groups.items():
        # Get model type subdirectory
        model_type = get_model_type_for_templates(model_templates)
        subdir = get_model_subdir(model_type)

        filename = f"{sanitize_filename(model_name)}.md"
        page_content = generate_model_page(model_name, model_templates)
        write_file(output_dir / subdir / filename, page_content, args.dry_run)

    print()
    print("Documentation generation complete!")
    if not args.dry_run:
        print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
