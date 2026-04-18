#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""
Update model_spec.py with commits from CI last_good_json results.

This script reads a last_good_json file and updates tt_metal_commit and vllm_commit
fields in model_spec.py for each ModelSpecTemplate based on CI results.

Usage:
    python3 update_model_spec_commits.py <last_good_json_path>
    python3 update_model_spec_commits.py --dry-run <last_good_json_path>
    python3 update_model_spec_commits.py --ignore-perf-status <last_good_json_path>
    python3 update_model_spec_commits.py --help

Example:
    python3 update_model_spec_commits.py release_logs/models_ci_last_good_*.json

The script:
- Parses all ModelSpecTemplate blocks in model_spec.py
- For each template, checks all weights to find matching model_ids in the JSON
- Validates that all devices in a template have consistent commits
- Updates commits in-place (7-character hash format)
- Updates status field based on perf_status (unless --ignore-perf-status is used)
- Skips templates with no CI data
- Errors if different devices have conflicting commits
"""

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

# Add repo root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from workflows.model_spec import (
    export_model_specs_json,
    spec_templates,
    ModelSpecTemplate,
)


def map_perf_status_to_model_status(perf_status):
    """Map CI perf_status string to ModelStatusTypes enum value."""
    status_map = {
        "experimental": "ModelStatusTypes.EXPERIMENTAL",
        "functional": "ModelStatusTypes.FUNCTIONAL",
        "complete": "ModelStatusTypes.COMPLETE",
        "top_perf": "ModelStatusTypes.TOP_PERF",
    }
    return status_map.get(perf_status.lower() if perf_status else None)


def model_name_from_weight(weight):
    """Extract model name from HuggingFace repo path."""
    return Path(weight).name


def find_template_in_content(content, impl_id, first_weight, devices):
    """
    Find a specific ModelSpecTemplate block in file content.

    Args:
        content: The raw text content of model_spec.py
        impl_id: The impl_id to search for
        first_weight: The first weight in the template's weights list
        devices: List of DeviceTypes to match against

    Returns:
        The matched template text, or None if not found
    """
    # Escape special regex characters in the weight string
    escaped_weight = re.escape(first_weight)

    # Find all ModelSpecTemplate( occurrences
    template_pattern = r"ModelSpecTemplate\("

    for match in re.finditer(template_pattern, content):
        # Found ModelSpecTemplate(, now extract the full template block
        start_pos = match.start()
        pos = match.end()
        paren_count = 1  # We're inside ModelSpecTemplate(

        # Scan forward to find the matching closing parenthesis
        while pos < len(content) and paren_count > 0:
            if content[pos] == "(":
                paren_count += 1
            elif content[pos] == ")":
                paren_count -= 1
            pos += 1

        if paren_count == 0:
            # Found the complete template block
            template_text = content[start_pos:pos]

            # Check if this template matches our criteria
            # 1. Must contain the first_weight in the weights list
            # 2. Must have the correct impl
            # 3. Must contain at least one of the specified devices
            weight_pattern = rf'weights=\[[^\]]*"{escaped_weight}"'
            impl_pattern = rf"impl={impl_id}_impl"

            if re.search(weight_pattern, template_text) and re.search(
                impl_pattern, template_text
            ):
                # Check if at least one device matches
                device_matched = False
                for device in devices:
                    device_pattern = rf"device=DeviceTypes\.{device.name}"
                    if re.search(device_pattern, template_text):
                        device_matched = True
                        break

                if device_matched:
                    # Include the trailing comma if present
                    if pos < len(content) and content[pos] == ",":
                        template_text += ","
                    return template_text

    return None


def get_commits_for_template(template: ModelSpecTemplate, last_good_data):
    """
    Get commits and status for a template from last_good_json data.

    Selection logic:
    - Use first weight (in template order) that has CI data
    - Use first device (in template order) from that weight
    - Warn if other weight/device combinations have conflicting commits

    Args:
        template: ModelSpecTemplate object
        last_good_data: Dictionary of CI results

    Returns tuple: (tt_metal_commit, vllm_commit, status, should_update, selected_model_id)
    """
    impl = template.impl
    weights = template.weights
    devices = [spec.device for spec in template.device_model_specs]

    if not weights:
        return None, None, None, False, None

    # Collect commits for each weight, preserving weight order
    weight_data = []
    all_device_commits = []

    for weight in weights:
        model_name = model_name_from_weight(weight)
        device_commits_for_weight = []

        for device in devices:
            model_id = f"id_{impl.impl_name}_{model_name}_{device.name.lower()}"

            if model_id in last_good_data:
                entry = last_good_data[model_id]

                if entry:
                    tt_metal = entry.get("tt_metal_commit", "")
                    vllm = entry.get("vllm_commit", "")
                    perf_status = entry.get("perf_status", "")

                    if tt_metal or vllm:
                        commit_data = {
                            "model_id": model_id,
                            "weight": weight,
                            "model_name": model_name,
                            "device": device.name,
                            "tt_metal_commit": tt_metal[:7] if tt_metal else "",
                            "vllm_commit": vllm[:7] if vllm else "",
                            "perf_status": perf_status,
                        }
                        device_commits_for_weight.append(commit_data)
                        all_device_commits.append(commit_data)
                        print(
                            f"  Found CI data: {model_id} -> tt_metal={tt_metal[:7] if tt_metal else 'N/A'}, vllm={vllm[:7] if vllm else 'N/A'}"
                        )

        if device_commits_for_weight:
            weight_data.append((weight, model_name, device_commits_for_weight))

    # If no weights have data, skip this template
    if not weight_data:
        return None, None, None, False, None

    # Select first weight with data, then first device from that weight
    selected_weight, selected_model_name, selected_devices = weight_data[0]
    selected_commit_data = selected_devices[0]

    print(
        f"\nSelected: weight={selected_weight}, device={selected_commit_data['device']}, model_id={selected_commit_data['model_id']}"
    )
    print(
        f"  tt_metal_commit={selected_commit_data['tt_metal_commit']}, vllm_commit={selected_commit_data['vllm_commit']}, status={selected_commit_data['perf_status']}"
    )

    # Check for conflicts across all collected data and warn
    tt_metal_commits = set(
        dc["tt_metal_commit"] for dc in all_device_commits if dc["tt_metal_commit"]
    )
    vllm_commits = set(
        dc["vllm_commit"] for dc in all_device_commits if dc["vllm_commit"]
    )
    perf_statuses = set(
        dc["perf_status"] for dc in all_device_commits if dc["perf_status"]
    )

    if len(tt_metal_commits) > 1:
        print("\nWarning: Multiple tt_metal_commits found across weights/devices:")
        for commit in sorted(tt_metal_commits):
            matching_ids = [
                dc["model_id"]
                for dc in all_device_commits
                if dc["tt_metal_commit"] == commit
            ]
            print(f"  {commit}: {matching_ids}")

    if len(vllm_commits) > 1:
        print("\nWarning: Multiple vllm_commits found across weights/devices:")
        for commit in sorted(vllm_commits):
            matching_ids = [
                dc["model_id"]
                for dc in all_device_commits
                if dc["vllm_commit"] == commit
            ]
            print(f"  {commit}: {matching_ids}")

    if len(perf_statuses) > 1:
        print("\nWarning: Multiple perf_statuses found across weights/devices:")
        for status in sorted(perf_statuses):
            matching_ids = [
                dc["model_id"]
                for dc in all_device_commits
                if dc["perf_status"] == status
            ]
            print(f"  {status}: {matching_ids}")

    # Extract selected values
    tt_metal_commit = (
        selected_commit_data["tt_metal_commit"]
        if selected_commit_data["tt_metal_commit"]
        else None
    )
    vllm_commit = (
        selected_commit_data["vllm_commit"]
        if selected_commit_data["vllm_commit"]
        else None
    )

    status = None
    if selected_commit_data["perf_status"]:
        status = map_perf_status_to_model_status(selected_commit_data["perf_status"])

    return tt_metal_commit, vllm_commit, status, True, selected_commit_data["model_id"]


def extract_status(template_text):
    """Extract current status from template text."""
    match = re.search(r"status=ModelStatusTypes\.(\w+)", template_text)
    if match:
        return match.group(1)
    return None


def extract_tt_metal_commit(template_text):
    """Extract current tt_metal_commit from template text."""
    match = re.search(r'tt_metal_commit="([^"]*)"', template_text)
    if match:
        return match.group(1)
    return None


def extract_vllm_commit(template_text):
    """Extract current vllm_commit from template text."""
    match = re.search(r'vllm_commit="([^"]*)"', template_text)
    if match:
        return match.group(1)
    return None


def get_ci_job_url_for_template(template: ModelSpecTemplate, last_good_data):
    """
    Get CI job URL and run number for a template from last_good_json data.

    Args:
        template: ModelSpecTemplate object
        last_good_data: Dictionary of CI results

    Returns tuple: (ci_job_url, ci_run_number)
    """
    # Get impl, weights, and devices directly from template
    impl = template.impl
    weights = template.weights
    devices = [spec.device for spec in template.device_model_specs]

    if not weights:
        return None, None

    # Look for ci_job_url and ci_run_number in any device with data
    for weight in weights:
        model_name = model_name_from_weight(weight)
        for device in devices:
            model_id = f"id_{impl.impl_name}_{model_name}_{device.name.lower()}"

            if model_id in last_good_data:
                entry = last_good_data[model_id]
                if entry:
                    ci_job_url = entry.get("ci_job_url")
                    ci_run_number = entry.get("ci_run_number")
                    if ci_job_url:
                        return ci_job_url, ci_run_number

    return None, None


def update_template_fields(template_text, tt_metal_commit, vllm_commit, status):
    """Update commit and status values in a template text."""
    updated = template_text

    if tt_metal_commit:
        # Replace tt_metal_commit value
        updated = re.sub(
            r'tt_metal_commit="[^"]*"', f'tt_metal_commit="{tt_metal_commit}"', updated
        )

    if vllm_commit:
        # Replace vllm_commit value
        updated = re.sub(
            r'vllm_commit="[^"]*"', f'vllm_commit="{vllm_commit}"', updated
        )

    if status:
        # Replace status value (matches ModelStatusTypes.XXXX format)
        updated = re.sub(r"status=ModelStatusTypes\.\w+", f"status={status}", updated)

    return updated


def generate_release_diff_markdown(update_records, output_path):
    """
    Generate release_models_diff.md markdown file with update details.

    Args:
        update_records: List of dicts with keys: impl, impl_id, model_arch,
                       weights, devices, status_before, status_after,
                       tt_metal_commit_before, tt_metal_commit_after,
                       vllm_commit_before, vllm_commit_after, ci_job_url, ci_run_number
        output_path: Path where markdown file should be written
    """
    lines = []
    lines.append("# Model Spec Release Updates\n")
    lines.append("\nThis document shows model specification updates.\n")

    if not update_records:
        lines.append("\nNo updates were made.\n")
    else:
        # Create single table header
        lines.append(
            "| Impl | Model Arch | Weights | Devices | TT-Metal Commit Change | Status Change | CI Job Link |"
        )
        lines.append(
            "|------|------------|---------|---------|------------------------|---------------|-------------|"
        )

        # Add rows for each record
        for record in update_records:
            # Impl ID
            impl_id = f"`{record['impl_id']}`"

            # Model architecture
            model_arch = f"`{record['model_arch']}`"

            # Weights (formatted list with line breaks)
            weights_formatted = "<br>".join([f"`{w}`" for w in record["weights"]])

            # Devices (comma-separated)
            devices = ", ".join(record["devices"])

            # TT-Metal Commit change
            tt_metal_before = record.get("tt_metal_commit_before")
            tt_metal_after = record.get("tt_metal_commit_after")

            if tt_metal_before and tt_metal_after:
                if tt_metal_before != tt_metal_after:
                    tt_metal_commit = f"`{tt_metal_before}` → `{tt_metal_after}`"
                else:
                    tt_metal_commit = f"`{tt_metal_after}`"
            elif tt_metal_after:
                tt_metal_commit = f"New: `{tt_metal_after}`"
            else:
                tt_metal_commit = "N/A"

            # Status change
            if record["status_before"] and record["status_after"]:
                status_before = record["status_before"]
                status_after_match = re.search(
                    r"ModelStatusTypes\.(\w+)", record["status_after"]
                )
                status_after = (
                    status_after_match.group(1)
                    if status_after_match
                    else record["status_after"]
                )

                if status_before != status_after:
                    status_change = f"{status_before} → {status_after}"
                else:
                    status_change = f"{status_after} (no change)"
            elif record["status_after"]:
                status_after_match = re.search(
                    r"ModelStatusTypes\.(\w+)", record["status_after"]
                )
                status_after = (
                    status_after_match.group(1)
                    if status_after_match
                    else record["status_after"]
                )
                status_change = f"New: {status_after}"
            else:
                status_change = "No change"

            # CI Job Link
            if record["ci_job_url"] and record.get("ci_run_number"):
                ci_link = f"[Run {record['ci_run_number']}]({record['ci_job_url']})"
            elif record["ci_job_url"]:
                ci_link = f"[View Job]({record['ci_job_url']})"
            else:
                ci_link = "N/A"

            # Add table row
            lines.append(
                f"| {impl_id} | {model_arch} | {weights_formatted} | {devices} | {tt_metal_commit} | {status_change} | {ci_link} |"
            )

    # Write to file
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nGenerated release diff markdown: {output_path}")


def reload_and_export_model_specs_json(model_spec_path, output_json_path):
    """
    Dynamically reimport MODEL_SPECS from updated model_spec.py and export to JSON.

    This reimports the module to pick up any in-place edits made by this script,
    then delegates to the shared export_model_specs_json utility.

    Args:
        model_spec_path: Path to the model_spec.py file
        output_json_path: Path where JSON output should be written
    """
    # Add the repository root to sys.path so imports work
    repo_root = model_spec_path.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Dynamically import the updated model_spec module
    spec = importlib.util.spec_from_file_location("model_spec", model_spec_path)
    model_spec_module = importlib.util.module_from_spec(spec)
    sys.modules["model_spec"] = model_spec_module
    spec.loader.exec_module(model_spec_module)

    model_specs = model_spec_module.MODEL_SPECS
    num_specs = export_model_specs_json(model_specs, Path(output_json_path))
    print(f"\nExported {num_specs} model specs to {output_json_path}")


def generate_model_support_docs(model_spec_path, output_dir="docs/model_support"):
    """
    Generate model support documentation by calling generate_model_support_docs.py.

    Args:
        model_spec_path: Path to the model_spec.py file
        output_dir: Output directory for model support docs (default: docs/model_support)
    """
    from scripts.release.generate_model_support_docs import (
        EXCLUDED_DEVICES,
        DEVICE_HARDWARE_PAGE_GROUPS_MAPPING,
        generate_models_by_hardware_page,
        generate_model_type_page,
        generate_model_page_group_page,
        group_templates_by_model,
        get_model_type_for_templates,
        get_model_subdir,
        get_model_page_group_filename,
        write_file,
    )
    from workflows.workflow_types import ModelType

    # Dynamically import the updated model_spec module to get fresh spec_templates
    repo_root = model_spec_path.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    spec = importlib.util.spec_from_file_location("model_spec_docs", model_spec_path)
    model_spec_module = importlib.util.module_from_spec(spec)
    sys.modules["model_spec_docs"] = model_spec_module
    spec.loader.exec_module(model_spec_module)

    templates = model_spec_module.spec_templates
    output_path = Path(output_dir)

    print(f"Generating Model Support documentation from {len(templates)} templates")
    print(f"Output directory: {output_path}")

    # Note: docs/model_support/README.md is no longer generated.
    # The model support content is maintained directly in root README.md
    # via update_readme_model_support() function.

    # Generate models by hardware page
    hardware_content = generate_models_by_hardware_page(templates)
    write_file(output_path / "models_by_hardware.md", hardware_content)

    # Generate model type table pages (in subdirectory as README.md)
    for model_type in ModelType:
        type_templates = [t for t in templates if t.model_type == model_type]
        if not type_templates:
            continue

        subdir = model_type.short_name.lower()
        page_content = generate_model_type_page(templates, model_type)
        write_file(output_path / subdir / "README.md", page_content)

    # Group templates by model name and generate per-page-group pages in subdirectories
    model_groups = group_templates_by_model(templates)

    for model_name, model_templates in model_groups.items():
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
                write_file(output_path / subdir / filename, page_content)

    print("Documentation generation complete!")


def update_readme_model_support(model_spec_path, readme_path="README.md"):
    """
    Update the Model Support section in README.md with links to docs/model_support/.

    Generates docs/model_support/ documentation (model type pages, hardware page,
    individual model pages), then generates the model support section content directly
    and adjusts paths to work from repo root. Uses HTML comment markers for idempotent
    replacement.

    Args:
        model_spec_path: Path to the model_spec.py file
        readme_path: Path to README.md file (default: README.md)
    """
    from scripts.release.generate_model_support_docs import generate_directory_readme

    readme_file = Path(readme_path)
    if not readme_file.exists():
        print(f"Warning: README.md not found at {readme_path}, skipping update")
        return

    # First, regenerate the model support docs (model type pages, hardware page, model pages)
    generate_model_support_docs(model_spec_path)

    # Load templates to generate the model support section content directly
    repo_root = model_spec_path.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    spec = importlib.util.spec_from_file_location("model_spec_readme", model_spec_path)
    model_spec_module = importlib.util.module_from_spec(spec)
    sys.modules["model_spec_readme"] = model_spec_module
    spec.loader.exec_module(model_spec_module)
    templates = model_spec_module.spec_templates

    # Generate the model support content directly (no intermediate file)
    model_support_content = generate_directory_readme(templates)

    # Adjust relative paths to work from repo root
    # - Links like (llm/README.md) -> (docs/model_support/llm/README.md)
    # - Links like (models_by_hardware.md#...) -> (docs/model_support/models_by_hardware.md#...)
    # - Links like (llm/Model.md) -> (docs/model_support/llm/Model.md)
    # Skip external links (http/https) and parent links (..)
    def adjust_link(match):
        link_text = match.group(1)
        link_path = match.group(2)

        # Skip external links and parent directory links
        if link_path.startswith(("http://", "https://", "..")):
            return match.group(0)

        # Prepend docs/model_support/ to relative paths
        return f"[{link_text}](docs/model_support/{link_path})"

    adjusted_content = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)", adjust_link, model_support_content
    )

    # Remove the "# Model Support" header since README.md already has "## Model Support"
    # and remove the intro line about "This directory contains..."
    lines = adjusted_content.split("\n")
    filtered_lines = []
    skip_next_empty = False
    for line in lines:
        if line.startswith("# Model Support"):
            skip_next_empty = True
            continue
        if line.startswith("This directory contains documentation"):
            skip_next_empty = True
            continue
        if skip_next_empty and line.strip() == "":
            skip_next_empty = False
            continue
        skip_next_empty = False
        filtered_lines.append(line)

    model_support_section = "\n".join(filtered_lines).strip()

    # Read current README content
    with open(readme_file, "r") as f:
        content = f.read()

    # Replace content between MODEL_SUPPORT markers
    start_marker = "<!-- MODEL_SUPPORT_START -->"
    end_marker = "<!-- MODEL_SUPPORT_END -->"

    start_pos = content.find(start_marker)
    end_pos = content.find(end_marker)

    if start_pos == -1 or end_pos == -1:
        print(
            f"Warning: Model Support markers not found in {readme_path}, skipping update"
        )
        return

    # Build new section with markers
    new_section = f"{start_marker}\n{model_support_section}\n{end_marker}"

    end_pos += len(end_marker)
    updated_content = content[:start_pos] + new_section + content[end_pos:]

    # Write back to file
    with open(readme_file, "w") as f:
        f.write(updated_content)

    print(f"\nSuccessfully updated Model Support section in {readme_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Update model_spec.py commits from last_good_json CI results"
    )
    parser.add_argument(
        "last_good_json", nargs="?", help="Path to last_good_json file with CI results"
    )
    parser.add_argument(
        "--model-spec-path",
        default="workflows/model_spec.py",
        help="Path to model_spec.py file (default: workflows/model_spec.py)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print changes without modifying files"
    )
    parser.add_argument(
        "--output-json",
        default="model_spec.json",
        help="Path to output JSON file with all model specs (default: model_spec.json)",
    )
    parser.add_argument(
        "--output-only",
        action="store_true",
        help="Read model_spec.py and generate outputs (README.md table and model_spec.json) without modifying model_spec.py",
    )
    parser.add_argument(
        "--readme-path",
        default="README.md",
        help="Path to README.md file (default: README.md)",
    )
    parser.add_argument(
        "--ignore-perf-status",
        action="store_true",
        help="Only update tt_metal_commit and vllm_commit, do not update status/perf_status",
    )

    args = parser.parse_args()

    # Handle --output-only mode
    if args.output_only:
        model_spec_path = Path(args.model_spec_path)
        if not model_spec_path.exists():
            raise FileNotFoundError(f"Error: File not found: {args.model_spec_path}")

        print(
            "Running in --output-only mode: generating outputs without modifying model_spec.py"
        )

        # Update README.md Model Support section and regenerate docs/model_support/
        update_readme_model_support(model_spec_path, args.readme_path)

        # Export MODEL_SPECS to JSON
        output_json_path = Path(args.output_json)
        reload_and_export_model_specs_json(model_spec_path, output_json_path)

        return

    # Validate that last_good_json is provided when not in --output-only mode
    if not args.last_good_json:
        raise ValueError(
            "Error: last_good_json argument is required when not using --output-only"
        )

    # Load last_good_json
    last_good_path = Path(args.last_good_json)
    if not last_good_path.exists():
        raise FileNotFoundError(f"Error: File not found: {args.last_good_json}")

    with open(last_good_path, "r") as f:
        last_good_data = json.load(f)

    # Read model_spec.py
    model_spec_path = Path(args.model_spec_path)
    if not model_spec_path.exists():
        raise FileNotFoundError(f"Error: File not found: {args.model_spec_path}")

    with open(model_spec_path, "r") as f:
        content = f.read()

    # Process each template by iterating spec_templates directly
    print(f"Processing {len(spec_templates)} ModelSpecTemplate objects")

    updated_content = content
    updates_made = 0
    update_records = []  # Track updates for markdown generation

    for template in spec_templates:
        # Log template being processed
        impl_name = template.impl.impl_name
        weights = template.weights
        devices = [spec.device.name for spec in template.device_model_specs]
        print(f"\n{'=' * 80}")
        print(
            f"Processing template: impl={impl_name}, weights={weights}, devices={devices}"
        )

        # Get commits and status for this template from CI data
        tt_metal_commit, vllm_commit, status, should_update, selected_model_id = (
            get_commits_for_template(template, last_good_data)
        )

        if not should_update:
            print("  No CI data found for this template. Skipping.")
            continue

        # Validate that selected commits come from this template's weights
        if selected_model_id:
            selected_model_name = selected_model_id.split("_")[2]
            template_model_names = [model_name_from_weight(w) for w in template.weights]

            if selected_model_name not in template_model_names:
                print(
                    f"\nError: Selected model_id '{selected_model_id}' does not match template weights: {template.weights}"
                )
                print("  Skipping update to prevent cross-contamination.")
                continue

        # Build unique identifier to find this template in text
        impl_id = template.impl.impl_id
        first_weight = template.weights[0] if template.weights else ""
        devices = [spec.device for spec in template.device_model_specs]

        if not first_weight:
            continue

        # Find and extract the template text in original content
        template_text = find_template_in_content(
            content, impl_id, first_weight, devices
        )

        if not template_text:
            print(
                f"\nWarning: Could not find template in file for impl={impl_id}, weight={first_weight}"
            )
            continue

        # Update the template fields
        # If --ignore-perf-status is set, don't update status
        status_to_update = None if args.ignore_perf_status else status
        updated_template = update_template_fields(
            template_text, tt_metal_commit, vllm_commit, status_to_update
        )

        if updated_template != template_text:
            # Get info from template object for logging
            weights = template.weights
            impl_name = template.impl.impl_name

            print("\nUpdating template:")
            print(f"  impl: {impl_name}")
            print(f"  weights[0]: {weights[0] if weights else 'unknown'}")
            if tt_metal_commit:
                print(f"  tt_metal_commit: {tt_metal_commit}")
            if vllm_commit:
                print(f"  vllm_commit: {vllm_commit}")
            if status_to_update:
                print(f"  status: {status_to_update}")
            elif args.ignore_perf_status and status:
                print(f"  status: {status} (ignored, not updating)")

            # Track update for markdown generation
            status_before = extract_status(template_text)
            tt_metal_commit_before = extract_tt_metal_commit(template_text)
            vllm_commit_before = extract_vllm_commit(template_text)
            devices = [spec.device.name for spec in template.device_model_specs]
            ci_job_url, ci_run_number = get_ci_job_url_for_template(
                template, last_good_data
            )
            model_arch = model_name_from_weight(weights[0]) if weights else "unknown"
            # For status_after, use None if ignoring perf status, otherwise use the status
            status_after = None if args.ignore_perf_status else status

            update_records.append(
                {
                    "impl": impl_name,
                    "impl_id": impl_id,
                    "model_arch": model_arch,
                    "weights": weights,
                    "devices": devices,
                    "status_before": status_before,
                    "status_after": status_after,
                    "tt_metal_commit_before": tt_metal_commit_before,
                    "tt_metal_commit_after": tt_metal_commit,
                    "vllm_commit_before": vllm_commit_before,
                    "vllm_commit_after": vllm_commit,
                    "ci_job_url": ci_job_url,
                    "ci_run_number": ci_run_number,
                }
            )

            # Apply replacement directly
            updated_content = updated_content.replace(
                template_text, updated_template, 1
            )
            updates_made += 1

    # Write updated content
    if updates_made > 0:
        if args.dry_run:
            print(
                f"\n[DRY RUN] Would update {updates_made} templates in {args.model_spec_path}"
            )
        else:
            with open(model_spec_path, "w") as f:
                f.write(updated_content)
            print(
                f"\nSuccessfully updated {updates_made} templates in {args.model_spec_path}"
            )

            # Export MODEL_SPECS to JSON
            output_json_path = Path(args.output_json)
            reload_and_export_model_specs_json(model_spec_path, output_json_path)

            # Generate release diff markdown
            if update_records:
                diff_markdown_path = last_good_path.parent / "release_models_diff.md"
                generate_release_diff_markdown(update_records, diff_markdown_path)

            # Update README.md Model Support section and regenerate docs/model_support/
            update_readme_model_support(model_spec_path)
    else:
        print("\nNo updates needed.")

        # Even if no updates were made, export the current MODEL_SPECS to JSON
        if not args.dry_run:
            output_json_path = Path(args.output_json)
            reload_and_export_model_specs_json(model_spec_path, output_json_path)


if __name__ == "__main__":
    main()
