#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Update model_spec.py with commits from CI last_good_json results.

This script reads a last_good_json file and updates tt_metal_commit and vllm_commit
fields in model_spec.py for each ModelSpecTemplate based on CI results.

Usage:
    python3 update_model_spec.py <last_good_json_path>
    python3 update_model_spec.py --dry-run <last_good_json_path>
    python3 update_model_spec.py --ignore-perf-status <last_good_json_path>
    python3 update_model_spec.py --help

Example:
    python3 update_model_spec.py release_logs/v{VERSION}/models_ci_last_good_*.json

The script:
- Parses all ModelSpecTemplate blocks in model_spec.py
- For each template, checks all weights to find matching model_ids in the JSON
- Validates that all devices in a template have consistent commits
- Updates commits in-place (7-character hash format)
- Updates status field based on perf_status (unless --ignore-perf-status is used)
- Stamps release_version on updated templates
- Skips templates with no CI data
- Errors if different devices have conflicting commits
"""

import argparse
import importlib.util
import json
import re
import subprocess
import sys
import types
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add repo root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from workflows.model_spec import (
    VERSION,
    export_model_specs_json,
    spec_templates,
    ModelSpecTemplate,
)

try:
    from release_diff import (
        CiMetadata,
        ReleaseDiffRecord,
        TemplateSnapshot,
        build_snapshot_index_by_template_key,
        build_template_key,
    )
    from release_paths import get_versioned_release_logs_dir, resolve_release_output_dir
except ImportError:
    from scripts.release.release_diff import (
        CiMetadata,
        ReleaseDiffRecord,
        TemplateSnapshot,
        build_snapshot_index_by_template_key,
        build_template_key,
    )
    from scripts.release.release_paths import (
        get_versioned_release_logs_dir,
        resolve_release_output_dir,
    )

RELEASE_BRANCH_PATTERN = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")


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


def update_template_fields(
    template_text, tt_metal_commit, vllm_commit, status, release_version=None
):
    """Update commit, status, and release_version values in a template text."""
    updated = template_text

    if tt_metal_commit:
        updated = re.sub(
            r'tt_metal_commit="[^"]*"', f'tt_metal_commit="{tt_metal_commit}"', updated
        )

    if vllm_commit:
        updated = re.sub(
            r'vllm_commit="[^"]*"', f'vllm_commit="{vllm_commit}"', updated
        )

    if status:
        updated = re.sub(r"status=ModelStatusTypes\.\w+", f"status={status}", updated)

    if release_version:
        lines = updated.splitlines(keepends=True)
        tt_metal_index = None
        release_version_index = None
        tt_metal_indent = None
        release_version_indent = None
        release_version_newline = "\n"

        for index, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('tt_metal_commit="') and stripped.endswith('",'):
                tt_metal_index = index
                tt_metal_indent = line[: len(line) - len(line.lstrip())]
            elif stripped.startswith('release_version="') and stripped.endswith('",'):
                release_version_index = index
                release_version_indent = line[: len(line) - len(line.lstrip())]
                release_version_newline = "\n" if line.endswith("\n") else ""

        if release_version_index is not None or tt_metal_index is not None:
            indent = release_version_indent or tt_metal_indent or ""
            newline = (
                release_version_newline
                if release_version_index is not None
                else ("\n" if lines[tt_metal_index].endswith("\n") else "")
            )
            release_version_line = (
                f'{indent}release_version="{release_version}",{newline}'
            )

            if release_version_index is not None:
                lines[release_version_index] = release_version_line
                if (
                    tt_metal_index is not None
                    and release_version_index > tt_metal_index
                ):
                    release_version_line = lines.pop(release_version_index)
                    lines.insert(tt_metal_index, release_version_line)
            else:
                lines.insert(tt_metal_index, release_version_line)

            updated = "".join(lines)

    return updated


def update_template_optional_string_field(template_text, field_name, value):
    """Update an optional quoted string field, preserving explicit `None` values."""
    quoted_pattern = rf'{field_name}="[^"]*"'
    none_pattern = rf"{field_name}=None"

    if value is None:
        return re.sub(quoted_pattern, f"{field_name}=None", template_text)

    replacement = f'{field_name}="{value}"'
    if re.search(quoted_pattern, template_text):
        return re.sub(quoted_pattern, replacement, template_text)
    if re.search(none_pattern, template_text):
        return re.sub(none_pattern, replacement, template_text)
    return template_text


def normalize_status_value(status_value):
    """Normalize `ModelStatusTypes.*` strings to their enum member names."""
    if not status_value:
        return None

    status_match = re.search(r"ModelStatusTypes\.(\w+)", status_value)
    if status_match:
        return status_match.group(1)
    return status_value


def generate_release_diff_markdown(update_records, output_path):
    """
    Generate pre_release_models_diff.md markdown file with update details.

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
                status_before = normalize_status_value(record["status_before"])
                status_after = normalize_status_value(record["status_after"])

                if status_before != status_after:
                    status_change = f"{status_before} → {status_after}"
                else:
                    status_change = f"{status_after} (no change)"
            elif record["status_after"]:
                status_after = normalize_status_value(record["status_after"])
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


def write_release_diff_json(update_records, output_path):
    """Write structured release diff records as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(update_records, f, indent=2)
        f.write("\n")

    print(f"\nGenerated release diff JSON: {output_path}")


def write_release_diff_outputs(update_records, output_dir):
    """Write markdown and JSON release diff artifacts from one record list."""
    output_dir.mkdir(parents=True, exist_ok=True)

    markdown_path = output_dir / "pre_release_models_diff.md"
    json_path = output_dir / "pre_release_models_diff.json"

    generate_release_diff_markdown(update_records, markdown_path)
    write_release_diff_json(update_records, json_path)

    return markdown_path, json_path


def extract_template_block_spans(content):
    """Extract raw `ModelSpecTemplate(...)` blocks and their spans."""
    blocks = []
    template_pattern = r"ModelSpecTemplate\("

    for match in re.finditer(template_pattern, content):
        start_pos = match.start()
        pos = match.end()
        paren_count = 1

        while pos < len(content) and paren_count > 0:
            if content[pos] == "(":
                paren_count += 1
            elif content[pos] == ")":
                paren_count -= 1
            pos += 1

        if paren_count != 0:
            raise ValueError("Unbalanced parentheses while parsing ModelSpecTemplate")

        end_pos = pos
        template_text = content[start_pos:end_pos]
        if pos < len(content) and content[pos] == ",":
            template_text += ","
            end_pos += 1
        blocks.append((start_pos, end_pos, template_text))

    return blocks


def extract_template_blocks(content):
    """Extract raw `ModelSpecTemplate(...)` blocks from `model_spec.py` content."""
    return [
        template_text for _, _, template_text in extract_template_block_spans(content)
    ]


def load_model_spec_module_from_content(
    model_spec_path: Path, content: str, module_name: str
):
    """Execute model_spec content in an isolated module namespace."""
    repo_root = model_spec_path.parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    module = types.ModuleType(module_name)
    module.__file__ = str(model_spec_path)
    sys.modules[module_name] = module
    exec(compile(content, str(model_spec_path), "exec"), module.__dict__)
    return module


def evaluate_template_block(model_spec_module, template_text: str):
    """Evaluate one raw `ModelSpecTemplate(...)` block in module context."""
    expression = template_text.rstrip()
    if expression.endswith(","):
        expression = expression[:-1]

    try:
        return eval(expression, model_spec_module.__dict__)
    except Exception as exc:
        raise ValueError(
            "Failed to evaluate ModelSpecTemplate block while building snapshots"
        ) from exc


def build_template_snapshots(
    model_spec_path: Path, content: str, module_name: str
) -> List[TemplateSnapshot]:
    """Build ordered template snapshots from raw template blocks in source order."""
    model_spec_module = load_model_spec_module_from_content(
        model_spec_path, content, module_name
    )
    template_blocks = extract_template_blocks(content)
    occurrence_counts = {}
    snapshots = []

    for template_text in template_blocks:
        template = evaluate_template_block(model_spec_module, template_text)
        impl_id = template.impl.impl_id
        occurrence_index = occurrence_counts.get(impl_id, 0)
        occurrence_counts[impl_id] = occurrence_index + 1

        weights = list(template.weights)
        devices = [spec.device.name for spec in template.device_model_specs]
        inference_engine = str(template.inference_engine)

        snapshots.append(
            {
                "impl": template.impl.impl_name,
                "impl_id": impl_id,
                "model_arch": model_name_from_weight(weights[0])
                if weights
                else "unknown",
                "inference_engine": inference_engine,
                "weights": weights,
                "devices": devices,
                "status": extract_status(template_text),
                "tt_metal_commit": extract_tt_metal_commit(template_text),
                "vllm_commit": extract_vllm_commit(template_text),
                "template_text": template_text,
                "template_key": build_template_key(
                    impl_id, weights, devices, inference_engine
                ),
                "occurrence_key": (impl_id, occurrence_index),
            }
        )

    return snapshots


def parse_release_branch_version(ref_name: str) -> Optional[Tuple[int, int, int]]:
    """Return semantic version tuple for exact release refs like `v0.10.0`."""
    match = RELEASE_BRANCH_PATTERN.fullmatch(ref_name)
    if not match:
        return None
    return tuple(int(part) for part in match.groups())


def extract_exact_release_ref_name(full_ref: str, short_ref: str) -> Optional[str]:
    """Return the exact `vMAJOR.MINOR.PATCH` ref name for supported ref kinds."""
    if full_ref.startswith("refs/remotes/"):
        if short_ref.count("/") != 1:
            return None
        _, release_ref_name = short_ref.split("/", 1)
    elif full_ref.startswith(("refs/heads/", "refs/tags/")):
        if "/" in short_ref:
            return None
        release_ref_name = short_ref
    else:
        return None

    if parse_release_branch_version(release_ref_name) is None:
        return None

    return release_ref_name


def format_release_ref_for_display(ref_name: str) -> str:
    """Render a git ref as its exact release name for user-facing logs."""
    if ref_name.startswith("refs/heads/"):
        return ref_name[len("refs/heads/") :]
    if ref_name.startswith("refs/tags/"):
        return ref_name[len("refs/tags/") :]
    if ref_name.startswith("refs/remotes/"):
        ref_name = ref_name[len("refs/remotes/") :]
    if ref_name.count("/") == 1:
        return ref_name.split("/", 1)[1]
    return ref_name


def list_release_branch_refs(
    repo_root: Path,
) -> List[Tuple[Tuple[int, int, int], str, str]]:
    """List local branches, remote branches, and tags matching `vMAJOR.MINOR.PATCH`."""
    result = subprocess.run(
        [
            "git",
            "for-each-ref",
            "--format=%(refname) %(refname:short)",
            "refs/heads",
            "refs/remotes",
            "refs/tags",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to list release branches: {result.stderr.strip() or result.stdout.strip()}"
        )

    release_refs = []
    for line in result.stdout.splitlines():
        stripped_line = line.strip()
        if not stripped_line:
            continue

        full_ref, short_ref = stripped_line.split(maxsplit=1)
        if short_ref.endswith("/HEAD"):
            continue

        release_ref_name = extract_exact_release_ref_name(full_ref, short_ref)
        if release_ref_name is None:
            continue

        version = parse_release_branch_version(release_ref_name)
        if version is None:
            continue

        release_refs.append((version, full_ref, short_ref))

    return release_refs


def resolve_latest_release_branch_ref(repo_root: Path) -> str:
    """Resolve the highest semantic version release branch or tag ref."""
    release_refs = list_release_branch_refs(repo_root)
    if not release_refs:
        raise RuntimeError(
            "Could not find any release branches or tags matching vMAJOR.MINOR.PATCH"
        )

    def sort_key(release_ref):
        version, full_ref, short_ref = release_ref
        if full_ref.startswith("refs/heads/"):
            ref_priority = 3
        elif full_ref.startswith("refs/remotes/origin/"):
            ref_priority = 2
        elif full_ref.startswith("refs/tags/"):
            ref_priority = 1
        else:
            ref_priority = 0
        return (
            version,
            ref_priority,
            short_ref,
        )

    _, _, resolved_ref = max(release_refs, key=sort_key)
    return resolved_ref


def read_git_base_model_spec_content(
    model_spec_path: Path, ref: Optional[str] = None
) -> str:
    """Read the git base version of `model_spec.py`."""
    repo_root = model_spec_path.parent.parent.resolve()
    resolved_ref = ref or resolve_latest_release_branch_ref(repo_root)
    relative_path = model_spec_path.resolve().relative_to(repo_root).as_posix()
    git_object = f"{resolved_ref}:{relative_path}"
    result = subprocess.run(
        ["git", "show", git_object],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to read git base content for {relative_path}: {result.stderr.strip()}"
        )
    return result.stdout


def build_release_diff_record(
    before_snapshot: Optional[TemplateSnapshot],
    after_snapshot: Optional[TemplateSnapshot],
    ci_metadata: Optional[CiMetadata] = None,
) -> ReleaseDiffRecord:
    """Create one release-diff record from before/after template snapshots."""
    current_snapshot = after_snapshot or before_snapshot
    if current_snapshot is None:
        raise ValueError(
            "Expected before_snapshot or after_snapshot when building diff"
        )
    ci_metadata = ci_metadata or {}

    return {
        "template_key": current_snapshot["template_key"],
        "impl": current_snapshot["impl"],
        "impl_id": current_snapshot["impl_id"],
        "model_arch": current_snapshot["model_arch"],
        "inference_engine": current_snapshot["inference_engine"],
        "weights": current_snapshot["weights"],
        "devices": current_snapshot["devices"],
        "status_before": before_snapshot["status"] if before_snapshot else None,
        "status_after": after_snapshot["status"] if after_snapshot else None,
        "tt_metal_commit_before": (
            before_snapshot["tt_metal_commit"] if before_snapshot else None
        ),
        "tt_metal_commit_after": (
            after_snapshot["tt_metal_commit"] if after_snapshot else None
        ),
        "vllm_commit_before": before_snapshot["vllm_commit"]
        if before_snapshot
        else None,
        "vllm_commit_after": after_snapshot["vllm_commit"] if after_snapshot else None,
        "ci_job_url": ci_metadata.get("ci_job_url"),
        "ci_run_number": ci_metadata.get("ci_run_number"),
    }


def build_release_diff_records_from_git(
    model_spec_path: Path,
    current_content: Optional[str] = None,
    ci_metadata_by_occurrence: Optional[Dict[Tuple[str, int], CiMetadata]] = None,
    base_ref: Optional[str] = None,
) -> List[ReleaseDiffRecord]:
    """Build final release-diff records from the git diff of `model_spec.py`."""
    current_text = (
        current_content if current_content is not None else model_spec_path.read_text()
    )
    base_text = read_git_base_model_spec_content(model_spec_path, ref=base_ref)
    before_snapshots = build_template_snapshots(
        model_spec_path, base_text, "model_spec_release_diff_before"
    )
    after_snapshots = build_template_snapshots(
        model_spec_path, current_text, "model_spec_release_diff_after"
    )

    ci_metadata_by_occurrence = ci_metadata_by_occurrence or {}
    before_by_template_key = build_snapshot_index_by_template_key(
        before_snapshots, "building release diff records from git (before snapshots)"
    )
    build_snapshot_index_by_template_key(
        after_snapshots, "building release diff records from git (after snapshots)"
    )
    matched_before_occurrence_keys = set()
    records = []
    unmatched_after_snapshots = []

    for after_snapshot in after_snapshots:
        before_snapshot = before_by_template_key.get(after_snapshot["template_key"])
        if before_snapshot is None:
            unmatched_after_snapshots.append(after_snapshot)
            continue

        matched_before_occurrence_keys.add(before_snapshot["occurrence_key"])
        if before_snapshot["template_text"] == after_snapshot["template_text"]:
            continue

        records.append(
            build_release_diff_record(
                before_snapshot,
                after_snapshot,
                ci_metadata_by_occurrence.get(after_snapshot["occurrence_key"]),
            )
        )

    unmatched_before_by_occurrence = {
        snapshot["occurrence_key"]: snapshot
        for snapshot in before_snapshots
        if snapshot["occurrence_key"] not in matched_before_occurrence_keys
    }

    for after_snapshot in unmatched_after_snapshots:
        before_snapshot = unmatched_before_by_occurrence.get(
            after_snapshot["occurrence_key"]
        )
        if (
            before_snapshot
            and before_snapshot["template_text"] == after_snapshot["template_text"]
        ):
            continue

        records.append(
            build_release_diff_record(
                before_snapshot,
                after_snapshot,
                ci_metadata_by_occurrence.get(after_snapshot["occurrence_key"]),
            )
        )

    return records


def generate_release_diff_outputs_from_git(
    model_spec_path: Path,
    output_dir: Path,
    current_content: Optional[str] = None,
    ci_metadata_by_occurrence: Optional[Dict[Tuple[str, int], CiMetadata]] = None,
    base_ref: Optional[str] = None,
):
    """Generate markdown and JSON release-diff artifacts from git-derived records."""
    update_records = build_release_diff_records_from_git(
        model_spec_path,
        current_content=current_content,
        ci_metadata_by_occurrence=ci_metadata_by_occurrence,
        base_ref=base_ref,
    )
    return write_release_diff_outputs(update_records, output_dir)


def apply_release_version_to_manual_updates_from_git(
    model_spec_path: Path,
    current_content: str,
    release_version: str,
    base_ref: Optional[str] = None,
):
    """Stamp release_version on templates whose tt_metal_commit changed in git."""
    base_text = read_git_base_model_spec_content(model_spec_path, ref=base_ref)
    before_snapshots = build_template_snapshots(
        model_spec_path, base_text, "model_spec_release_version_before"
    )
    after_snapshots = build_template_snapshots(
        model_spec_path, current_content, "model_spec_release_version_after"
    )
    template_spans = extract_template_block_spans(current_content)

    if len(template_spans) != len(after_snapshots):
        raise ValueError(
            "Template block count does not match loaded spec_templates count"
        )

    before_by_template_key = build_snapshot_index_by_template_key(
        before_snapshots, "stamping release_version from git (before snapshots)"
    )
    build_snapshot_index_by_template_key(
        after_snapshots, "stamping release_version from git (after snapshots)"
    )
    matched_before_occurrence_keys = set()
    unmatched_after_snapshots = []
    indices_to_update = []

    def maybe_mark_for_release_version(index, before_snapshot, after_snapshot):
        before_commit = before_snapshot["tt_metal_commit"] if before_snapshot else None
        after_commit = after_snapshot["tt_metal_commit"]
        if not after_commit or before_commit == after_commit:
            return
        indices_to_update.append(index)

    for index, after_snapshot in enumerate(after_snapshots):
        before_snapshot = before_by_template_key.get(after_snapshot["template_key"])
        if before_snapshot is None:
            unmatched_after_snapshots.append((index, after_snapshot))
            continue

        matched_before_occurrence_keys.add(before_snapshot["occurrence_key"])
        maybe_mark_for_release_version(index, before_snapshot, after_snapshot)

    unmatched_before_by_occurrence = {
        snapshot["occurrence_key"]: snapshot
        for snapshot in before_snapshots
        if snapshot["occurrence_key"] not in matched_before_occurrence_keys
    }

    for index, after_snapshot in unmatched_after_snapshots:
        before_snapshot = unmatched_before_by_occurrence.get(
            after_snapshot["occurrence_key"]
        )
        maybe_mark_for_release_version(index, before_snapshot, after_snapshot)

    updated_content = current_content
    release_version_updates = 0

    for index in sorted(set(indices_to_update), reverse=True):
        start_pos, end_pos, template_text = template_spans[index]
        updated_template = update_template_fields(
            template_text, None, None, None, release_version=release_version
        )
        if updated_template == template_text:
            continue
        updated_content = (
            updated_content[:start_pos] + updated_template + updated_content[end_pos:]
        )
        release_version_updates += 1

    return updated_content, release_version_updates


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
    from scripts.release.release_performance import load_release_performance_data
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
    release_performance_data = load_release_performance_data()

    # Note: docs/model_support/README.md is no longer generated.
    # The model support content is maintained directly in root README.md
    # via regenerate_model_support_docs_and_update_readme().

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
                    model_name,
                    model_templates,
                    group,
                    release_performance_data=release_performance_data,
                )
                write_file(output_path / subdir / filename, page_content)

    print("Documentation generation complete!")


def regenerate_model_support_docs_and_update_readme(
    model_spec_path, readme_path="README.md"
):
    """
    Regenerate docs/model_support/ and update the Model Support section in README.md.

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


def update_readme_model_support(model_spec_path, readme_path="README.md"):
    """Backward-compatible wrapper for the explicit README/docs regeneration helper."""
    return regenerate_model_support_docs_and_update_readme(model_spec_path, readme_path)


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
        default="default_model_spec.json",
        help=(
            "Path to output JSON file with all model specs "
            "(default: default_model_spec.json)"
        ),
    )
    parser.add_argument(
        "--output-only",
        action="store_true",
        help=(
            "Regenerate release outputs and stamp release_version for templates "
            "whose tt_metal_commit changed in manual edits"
        ),
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
    parser.add_argument(
        "--models-ci-run-id",
        type=int,
        default=None,
        help="GitHub Actions workflow run ID; automatically runs models_ci_reader pipeline to produce the last_good JSON",
    )
    parser.add_argument(
        "--out-root",
        default=None,
        help=(
            "Output directory for CI reader artifacts when using --models-ci-run-id "
            f"(default: {get_versioned_release_logs_dir()})"
        ),
    )

    args = parser.parse_args()

    # Handle --output-only mode
    if args.output_only:
        model_spec_path = Path(args.model_spec_path)
        if not model_spec_path.exists():
            raise FileNotFoundError(f"Error: File not found: {args.model_spec_path}")

        base_release_ref = resolve_latest_release_branch_ref(
            model_spec_path.parent.parent.resolve()
        )
        print(
            "Using latest release branch base ref: "
            f"{format_release_ref_for_display(base_release_ref)}"
        )

        current_content = model_spec_path.read_text()
        updated_content, release_version_updates = (
            apply_release_version_to_manual_updates_from_git(
                model_spec_path, current_content, VERSION, base_ref=base_release_ref
            )
        )

        if updated_content != current_content:
            model_spec_path.write_text(updated_content)
            print(
                "Running in --output-only mode: updated release_version for "
                f"{release_version_updates} manually changed templates"
            )
        else:
            print(
                "Running in --output-only mode: no release_version updates were "
                "needed for manual tt_metal_commit changes"
            )

        release_output_dir = resolve_release_output_dir(args.out_root)
        generate_release_diff_outputs_from_git(
            model_spec_path,
            release_output_dir,
            current_content=updated_content,
            base_ref=base_release_ref,
        )

        # Regenerate docs/model_support/ and update README.md Model Support section.
        regenerate_model_support_docs_and_update_readme(
            model_spec_path, args.readme_path
        )

        # Export MODEL_SPECS to JSON
        output_json_path = Path(args.output_json)
        reload_and_export_model_specs_json(model_spec_path, output_json_path)

        return

    # Validate that exactly one of --models-ci-run-id or last_good_json is provided
    if args.models_ci_run_id and args.last_good_json:
        raise ValueError(
            "Error: provide --models-ci-run-id or last_good_json, not both."
        )
    if not args.models_ci_run_id and not args.last_good_json:
        raise ValueError(
            "Error: last_good_json or --models-ci-run-id is required when not using --output-only"
        )

    release_output_dir = resolve_release_output_dir(args.out_root)

    # Resolve last_good_path: either run the CI pipeline or use the supplied file
    if args.models_ci_run_id:
        from scripts.release.models_ci_reader import run_ci_pipeline

        last_good_path = run_ci_pipeline(
            args.models_ci_run_id,
            release_output_dir,
        )
    else:
        last_good_path = Path(args.last_good_json)
        if not last_good_path.exists():
            raise FileNotFoundError(f"Error: File not found: {args.last_good_json}")

    with open(last_good_path, "r") as f:
        last_good_data = json.load(f)

    # Read model_spec.py
    model_spec_path = Path(args.model_spec_path)
    if not model_spec_path.exists():
        raise FileNotFoundError(f"Error: File not found: {args.model_spec_path}")
    base_release_ref = resolve_latest_release_branch_ref(
        model_spec_path.parent.parent.resolve()
    )
    print(
        "Using latest release branch base ref: "
        f"{format_release_ref_for_display(base_release_ref)}"
    )

    with open(model_spec_path, "r") as f:
        content = f.read()

    # Process each template by iterating spec_templates directly
    print(f"Processing {len(spec_templates)} ModelSpecTemplate objects")

    updated_content = content
    updates_made = 0
    ci_metadata_by_occurrence = {}
    template_occurrence_counts = {}

    for template in spec_templates:
        # Log template being processed
        impl_name = template.impl.impl_name
        weights = template.weights
        devices = [spec.device.name for spec in template.device_model_specs]
        impl_id = template.impl.impl_id
        occurrence_index = template_occurrence_counts.get(impl_id, 0)
        template_occurrence_counts[impl_id] = occurrence_index + 1
        occurrence_key = (impl_id, occurrence_index)
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
            template_text,
            tt_metal_commit,
            vllm_commit,
            status_to_update,
            release_version=VERSION,
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

            ci_job_url, ci_run_number = get_ci_job_url_for_template(
                template, last_good_data
            )

            ci_metadata_by_occurrence[occurrence_key] = {
                "ci_job_url": ci_job_url,
                "ci_run_number": ci_run_number,
            }

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

            generate_release_diff_outputs_from_git(
                model_spec_path,
                release_output_dir,
                current_content=updated_content,
                ci_metadata_by_occurrence=ci_metadata_by_occurrence,
                base_ref=base_release_ref,
            )

            # Regenerate docs/model_support/ and update README.md Model Support section.
            regenerate_model_support_docs_and_update_readme(model_spec_path)
    else:
        print("\nNo updates needed.")

        # Even if no updates were made, export the current MODEL_SPECS to JSON
        if not args.dry_run:
            output_json_path = Path(args.output_json)
            reload_and_export_model_specs_json(model_spec_path, output_json_path)
            generate_release_diff_outputs_from_git(
                model_spec_path,
                release_output_dir,
                current_content=content,
                ci_metadata_by_occurrence=ci_metadata_by_occurrence,
                base_ref=base_release_ref,
            )


if __name__ == "__main__":
    main()
