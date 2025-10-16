#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Update model_spec.py with commits from CI last_good_json results.

This script reads a last_good_json file and updates tt_metal_commit and vllm_commit
fields in model_spec.py for each ModelSpecTemplate based on CI results.

Usage:
    python3 update_model_spec_commits.py <last_good_json_path>
    python3 update_model_spec_commits.py --dry-run <last_good_json_path>
    python3 update_model_spec_commits.py --help

Example:
    python3 update_model_spec_commits.py release_logs/models_ci_last_good_*.json

The script:
- Parses all ModelSpecTemplate blocks in model_spec.py
- For each template, checks all weights to find matching model_ids in the JSON
- Validates that all devices in a template have consistent commits
- Updates commits in-place (7-character hash format)
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
from workflows.model_spec import spec_templates, ModelSpecTemplate


def map_perf_status_to_model_status(perf_status):
    """Map CI perf_status string to ModelStatusTypes enum value."""
    status_map = {
        'experimental': 'ModelStatusTypes.EXPERIMENTAL',
        'functional': 'ModelStatusTypes.FUNCTIONAL',
        'complete': 'ModelStatusTypes.COMPLETE',
        'top_perf': 'ModelStatusTypes.TOP_PERF',
    }
    return status_map.get(perf_status.lower() if perf_status else None)


def model_name_from_weight(weight):
    """Extract model name from HuggingFace repo path."""
    return Path(weight).name


def find_template_in_content(content, impl_id, first_weight):
    """
    Find a specific ModelSpecTemplate block in file content.
    
    Args:
        content: The raw text content of model_spec.py
        impl_id: The impl_id to search for
        first_weight: The first weight in the template's weights list
    
    Returns:
        The matched template text, or None if not found
    """
    # Escape special regex characters in the weight string
    escaped_weight = re.escape(first_weight)
    
    # Find all ModelSpecTemplate blocks and match them manually
    # This approach handles nested parentheses correctly
    start_pattern = rf'ModelSpecTemplate\(\s*weights=\[.*?"{escaped_weight}".*?\].*?impl={impl_id}_impl'
    
    for match in re.finditer(start_pattern, content, re.DOTALL):
        # Found a potential match, now find the complete template block
        start_pos = match.start()
        pos = match.end()
        paren_count = 1  # We're inside ModelSpecTemplate(
        
        # Scan forward to find the matching closing parenthesis
        while pos < len(content) and paren_count > 0:
            if content[pos] == '(':
                paren_count += 1
            elif content[pos] == ')':
                paren_count -= 1
            pos += 1
        
        if paren_count == 0:
            # Found the complete template block
            # Include the trailing comma if present
            template_text = content[start_pos:pos]
            if pos < len(content) and content[pos] == ',':
                template_text += ','
            return template_text
    
    return None


def get_commits_for_template(template: ModelSpecTemplate, last_good_data):
    """
    Get commits and status for a template from last_good_json data.
    
    Args:
        template: ModelSpecTemplate object
        last_good_data: Dictionary of CI results
    
    Returns tuple: (tt_metal_commit, vllm_commit, status, should_update)
    """
    # Get impl directly from template
    impl = template.impl
    weights = template.weights
    devices = [spec.device for spec in template.device_model_specs]
    
    if not weights:
        return None, None, None, False
    
    # Collect commits and status from all devices and all weights
    # We check all weights to find matching CI data, not just weights[0]
    device_commits = []
    for weight in weights:
        model_name = model_name_from_weight(weight)
        for device in devices:
            model_id = f"id_{impl.impl_name}_{model_name}_{device.name.lower()}"
            if model_id in last_good_data:
                print(f"model_id: {model_id}")
                entry = last_good_data[model_id]
                # Skip empty entries
                if entry:
                    tt_metal = entry.get('tt_metal_commit', '')
                    vllm = entry.get('vllm_commit', '')
                    perf_status = entry.get('perf_status', '')
                    if tt_metal or vllm:
                        device_commits.append({
                            'model_id': model_id,
                            'tt_metal_commit': tt_metal[:7] if tt_metal else '',
                            'vllm_commit': vllm[:7] if vllm else '',
                            'perf_status': perf_status,
                        })
    
    # If no devices have data, skip this template
    if not device_commits:
        return None, None, None, False
    
    # Check for conflicting commits across devices and warn
    tt_metal_commits = set(dc['tt_metal_commit'] for dc in device_commits if dc['tt_metal_commit'])
    vllm_commits = set(dc['vllm_commit'] for dc in device_commits if dc['vllm_commit'])
    perf_statuses = set(dc['perf_status'] for dc in device_commits if dc['perf_status'])
    
    if len(tt_metal_commits) > 1:
        model_ids = [dc['model_id'] for dc in device_commits]
        print(f"\nWarning: Multiple tt_metal_commits found for template. Using first device. model_ids: {model_ids}")
        print(f"  Found commits: {tt_metal_commits}")
    
    if len(vllm_commits) > 1:
        model_ids = [dc['model_id'] for dc in device_commits]
        print(f"\nWarning: Multiple vllm_commits found for template. Using first device. model_ids: {model_ids}")
        print(f"  Found commits: {vllm_commits}")
    
    if len(perf_statuses) > 1:
        model_ids = [dc['model_id'] for dc in device_commits]
        print(f"\nWarning: Multiple perf_statuses found for template. Using first device. model_ids: {model_ids}")
        print(f"  Found statuses: {perf_statuses}")
    
    # Use the first device's commits and status deterministically
    first_device = device_commits[0]
    tt_metal_commit = first_device['tt_metal_commit'] if first_device['tt_metal_commit'] else None
    vllm_commit = first_device['vllm_commit'] if first_device['vllm_commit'] else None
    
    status = None
    if first_device['perf_status']:
        status = map_perf_status_to_model_status(first_device['perf_status'])
    
    return tt_metal_commit, vllm_commit, status, True


def extract_status(template_text):
    """Extract current status from template text."""
    match = re.search(r'status=ModelStatusTypes\.(\w+)', template_text)
    if match:
        return match.group(1)
    return None


def get_ci_job_url_for_template(template: ModelSpecTemplate, last_good_data):
    """
    Get CI job URL for a template from last_good_json data.
    
    Args:
        template: ModelSpecTemplate object
        last_good_data: Dictionary of CI results
    
    Returns the ci_job_url from the first device with data.
    """
    # Get impl, weights, and devices directly from template
    impl = template.impl
    weights = template.weights
    devices = [spec.device for spec in template.device_model_specs]
    
    if not weights:
        return None
    
    # Look for ci_job_url in any device with data
    for weight in weights:
        model_name = model_name_from_weight(weight)
        for device in devices:
            model_id = f"id_{impl.impl_name}_{model_name}_{device.name.lower()}"
            
            if model_id in last_good_data:
                entry = last_good_data[model_id]
                if entry and 'ci_job_url' in entry:
                    return entry['ci_job_url']
    
    return None


def update_template_fields(template_text, tt_metal_commit, vllm_commit, status):
    """Update commit and status values in a template text."""
    updated = template_text
    
    if tt_metal_commit:
        # Replace tt_metal_commit value
        updated = re.sub(
            r'tt_metal_commit="[^"]*"',
            f'tt_metal_commit="{tt_metal_commit}"',
            updated
        )
    
    if vllm_commit:
        # Replace vllm_commit value
        updated = re.sub(
            r'vllm_commit="[^"]*"',
            f'vllm_commit="{vllm_commit}"',
            updated
        )
    
    if status:
        # Replace status value (matches ModelStatusTypes.XXXX format)
        updated = re.sub(
            r'status=ModelStatusTypes\.\w+',
            f'status={status}',
            updated
        )
    
    return updated


def generate_release_diff_markdown(update_records, output_path):
    """
    Generate release_models_diff.md markdown file with update details.
    
    Args:
        update_records: List of dicts with keys: impl, weights, devices, 
                       status_before, status_after, ci_job_url
        output_path: Path where markdown file should be written
    """
    lines = []
    lines.append("# Model Spec Release Updates\n")
    lines.append(f"\nThis document shows model specification updates.\n")
    
    if not update_records:
        lines.append("\nNo updates were made.\n")
    else:
        # Create single table header
        lines.append("| Model Arch | Weights | Devices | Status Change | CI Job Link |")
        lines.append("|------------|---------|---------|---------------|-------------|")
        
        # Add rows for each record
        for record in update_records:
            # Model architecture
            model_arch = f"`{record['model_arch']}`"
            
            # Weights (formatted list with line breaks)
            weights_formatted = "<br>".join([f"`{w}`" for w in record['weights']])
            
            # Devices (comma-separated)
            devices = ", ".join(record['devices'])
            
            # Status change
            if record['status_before'] and record['status_after']:
                status_before = record['status_before']
                status_after_match = re.search(r'ModelStatusTypes\.(\w+)', record['status_after'])
                status_after = status_after_match.group(1) if status_after_match else record['status_after']
                
                if status_before != status_after:
                    status_change = f"{status_before} → {status_after}"
                else:
                    status_change = "No change"
            elif record['status_after']:
                status_after_match = re.search(r'ModelStatusTypes\.(\w+)', record['status_after'])
                status_after = status_after_match.group(1) if status_after_match else record['status_after']
                status_change = f"New: {status_after}"
            else:
                status_change = "No change"
            
            # CI Job Link
            if record['ci_job_url']:
                ci_link = f"[View Job]({record['ci_job_url']})"
            else:
                ci_link = "N/A"
            
            # Add table row
            lines.append(f"| {model_arch} | {weights_formatted} | {devices} | {status_change} | {ci_link} |")
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write("\n".join(lines))
    
    print(f"\nGenerated release diff markdown: {output_path}")


def export_model_specs_json(model_spec_path, output_json_path):
    """
    Dynamically import MODEL_SPECS from updated model_spec.py and export to JSON.
    
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
    spec.loader.exec_module(model_spec_module)
    
    # Get MODEL_SPECS dictionary from the module
    model_specs = model_spec_module.MODEL_SPECS
    
    # Serialize all ModelSpec instances
    serialized_specs = {}
    for model_id, model_spec in model_specs.items():
        serialized_specs[model_id] = model_spec.get_serialized_dict()
    
    # Write to JSON file
    with open(output_json_path, 'w') as f:
        json.dump(serialized_specs, f, indent=2)
    
    print(f"\nExported {len(serialized_specs)} model specs to {output_json_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Update model_spec.py commits from last_good_json CI results'
    )
    parser.add_argument(
        'last_good_json',
        help='Path to last_good_json file with CI results'
    )
    parser.add_argument(
        '--model-spec-path',
        default='workflows/model_spec.py',
        help='Path to model_spec.py file (default: workflows/model_spec.py)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print changes without modifying files'
    )
    parser.add_argument(
        '--output-json',
        default='model_specs_output.json',
        help='Path to output JSON file with all model specs (default: model_specs_output.json)'
    )
    
    args = parser.parse_args()
    
    # Load last_good_json
    last_good_path = Path(args.last_good_json)
    if not last_good_path.exists():
        raise FileNotFoundError(f"Error: File not found: {args.last_good_json}")

    with open(last_good_path, 'r') as f:
        last_good_data = json.load(f)
    
    # Read model_spec.py
    model_spec_path = Path(args.model_spec_path)
    if not model_spec_path.exists():
        raise FileNotFoundError(f"Error: File not found: {args.model_spec_path}")
    
    with open(model_spec_path, 'r') as f:
        content = f.read()
    
    # Process each template by iterating spec_templates directly
    print(f"Processing {len(spec_templates)} ModelSpecTemplate objects")
    
    updated_content = content
    updates_made = 0
    update_records = []  # Track updates for markdown generation
    
    for template in spec_templates:
        # Get commits and status for this template from CI data
        tt_metal_commit, vllm_commit, status, should_update = get_commits_for_template(
            template, last_good_data
        )
        
        if should_update:
            # Build unique identifier to find this template in text
            impl_id = template.impl.impl_id
            first_weight = template.weights[0] if template.weights else ""
            
            if not first_weight:
                continue
            
            # Find and extract the template text on-demand
            template_text = find_template_in_content(updated_content, impl_id, first_weight)
            
            if not template_text:
                print(f"\nWarning: Could not find template in file for impl={impl_id}, weight={first_weight}")
                continue
            
            # Update the template fields
            updated_template = update_template_fields(template_text, tt_metal_commit, vllm_commit, status)
            
            if updated_template != template_text:
                # Get info from template object for logging
                weights = template.weights
                impl_name = template.impl.impl_name
                
                print(f"\nUpdating template:")
                print(f"  impl: {impl_name}")
                print(f"  weights[0]: {weights[0] if weights else 'unknown'}")
                if tt_metal_commit:
                    print(f"  tt_metal_commit: {tt_metal_commit}")
                if vllm_commit:
                    print(f"  vllm_commit: {vllm_commit}")
                if status:
                    print(f"  status: {status}")
                
                # Track update for markdown generation
                status_before = extract_status(template_text)
                devices = [spec.device.name for spec in template.device_model_specs]
                ci_job_url = get_ci_job_url_for_template(template, last_good_data)
                model_arch = model_name_from_weight(weights[0]) if weights else 'unknown'
                
                update_records.append({
                    'impl': impl_name,
                    'model_arch': model_arch,
                    'weights': weights,
                    'devices': devices,
                    'status_before': status_before,
                    'status_after': status,
                    'ci_job_url': ci_job_url,
                })
                
                # Replace this specific template in content
                updated_content = updated_content.replace(template_text, updated_template, 1)
                updates_made += 1
    
    # Write updated content
    if updates_made > 0:
        if args.dry_run:
            print(f"\n[DRY RUN] Would update {updates_made} templates in {args.model_spec_path}")
        else:
            with open(model_spec_path, 'w') as f:
                f.write(updated_content)
            print(f"\nSuccessfully updated {updates_made} templates in {args.model_spec_path}")
            
            # Export MODEL_SPECS to JSON
            output_json_path = Path(args.output_json)
            export_model_specs_json(model_spec_path, output_json_path)
            
            # Generate release diff markdown
            if update_records:
                diff_markdown_path = last_good_path.parent / "release_models_diff.md"
                generate_release_diff_markdown(update_records, diff_markdown_path)
    else:
        print("\nNo updates needed.")
        
        # Even if no updates were made, export the current MODEL_SPECS to JSON
        if not args.dry_run:
            output_json_path = Path(args.output_json)
            export_model_specs_json(model_spec_path, output_json_path)


if __name__ == '__main__':
    main()

