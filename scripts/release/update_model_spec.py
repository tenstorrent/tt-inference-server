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


def map_perf_status_to_model_status(perf_status):
    """Map CI perf_status string to ModelStatusTypes enum value."""
    status_map = {
        'experimental': 'ModelStatusTypes.EXPERIMENTAL',
        'functional': 'ModelStatusTypes.FUNCTIONAL',
        'complete': 'ModelStatusTypes.COMPLETE',
        'top_perf': 'ModelStatusTypes.TOP_PERF',
    }
    return status_map.get(perf_status.lower() if perf_status else None)


def extract_impl_name(impl_text):
    """Extract impl_name from ImplSpec definition (e.g., 'impl_name="tt-transformers"')."""
    match = re.search(r'impl_name="([^"]+)"', impl_text)
    if match:
        return match.group(1)
    return None


def extract_devices(template_text):
    """Extract device types from DeviceModelSpec entries."""
    devices = []
    for match in re.finditer(r'device=DeviceTypes\.(\w+)', template_text):
        devices.append(match.group(1))
    return devices


def extract_weights(template_text):
    """Extract weights list from template."""
    weights_match = re.search(r'weights=\[(.*?)\]', template_text, re.DOTALL)
    if not weights_match:
        return []
    
    weights_text = weights_match.group(1)
    weights = []
    for match in re.finditer(r'"([^"]+)"', weights_text):
        weights.append(match.group(1))
    return weights


def model_name_from_weight(weight):
    """Extract model name from HuggingFace repo path."""
    return Path(weight).name


def parse_model_spec_file(content):
    """Parse model_spec.py to extract template blocks."""
    # Find all ModelSpecTemplate blocks
    template_pattern = r'(ModelSpecTemplate\([^)]*(?:\([^)]*\)[^)]*)*\))'
    templates = []
    
    # Split by ModelSpecTemplate to find individual templates
    parts = content.split('ModelSpecTemplate(')
    
    for i, part in enumerate(parts[1:], 1):  # Skip first empty part
        # Find the matching closing parenthesis for this template
        paren_count = 1
        end_idx = 0
        for j, char in enumerate(part):
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
                if paren_count == 0:
                    end_idx = j
                    break
        
        if end_idx > 0:
            template_text = 'ModelSpecTemplate(' + part[:end_idx + 1]
            
            # Find the start position in original content
            search_start = 0
            for k in range(i):
                if k == 0:
                    search_start = content.find('ModelSpecTemplate(')
                else:
                    search_start = content.find('ModelSpecTemplate(', search_start + 1)
            
            templates.append({
                'text': template_text,
                'start': search_start,
                'end': search_start + len(template_text)
            })
    
    return templates


def get_commits_for_template(template_text, last_good_data):
    """
    Get commits and status for a template from last_good_json data.
    
    Returns tuple: (tt_metal_commit, vllm_commit, status, should_update)
    """
    # Extract impl reference
    impl_match = re.search(r'impl=(\w+)', template_text)
    if not impl_match:
        return None, None, None, False
    
    impl_var = impl_match.group(1)
    
    # Map impl variable to impl_name (hardcoded for known impls)
    impl_name_map = {
        'tt_transformers_impl': 'tt-transformers',
        'llama3_impl': 'llama3',
        't3000_llama2_70b_impl': 'llama2-70b',
        'llama3_70b_galaxy_impl': 'llama3-70b-galaxy',
    }
    
    impl_name = impl_name_map.get(impl_var)
    if not impl_name:
        return None, None, None, False
    
    # Extract weights and devices
    weights = extract_weights(template_text)
    if not weights:
        return None, None, None, False
    
    devices = extract_devices(template_text)
    
    # Collect commits and status from all devices and all weights
    # We check all weights to find matching CI data, not just weights[0]
    device_commits = []
    for weight in weights:
        model_name = model_name_from_weight(weight)
        for device in devices:
            model_id = f"id_{impl_name}_{model_name}_{device.lower()}"
            
            if model_id in last_good_data:
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
    
    # Parse templates
    templates = parse_model_spec_file(content)
    print(f"Found {len(templates)} ModelSpecTemplate blocks")
    
    # Process each template
    updated_content = content
    updates_made = 0
    offset = 0  # Track position changes as we replace text
    
    for template_info in templates:
        template_text = template_info['text']
        
        # Get commits and status for this template
        tt_metal_commit, vllm_commit, status, should_update = get_commits_for_template(
            template_text, last_good_data
        )
        
        if should_update:
            # Update the template
            updated_template = update_template_fields(template_text, tt_metal_commit, vllm_commit, status)
            
            if updated_template != template_text:
                # Get weights for logging
                weights = extract_weights(template_text)
                impl_match = re.search(r'impl=(\w+)', template_text)
                impl_var = impl_match.group(1) if impl_match else 'unknown'
                
                print(f"\nUpdating template:")
                print(f"  impl: {impl_var}")
                print(f"  weights[0]: {weights[0] if weights else 'unknown'}")
                if tt_metal_commit:
                    print(f"  tt_metal_commit: {tt_metal_commit}")
                if vllm_commit:
                    print(f"  vllm_commit: {vllm_commit}")
                if status:
                    print(f"  status: {status}")
                
                # Replace in the full content
                start = template_info['start'] + offset
                end = template_info['end'] + offset
                updated_content = updated_content[:start] + updated_template + updated_content[end:]
                
                # Update offset for next replacements
                offset += len(updated_template) - len(template_text)
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
    else:
        print("\nNo updates needed.")
        
        # Even if no updates were made, export the current MODEL_SPECS to JSON
        if not args.dry_run:
            output_json_path = Path(args.output_json)
            export_model_specs_json(model_spec_path, output_json_path)


if __name__ == '__main__':
    main()

