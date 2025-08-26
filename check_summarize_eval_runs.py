#!/usr/bin/env python3
"""
Python 3.10 script to check for failed accuracy evaluations in workflow report files.

This script searches for report files matching the pattern:
workflow_logs/reports_output/release/report_id_llama3-70b-galaxy_Llama-3.3-70B-Instruct_galaxy_*.md

It looks for "Accuracy Evaluations for Llama-3.3-70B-Instruct on galaxy" sections
and identifies any evaluations marked as "FAIL".
"""

import os
import re
import sys
from pathlib import Path
from glob import glob


def find_report_files(base_dir: str) -> list[str]:
    """Find all matching report files."""
    pattern = os.path.join(
        base_dir,
        "workflow_logs/reports_output/release/report_id_llama3-70b-galaxy_Llama-3.3-70B-Instruct_galaxy_*.md"
    )
    return glob(pattern)


def parse_accuracy_section(content: str) -> list[dict]:
    """
    Parse the accuracy evaluations section from markdown content.
    Returns a list of failed evaluations with details.
    """
    failed_evals = []
    
    # Look for the accuracy evaluations section
    accuracy_pattern = r"### Accuracy Evaluations for Llama-3\.3-70B-Instruct on galaxy\s*\n\n(.*?)(?=\n###|\n##|\Z)"
    accuracy_match = re.search(accuracy_pattern, content, re.DOTALL | re.IGNORECASE)
    
    if not accuracy_match:
        return failed_evals
    
    accuracy_section = accuracy_match.group(1)
    
    # Find the markdown table within the accuracy section
    # Look for table rows that contain "FAIL"
    table_pattern = r"\|.*?\|.*?\|.*?\|.*?\|"
    table_rows = re.findall(table_pattern, accuracy_section)
    
    # Parse table header to understand column positions
    header_found = False
    column_indices = {}
    
    for row in table_rows:
        cells = [cell.strip() for cell in row.split('|')[1:-1]]  # Remove empty first/last elements
        
        if not header_found and cells:
            # Check if this looks like a header row
            if any(header_word in cell.lower() for cell in cells for header_word in ['model', 'device', 'task', 'accuracy']):
                for i, cell in enumerate(cells):
                    column_indices[cell.lower().replace('_', ' ').strip()] = i
                header_found = True
                continue
        
        # Skip separator rows (contain only dashes and pipes)
        if all(set(cell.strip()) <= {'-', ' '} for cell in cells if cell.strip()):
            continue
        
        # Check for FAIL in accuracy_check column or any cell containing "FAIL"
        if any("FAIL" in cell for cell in cells):
            eval_info = {
                'model': cells[column_indices.get('model', 0)] if 'model' in column_indices else cells[0] if cells else 'Unknown',
                'device': cells[column_indices.get('device', 1)] if 'device' in column_indices else cells[1] if len(cells) > 1 else 'Unknown',
                'task_name': cells[column_indices.get('task name', 2)] if 'task name' in column_indices else cells[2] if len(cells) > 2 else 'Unknown',
                'accuracy_check': cells[column_indices.get('accuracy check', 3)] if 'accuracy check' in column_indices else 'Unknown',
                'raw_row': '|'.join(cells)
            }
            
            # Only add if it actually contains FAIL
            if "FAIL" in eval_info['accuracy_check'] or any("FAIL" in cell for cell in cells):
                failed_evals.append(eval_info)
    
    return failed_evals


def parse_benchmark_section(content: str) -> dict:
    """
    Parse the benchmark targets section to extract TTFT and Tput User metrics for ISL=128/OSL=128.
    Returns a dictionary with min/max values or None if not found.
    Note: TPOT is not available in the targets section, only in sweeps section.
    """
    benchmark_metrics = {
        'ttft_values': [],
        'tpot_values': [],  # Will remain empty since targets section doesn't have TPOT
        'tput_user_values': [],  # Add Tput User values
        'min_ttft': None,
        'max_ttft': None,
        'min_tpot': None,
        'max_tpot': None,
        'min_tput_user': None,
        'max_tput_user': None
    }
    
    # Look for benchmark targets section specifically
    benchmark_pattern = r"#### Text-to-Text Performance Benchmark Targets Llama-3\.3-70B-Instruct on galaxy\s*\n\n(.*?)(?=\n####|\n###|\n##|\Z)"
    benchmark_match = re.search(benchmark_pattern, content, re.DOTALL | re.IGNORECASE)
    
    if not benchmark_match:
        return benchmark_metrics
    
    benchmark_section = benchmark_match.group(1)
    
    # Find table rows - match entire rows that contain pipes
    table_rows = []
    for line in benchmark_section.split('\n'):
        if '|' in line and line.strip():
            table_rows.append(line.strip())
    
    # Parse table header to find column positions
    header_found = False
    column_indices = {}
    
    for row in table_rows:
        cells = [cell.strip() for cell in row.split('|')[1:-1]]  # Remove empty first/last elements
        
        if not header_found and cells:
            # Check if this looks like a header row (contains ISL, OSL, TTFT, Tput User)
            if any(header_word in cell.upper() for cell in cells for header_word in ['ISL', 'OSL', 'TTFT', 'TPUT USER']):
                for i, cell in enumerate(cells):
                    cell_clean = cell.strip().upper()
                    if 'ISL' in cell_clean:
                        column_indices['isl'] = i
                    elif 'OSL' in cell_clean:
                        column_indices['osl'] = i
                    elif 'TTFT' in cell_clean and 'TARGET' not in cell_clean and 'CHECK' not in cell_clean and 'FUNCTIONAL' not in cell_clean and 'COMPLETE' not in cell_clean:
                        # Get the first/main TTFT column (actual measured values)
                        if 'ttft' not in column_indices:  # Only take the first one
                            column_indices['ttft'] = i
                    elif 'TPUT USER' in cell_clean and 'TARGET' not in cell_clean and 'CHECK' not in cell_clean and 'FUNCTIONAL' not in cell_clean and 'COMPLETE' not in cell_clean:
                        # Get the first/main Tput User column (actual measured values)
                        if 'tput_user' not in column_indices:  # Only take the first one
                            column_indices['tput_user'] = i
                header_found = True
                continue
        
        # Skip separator rows (contain only dashes and pipes)
        if all(set(cell.strip()) <= {'-', ' '} for cell in cells if cell.strip()):
            continue
        
        # Look for ISL=128, OSL=128 rows
        if len(cells) > max(column_indices.values()) if column_indices else False:
            try:
                isl_val = cells[column_indices.get('isl', 0)].strip()
                osl_val = cells[column_indices.get('osl', 1)].strip()
                
                if isl_val == '128' and osl_val == '128':
                    # Extract TTFT and Tput User values (TPOT not available in targets section)
                    ttft_cell = cells[column_indices.get('ttft', -1)].strip()
                    tput_user_cell = cells[column_indices.get('tput_user', -1)].strip()
                    
                    # Parse numeric values
                    try:
                        ttft_val = float(ttft_cell)
                        benchmark_metrics['ttft_values'].append(ttft_val)
                    except (ValueError, IndexError):
                        pass
                    
                    try:
                        tput_user_val = float(tput_user_cell)
                        benchmark_metrics['tput_user_values'].append(tput_user_val)
                    except (ValueError, IndexError):
                        pass
            except (IndexError, ValueError):
                continue
    
    # Calculate min/max values
    if benchmark_metrics['ttft_values']:
        benchmark_metrics['min_ttft'] = min(benchmark_metrics['ttft_values'])
        benchmark_metrics['max_ttft'] = max(benchmark_metrics['ttft_values'])
    
    if benchmark_metrics['tpot_values']:
        benchmark_metrics['min_tpot'] = min(benchmark_metrics['tpot_values'])
        benchmark_metrics['max_tpot'] = max(benchmark_metrics['tpot_values'])
    
    if benchmark_metrics['tput_user_values']:
        benchmark_metrics['min_tput_user'] = min(benchmark_metrics['tput_user_values'])
        benchmark_metrics['max_tput_user'] = max(benchmark_metrics['tput_user_values'])
    
    return benchmark_metrics


def check_report_file(file_path: str) -> dict:
    """
    Check a single report file for failed accuracy evaluations and benchmark metrics.
    Returns a dictionary with file info, failed evaluations, and benchmark data.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        failed_evals = parse_accuracy_section(content)
        benchmark_metrics = parse_benchmark_section(content)
        
        return {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'failed_evaluations': failed_evals,
            'has_failures': len(failed_evals) > 0,
            'benchmark_metrics': benchmark_metrics,
            'error': None
        }
    
    except Exception as e:
        return {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'failed_evaluations': [],
            'has_failures': False,
            'benchmark_metrics': {
                'ttft_values': [],
                'tpot_values': [],
                'tput_user_values': [],
                'min_ttft': None,
                'max_ttft': None,
                'min_tpot': None,
                'max_tpot': None,
                'min_tput_user': None,
                'max_tput_user': None
            },
            'error': str(e)
        }


def main():
    """Main function to check all report files for failed evaluations."""
    # Use current directory as base, or accept command line argument
    base_dir = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    
    print(f"Checking for failed accuracy evaluations in: {base_dir}")
    print("=" * 80)
    
    # Find all matching report files
    report_files = find_report_files(base_dir)
    
    if not report_files:
        print("No matching report files found.")
        print(f"Looking for pattern: workflow_logs/reports_output/release/report_id_llama3-70b-galaxy_Llama-3.3-70B-Instruct_galaxy_*.md")
        return 0
    
    print(f"Found {len(report_files)} report file(s):")
    for file_path in report_files:
        print(f"  - {file_path}")
    print()
    
    # Check each file
    total_failures = 0
    files_with_failures = 0
    all_ttft_values = []
    all_tpot_values = []
    all_tput_user_values = []
    ttft_file_mapping = []  # List of (value, file_path) tuples
    tput_user_file_mapping = []  # List of (value, file_path) tuples
    
    for file_path in report_files:
        result = check_report_file(file_path)
        
        # Collect benchmark metrics from all files with file path tracking
        if result['benchmark_metrics']['ttft_values']:
            all_ttft_values.extend(result['benchmark_metrics']['ttft_values'])
            # Track which file each TTFT value came from
            for ttft_val in result['benchmark_metrics']['ttft_values']:
                ttft_file_mapping.append((ttft_val, file_path))
        if result['benchmark_metrics']['tpot_values']:
            all_tpot_values.extend(result['benchmark_metrics']['tpot_values'])
        if result['benchmark_metrics']['tput_user_values']:
            all_tput_user_values.extend(result['benchmark_metrics']['tput_user_values'])
            # Track which file each Tput User value came from
            for tput_user_val in result['benchmark_metrics']['tput_user_values']:
                tput_user_file_mapping.append((tput_user_val, file_path))
        
        if result['error']:
            print(f"File: {result['file_name']}")
            print(f"Path: {result['file_path']}")
            print(f"ERROR: {result['error']}")
            print("-" * 80)
        elif result['has_failures']:
            files_with_failures += 1
            print(f"File: {result['file_name']}")
            print(f"Path: {result['file_path']}")
            print(f"FAILED EVALUATIONS FOUND: {len(result['failed_evaluations'])}")
            
            for i, eval_info in enumerate(result['failed_evaluations'], 1):
                print(f"  {i}. Task: {eval_info['task_name']}")
                print(f"     Model: {eval_info['model']}")
                print(f"     Device: {eval_info['device']}")
                print(f"     Status: {eval_info['accuracy_check']}")
                print(f"     Raw: {eval_info['raw_row']}")
                print()
            
            total_failures += len(result['failed_evaluations'])
            print("-" * 80)
        # Skip files with no failures - don't print anything for them
    
    # Summary
    print(f"\nSUMMARY:")
    print(f"Total files checked: {len(report_files)}")
    print(f"Files with failures: {files_with_failures}")
    print(f"Total failed evaluations: {total_failures}")
    
    # Benchmark metrics summary for ISL=128/OSL=128 from targets section
    print(f"\nBENCHMARK METRICS (ISL=128/OSL=128) from Targets Section:")
    if all_ttft_values:
        min_ttft = min(all_ttft_values)
        max_ttft = max(all_ttft_values)
        print(f"TTFT (Time To First Token): Min={min_ttft:.1f}ms, Max={max_ttft:.1f}ms")
        print(f"TTFT data points: {len(all_ttft_values)} values across all reports")
        
        # Find file paths for min and max values
        min_file = None
        max_file = None
        for ttft_val, file_path in ttft_file_mapping:
            if ttft_val == min_ttft and min_file is None:
                min_file = file_path
            if ttft_val == max_ttft and max_file is None:
                max_file = file_path
        
        if min_file:
            print(f"Min TTFT file: {min_file}")
        if max_file:
            print(f"Max TTFT file: {max_file}")
    else:
        print("TTFT: No data found for ISL=128/OSL=128 in targets section")
    
    # Tput User metrics
    if all_tput_user_values:
        min_tput_user = min(all_tput_user_values)
        max_tput_user = max(all_tput_user_values)
        print(f"Tput User (Throughput per User): Min={min_tput_user:.2f} TPS, Max={max_tput_user:.2f} TPS")
        print(f"Tput User data points: {len(all_tput_user_values)} values across all reports")
        
        # Find file paths for min and max values
        min_tput_user_file = None
        max_tput_user_file = None
        for tput_user_val, file_path in tput_user_file_mapping:
            if tput_user_val == min_tput_user and min_tput_user_file is None:
                min_tput_user_file = file_path
            if tput_user_val == max_tput_user and max_tput_user_file is None:
                max_tput_user_file = file_path
        
        if min_tput_user_file:
            print(f"Min Tput User file: {min_tput_user_file}")
        if max_tput_user_file:
            print(f"Max Tput User file: {max_tput_user_file}")
    else:
        print("Tput User: No data found for ISL=128/OSL=128 in targets section")
    
    print("TPOT: Not available in targets section (only available in sweeps section)")
    
    # Return non-zero exit code if failures found
    return 1 if total_failures > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
