#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Core parsing module for workflow logs artifacts.

This module provides functions to parse workflow_logs_* directories produced by
CI runs, extracting model specifications, performance reports, and status information.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def parse_commits_from_docker_image(docker_image: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract tt-metal and vllm commits from docker image tag.
    
    Expected format: registry/image:version-tt_metal_commit-vllm_commit-timestamp
    Example: ghcr.io/.../image:0.0.5-55fd115b891c705b86a41d265ae1638b2a71fa7d-aa4ae1e-52275478061
    
    The tag format typically includes:
    - Version number (e.g., 0.0.5)
    - tt-metal commit (40-character hex hash)
    - vllm commit (7-character hex hash)
    - timestamp (numeric)
    
    Args:
        docker_image: Full docker image string with tag
    
    Returns:
        Tuple of (tt_metal_commit, vllm_commit) or (None, None) if parsing fails
    """
    if not docker_image or ':' not in docker_image:
        return None, None
    
    try:
        # Extract tag portion after the last ':'
        tag = docker_image.split(':')[-1]
        
        # Pattern to match: version-tt_metal_commit(40)-vllm_commit(7)-timestamp
        # Example: 0.0.5-55fd115b891c705b86a41d265ae1638b2a71fa7d-aa4ae1e-52275478061
        pattern = r'^([0-9.]+)-([0-9a-fA-F]{40})-([0-9a-fA-F]{7})-(\d+)$'
        match = re.match(pattern, tag)
        
        if match:
            version, tt_metal_commit, vllm_commit, timestamp = match.groups()
            logger.info(f"Parsed commits from docker image tag: tt-metal={tt_metal_commit}, vllm={vllm_commit}")
            return tt_metal_commit, vllm_commit
        else:
            logger.debug(f"Docker image tag does not match expected format: {tag}")
            return None, None
            
    except Exception as e:
        logger.debug(f"Failed to parse commits from docker image '{docker_image}': {e}")
        return None, None


def latest_json_by_mtime(dir_path: Path, pattern: str) -> Optional[Path]:
    """Find the most recently modified JSON file matching a pattern."""
    logger.debug(f"Globbing for pattern '{pattern}' in directory: {dir_path}")
    files = list(dir_path.glob(pattern))
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def load_model_spec_json(run_specs_dir: Path) -> Tuple[Optional[dict], Optional[str]]:
    """Load model spec JSON from run_specs directory.
    
    Args:
        run_specs_dir: Path to run_specs directory
        
    Returns:
        Tuple of (model_spec_dict, model_id)
    """
    spec_file = latest_json_by_mtime(run_specs_dir, "*.json")
    if not spec_file:
        return None, None
    try:
        logger.info(f"Reading model spec JSON: {spec_file}")
        data = json.loads(spec_file.read_text())
    except Exception as e:
        logger.warning(f"Failed to parse model spec JSON from {spec_file}: {e}")
        return None, None
    model_id = data.get("model_id")
    return data, model_id


def load_report_data_json(reports_root: Path, model_id: str) -> Optional[dict]:
    """Load report_data JSON from reports_output directory.
    
    Args:
        reports_root: Path to reports_output directory
        model_id: Model identifier to find specific report
        
    Returns:
        Report data dict or None if not found
    """
    if not reports_root.exists():
        logger.debug(f"Reports root does not exist: {reports_root}")
        return None
    
    # Workflow subdirectory can vary (release, benchmarks, evals)
    for workflow_dir in reports_root.iterdir():
        if not workflow_dir.is_dir():
            continue
        data_dir = workflow_dir / "data"
        if not data_dir.is_dir():
            continue
        
        # Try model-specific report first
        report_file = latest_json_by_mtime(data_dir, f"report_data_{model_id}_*.json")
        if not report_file:
            # Fallback: any report_data_*.json
            report_file = latest_json_by_mtime(data_dir, "report_data_*.json")
        
        if report_file:
            try:
                logger.info(f"Reading report data JSON: {report_file}")
                return json.loads(report_file.read_text())
            except Exception as e:
                logger.warning(f"Failed to parse report data JSON from {report_file}: {e}")
                continue
    
    logger.debug(f"No report data found for model_id: {model_id}")
    return None


def parse_perf_status(report_data: dict) -> str:
    """Extract performance status from report_data.
    
    Determines highest target achieved among target, complete, functional.
    Pass condition for a level: all checks != 3
    
    Args:
        report_data: Report data dictionary
        
    Returns:
        Performance status: "target", "complete", "functional", or "experimental"
    """
    try:
        summaries = report_data.get("benchmarks_summary", [])
        if not summaries:
            return "experimental"
        target_checks = summaries[0].get("target_checks", {})
        
        def passes(checks: dict) -> bool:
            if not isinstance(checks, dict):
                return False
            ttft_check = checks.get("ttft_check")
            tput_user_check = checks.get("tput_user_check")
            tput_check = checks.get("tput_check")
            return all(x is not None and x != 3 for x in (ttft_check, tput_user_check, tput_check))

        # Order of highest to lowest
        if passes(target_checks.get("target", {})):
            return "target"
        if passes(target_checks.get("complete", {})):
            return "complete"
        if passes(target_checks.get("functional", {})):
            return "functional"
        return "experimental"
    except Exception as e:
        logger.debug(f"Error parsing perf status: {e}")
        return "experimental"


def parse_accuracy_status(report_data: dict) -> bool:
    """Extract accuracy status from report_data.
    
    Args:
        report_data: Report data dictionary
        
    Returns:
        True if all accuracy checks pass (no check == 3), False otherwise
    """
    try:
        evals = report_data.get("evals", [])
        if not evals:
            return False
        for e in evals:
            if e.get("accuracy_check") == 3:
                return False
        return True
    except Exception as e:
        logger.debug(f"Error parsing accuracy status: {e}")
        return False


def parse_workflow_logs_dir(workflow_logs_dir: Path, last_run_only: bool = True) -> Optional[dict]:
    """Parse a single workflow_logs_* directory and return structured data.
    
    This is the main entry point for parsing a workflow logs directory.
    It extracts model specifications and performance/accuracy reports.
    
    Args:
        workflow_logs_dir: Path to workflow_logs_* directory
        last_run_only: Optional boolean to only parse the last run
        
    Returns:
        Dict with structure organized by directory:
        {
            "dir_name": str,       # directory name
            "summary": {
                "model_id": str,
                "perf_status": str,     # "target"|"complete"|"functional"|"experimental"
                "accuracy_status": bool,
                "is_passing": bool,     # True if perf_status != "experimental" and accuracy_status == True
                "docker_image": str,    # Docker image from model_spec
                "tt_metal_commit": str, # tt-metal commit parsed from docker image tag
                "vllm_commit": str      # vllm commit parsed from docker image tag
            },
            "run_specs": {
                "model_spec": dict      # Model specification JSON
            },
            "reports_output": {
                "report_data": dict     # Performance and evaluation report data
            },
            "tt_smi_output": {
                "firmware_bundle": str, # Firmware bundle version
                "kmd_version": str,     # KMD driver version
                "tt_smi": dict          # tt-smi output without device_info
            }
        }
        or None if parsing fails
    """

    if not last_run_only:
        raise NotImplementedError("Only supports last_run_only=True")
    
    logger.info(f"Parsing workflow logs directory: {workflow_logs_dir}")
    
    if not workflow_logs_dir.exists():
        logger.error(f"Directory does not exist: {workflow_logs_dir}")
        return None
    
    if not workflow_logs_dir.is_dir():
        logger.error(f"Path is not a directory: {workflow_logs_dir}")
        return None
    
    # Load model spec
    run_specs_dir = workflow_logs_dir / "run_specs"
    model_spec_json, model_id = load_model_spec_json(run_specs_dir)
    if not model_id:
        logger.warning(f"Could not find model_id in {workflow_logs_dir}")
        return None
    
    # Load report data
    reports_root = workflow_logs_dir / "reports_output"
    report_data_json = load_report_data_json(reports_root, model_id)
    if not report_data_json:
        logger.warning(f"Could not find report data in {workflow_logs_dir}")
        return None
    
    # Parse status
    perf_status = parse_perf_status(report_data_json)
    accuracy_status = parse_accuracy_status(report_data_json)
    is_passing = (perf_status != "experimental") and accuracy_status
    
    # Extract docker image and parse commits
    spec_docker_image = model_spec_json.get("docker_image") if model_spec_json else None
    override_docker_image = model_spec_json.get("cli_args", {}).get("override_docker_image") if model_spec_json else None
    docker_image = override_docker_image if override_docker_image else spec_docker_image
    
    tt_metal_commit = None
    vllm_commit = None
    if docker_image:
        tt_metal_commit, vllm_commit = parse_commits_from_docker_image(docker_image)
    
    logger.info(f"Successfully parsed {workflow_logs_dir.name}: model_id={model_id}, "
                f"perf={perf_status}, accuracy={accuracy_status}, passing={is_passing}")
    

    # TODO: add tt_smi_output parsing
    tt_smi_output = None

    result = {
        "dir_name": workflow_logs_dir.name,
        "summary":
            {
            "model_id": model_id,
            "perf_status": perf_status,
            "accuracy_status": accuracy_status,
            "is_passing": is_passing,
            "docker_image": docker_image,
            "tt_metal_commit": tt_metal_commit,
            "vllm_commit": vllm_commit,
            },
        "run_specs": model_spec_json,
        "reports_output": report_data_json,
        "tt_smi_output": tt_smi_output,
    }
    # TODO: return lists of results if last_run_only is False
    
    return result


def write_workflow_logs_output(parsed_data: dict, output_path: Path) -> None:
    """Write parsed workflow logs data to JSON file.
    
    Args:
        parsed_data: Dict returned by parse_workflow_logs_dir()
        output_path: Path to output JSON file
    """
    logger.info(f"Writing workflow logs output to: {output_path}")
    
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write JSON with indentation
    output_text = json.dumps(parsed_data, indent=2)
    output_path.write_text(output_text)
    
    logger.info(f"Wrote {len(output_text.encode('utf-8'))} bytes to {output_path}")

