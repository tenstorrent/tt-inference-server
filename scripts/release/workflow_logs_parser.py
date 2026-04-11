#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Core parsing module for workflow logs artifacts.

This module provides functions to parse workflow_logs_* directories produced by
CI runs, extracting model specifications, performance reports, and status information.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional, Tuple

# Add project root to Python path to allow imports from workflows
import sys

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
from workflows.acceptance_criteria import evaluate_benchmark_targets
from workflows.utils import parse_commits_from_docker_image

logger = logging.getLogger(__name__)


def latest_json_by_mtime(dir_path: Path, pattern: str) -> Optional[Path]:
    """Find the most recently modified JSON file matching a pattern."""
    logger.debug(f"Globbing for pattern '{pattern}' in directory: {dir_path}")
    files = list(dir_path.glob(pattern))
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def load_model_spec_json(model_specs_dir: Path) -> Tuple[Optional[dict], Optional[str]]:
    """Load model spec JSON from model_specs directory.

    Supports both the new model_specs/ directory and legacy run_specs/ as fallback.

    Args:
        model_specs_dir: Path to model_specs directory

    Returns:
        Tuple of (model_spec_dict, model_id)
    """
    spec_file = latest_json_by_mtime(model_specs_dir, "*.json")
    if not spec_file:
        return None, None
    try:
        logger.info(f"Reading model spec JSON: {spec_file}")
        data = json.loads(spec_file.read_text())
    except Exception as e:
        logger.warning(f"Failed to parse model spec JSON from {spec_file}: {e}")
        return None, None
    model_spec_json = data.get("runtime_model_spec")
    if isinstance(model_spec_json, dict):
        flattened_model_spec_json = dict(model_spec_json)
        runtime_config = data.get("runtime_config")
        if runtime_config is not None:
            flattened_model_spec_json["runtime_config"] = runtime_config
        model_spec_json = flattened_model_spec_json
    elif isinstance(data, dict):
        model_spec_json = data
    else:
        return None, None

    model_id = model_spec_json.get("model_id")
    return model_spec_json, model_id


def find_report_data_json_path(reports_root: Path, model_id: str) -> Optional[Path]:
    """Locate the newest report_data JSON for a model inside reports_output."""
    if not reports_root.exists():
        logger.debug(f"Reports root does not exist: {reports_root}")
        return None

    for workflow_dir in reports_root.iterdir():
        if not workflow_dir.is_dir():
            continue
        data_dir = workflow_dir / "data"
        if not data_dir.is_dir():
            continue

        report_file = latest_json_by_mtime(data_dir, f"report_data_{model_id}_*.json")
        if not report_file:
            report_file = latest_json_by_mtime(data_dir, "report_data_*.json")
        if report_file:
            return report_file

    logger.debug(f"No report data found for model_id: {model_id}")
    return None


def load_report_data_json(reports_root: Path, model_id: str) -> Optional[dict]:
    """Load report_data JSON from reports_output directory.

    Args:
        reports_root: Path to reports_output directory
        model_id: Model identifier to find specific report

    Returns:
        Report data dict or None if not found
    """
    report_file = find_report_data_json_path(reports_root, model_id)
    if not report_file:
        return None
    try:
        logger.info(f"Reading report data JSON: {report_file}")
        return json.loads(report_file.read_text())
    except Exception as e:
        logger.warning(f"Failed to parse report data JSON from {report_file}: {e}")
        return None


def parse_perf_status(report_data: dict) -> str:
    """Extract performance status from report_data.

    Returns the support-tier benchmark status from benchmark_target_evaluation.

    Args:
        report_data: Report data dictionary

    Returns:
        Performance status: "target", "complete", "functional", or "experimental"
    """
    try:
        benchmark_target_evaluation = _get_benchmark_target_evaluation(report_data)
        return benchmark_target_evaluation.get("status", "experimental")
    except Exception as e:
        logger.debug(f"Error parsing perf status: {e}")
        return "experimental"


def parse_accuracy_status(report_data: dict) -> bool:
    """Extract accuracy status from report_data.

    Args:
        report_data: Report data dictionary

    Returns:
        True if all accuracy checks pass (no check == 3) AND all entries have accuracy_check field, False otherwise
    """
    try:
        evals = report_data.get("evals", [])
        if not evals:
            return False
        for e in evals:
            accuracy_check = e.get("accuracy_check")
            if accuracy_check is None:
                return False
            if accuracy_check == 3:
                return False
        return True
    except Exception as e:
        logger.debug(f"Error parsing accuracy status: {e}")
        return False


def parse_evals_completed(report_data: dict) -> bool:
    """Check if evaluation data is complete (accuracy_check field exists).

    Args:
        report_data: Report data dictionary

    Returns:
        True if all eval entries have accuracy_check field, False otherwise
    """
    try:
        evals = report_data.get("evals", [])
        if not evals:
            return False
        for e in evals:
            if e.get("accuracy_check") is None:
                return False
        return True
    except Exception as e:
        logger.debug(f"Error parsing evals completed status: {e}")
        return False


def parse_benchmarks_completed(report_data: dict) -> bool:
    """Check if benchmark data is complete enough for target evaluation.

    Args:
        report_data: Report data dictionary

    Returns:
        True if benchmark references are available and benchmark evaluation has no errors.
    """
    try:
        benchmark_target_evaluation = _get_benchmark_target_evaluation(report_data)
        return benchmark_target_evaluation.get(
            "reference_available", False
        ) and not bool(benchmark_target_evaluation.get("errors"))
    except Exception as e:
        logger.debug(f"Error parsing benchmarks completed status: {e}")
        return False


def _get_benchmark_target_evaluation(
    report_data: dict,
    model_spec: Optional[Any] = None,
    *,
    prefer_report_evaluation: bool = True,
) -> dict:
    benchmark_target_evaluation = report_data.get("benchmark_target_evaluation")
    if prefer_report_evaluation and isinstance(benchmark_target_evaluation, dict):
        return benchmark_target_evaluation
    return evaluate_benchmark_targets(report_data, model_spec=model_spec)


def _get_regression_summary(
    benchmark_target_evaluation: dict,
) -> Tuple[bool, bool, bool]:
    regression = benchmark_target_evaluation.get("regression", {})
    regression_checked = bool(regression.get("checked"))
    regression_passed = bool(regression.get("passed")) if regression_checked else True
    regression_ok = regression_passed if regression_checked else True
    return regression_checked, regression_passed, regression_ok


def build_parsed_workflow_logs_data(
    dir_name: str,
    model_spec_json: dict,
    report_data_json: dict,
    *,
    resolved_model_spec: Optional[Any] = None,
    prefer_report_benchmark_target_evaluation: bool = True,
    report_data_json_path: Optional[Path] = None,
) -> Optional[dict]:
    """Build parsed workflow-log style data from already-loaded JSON payloads."""
    model_id = model_spec_json.get("model_id")
    if not model_id:
        logger.warning(f"Could not find model_id in parsed model spec for {dir_name}")
        return None

    benchmark_target_evaluation = _get_benchmark_target_evaluation(
        report_data_json,
        model_spec=resolved_model_spec,
        prefer_report_evaluation=prefer_report_benchmark_target_evaluation,
    )
    perf_status = benchmark_target_evaluation.get("status", "experimental")
    benchmarks_completed = benchmark_target_evaluation.get(
        "reference_available", False
    ) and not bool(benchmark_target_evaluation.get("errors"))
    accuracy_status = parse_accuracy_status(report_data_json)
    evals_completed = parse_evals_completed(report_data_json)
    regression_checked, regression_passed, regression_ok = _get_regression_summary(
        benchmark_target_evaluation
    )
    is_passing = benchmarks_completed and accuracy_status and regression_ok

    spec_docker_image = model_spec_json.get("docker_image")
    override_docker_image = model_spec_json.get("cli_args", {}).get(
        "override_docker_image"
    )
    docker_image = override_docker_image if override_docker_image else spec_docker_image

    tt_metal_commit = None
    vllm_commit = None
    if docker_image:
        tt_metal_commit, vllm_commit = parse_commits_from_docker_image(docker_image)

    logger.info(
        f"Successfully parsed {dir_name}: model_id={model_id}, "
        f"perf={perf_status}, benchmarks_completed={benchmarks_completed}, "
        f"accuracy={accuracy_status}, evals_completed={evals_completed}, "
        f"regression_ok={regression_ok}, passing={is_passing}"
    )

    return {
        "dir_name": dir_name,
        "summary": {
            "model_id": model_id,
            "perf_status": perf_status,
            "benchmarks_completed": benchmarks_completed,
            "accuracy_status": accuracy_status,
            "evals_completed": evals_completed,
            "regression_checked": regression_checked,
            "regression_passed": regression_passed,
            "regression_ok": regression_ok,
            "is_passing": is_passing,
            "docker_image": docker_image,
            "tt_metal_commit": tt_metal_commit,
            "vllm_commit": vllm_commit,
        },
        "model_specs": model_spec_json,
        "reports_output": report_data_json,
        "report_data_json_path": str(report_data_json_path)
        if report_data_json_path
        else None,
        "tt_smi_output": None,
    }


def parse_workflow_logs_dir(
    workflow_logs_dir: Path, last_run_only: bool = True
) -> Optional[dict]:
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
                "benchmarks_completed": bool, # True if benchmark_target_evaluation can be computed
                "accuracy_status": bool,
                "evals_completed": bool, # True if eval entries have accuracy_check field
                "regression_checked": bool,
                "regression_passed": bool,
                "regression_ok": bool,
                "is_passing": bool,     # True if benchmarks, accuracy, and regression checks all pass
                "docker_image": str,    # Docker image from model_spec
                "tt_metal_commit": str, # tt-metal commit parsed from docker image tag
                "vllm_commit": str      # vllm commit parsed from docker image tag
            },
            "model_specs": {
                "model_spec": dict      # Model specification JSON
            },
            "reports_output": dict,     # Full report_data_*.json payload
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

    # Load model spec (try new model_specs/ first, fall back to legacy run_specs/)
    model_specs_dir = workflow_logs_dir / "runtime_model_specs"
    if not model_specs_dir.is_dir():
        model_specs_dir = workflow_logs_dir / "run_specs"
    model_spec_json, model_id = load_model_spec_json(model_specs_dir)
    if not model_id:
        logger.warning(f"Could not find model_id in {workflow_logs_dir}")
        return None

    # Load report data
    reports_root = workflow_logs_dir / "reports_output"
    report_data_json_path = find_report_data_json_path(reports_root, model_id)
    if not report_data_json_path:
        logger.warning(f"Could not find report data in {workflow_logs_dir}")
        return None
    try:
        logger.info(f"Reading report data JSON: {report_data_json_path}")
        report_data_json = json.loads(report_data_json_path.read_text())
    except Exception as exc:
        logger.warning(
            f"Failed to parse report data JSON from {report_data_json_path}: {exc}"
        )
        return None
    if not report_data_json:
        logger.warning(f"Could not find report data in {workflow_logs_dir}")
        return None

    return build_parsed_workflow_logs_data(
        workflow_logs_dir.name,
        model_spec_json,
        report_data_json,
        report_data_json_path=report_data_json_path,
    )


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
