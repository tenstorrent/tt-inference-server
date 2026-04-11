#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Generate Step-5 release compatibility, documentation, and performance artifacts.

This script owns the release-artifact generation step by:
- Loading CI or standalone report data for the finalized release branch
- Recomputing acceptance warnings from raw release report data
- Updating the checked-in release performance baseline and diff output
- Exporting `release_model_spec.json`
- Regenerating `docs/model_support/` and the root `README.md`

Usage:
    python3 generate_release_artifacts.py <ci_artifacts_path> --release [--dry-run]
    python3 generate_release_artifacts.py --models-ci-run-id <run_id> --release [--dry-run]
    python3 generate_release_artifacts.py --report-data-json <path> --release [--dry-run]
    python3 generate_release_artifacts.py --help
"""

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

# Add project root to Python path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
from workflows.model_spec import MODEL_SPECS
from workflows.acceptance_criteria import (
    acceptance_criteria_check,
    evaluate_benchmark_targets,
    format_acceptance_summary_markdown,
)
from workflows.utils import get_version, parse_commits_from_docker_image

try:
    from generate_model_support_docs import (
        regenerate_model_support_docs_and_update_readme,
    )
    from generate_release_notes import (
        build_release_notes,
        load_git_release_performance_data,
        load_optional_json,
    )
    from release_performance import (
        build_release_performance_diff_data,
        get_release_performance_path,
        iter_release_performance_items,
        load_release_performance_data,
        ReleasePerformanceWriteMode,
        update_release_performance_outputs,
        write_release_performance_diff_data,
        write_release_performance_data,
    )
    from release_paths import get_versioned_release_logs_dir, resolve_release_output_dir
    from update_model_spec import reload_and_export_model_specs_json
    from models_ci_reader import (
        DEFAULT_OWNER as MODELS_CI_DEFAULT_OWNER,
        DEFAULT_REPO as MODELS_CI_DEFAULT_REPO,
        check_auth as models_ci_check_auth,
        download_runs as models_ci_download_runs,
        process_run_directory as process_ci_run_directory,
    )
    from workflow_logs_parser import build_parsed_workflow_logs_data
    from workflows.release_report_markdown import build_release_report_markdown
    from workflows.reports_schema import validate_report_file
except ImportError:
    from scripts.release.generate_model_support_docs import (
        regenerate_model_support_docs_and_update_readme,
    )
    from scripts.release.generate_release_notes import (
        build_release_notes,
        load_git_release_performance_data,
        load_optional_json,
    )
    from scripts.release.release_performance import (
        build_release_performance_diff_data,
        get_release_performance_path,
        iter_release_performance_items,
        load_release_performance_data,
        ReleasePerformanceWriteMode,
        update_release_performance_outputs,
        write_release_performance_diff_data,
        write_release_performance_data,
    )
    from scripts.release.release_paths import (
        get_versioned_release_logs_dir,
        resolve_release_output_dir,
    )
    from scripts.release.update_model_spec import reload_and_export_model_specs_json
    from scripts.release.models_ci_reader import (
        DEFAULT_OWNER as MODELS_CI_DEFAULT_OWNER,
        DEFAULT_REPO as MODELS_CI_DEFAULT_REPO,
        check_auth as models_ci_check_auth,
        download_runs as models_ci_download_runs,
        process_run_directory as process_ci_run_directory,
    )
    from scripts.release.workflow_logs_parser import build_parsed_workflow_logs_data
    from workflows.release_report_markdown import build_release_report_markdown
    from workflows.reports_schema import validate_report_file

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
IMAGE_EXISTS_TIMEOUT_SECONDS = 30
IMAGE_COPY_TIMEOUT_SECONDS = 600
IMAGE_STATUS_EXISTS_WITH_CI = "exists_with_ci"
IMAGE_STATUS_EXISTS_WITHOUT_CI = "exists_without_ci"
IMAGE_STATUS_COPY_FROM_CI = "copy_from_ci"
IMAGE_STATUS_COPIED = "copied"
IMAGE_STATUS_NEEDS_BUILD = "needs_build"
RELEASE_WORKFLOW_FILE = "release.yml"
CommitPair = Tuple[Optional[str], Optional[str]]
REPORT_RENDERER_DIRNAME = "release_report_renderer"
LEGACY_REPORT_SECTION_MARKDOWN_FIELDS = (
    "benchmarks_markdown",
    "aiperf_benchmarks_markdown",
    "genai_perf_benchmarks_markdown",
    "evals_markdown",
    "parameter_support_tests_markdown",
    "stress_tests_markdown",
    "server_tests_markdown",
)


@dataclass(frozen=True)
class MergedModelRecord:
    """Release-artifact inputs for a single model."""

    model_id: str
    model_spec: Any
    ci_data: Dict[str, Any]
    target_docker_image: str


@dataclass(frozen=True)
class ImagePlan:
    """One image-level decision shared by all models using the same docker image."""

    target_image: str
    model_ids: Tuple[str, ...]
    status: str
    ci_source_image: Optional[str] = None


@dataclass(frozen=True)
class ReleaseArtifactsRequest:
    """Typed Step-5 request built from CLI inputs."""

    merged_spec: Dict[str, MergedModelRecord]
    output_dir: Path
    model_spec_path: Path
    readme_path: Path
    release_model_spec_path: Path
    version: str
    dry_run: bool
    release: bool


@dataclass(frozen=True)
class ReleaseArtifactsResult:
    """Typed Step-5 result used by wrappers and tests."""

    generated_artifacts: Tuple[str, ...]
    acceptance_warnings: Tuple[Dict[str, Any], ...]
    acceptance_warnings_path: Optional[Path]
    release_performance_path: Optional[Path]
    release_performance_diff_path: Optional[Path]


def configure_logging() -> None:
    """Configure CLI logging only when the script is executed."""
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def get_target_docker_image(model_id: str, model_spec: Any, is_dev: bool) -> str:
    """Return the release/dev target image for a model without mutating MODEL_SPECS."""
    docker_image = model_spec.docker_image
    if not docker_image:
        raise ValueError(f"MODEL_SPECS entry {model_id} is missing docker_image")
    if is_dev:
        return docker_image.replace("-release-", "-dev-")
    return docker_image


def check_crane_installed() -> bool:
    """Check if crane tool is installed and available."""
    if not shutil.which("crane"):
        logger.error("crane tool not found in PATH")
        logger.error(
            "Please install crane: https://github.com/google/go-containerregistry/blob/main/cmd/crane/README.md"
        )
        return False
    return True


def check_docker_installed() -> bool:
    """Check if docker is installed and available."""
    if not shutil.which("docker"):
        logger.error("docker not found in PATH")
        logger.error("Please install docker: https://docs.docker.com/get-docker/")
        return False
    return True


def _load_run_ci_metadata(run_out_dir: Path) -> Optional[Dict[str, Any]]:
    metadata_path = run_out_dir / "run_ci_metadata.json"
    if not metadata_path.exists():
        return None
    with metadata_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _build_raw_ci_entry_sort_key(
    entry: Dict[str, Any],
) -> Tuple[int, str, str, str, bool]:
    ci_metadata = entry.get("ci_metadata") or {}
    ci_job_metadata = ci_metadata.get("ci_job_metadata") or {}
    run_number = ci_metadata.get("run_number")
    try:
        numeric_run_number = int(run_number)
    except (TypeError, ValueError):
        numeric_run_number = -1
    return (
        numeric_run_number,
        str(ci_job_metadata.get("completed_at") or ""),
        str(ci_job_metadata.get("started_at") or ""),
        str(entry.get("job_run_datetimestamp") or ""),
        bool((entry.get("workflow_logs") or {}).get("reports_output")),
    )


def _select_release_ci_entry(entries: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Pick the newest raw CI entry for a model from release workflow artifacts."""
    if not entries:
        return None
    return max(entries, key=_build_raw_ci_entry_sort_key)


def _flatten_raw_ci_entry(raw_entry: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize one raw CI entry into the shape expected by release generators."""
    if not raw_entry:
        return {}

    workflow_logs = raw_entry.get("workflow_logs") or {}
    workflow_summary = workflow_logs.get("summary") or {}
    release_report = workflow_logs.get("reports_output") or {}
    report_data_json_path = workflow_logs.get("report_data_json_path")
    ci_metadata = raw_entry.get("ci_metadata") or {}
    ci_job_metadata = ci_metadata.get("ci_job_metadata") or {}
    ci_logs = raw_entry.get("ci_logs") or {}
    return {
        "tt_metal_commit": workflow_summary.get("tt_metal_commit"),
        "vllm_commit": workflow_summary.get("vllm_commit"),
        "docker_image": workflow_summary.get("docker_image"),
        "perf_status": workflow_summary.get("perf_status"),
        "benchmarks_completed": workflow_summary.get("benchmarks_completed"),
        "accuracy_status": workflow_summary.get("accuracy_status"),
        "evals_completed": workflow_summary.get("evals_completed"),
        "regression_checked": workflow_summary.get("regression_checked"),
        "regression_passed": workflow_summary.get("regression_passed"),
        "regression_ok": workflow_summary.get("regression_ok"),
        "is_passing": workflow_summary.get("is_passing"),
        "ci_run_id": ci_metadata.get("run_id"),
        "ci_run_number": ci_metadata.get("run_number"),
        "ci_run_url": ci_metadata.get("ci_run_url"),
        "ci_job_id": ci_job_metadata.get("job_id"),
        "ci_job_url": ci_job_metadata.get("job_url"),
        "ci_job_name": ci_job_metadata.get("job_name"),
        "ci_job_status": ci_job_metadata.get("job_status"),
        "ci_job_conclusion": ci_job_metadata.get("job_conclusion"),
        "ci_job_started_at": ci_job_metadata.get("started_at"),
        "ci_job_completed_at": ci_job_metadata.get("completed_at"),
        "firmware_bundle": ci_logs.get("firmware_bundle"),
        "driver_version": ci_logs.get("kmd_version"),
        "release_report": release_report if isinstance(release_report, dict) else {},
        "release_report_json_path": report_data_json_path,
    }


def build_ci_data_from_raw_results(
    all_models_dict: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    """Collapse raw workflow-log entries to one normalized CI record per model."""
    ci_data_by_model: Dict[str, Dict[str, Any]] = {}
    for model_id, entries in all_models_dict.items():
        ci_data_by_model[model_id] = _flatten_raw_ci_entry(
            _select_release_ci_entry(entries)
        )
    return ci_data_by_model


def resolve_ci_run_directory(ci_artifacts_path: Path) -> Path:
    """Resolve a raw CI artifacts path to exactly one downloaded run directory."""
    resolved_path = ci_artifacts_path.resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"CI artifacts path not found: {resolved_path}")
    if not resolved_path.is_dir():
        raise ValueError(f"CI artifacts path must be a directory, got: {resolved_path}")

    if (resolved_path / "run_ci_metadata.json").exists():
        return resolved_path

    ci_run_logs_dir = resolved_path / "ci_run_logs"
    if ci_run_logs_dir.is_dir():
        run_dirs = sorted(
            path
            for path in ci_run_logs_dir.iterdir()
            if path.is_dir() and path.name.startswith("On_nightly_")
        )
        if len(run_dirs) != 1:
            raise ValueError(
                "Expected exactly one CI run directory under "
                f"{ci_run_logs_dir}, found {len(run_dirs)}. "
                "Pass the specific On_nightly_* directory instead."
            )
        return run_dirs[0]

    raise ValueError(
        "Could not resolve a CI run directory from path: "
        f"{resolved_path}. Expected a downloaded On_nightly_* directory or a "
        "release_logs/vX.Y.Z directory containing one ci_run_logs/ run."
    )


def load_raw_ci_results_from_run_directory(
    run_out_dir: Path,
) -> Dict[str, List[Dict[str, Any]]]:
    """Parse one downloaded CI run directory into raw per-model entries."""
    run_ci_metadata = _load_run_ci_metadata(run_out_dir)
    run_timestamp = datetime.fromtimestamp(run_out_dir.stat().st_mtime).strftime(
        "%Y-%m-%d_%H-%M-%S"
    )
    return process_ci_run_directory(run_out_dir, run_timestamp, run_ci_metadata)


def _find_run_directory_for_run_id(out_root: Path, run_id: int) -> Path:
    ci_run_logs_dir = out_root / "ci_run_logs"
    if not ci_run_logs_dir.is_dir():
        raise FileNotFoundError(f"No ci_run_logs directory found under {out_root}")
    matching_dirs = sorted(
        path
        for path in ci_run_logs_dir.iterdir()
        if path.is_dir() and path.name.endswith(f"_{run_id}")
    )
    if len(matching_dirs) != 1:
        raise ValueError(
            "Expected exactly one downloaded CI run directory for run id "
            f"{run_id}, found {len(matching_dirs)}"
        )
    return matching_dirs[0]


def load_ci_data_from_run_id(
    run_id: int,
    out_root: Path,
    *,
    workflow_file: str,
) -> Dict[str, Dict[str, Any]]:
    """Download raw CI artifacts for one run id and build normalized model records."""
    token = models_ci_check_auth(MODELS_CI_DEFAULT_OWNER, MODELS_CI_DEFAULT_REPO)
    models_ci_download_runs(
        MODELS_CI_DEFAULT_OWNER,
        MODELS_CI_DEFAULT_REPO,
        workflow_file,
        token,
        out_root,
        max_runs=1,
        run_id=run_id,
    )
    run_out_dir = _find_run_directory_for_run_id(out_root, run_id)
    return build_ci_data_from_raw_results(
        load_raw_ci_results_from_run_directory(run_out_dir)
    )


def load_ci_data_from_artifacts_path(
    ci_artifacts_path: Path,
) -> Dict[str, Dict[str, Any]]:
    """Load normalized CI data directly from downloaded raw workflow-log artifacts."""
    run_out_dir = resolve_ci_run_directory(ci_artifacts_path)
    return build_ci_data_from_raw_results(
        load_raw_ci_results_from_run_directory(run_out_dir)
    )


def _load_report_data_json(report_data_json_path: Path) -> Dict[str, Any]:
    resolved_path = report_data_json_path.resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Report data JSON not found: {resolved_path}")
    if not resolved_path.is_file():
        raise ValueError(f"Report data JSON must be a file, got: {resolved_path}")

    with resolved_path.open("r", encoding="utf-8") as file:
        report_data = json.load(file)
    if not isinstance(report_data, dict):
        raise ValueError(f"Report data JSON must contain an object: {resolved_path}")
    return report_data


def _build_report_only_ci_entry(
    report_path: Path,
    model_spec: Any,
    report_data: Dict[str, Any],
) -> Dict[str, Any]:
    parsed_workflow_logs = build_parsed_workflow_logs_data(
        report_path.stem,
        model_spec.get_serialized_dict(),
        report_data,
        resolved_model_spec=model_spec,
        prefer_report_benchmark_target_evaluation=False,
        report_data_json_path=report_path,
    )
    if not parsed_workflow_logs:
        raise ValueError(f"Could not parse standalone report data: {report_path}")

    return _flatten_raw_ci_entry(
        {
            "workflow_logs": {
                "summary": parsed_workflow_logs.get("summary") or {},
                "reports_output": parsed_workflow_logs.get("reports_output") or {},
                "report_data_json_path": parsed_workflow_logs.get(
                    "report_data_json_path"
                ),
            },
            "ci_metadata": {},
            "ci_logs": {},
        }
    )


def load_ci_data_from_report_data_json(
    report_data_json_path: Path,
) -> Dict[str, Dict[str, Any]]:
    """Build one normalized CI entry from a standalone report_data JSON file."""
    report_data = _load_report_data_json(report_data_json_path)
    metadata = report_data.get("metadata") or {}
    model_id = metadata.get("model_id")
    if not model_id:
        raise ValueError(
            f"Report data JSON is missing metadata.model_id: {report_data_json_path}"
        )

    model_spec = MODEL_SPECS.get(model_id)
    if model_spec is None:
        raise ValueError(
            "Report data JSON references model_id not present in checked-in "
            f"MODEL_SPECS: {model_id}"
        )

    return {
        model_id: _build_report_only_ci_entry(
            report_data_json_path, model_spec, report_data
        )
    }


def build_merged_spec_from_report_data_json(
    report_data_json_path: Path, is_dev: bool
) -> Dict[str, MergedModelRecord]:
    """Build merged Step-5 inputs from one standalone report_data JSON file."""
    return merge_specs_with_ci_data(
        load_ci_data_from_report_data_json(report_data_json_path),
        is_dev=is_dev,
    )


def merge_specs_with_ci_data(
    ci_data: Dict[str, Dict[str, Any]], is_dev: bool
) -> Dict[str, MergedModelRecord]:
    """Merge normalized CI model data with MODEL_SPECS."""
    logger.info(f"Loaded normalized CI data for {len(ci_data)} model entries")

    merged: Dict[str, MergedModelRecord] = {}
    for model_id, model_spec in MODEL_SPECS.items():
        ci_entry = ci_data.get(model_id, {})
        merged[model_id] = MergedModelRecord(
            model_id=model_id,
            model_spec=model_spec,
            ci_data=ci_entry,
            target_docker_image=get_target_docker_image(model_id, model_spec, is_dev),
        )

    logger.info(f"Merged {len(merged)} model specs with CI data")
    logger.info(
        f"Models with CI data: {len([m for m in merged.values() if m.ci_data])}"
    )
    logger.info(
        f"Models without CI data: {len([m for m in merged.values() if not m.ci_data])}"
    )

    return merged


def run_registry_command(
    command: List[str], timeout_seconds: int, action_description: str
) -> Optional[subprocess.CompletedProcess]:
    """Run a registry command with consistent timeout and exception handling."""
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        logger.warning(
            f"Timeout while {action_description} ({timeout_seconds} seconds exceeded)"
        )
        return None
    except Exception as exc:
        logger.error(f"Exception while {action_description}: {exc}")
        return None


def check_image_exists(image: str, cache: Optional[Dict[str, bool]] = None) -> bool:
    """
    Check if a docker image exists on remote using docker manifest inspect.

    Args:
        image: Full docker image name with registry and tag
        cache: Optional cache dictionary to store and reuse results

    Returns:
        True if image exists, False otherwise
    """
    if cache is not None and image in cache:
        logger.debug(f"Cache HIT for image: {image}")
        return cache[image]

    logger.debug(f"Cache MISS for image: {image}")

    result = run_registry_command(
        ["docker", "manifest", "inspect", image],
        IMAGE_EXISTS_TIMEOUT_SECONDS,
        f"checking image existence for {image}",
    )
    exists = result is not None and result.returncode == 0

    if cache is not None:
        cache[image] = exists

    return exists


def extract_commits_from_tag(docker_image: str) -> Optional[CommitPair]:
    """
    Extract tt-metal and vllm commit hashes from docker image tag.

    Expected format: registry/name:version-tt_metal_commit-vllm_commit[-build_id]

    Args:
        docker_image: Full docker image name with registry and tag

    Returns:
        Tuple of (tt_metal_commit, vllm_commit) or None if parsing fails
        vllm_commit can be None for images that don't have a vllm component
    """
    tt_metal_commit, vllm_commit = parse_commits_from_docker_image(docker_image)
    if not tt_metal_commit:
        logger.debug(f"Unable to parse commits from docker image tag: {docker_image}")
        return None
    return (tt_metal_commit, vllm_commit)


def commits_match(
    release_commits: Optional[CommitPair],
    ci_commits: Optional[CommitPair],
    model_spec: Any,
) -> bool:
    """
    Check if commits from release and CI docker images match.

    Compares the prefixes of commit hashes since CI images may have full hashes
    while MODEL_SPECS may have short hashes (or vice versa).
    """
    if release_commits is None or ci_commits is None:
        logger.debug("Cannot validate commits: parsing failed for one or both images")
        return False

    release_tt_metal, release_vllm = release_commits
    ci_tt_metal, ci_vllm = ci_commits

    expected_tt_metal = model_spec.tt_metal_commit
    expected_vllm = model_spec.vllm_commit

    if not (release_tt_metal and expected_tt_metal):
        logger.debug("Missing tt-metal commit in release image or model spec")
        return False

    if not (
        release_tt_metal.startswith(expected_tt_metal)
        or expected_tt_metal.startswith(release_tt_metal)
    ):
        logger.debug(
            f"Release tt-metal commit mismatch: {release_tt_metal} vs {expected_tt_metal}"
        )
        return False

    if not ci_tt_metal:
        logger.debug("Missing tt-metal commit in CI image")
        return False

    if not (
        ci_tt_metal.startswith(expected_tt_metal)
        or expected_tt_metal.startswith(ci_tt_metal)
    ):
        logger.debug(
            f"CI tt-metal commit mismatch: {ci_tt_metal} vs {expected_tt_metal}"
        )
        return False

    if expected_vllm:
        if not release_vllm:
            logger.debug("Missing vllm commit in release image")
            return False

        if not (
            release_vllm.startswith(expected_vllm)
            or expected_vllm.startswith(release_vllm)
        ):
            logger.debug(
                f"Release vllm commit mismatch: {release_vllm} vs {expected_vllm}"
            )
            return False

        if not ci_vllm:
            logger.debug("Missing vllm commit in CI image")
            return False

        if not (ci_vllm.startswith(expected_vllm) or expected_vllm.startswith(ci_vllm)):
            logger.debug(f"CI vllm commit mismatch: {ci_vllm} vs {expected_vllm}")
            return False

    return True


def copy_docker_image(src: str, dst: str, dry_run: bool = False) -> bool:
    """
    Copy docker image from source to destination using crane.

    Args:
        src: Source docker image (CI registry)
        dst: Destination docker image (release registry)
        dry_run: If True, only log the action without executing

    Returns:
        True if copy successful, False otherwise
    """
    if dry_run:
        logger.info(f"[DRY-RUN] Would copy: {src} -> {dst}")
        return True

    logger.info(f"Copying image: {src} -> {dst}")
    result = run_registry_command(
        ["crane", "copy", src, dst],
        IMAGE_COPY_TIMEOUT_SECONDS,
        f"copying image {src} -> {dst}",
    )
    if result is None:
        return False

    if result.returncode == 0:
        return True

    logger.error(f"crane copy failed with exit code {result.returncode}")
    if result.stderr:
        logger.error(f"Error output: {result.stderr}")
    return False


def group_models_by_target_image(
    merged_spec: Dict[str, MergedModelRecord],
) -> Dict[str, List[MergedModelRecord]]:
    """Group merged model records by target docker image."""
    grouped: DefaultDict[str, List[MergedModelRecord]] = defaultdict(list)
    for record in merged_spec.values():
        grouped[record.target_docker_image].append(record)
    return {image: grouped[image] for image in sorted(grouped)}


def log_commit_mismatch(
    record: MergedModelRecord,
    target_image: str,
    ci_image: str,
    ci_commits: Optional[CommitPair],
) -> None:
    """Log a model-specific mismatch against a candidate shared image."""
    logger.warning(
        f"  Commit mismatch between release and CI images for model: {record.model_id}"
    )
    logger.warning(f"     Release image: {target_image}")
    logger.warning(
        f"     Release expects: tt-metal={record.model_spec.tt_metal_commit}, vllm={record.model_spec.vllm_commit}"
    )
    logger.warning(f"     CI image: {ci_image}")
    logger.warning(
        f"     CI image has: tt-metal={ci_commits[0] if ci_commits else 'unknown'}, vllm={ci_commits[1] if ci_commits else 'unknown'}"
    )
    logger.warning(
        "     Shared docker images must resolve to a single commit-compatible Models CI source."
    )


def image_matches_all_specs(
    target_image: str, ci_image: str, records: List[MergedModelRecord]
) -> bool:
    """Return True when one CI image is compatible with every model sharing a target image."""
    target_commits = extract_commits_from_tag(target_image)
    ci_commits = extract_commits_from_tag(ci_image)
    if target_commits is None or ci_commits is None:
        logger.info(
            "  Unable to validate commits because one image tag could not be parsed"
        )
        return False

    all_records_match = True
    for record in records:
        if not commits_match(target_commits, ci_commits, record.model_spec):
            log_commit_mismatch(record, target_image, ci_image, ci_commits)
            all_records_match = False

    return all_records_match


def select_ci_source_for_image(
    target_image: str,
    records: List[MergedModelRecord],
    image_exists_cache: Dict[str, bool],
) -> Optional[str]:
    """Select the first valid CI image that is compatible with all models sharing a target image."""
    seen_ci_images = set()
    has_ci_data = False
    has_ci_image = False

    for record in records:
        if record.ci_data:
            has_ci_data = True
        ci_docker_image = record.ci_data.get("docker_image") if record.ci_data else None
        if not ci_docker_image:
            continue
        has_ci_image = True
        if ci_docker_image in seen_ci_images:
            continue
        seen_ci_images.add(ci_docker_image)

        logger.info(f"  Candidate CI image: {ci_docker_image}")
        if not check_image_exists(ci_docker_image, cache=image_exists_cache):
            logger.info("  Candidate CI image not found on remote container registry")
            continue

        if image_matches_all_specs(target_image, ci_docker_image, records):
            logger.info("  Found valid Models CI reference")
            return ci_docker_image

    if not has_ci_data:
        logger.info("  No CI data available for models sharing this image")
    elif not has_ci_image:
        logger.info("  No CI docker_image available for models sharing this image")
    else:
        logger.info("  No valid Models CI source found for shared image")

    return None


def plan_image_action(
    target_image: str,
    records: List[MergedModelRecord],
    image_exists_cache: Dict[str, bool],
    *,
    allow_ci_promotion: bool = True,
) -> ImagePlan:
    """Compute a single image-level action for all models sharing one target image."""
    model_ids = tuple(record.model_id for record in records)
    release_exists = check_image_exists(target_image, cache=image_exists_cache)
    if release_exists:
        logger.info("  Found image on remote container registry")

    if not allow_ci_promotion:
        logger.info("  Standalone report mode: skipping Models CI source lookup")
        if release_exists:
            return ImagePlan(
                target_image=target_image,
                model_ids=model_ids,
                status=IMAGE_STATUS_EXISTS_WITHOUT_CI,
            )
        logger.info("  No existing release image found, image needs building")
        return ImagePlan(
            target_image=target_image,
            model_ids=model_ids,
            status=IMAGE_STATUS_NEEDS_BUILD,
        )

    ci_source_image = select_ci_source_for_image(
        target_image, records, image_exists_cache
    )
    if release_exists:
        if ci_source_image:
            return ImagePlan(
                target_image=target_image,
                model_ids=model_ids,
                status=IMAGE_STATUS_EXISTS_WITH_CI,
                ci_source_image=ci_source_image,
            )
        logger.info("  Existing image without Models CI reference")
        return ImagePlan(
            target_image=target_image,
            model_ids=model_ids,
            status=IMAGE_STATUS_EXISTS_WITHOUT_CI,
        )

    if ci_source_image:
        return ImagePlan(
            target_image=target_image,
            model_ids=model_ids,
            status=IMAGE_STATUS_COPY_FROM_CI,
            ci_source_image=ci_source_image,
        )

    logger.info("  No valid Models CI source found, image needs building")
    return ImagePlan(
        target_image=target_image,
        model_ids=model_ids,
        status=IMAGE_STATUS_NEEDS_BUILD,
    )


def execute_image_plan(plan: ImagePlan, dry_run: bool) -> ImagePlan:
    """Execute side effects for an image plan and return the final status."""
    if plan.status != IMAGE_STATUS_COPY_FROM_CI:
        return plan

    logger.info(
        "  Copying from Models CI container registry to release container registry"
    )
    if plan.ci_source_image and copy_docker_image(
        plan.ci_source_image, plan.target_image, dry_run
    ):
        logger.info("  Successfully copied to release container registry")
        return replace(plan, status=IMAGE_STATUS_COPIED)

    logger.error("  Failed to copy image")
    return replace(plan, status=IMAGE_STATUS_NEEDS_BUILD)


def apply_image_plan(
    plan: ImagePlan,
    images_to_build: DefaultDict[str, List[str]],
    copied_images: Dict[str, str],
    existing_with_ci_ref: Dict[str, str],
    existing_without_ci_ref: DefaultDict[str, List[str]],
) -> None:
    """Apply one image-level result into the output collections."""
    if plan.status == IMAGE_STATUS_NEEDS_BUILD:
        images_to_build[plan.target_image].extend(plan.model_ids)
        return

    if plan.status == IMAGE_STATUS_COPIED:
        if not plan.ci_source_image:
            raise ValueError("Copied image plans must include a CI source image")
        copied_images[plan.target_image] = plan.ci_source_image
        return

    if plan.status == IMAGE_STATUS_EXISTS_WITH_CI:
        if not plan.ci_source_image:
            raise ValueError("Existing-with-CI plans must include a CI source image")
        existing_with_ci_ref[plan.target_image] = plan.ci_source_image
        return

    if plan.status == IMAGE_STATUS_EXISTS_WITHOUT_CI:
        existing_without_ci_ref[plan.target_image].extend(plan.model_ids)
        return

    raise ValueError(f"Unknown image plan status: {plan.status}")


def log_summary(
    total_models: int,
    unique_images_processed: int,
    images_to_build: DefaultDict[str, List[str]],
    copied_images: Dict[str, str],
    existing_with_ci_ref: Dict[str, str],
    existing_without_ci_ref: DefaultDict[str, List[str]],
) -> None:
    """Log final processing summary."""
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total models processed: {total_models}")
    logger.info(f"Unique docker images processed: {unique_images_processed}")
    logger.info(
        f"Efficiency gain: {total_models - unique_images_processed} redundant checks avoided"
    )
    logger.info(f"Existing images with CI backing: {len(existing_with_ci_ref)}")
    logger.info(f"Existing images without CI backing: {len(existing_without_ci_ref)}")
    logger.info(f"Images copied from CI: {len(copied_images)}")
    logger.info(f"Images that need building: {len(images_to_build)}")
    logger.info(f"Unique images to build: {len(images_to_build)}")
    logger.info("=" * 80)


def generate_release_artifacts(
    merged_spec: Dict[str, MergedModelRecord],
    dry_run: bool,
    image_exists_cache: Optional[Dict[str, bool]] = None,
    *,
    allow_ci_promotion: bool = True,
) -> Tuple[
    DefaultDict[str, List[str]],
    int,
    Dict[str, str],
    Dict[str, str],
    DefaultDict[str, List[str]],
]:
    """
    Process images and create release artifacts.

    Returns:
        Tuple of image build/copy state needed for output files.
    """
    images_to_build = defaultdict(list)
    copied_images: Dict[str, str] = {}
    existing_with_ci_ref: Dict[str, str] = {}
    existing_without_ci_ref = defaultdict(list)
    if image_exists_cache is None:
        image_exists_cache = {}
    grouped_records = group_models_by_target_image(merged_spec)

    logger.info(
        f"Processing {len(merged_spec)} models across {len(grouped_records)} docker images..."
    )

    for index, (target_image, records) in enumerate(grouped_records.items(), start=1):
        logger.info(
            f"[{index}/{len(grouped_records)}] Processing docker image: {target_image}"
        )
        logger.info(
            f"  Shared by models: {', '.join(record.model_id for record in records)}"
        )
        image_plan = plan_image_action(
            target_image,
            records,
            image_exists_cache,
            allow_ci_promotion=allow_ci_promotion,
        )
        image_plan = execute_image_plan(image_plan, dry_run)
        apply_image_plan(
            image_plan,
            images_to_build,
            copied_images,
            existing_with_ci_ref,
            existing_without_ci_ref,
        )

    unique_images_count = len(images_to_build)
    unique_images_processed = len(grouped_records)
    log_summary(
        total_models=len(merged_spec),
        unique_images_processed=unique_images_processed,
        images_to_build=images_to_build,
        copied_images=copied_images,
        existing_with_ci_ref=existing_with_ci_ref,
        existing_without_ci_ref=existing_without_ci_ref,
    )
    return (
        images_to_build,
        unique_images_count,
        copied_images,
        existing_with_ci_ref,
        existing_without_ci_ref,
    )


def write_output(
    images_to_build: DefaultDict[str, List[str]],
    copied_images: Dict[str, str],
    existing_with_ci_ref: Dict[str, str],
    existing_without_ci_ref: DefaultDict[str, List[str]],
    output_dir: Path,
    prefix: str,
    *,
    generated_artifacts: Optional[List[str]] = None,
    acceptance_warnings: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, object]:
    """
    Write the artifact summary JSON and markdown files.

    Returns the structured summary dictionary written to disk.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{prefix}_artifacts_summary.json"

    unique_images = sorted(images_to_build.keys())
    unique_existing_without_ci = sorted(existing_without_ci_ref.keys())

    output_data = {
        "images_to_build": unique_images,
        "copied_images": copied_images,
        "existing_with_ci_ref": existing_with_ci_ref,
        "existing_without_ci_ref": unique_existing_without_ci,
        "generated_artifacts": sorted(generated_artifacts or []),
        "acceptance_warnings": acceptance_warnings or [],
        "summary": {
            "total_to_build": len(unique_images),
            "total_copied": len(copied_images),
            "total_existing_with_ci": len(existing_with_ci_ref),
            "total_existing_without_ci": len(existing_without_ci_ref),
            "total_generated_artifacts": len(generated_artifacts or []),
            "total_acceptance_warnings": len(acceptance_warnings or []),
        },
    }

    output_file.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
    logger.info(f"Written JSON summary to {output_file}")

    markdown_file = output_dir / f"{prefix}_artifacts_summary.md"
    markdown_content = "# Release Artifacts Summary\n\n"

    if acceptance_warnings:
        markdown_content += "## Release Acceptance Warnings\n\n"
        markdown_content += (
            "These warnings are computed locally from release report data and do not block "
            "artifact generation.\n\n"
        )
        for warning in acceptance_warnings:
            heading = (
                warning.get("heading") or warning.get("model_id") or "Unknown model"
            )
            markdown_content += f"### {heading}\n\n"
            if warning.get("ci_job_url"):
                markdown_content += f"CI job: {warning['ci_job_url']}\n\n"
            markdown_content += warning.get("summary_markdown", "").strip()
            markdown_content += "\n\n"
    else:
        markdown_content += "## Release Acceptance Warnings\n\n"
        markdown_content += "No release acceptance warnings were generated.\n\n"

    markdown_content += "## Generated Release Artifacts\n\n"
    if generated_artifacts:
        for artifact_path in sorted(generated_artifacts):
            markdown_content += f"- `{artifact_path}`\n"
        markdown_content += f"\n**Total:** {len(generated_artifacts)}\n\n"
    else:
        markdown_content += (
            "No additional generated release artifacts were recorded.\n\n"
        )

    markdown_content += "## Images Promoted from Models CI\n\n"
    if copied_images:
        for dst, src in sorted(copied_images.items()):
            dst_link = dst.replace("ghcr.io/", "https://ghcr.io/")
            src_link = src.replace("ghcr.io/", "https://ghcr.io/")
            markdown_content += f"- {dst_link}\n"
            markdown_content += f"  - from: {src_link}\n\n"
        markdown_content += f"**Total:** {len(copied_images)}\n\n"
    else:
        markdown_content += "No images were copied from Models CI.\n\n"

    markdown_content += "## Existing Images with Models CI reference\n\n"
    markdown_content += "Images that already exist on remote and have a valid Models CI image available.\n\n"
    if existing_with_ci_ref:
        for dst, src in sorted(existing_with_ci_ref.items()):
            dst_link = dst.replace("ghcr.io/", "https://ghcr.io/")
            src_link = src.replace("ghcr.io/", "https://ghcr.io/")
            markdown_content += f"- {dst_link}\n"
            markdown_content += f"  - CI source: {src_link}\n\n"
        markdown_content += f"**Total:** {len(existing_with_ci_ref)}\n\n"
    else:
        markdown_content += "No existing images with Models CI reference.\n\n"

    markdown_content += "## Existing Images without Models CI reference\n\n"
    markdown_content += "Images that already exist on remote but have no valid Models CI reference (manually built/pushed).\n\n"
    if unique_existing_without_ci:
        for img in unique_existing_without_ci:
            img_link = img.replace("ghcr.io/", "https://ghcr.io/")
            markdown_content += f"- {img_link}\n"
        markdown_content += f"\n**Total:** {len(unique_existing_without_ci)}\n\n"
    else:
        markdown_content += "No existing images without Models CI reference.\n\n"

    markdown_content += "## Docker Images Requiring New Builds\n\n"
    markdown_content += (
        "**Note:** Model Specs added outside of Models CI will need to have Docker images "
        "built manually and will show up here if not already existing. This will happen by "
        "design when the VERSION file is incremented.\n\n"
    )
    if unique_images:
        for img in unique_images:
            img_link = img.replace("ghcr.io/", "https://ghcr.io/")
            markdown_content += f"- {img_link}\n"
        markdown_content += f"\n**Total:** {len(unique_images)}\n"
    else:
        markdown_content += "No images need to be built.\n"

    markdown_file.write_text(markdown_content, encoding="utf-8")
    logger.info(f"Written markdown summary to {markdown_file}")

    return output_data


def write_release_performance_outputs(
    merged_spec: Dict[str, MergedModelRecord],
    output_dir: Path,
    dry_run: bool,
) -> Dict[str, Any]:
    """Write the schema-valid baseline and return rich data for downstream docs."""
    update_result = update_release_performance_outputs(
        merged_spec.values(),
        mode=ReleasePerformanceWriteMode.REPLACE,
    )
    release_performance_data = update_result.artifacts.rich_data
    baseline_release_performance_data = update_result.final_baseline_data

    release_performance_path = get_release_performance_path()
    if not baseline_release_performance_data.get("models"):
        logger.warning(
            "Skipping checked-in release performance baseline update because no release data was found"
        )
    elif dry_run:
        logger.info(
            "Dry-run mode: skipping checked-in release performance baseline update"
        )
    else:
        write_release_performance_data(
            baseline_release_performance_data, path=release_performance_path
        )
        logger.info(
            f"Written checked-in release performance baseline to {release_performance_path}"
        )

    return release_performance_data


def _sanitize_report_identifier(value: str) -> str:
    sanitized_value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return sanitized_value or "report"


def _build_release_report_id(
    entry: Dict[str, Any],
    model_name: str,
    device: str,
    impl_id: str,
    inference_engine: str,
) -> str:
    report_data = entry.get("report_data")
    metadata = report_data.get("metadata") if isinstance(report_data, dict) else {}
    if isinstance(metadata, dict):
        report_id = str(metadata.get("report_id") or "").strip()
        if report_id:
            return _sanitize_report_identifier(report_id)
    return _sanitize_report_identifier(
        f"{model_name}_{device}_{impl_id}_{inference_engine}_release_report"
    )


def _has_legacy_report_section_markdown(report_data: Dict[str, Any]) -> bool:
    return any(
        str(report_data.get(field_name) or "").strip()
        for field_name in LEGACY_REPORT_SECTION_MARKDOWN_FIELDS
    )


def _write_report_companion_markdown(
    renderer_root: Path,
    report_id: str,
    *,
    subdir: str,
    filename_template: str,
    markdown: str,
) -> None:
    markdown_text = str(markdown or "").strip()
    if not markdown_text:
        return
    output_dir = renderer_root / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename_template.format(report_id=report_id)
    output_path.write_text(markdown_text, encoding="utf-8")


def _materialize_release_report_json(
    output_dir: Path,
    entry: Dict[str, Any],
    model_name: str,
    device: str,
    impl_id: str,
    inference_engine: str,
) -> Path:
    report_data = entry.get("report_data")
    if not isinstance(report_data, dict) or not report_data:
        raise ValueError("release performance entry is missing report_data")

    renderer_root = output_dir / REPORT_RENDERER_DIRNAME
    release_data_dir = renderer_root / "release" / "data"
    release_data_dir.mkdir(parents=True, exist_ok=True)

    report_id = _build_release_report_id(
        entry, model_name, device, impl_id, inference_engine
    )
    materialized_report_data = deepcopy(report_data)
    legacy_markdown = {
        field_name: str(materialized_report_data.pop(field_name, "") or "").strip()
        for field_name in LEGACY_REPORT_SECTION_MARKDOWN_FIELDS
    }

    _write_report_companion_markdown(
        renderer_root,
        report_id,
        subdir="benchmarks",
        filename_template="benchmark_display_{report_id}.md",
        markdown=legacy_markdown.get("benchmarks_markdown", ""),
    )
    _write_report_companion_markdown(
        renderer_root,
        report_id,
        subdir="benchmarks_aiperf",
        filename_template="aiperf_benchmark_display_{report_id}.md",
        markdown=legacy_markdown.get("aiperf_benchmarks_markdown", ""),
    )
    _write_report_companion_markdown(
        renderer_root,
        report_id,
        subdir="benchmarks_genai_perf",
        filename_template="genai_perf_benchmark_display_{report_id}.md",
        markdown=legacy_markdown.get("genai_perf_benchmarks_markdown", ""),
    )

    report_json_path = release_data_dir / f"report_data_{report_id}.json"
    report_json_path.write_text(
        json.dumps(materialized_report_data, indent=2) + "\n",
        encoding="utf-8",
    )
    return report_json_path


def _resolve_release_report_json_path(
    output_dir: Path,
    entry: Dict[str, Any],
    model_name: str,
    device: str,
    impl_id: str,
    inference_engine: str,
) -> Optional[Path]:
    report_data = entry.get("report_data")
    report_data_json_path = entry.get("report_data_json_path")
    if (
        isinstance(report_data_json_path, str)
        and report_data_json_path
        and isinstance(report_data, dict)
        and not _has_legacy_report_section_markdown(report_data)
    ):
        candidate_path = Path(report_data_json_path)
        if candidate_path.exists():
            return candidate_path
    if isinstance(report_data, dict) and report_data:
        return _materialize_release_report_json(
            output_dir, entry, model_name, device, impl_id, inference_engine
        )
    return None


def render_release_report_markdown_outputs(
    release_performance_data: Dict[str, Any],
    output_dir: Path,
) -> Tuple[str, ...]:
    """Render report markdown for rich release-performance entries."""
    generated_paths: List[str] = []
    renderer_release_dir = output_dir / REPORT_RENDERER_DIRNAME / "release"
    renderer_release_dir.mkdir(parents=True, exist_ok=True)

    for (
        model_name,
        device,
        impl_id,
        inference_engine,
        entry,
    ) in iter_release_performance_items(release_performance_data):
        report_json_path = _resolve_release_report_json_path(
            output_dir, entry, model_name, device, impl_id, inference_engine
        )
        if not report_json_path:
            continue

        validate_report_file(report_json_path)
        report_markdown = build_release_report_markdown(report_json_path)
        entry["report_markdown"] = report_markdown

        report_id = report_json_path.stem.replace("report_data_", "", 1)
        report_markdown_path = renderer_release_dir / f"report_{report_id}.md"
        report_markdown_path.write_text(report_markdown, encoding="utf-8")
        generated_paths.append(str(report_markdown_path))

    return tuple(generated_paths)


def build_acceptance_warnings(
    merged_spec: Dict[str, MergedModelRecord],
) -> List[Dict[str, Any]]:
    """Recompute release acceptance from raw reports and return warning entries."""
    warnings: List[Dict[str, Any]] = []
    for record in sorted(merged_spec.values(), key=lambda item: item.model_id):
        release_report = record.ci_data.get("release_report") or {}
        if not isinstance(release_report, dict) or not release_report:
            continue

        benchmark_target_evaluation = evaluate_benchmark_targets(
            release_report, model_spec=record.model_spec
        )
        accepted, acceptance_blockers = acceptance_criteria_check(
            release_report,
            benchmark_target_evaluation=benchmark_target_evaluation
            if isinstance(benchmark_target_evaluation, dict)
            else None,
            model_spec=record.model_spec,
        )
        if accepted:
            continue

        summary_markdown = format_acceptance_summary_markdown(
            accepted,
            acceptance_blockers,
            benchmark_target_evaluation=benchmark_target_evaluation
            if isinstance(benchmark_target_evaluation, dict)
            else None,
        )
        heading = f"{record.model_spec.model_name} on {getattr(record.model_spec.device_type, 'name', 'unknown')}"
        ci_job_url = record.ci_data.get("ci_job_url")
        logger.warning("Release acceptance warning for %s", heading)
        if ci_job_url:
            logger.warning("CI job: %s", ci_job_url)
        for line in summary_markdown.splitlines():
            logger.warning("%s", line)
        warnings.append(
            {
                "model_id": record.model_id,
                "heading": heading,
                "ci_job_url": ci_job_url,
                "acceptance_blockers": acceptance_blockers,
                "summary_markdown": summary_markdown,
            }
        )

    return warnings


def write_acceptance_warnings_output(
    warnings: List[Dict[str, Any]], output_dir: Path, dry_run: bool
) -> Optional[Path]:
    """Persist Step-5 acceptance warnings for later Step-6 summary generation."""
    if dry_run:
        logger.info(
            "Dry-run mode: skipping acceptance warnings output in %s",
            output_dir / "release_acceptance_warnings.json",
        )
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "release_acceptance_warnings.json"
    output_path.write_text(json.dumps(warnings, indent=2), encoding="utf-8")
    logger.info("Written release acceptance warnings to %s", output_path)
    return output_path


def require_release_diff_path(output_dir: Path) -> Path:
    """Require the pre-release diff artifact before release-only diff generation."""
    release_diff_path = output_dir / "pre_release_models_diff.json"
    if not release_diff_path.exists():
        raise FileNotFoundError(
            "Missing required pre-release diff artifact: "
            f"{release_diff_path}. Run pre-release diff generation first."
        )
    return release_diff_path


def write_release_performance_diff_output(
    output_dir: Path,
    release_performance_data: Dict[str, Any],
) -> Path:
    """Write the release-scoped performance diff JSON artifact."""
    release_diff_records = load_optional_json(
        require_release_diff_path(output_dir), default=[]
    )
    release_performance_path = get_release_performance_path()
    base_release_performance_data = load_git_release_performance_data(
        release_performance_path
    )
    release_performance_diff_data = build_release_performance_diff_data(
        release_diff_records=release_diff_records,
        release_performance_data=release_performance_data,
        base_release_performance_data=base_release_performance_data,
        base_ref="HEAD",
        compared_path=release_performance_path,
    )
    output_path = output_dir / "release_performance_diff.json"
    write_release_performance_diff_data(release_performance_diff_data, output_path)
    logger.info(f"Written release performance diff JSON to {output_path}")
    return output_path


def write_release_model_spec_output(
    model_spec_path: Path,
    output_path: Path,
    dry_run: bool,
) -> None:
    """Write release_model_spec.json from the finalized model_spec.py."""
    if dry_run:
        count = len(MODEL_SPECS)
        logger.info(
            "Dry-run mode: skipping compatibility export of "
            f"{count} model specs to {output_path}"
        )
        return

    reload_and_export_model_specs_json(model_spec_path, output_path)
    logger.info(f"Written release model spec compatibility artifact to {output_path}")


def write_release_notes(
    output_dir: Path,
    version: str,
    release_performance_data: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write versioned release notes using structured release artifacts."""
    notes_path = output_dir / f"release_notes_v{version}.md"
    model_diff_records = load_optional_json(
        output_dir / "pre_release_models_diff.json", default=[]
    )
    artifacts_summary_data = load_optional_json(
        output_dir / "release_artifacts_summary.json", default={}
    )
    current_release_performance_data = (
        release_performance_data or load_release_performance_data()
    )
    base_release_performance_data = load_git_release_performance_data(
        get_release_performance_path()
    )

    notes = build_release_notes(
        version=version,
        model_diff_records=model_diff_records,
        artifacts_summary_data=artifacts_summary_data,
        release_performance_data=current_release_performance_data,
        base_release_performance_data=base_release_performance_data,
    )
    notes_path.write_text(notes, encoding="utf-8")
    logger.info(f"Written release notes to {notes_path}")
    return notes_path


def emit_markdown_summary(markdown_path: Path) -> None:
    """Write the generated markdown summary to stdout."""
    sys.stdout.write(markdown_path.read_text(encoding="utf-8"))


def build_generated_artifact_paths(
    *,
    output_dir: Path,
    release_model_spec_path: Path,
    readme_path: Path,
) -> List[str]:
    """Return repo-relative paths for the Step-5 artifacts produced for a release."""
    return [
        str(release_model_spec_path),
        "docs/model_support/",
        str(readme_path),
        str(output_dir / "release_performance_diff.json"),
        str(get_release_performance_path()),
    ]


def add_cli_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_output_dir: Path,
    include_target_flags: bool = True,
) -> None:
    """Add CLI arguments shared by release-artifact entrypoints."""
    parser.add_argument(
        "ci_artifacts_path",
        nargs="?",
        help=(
            "Path to downloaded raw CI artifacts. Expected either "
            "release_logs/v{VERSION} or one ci_run_logs/On_nightly_* directory."
        ),
    )
    parser.add_argument(
        "--models-ci-run-id",
        type=int,
        default=None,
        help=(
            "GitHub Actions workflow run ID; automatically downloads raw workflow "
            "artifacts and processes them directly."
        ),
    )
    parser.add_argument(
        "--report-data-json",
        default=None,
        help=(
            "Path to one standalone report_data_*.json file. This runs the full "
            "artifact-generation flow using the checked-in workflows/model_spec.py "
            "entry for metadata.model_id and skips image promotion from Models CI."
        ),
    )
    parser.add_argument(
        "--out-root",
        default=None,
        help=(
            "Output directory for CI reader artifacts when using --models-ci-run-id "
            f"(default: {default_output_dir})"
        ),
    )
    if include_target_flags:
        parser.add_argument("--dev", action="store_true", help="Target -dev- images")
        parser.add_argument(
            "--release", action="store_true", help="Target -release- images"
        )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Directory for output files (default: {default_output_dir})",
    )
    parser.add_argument(
        "--model-spec-path",
        default="workflows/model_spec.py",
        help="Path to model_spec.py file (default: workflows/model_spec.py)",
    )
    parser.add_argument(
        "--readme-path",
        default="README.md",
        help="Path to README.md file (default: README.md)",
    )
    parser.add_argument(
        "--release-model-spec-path",
        default="release_model_spec.json",
        help=(
            "Path to release_model_spec.json compatibility artifact "
            "(default: release_model_spec.json)"
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview actions without executing them"
    )


def validate_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    require_target_flag: bool,
) -> None:
    """Validate parsed CLI arguments before execution."""
    if require_target_flag:
        if args.dev and args.release:
            parser.error("--dev and --release are mutually exclusive")
        if not args.dev and not args.release:
            parser.error("Either --dev or --release must be specified")

    provided_inputs = [
        input_name
        for input_name, is_provided in (
            ("ci_artifacts_path", bool(args.ci_artifacts_path)),
            ("--models-ci-run-id", args.models_ci_run_id is not None),
            ("--report-data-json", bool(args.report_data_json)),
        )
        if is_provided
    ]
    if len(provided_inputs) > 1:
        parser.error(
            "Provide exactly one of ci_artifacts_path, --models-ci-run-id, "
            "or --report-data-json."
        )
    if not provided_inputs:
        parser.error(
            "Provide one of ci_artifacts_path, --models-ci-run-id, "
            "or --report-data-json."
        )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse and validate CLI arguments."""
    default_output_dir = get_versioned_release_logs_dir()
    parser = argparse.ArgumentParser(
        description="Generate Step-5 release artifacts from CI or report data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    add_cli_arguments(
        parser,
        default_output_dir=default_output_dir,
        include_target_flags=True,
    )
    args = parser.parse_args(argv)
    validate_args(parser, args, require_target_flag=True)
    return args


def build_release_artifacts_request(
    args: argparse.Namespace, merged_spec: Dict[str, MergedModelRecord]
) -> ReleaseArtifactsRequest:
    """Build the typed Step-5 request from parsed CLI inputs."""
    return ReleaseArtifactsRequest(
        merged_spec=merged_spec,
        output_dir=resolve_release_output_dir(args.output_dir),
        model_spec_path=Path(args.model_spec_path),
        readme_path=Path(args.readme_path),
        release_model_spec_path=Path(args.release_model_spec_path),
        version=get_version(),
        dry_run=args.dry_run,
        release=bool(args.release),
    )


def generate_release_artifact_outputs(
    request: ReleaseArtifactsRequest,
) -> ReleaseArtifactsResult:
    """Execute Step 5 and return the structured release-artifact result."""
    if not request.release:
        logger.info(
            "Dev mode no longer performs image promotion here; no Step-5 release artifacts generated."
        )
        return ReleaseArtifactsResult(
            generated_artifacts=(),
            acceptance_warnings=(),
            acceptance_warnings_path=None,
            release_performance_path=None,
            release_performance_diff_path=None,
        )

    logger.info("\nStep 2: Recomputing release acceptance warnings...")
    acceptance_warnings = build_acceptance_warnings(request.merged_spec)
    acceptance_warnings_path = write_acceptance_warnings_output(
        acceptance_warnings,
        request.output_dir,
        request.dry_run,
    )

    logger.info("\nStep 3: Writing release performance outputs...")
    release_performance_data = write_release_performance_outputs(
        request.merged_spec,
        request.output_dir,
        request.dry_run,
    )
    report_markdown_artifacts = render_release_report_markdown_outputs(
        release_performance_data, request.output_dir
    )

    logger.info("\nStep 4: Writing release performance diff output...")
    release_performance_diff_path = write_release_performance_diff_output(
        request.output_dir,
        release_performance_data,
    )

    logger.info("\nStep 5: Writing release model spec compatibility artifact...")
    write_release_model_spec_output(
        model_spec_path=request.model_spec_path,
        output_path=request.release_model_spec_path,
        dry_run=request.dry_run,
    )

    logger.info("\nStep 6: Regenerating model support docs and README...")
    regenerate_model_support_docs_and_update_readme(
        model_spec_path=request.model_spec_path,
        readme_path=request.readme_path,
        release_performance_data=release_performance_data,
        dry_run=request.dry_run,
    )

    generated_artifacts = tuple(
        [
            *build_generated_artifact_paths(
                output_dir=request.output_dir,
                release_model_spec_path=request.release_model_spec_path,
                readme_path=request.readme_path,
            ),
            *report_markdown_artifacts,
        ]
    )
    return ReleaseArtifactsResult(
        generated_artifacts=generated_artifacts,
        acceptance_warnings=tuple(acceptance_warnings),
        acceptance_warnings_path=acceptance_warnings_path,
        release_performance_path=get_release_performance_path(),
        release_performance_diff_path=release_performance_diff_path,
    )


def run_from_args(args: argparse.Namespace) -> int:
    """Run the release-artifact flow from parsed CLI arguments."""
    ci_data: Dict[str, Dict[str, Any]]
    merged_spec: Dict[str, MergedModelRecord]

    if args.models_ci_run_id is not None:
        ci_data = load_ci_data_from_run_id(
            args.models_ci_run_id,
            resolve_release_output_dir(args.out_root),
            workflow_file=RELEASE_WORKFLOW_FILE if args.release else "on-nightly.yml",
        )
        merged_spec = merge_specs_with_ci_data(ci_data, args.dev)
    elif args.report_data_json:
        merged_spec = build_merged_spec_from_report_data_json(
            Path(args.report_data_json),
            is_dev=args.dev,
        )
        ci_data = {
            model_id: record.ci_data
            for model_id, record in merged_spec.items()
            if record.ci_data
        }
    else:
        ci_data = load_ci_data_from_artifacts_path(Path(args.ci_artifacts_path))
        merged_spec = merge_specs_with_ci_data(ci_data, args.dev)

    output_dir = resolve_release_output_dir(args.output_dir)
    run_target = "dev" if args.dev else "release"

    logger.info("=" * 80)
    logger.info("RELEASE ARTIFACT GENERATION")
    logger.info("=" * 80)
    logger.info(f"Loaded CI entries: {len(ci_data)}")
    logger.info(f"Run target:       {run_target}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Dry-run mode:     {args.dry_run}")
    logger.info("=" * 80 + "\n")

    logger.info("\nStep 1: Merging CI data with MODEL_SPECS...")
    request = build_release_artifacts_request(args, merged_spec)
    result = generate_release_artifact_outputs(request)

    logger.info("\n" + "=" * 80)
    logger.info("COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    if result.generated_artifacts:
        logger.info("Generated artifacts:")
        for artifact_path in result.generated_artifacts:
            logger.info(" - %s", artifact_path)
    if result.acceptance_warnings_path:
        logger.info("Acceptance warnings: %s", result.acceptance_warnings_path)
    logger.info("Release acceptance warnings: %s", len(result.acceptance_warnings))

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for the script."""
    configure_logging()
    args = parse_args(argv)
    return run_from_args(args)


if __name__ == "__main__":
    sys.exit(main())
