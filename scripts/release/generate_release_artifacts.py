#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Create release image artifacts by copying CI docker images to release registry.

This script manages docker image artifacts for releases by:
- Merging models_ci_last_good JSON with MODEL_SPECS
- Checking if release docker images exist on remote
- Copying CI docker images to release registry using crane
- Tracking images that need building in JSON and markdown output
- Generating versioned release notes for release runs

Usage:
    python3 generate_release_artifacts.py <models_ci_last_good_json> [--dev | --release] [--dry-run]
    python3 generate_release_artifacts.py --models-ci-run-id <run_id> [--dev | --release] [--dry-run]
    python3 generate_release_artifacts.py --help

Examples:
    # Update dev images from a last_good JSON file
    python3 generate_release_artifacts.py release_logs/v{VERSION}/models_ci_last_good_*.json --dev

    # Update release images from a last_good JSON file
    python3 generate_release_artifacts.py release_logs/v{VERSION}/models_ci_last_good_*.json --release

    # Automatically fetch CI data by run ID and update dev images
    python3 generate_release_artifacts.py --models-ci-run-id 19339722549 --dev

    # Dry-run dev images
    python3 generate_release_artifacts.py release_logs/v{VERSION}/models_ci_last_good_*.json --dev --dry-run
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

# Add project root to Python path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
from workflows.model_spec import MODEL_SPECS
from workflows.utils import get_version, parse_commits_from_docker_image

try:
    from generate_release_notes import (
        build_release_notes,
        load_git_release_performance_data,
        load_optional_json,
    )
    from release_performance import (
        build_release_performance_data,
        get_release_performance_path,
        load_release_performance_data,
        write_release_performance_data,
    )
    from release_paths import get_versioned_release_logs_dir, resolve_release_output_dir
except ImportError:
    from scripts.release.generate_release_notes import (
        build_release_notes,
        load_git_release_performance_data,
        load_optional_json,
    )
    from scripts.release.release_performance import (
        build_release_performance_data,
        get_release_performance_path,
        load_release_performance_data,
        write_release_performance_data,
    )
    from scripts.release.release_paths import (
        get_versioned_release_logs_dir,
        resolve_release_output_dir,
    )

logger = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
IMAGE_EXISTS_TIMEOUT_SECONDS = 30
IMAGE_COPY_TIMEOUT_SECONDS = 600
IMAGE_STATUS_EXISTS_WITH_CI = "exists_with_ci"
IMAGE_STATUS_EXISTS_WITHOUT_CI = "exists_without_ci"
IMAGE_STATUS_COPY_FROM_CI = "copy_from_ci"
IMAGE_STATUS_COPIED = "copied"
IMAGE_STATUS_NEEDS_BUILD = "needs_build"
CommitPair = Tuple[Optional[str], Optional[str]]


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


def merge_specs_with_ci_data(
    ci_json_path: Path, is_dev: bool
) -> Dict[str, MergedModelRecord]:
    """
    Merge models_ci_last_good JSON with MODEL_SPECS.

    Args:
        ci_json_path: Path to the CI results JSON file
        is_dev: If True, derive '-dev-' image targets instead of '-release-'

    Returns:
        Dictionary mapping model_id to merged data containing model_spec, ci_data,
        and the derived target docker image.
    """
    logger.info(f"Loading CI data from: {ci_json_path}")

    if not ci_json_path.exists():
        raise FileNotFoundError(f"CI JSON file not found: {ci_json_path}")

    with ci_json_path.open("r", encoding="utf-8") as f:
        ci_data = json.load(f)

    logger.info(f"Loaded CI data for {len(ci_data)} model entries")

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
) -> ImagePlan:
    """Compute a single image-level action for all models sharing one target image."""
    model_ids = tuple(record.model_id for record in records)
    release_exists = check_image_exists(target_image, cache=image_exists_cache)
    if release_exists:
        logger.info("  Found image on remote container registry")

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
    merged_spec: Dict[str, MergedModelRecord], dry_run: bool
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
    image_exists_cache: Dict[str, bool] = {}
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
        image_plan = plan_image_action(target_image, records, image_exists_cache)
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
        "summary": {
            "total_to_build": len(unique_images),
            "total_copied": len(copied_images),
            "total_existing_with_ci": len(existing_with_ci_ref),
            "total_existing_without_ci": len(existing_without_ci_ref),
        },
    }

    output_file.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
    logger.info(f"Written JSON summary to {output_file}")

    markdown_file = output_dir / f"{prefix}_artifacts_summary.md"
    markdown_content = "# Release Artifacts Summary\n\n"

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
    """Write the release-performance baseline and return its raw data."""
    release_performance_data = build_release_performance_data(merged_spec.values())

    release_performance_path = get_release_performance_path()
    if not release_performance_data.get("models"):
        logger.warning(
            "Skipping checked-in release performance baseline update because no release data was found"
        )
    elif dry_run:
        logger.info(
            "Dry-run mode: skipping checked-in release performance baseline update"
        )
    else:
        write_release_performance_data(
            release_performance_data, path=release_performance_path
        )
        logger.info(
            f"Written checked-in release performance baseline to {release_performance_path}"
        )

    return release_performance_data


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


def main():
    """Main entry point for the script."""
    configure_logging()
    default_output_dir = get_versioned_release_logs_dir()
    parser = argparse.ArgumentParser(
        description="Create release image artifacts by copying CI docker images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "models_ci_last_good_json",
        nargs="?",
        help="Path to models_ci_last_good JSON file with CI results",
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
            f"(default: {default_output_dir})"
        ),
    )
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
        "--dry-run", action="store_true", help="Preview actions without executing them"
    )

    args = parser.parse_args()

    if args.dev and args.release:
        parser.error("--dev and --release are mutually exclusive")
    if not args.dev and not args.release:
        parser.error("Either --dev or --release must be specified")

    if args.models_ci_run_id and args.models_ci_last_good_json:
        parser.error(
            "Provide --models-ci-run-id or models_ci_last_good_json, not both."
        )
    if not args.models_ci_run_id and not args.models_ci_last_good_json:
        parser.error("Provide --models-ci-run-id or models_ci_last_good_json.")

    output_dir = resolve_release_output_dir(args.output_dir)

    if args.models_ci_run_id:
        from scripts.release.models_ci_reader import run_ci_pipeline

        ci_json_path = run_ci_pipeline(
            args.models_ci_run_id,
            resolve_release_output_dir(args.out_root),
        )
    else:
        ci_json_path = Path(args.models_ci_last_good_json).resolve()

    image_target = "dev" if args.dev else "release"

    logger.info("=" * 80)
    logger.info("RELEASE IMAGE ARTIFACTS SCRIPT")
    logger.info("=" * 80)
    logger.info(f"CI JSON:          {ci_json_path}")
    logger.info(f"Image target:     {image_target}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Dry-run mode:     {args.dry_run}")
    logger.info("=" * 80 + "\n")

    if not check_docker_installed():
        return 1
    if not check_crane_installed():
        return 1

    logger.info("\nStep 1: Merging CI data with MODEL_SPECS...")
    merged_spec = merge_specs_with_ci_data(ci_json_path, args.dev)

    logger.info("\nStep 2: Creating release artifacts...")
    (
        images_to_build,
        unique_images_count,
        copied_images,
        existing_with_ci_ref,
        existing_without_ci_ref,
    ) = generate_release_artifacts(merged_spec, args.dry_run)

    logger.info("\nStep 3: Writing output files...")
    write_output(
        images_to_build,
        copied_images,
        existing_with_ci_ref,
        existing_without_ci_ref,
        output_dir,
        image_target,
    )

    notes_path = None
    if args.release:
        logger.info("\nStep 4: Writing release performance outputs...")
        release_performance_data = write_release_performance_outputs(
            merged_spec, output_dir, args.dry_run
        )
        logger.info("\nStep 5: Writing release notes...")
        notes_path = write_release_notes(
            output_dir,
            get_version(),
            release_performance_data=release_performance_data,
        )

    logger.info("\n" + "=" * 80)
    logger.info("COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    if args.dry_run:
        logger.info("This was a DRY-RUN - no release images were copied")
        logger.info("=" * 80)

    logger.info(f"Output JSON: {output_dir / f'{image_target}_artifacts_summary.json'}")
    logger.info(
        f"Output Markdown: {output_dir / f'{image_target}_artifacts_summary.md'}"
    )
    if notes_path:
        logger.info(f"Release notes: {notes_path}")
    logger.info(f"Unique images to build: {unique_images_count}")
    logger.info(f"Images promoted from Models CI: {len(copied_images)}")
    logger.info(f"Existing images with CI backing: {len(existing_with_ci_ref)}")
    logger.info(f"Existing images without CI backing: {len(existing_without_ci_ref)}")

    emit_markdown_summary(output_dir / f"{image_target}_artifacts_summary.md")

    return 0


if __name__ == "__main__":
    sys.exit(main())
