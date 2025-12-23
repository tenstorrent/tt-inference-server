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
- Optionally incrementing VERSION file

Usage:
    python3 make_release_image_artifacts.py <models_ci_last_good_json> [--dev | --release] [--increment {major|minor|patch}] [--dry-run]
    python3 make_release_image_artifacts.py --help

Examples:
    # Update dev images without incrementing VERSION
    python3 make_release_image_artifacts.py release_logs/models_ci_last_good_*.json --dev

    # Update release images and increment VERSION (minor)
    python3 make_release_image_artifacts.py release_logs/models_ci_last_good_*.json --release --increment minor

    # Dry-run dev images
    python3 make_release_image_artifacts.py release_logs/models_ci_last_good_*.json --dev --dry-run

    # Do a major release (increments VERSION and targets release images)
    python3 make_release_image_artifacts.py release_logs/models_ci_last_good_*.json --release --increment major
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Tuple

# Add project root to Python path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
from workflows.model_spec import MODEL_SPECS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


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
    ci_json_path: Path, is_dev: bool, old_version: str, new_version: str
) -> Dict[str, dict]:
    """
    Merge models_ci_last_good JSON with MODEL_SPECS.

    Args:
        ci_json_path: Path to the CI results JSON file
        is_dev: If True, modifies docker_image to use '-dev-' instead of '-release-'
        old_version: The version that was in MODEL_SPECS at import time
        new_version: The version to use (may be same as old_version if no increment)

    Returns:
        Dictionary mapping model_id to merged data containing model_spec and ci_data
    """
    logger.info(f"Loading CI data from: {ci_json_path}")

    if not ci_json_path.exists():
        raise FileNotFoundError(f"CI JSON file not found: {ci_json_path}")

    with open(ci_json_path, "r") as f:
        ci_data = json.load(f)

    logger.info(f"Loaded CI data for {len(ci_data)} model entries")

    # Log version update if applicable
    if old_version != new_version:
        logger.info(
            f"Updating docker_image versions from {old_version} to {new_version}"
        )

    # Merge with MODEL_SPECS
    merged = {}
    version_updated_count = 0
    for model_id, model_spec in MODEL_SPECS.items():
        ci_entry = ci_data.get(model_id, {})

        docker_image = model_spec.docker_image

        # Update version in docker_image if it changed
        if old_version != new_version and old_version in docker_image:
            docker_image = docker_image.replace(old_version, new_version)
            version_updated_count += 1
            logger.debug(
                f"Updated version in docker_image for {model_id}: {old_version} -> {new_version}"
            )

        # Convert to dev image if requested
        if is_dev:
            docker_image = docker_image.replace("-release-", "-dev-")

        object.__setattr__(model_spec, "docker_image", docker_image)

        merged[model_id] = {"model_spec": model_spec, "ci_data": ci_entry}

    logger.info(f"Merged {len(merged)} model specs with CI data")
    if version_updated_count > 0:
        logger.info(f"Updated version in {version_updated_count} docker_image strings")
    logger.info(
        f"Models with CI data: {len([m for m in merged.values() if m['ci_data']])}"
    )
    logger.info(
        f"Models without CI data: {len([m for m in merged.values() if not m['ci_data']])}"
    )

    return merged


def check_image_exists(image: str, cache: Optional[Dict[str, bool]] = None) -> bool:
    """
    Check if a docker image exists on remote using docker manifest inspect.

    Args:
        image: Full docker image name with registry and tag
        cache: Optional cache dictionary to store and reuse results

    Returns:
        True if image exists, False otherwise
    """
    # Check cache first
    if cache is not None and image in cache:
        logger.debug(f"Cache HIT for image: {image}")
        return cache[image]

    logger.debug(f"Cache MISS for image: {image}")

    try:
        result = subprocess.run(
            ["crane", "manifest", image],
            capture_output=True,
            text=True,
            timeout=30,
        )
        exists = result.returncode == 0
    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout checking image existence: {image}")
        exists = False
    except Exception as e:
        logger.debug(f"Error checking image {image}: {e}")
        exists = False

    # Store in cache if provided
    if cache is not None:
        cache[image] = exists

    return exists


def extract_commits_from_tag(docker_image: str) -> Optional[Tuple[str, Optional[str]]]:
    """
    Extract tt-metal and vllm commit hashes from docker image tag.

    Expected format: registry/name:version-tt_metal_commit-vllm_commit[-build_id]
    Examples:
        ghcr.io/.../image:0.2.0-17a5973-aa4ae1e -> ("17a5973", "aa4ae1e")
        ghcr.io/.../image:0.0.5-ef93cf18b3aee66cc9ec703423de0ad3c6fde844-1d799da-52729064622
            -> ("ef93cf18b3aee66cc9ec703423de0ad3c6fde844", "1d799da")
        ghcr.io/.../image:0.2.0-17a5973 -> ("17a5973", None)

    Args:
        docker_image: Full docker image name with registry and tag

    Returns:
        Tuple of (tt_metal_commit, vllm_commit) or None if parsing fails
        vllm_commit can be None for images that don't have a vllm component
    """
    try:
        # Extract tag from image (everything after the last ':')
        if ":" not in docker_image:
            logger.debug(f"No tag found in docker image: {docker_image}")
            return None

        tag = docker_image.split(":")[-1]

        # Split tag by '-' to get components
        # Expected: version-tt_metal_commit-vllm_commit[-build_id]
        parts = tag.split("-")

        if len(parts) < 2:
            logger.debug(f"Tag format unexpected (too few parts): {tag}")
            return None

        # First part is version (e.g., "0.2.0"), skip it
        # Second part is tt-metal commit
        tt_metal_commit = parts[1]

        # Third part is vllm commit (if present)
        vllm_commit = parts[2] if len(parts) >= 3 else None

        return (tt_metal_commit, vllm_commit)

    except Exception as e:
        logger.debug(f"Error extracting commits from tag {docker_image}: {e}")
        return None


def commits_match(
    release_commits: Optional[Tuple], ci_commits: Optional[Tuple], model_spec
) -> bool:
    """
    Check if commits from release and CI docker images match.

    Compares the prefixes of commit hashes since CI images may have full hashes
    while MODEL_SPECS may have short hashes (or vice versa).

    Args:
        release_commits: Tuple of (tt_metal_commit, vllm_commit) from release image
        ci_commits: Tuple of (tt_metal_commit, vllm_commit) from CI image
        model_spec: ModelSpec with expected tt_metal_commit and vllm_commit

    Returns:
        True if both tt-metal and vllm commits match, False otherwise
    """
    if release_commits is None or ci_commits is None:
        logger.debug("Cannot validate commits: parsing failed for one or both images")
        return False

    release_tt_metal, release_vllm = release_commits
    ci_tt_metal, ci_vllm = ci_commits

    # Get expected commits from model_spec
    expected_tt_metal = model_spec.tt_metal_commit
    expected_vllm = model_spec.vllm_commit

    # Validate tt-metal commit matches
    # Check if release image has expected tt-metal commit
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

    # Check if CI image has matching tt-metal commit
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

    # Validate vllm commit matches (if present in model_spec)
    if expected_vllm:
        # Check release image
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

        # Check CI image
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

    try:
        logger.info(f"Copying image: {src} -> {dst}")
        result = subprocess.run(
            ["crane", "copy", src, dst],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout for large images
        )

        if result.returncode == 0:
            return True
        else:
            logger.error(f"crane copy failed with exit code {result.returncode}")
            if result.stderr:
                logger.error(f"Error output: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        logger.error("Timeout copying image (10 minutes exceeded)")
        return False
    except Exception as e:
        logger.error(f"Exception during crane copy: {e}")
        return False


def make_release_artifacts(
    merged_spec: Dict, dry_run: bool
) -> Tuple[
    DefaultDict[str, List[str]],
    int,
    Dict[str, str],
    Dict[str, str],
    DefaultDict[str, List[str]],
]:
    """
    Process each model and create release artifacts.

    For each model:
    - Check if release docker image exists on remote
    - If not, try to copy from CI docker image
    - Track images that need to be built

    Args:
        merged_spec: Dictionary of merged model specs and CI data
        dry_run: If True, only preview actions

    Returns:
        Tuple of (defaultdict mapping docker_image to list of model_ids that need building,
                  count of unique images,
                  dictionary mapping destination to source for successfully copied images,
                  dictionary mapping existing images with CI backing to their CI source,
                  defaultdict mapping existing images without CI backing to list of model_ids)
    """
    images_to_build = defaultdict(list)
    copied_images = {}
    existing_with_ci_ref = {}
    existing_without_ci_ref = defaultdict(list)
    processed = 0
    found_with_ci = 0
    found_without_ci = 0
    copied_from_ci = 0
    needs_building = 0
    processed_images = set()
    image_results = {}
    image_exists_cache: Dict[str, bool] = {}

    logger.info(f"Processing {len(merged_spec)} models...")

    for model_id, data in merged_spec.items():
        processed += 1
        model_spec = data["model_spec"]
        ci_data = data["ci_data"]
        docker_image = model_spec.docker_image

        logger.info(f"[{processed}/{len(merged_spec)}] Processing {model_id}")
        logger.info(f"  Release image: {docker_image}")

        # Check if this image has already been processed
        if docker_image in processed_images:
            logger.info("  ⚡ Image already processed, using cached result")
            cached_result = image_results.get(docker_image)
            if cached_result == "needs_building":
                images_to_build[docker_image].append(model_id)
                needs_building += 1
            elif cached_result == "exists_with_ci":
                found_with_ci += 1
            elif cached_result == "exists_without_ci":
                existing_without_ci_ref[docker_image].append(model_id)
                found_without_ci += 1
            elif cached_result == "copied":
                copied_from_ci += 1
            continue

        # Check if release image already exists
        if check_image_exists(docker_image, cache=image_exists_cache):
            logger.info("  ✓ Found image on remote container registry")

            # Check if it has valid CI backing
            has_ci_backing = False
            ci_docker_image = None

            if ci_data:
                ci_docker_image = ci_data.get("docker_image")
                if ci_docker_image:
                    logger.info(f"  CI image: {ci_docker_image}")

                    # Check if CI image exists
                    if check_image_exists(ci_docker_image, cache=image_exists_cache):
                        # Validate that commit hashes match between release and CI images
                        release_commits = extract_commits_from_tag(docker_image)
                        ci_commits = extract_commits_from_tag(ci_docker_image)

                        if commits_match(release_commits, ci_commits, model_spec):
                            logger.info("  ✓ Has valid Models CI reference")
                            existing_with_ci_ref[docker_image] = ci_docker_image
                            processed_images.add(docker_image)
                            image_results[docker_image] = "exists_with_ci"
                            found_with_ci += 1
                            has_ci_backing = True
                        else:
                            logger.info(
                                "  ⚠ Commit mismatch between release and CI images"
                            )
                    else:
                        logger.info("  ⚠ CI image not found on remote")
                else:
                    logger.info("  ⚠ No CI docker_image in CI data")
            else:
                logger.info("  ⚠ No CI data available")

            if not has_ci_backing:
                logger.info("  → Existing image without Models CI reference")
                existing_without_ci_ref[docker_image].append(model_id)
                processed_images.add(docker_image)
                image_results[docker_image] = "exists_without_ci"
                found_without_ci += 1

            continue

        # Check if we have CI data
        if not ci_data:
            if docker_image in copied_images:
                logger.warning(
                    f"  ⚠ Image already copied from {copied_images[docker_image]}, skipping build list"
                )
                logger.debug("  → Checked copied_images, found existing entry")
                continue
            logger.debug(
                f"  → Checked copied_images (size={len(copied_images)}), not found"
            )
            images_to_build[docker_image].append(model_id)
            # processed_images.add(docker_image)
            image_results[docker_image] = "needs_building"
            logger.info("  ⚠ No CI data available, added to images_to_be_built.json")
            needs_building += 1
            continue

        # Get CI docker image
        ci_docker_image = ci_data.get("docker_image")
        if not ci_docker_image:
            if docker_image in copied_images:
                logger.warning(
                    f"  ⚠ Image already copied from {copied_images[docker_image]}, skipping build list"
                )
                continue
            images_to_build[docker_image].append(model_id)
            processed_images.add(docker_image)
            image_results[docker_image] = "needs_building"
            logger.info(
                "  ⚠ No CI docker_image in CI data, added to images_to_be_built.json"
            )
            needs_building += 1
            continue

        logger.info(f"  CI image: {ci_docker_image}")

        # Check if CI image exists
        if not check_image_exists(ci_docker_image, cache=image_exists_cache):
            logger.error("  ✗ CI image not found on remote container registry")
            if docker_image in copied_images:
                logger.warning(
                    f"  ⚠ Image already copied from {copied_images[docker_image]}, skipping build list"
                )
                continue
            images_to_build[docker_image].append(model_id)
            processed_images.add(docker_image)
            image_results[docker_image] = "needs_building"
            needs_building += 1
            continue

        # Validate that commit hashes match between release and CI images
        release_commits = extract_commits_from_tag(docker_image)
        ci_commits = extract_commits_from_tag(ci_docker_image)

        if not commits_match(release_commits, ci_commits, model_spec):
            logger.warning(
                f"  ⚠ Commit mismatch between release and CI images for model: {model_id}"
            )
            logger.warning(f"     Release image: {docker_image}")
            logger.warning(
                f"     Release expects: tt-metal={model_spec.tt_metal_commit}, vllm={model_spec.vllm_commit}"
            )
            logger.warning(f"     CI image: {ci_docker_image}")
            logger.warning(
                f"     CI image has: tt-metal={ci_commits[0] if ci_commits else 'unknown'}, vllm={ci_commits[1] if ci_commits and len(ci_commits) > 1 else 'unknown'}"
            )
            logger.warning(
                "     → This happens when multiple models in MODEL_SPECS share the same docker_image"
            )
            logger.warning(
                "        but have different commits in CI data. Skipping copy to avoid mismatch."
            )
            if docker_image in copied_images:
                logger.warning(
                    f"  ⚠ Image already copied from {copied_images[docker_image]}, skipping build list"
                )
                continue
            images_to_build[docker_image].append(model_id)
            processed_images.add(docker_image)
            image_results[docker_image] = "needs_building"
            needs_building += 1
            continue

        # Copy CI image to release location
        logger.info(
            "  → Copying from Models CI container registry to release container registry"
        )
        if copy_docker_image(ci_docker_image, docker_image, dry_run):
            logger.info("  ✓ Successfully copied to release container registry")
            copied_images[docker_image] = ci_docker_image
            logger.debug(
                f"  → Added to copied_images dict: {docker_image} <- {ci_docker_image}"
            )

            # Remove this image from images_to_build if it was added by a previous model
            if docker_image in images_to_build:
                removed_models = images_to_build[docker_image]
                del images_to_build[docker_image]
                logger.debug(
                    f"  → Removed {len(removed_models)} model(s) from images_to_build (image was copied): {removed_models}"
                )

            processed_images.add(docker_image)
            image_results[docker_image] = "copied"
            copied_from_ci += 1
        else:
            logger.error("  ✗ Failed to copy image")
            if docker_image in copied_images:
                logger.warning(
                    f"  ⚠ Image already copied from {copied_images[docker_image]}, skipping build list"
                )
                continue
            images_to_build[docker_image].append(model_id)
            processed_images.add(docker_image)
            image_results[docker_image] = "needs_building"
            needs_building += 1

    # Calculate unique images
    unique_images_count = len(images_to_build)
    unique_images_processed = len(processed_images)

    # Log summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total models processed: {processed}")
    logger.info(f"Unique docker images processed: {unique_images_processed}")
    logger.info(
        f"Efficiency gain: {processed - unique_images_processed} redundant checks avoided"
    )
    logger.info(f"Existing images with CI backing: {found_with_ci}")
    logger.info(f"Existing images without CI backing: {found_without_ci}")
    logger.info(f"Images copied from CI: {copied_from_ci}")
    logger.info(f"Images that need building: {needs_building}")
    logger.info(f"Unique images to build: {unique_images_count}")
    logger.info(f"Tracked copied images (dict size): {len(copied_images)}")
    logger.info(
        f"Tracked existing with CI backing (dict size): {len(existing_with_ci_ref)}"
    )
    logger.info(
        f"Tracked existing without CI backing (dict size): {len(existing_without_ci_ref)}"
    )
    logger.info(f"Image existence cache entries: {len(image_exists_cache)}")
    logger.info("=" * 80)
    return (
        images_to_build,
        unique_images_count,
        copied_images,
        existing_with_ci_ref,
        existing_without_ci_ref,
    )


def increment_version(
    version_file: Path, increment_type: Optional[str], dry_run: bool
) -> str:
    """
    Read, increment, and write VERSION file based on increment type.

    Args:
        version_file: Path to VERSION file
        increment_type: One of 'major', 'minor', 'patch', or None (no increment)
        dry_run: If True, don't write changes

    Returns:
        New version string (or current version if increment_type is None)
    """
    if not version_file.exists():
        raise FileNotFoundError(f"VERSION file not found: {version_file}")

    current = version_file.read_text().strip()
    logger.info(f"Current VERSION: {current}")

    if increment_type is None:
        logger.info("No increment requested: VERSION remains unchanged")
        return current

    # Parse version (major.minor.patch)
    try:
        parts = current.split(".")
        if len(parts) != 3:
            raise ValueError(
                f"VERSION format must be major.minor.patch, got: {current}"
            )
        major, minor, patch = map(int, parts)
    except Exception as e:
        raise ValueError(f"Invalid VERSION format '{current}': {e}")

    # Increment based on increment type
    if increment_type == "major":
        new_version = f"{major + 1}.0.0"
    elif increment_type == "minor":
        new_version = f"{major}.{minor + 1}.0"
    elif increment_type == "patch":
        new_version = f"{major}.{minor}.{patch + 1}"
    else:
        raise ValueError(f"Invalid increment_type: {increment_type}")

    # Write new version (or log what would be written in dry-run)
    if dry_run:
        logger.info(f"[DRY-RUN] Would update VERSION: {current} -> {new_version}")
    else:
        version_file.write_text(new_version + "\n")
        logger.info(f"Updated VERSION: {current} -> {new_version}")

    # Return new_version regardless of dry_run so downstream logic uses incremented version
    return new_version


def write_output(
    images_to_build: DefaultDict[str, List[str]],
    copied_images: Dict[str, str],
    existing_with_ci_ref: Dict[str, str],
    existing_without_ci_ref: DefaultDict[str, List[str]],
    output_dir: Path,
    prefix: str,
    dry_run: bool,
):
    """
    Write release_artifacts_summary.json and release_artifacts_summary.md files.

    Args:
        images_to_build: DefaultDict mapping docker_image to list of model_ids
        copied_images: Dictionary mapping destination to source for successfully copied images
        existing_with_ci_ref: Dictionary mapping existing images with CI backing to their CI source
        existing_without_ci_ref: DefaultDict mapping existing images without CI backing to list of model_ids
        output_dir: Directory for output files
        prefix: Prefix for output files ('dev' or 'release')
        dry_run: If True, don't write files

    Returns:
        Dictionary containing 'images_to_build', 'copied_images', 'existing_with_ci_ref', 'existing_without_ci_ref', and 'summary' keys
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{prefix}_artifacts_summary.json"

    # Calculate unique images
    unique_images = sorted(images_to_build.keys())
    unique_existing_without_ci = sorted(existing_without_ci_ref.keys())

    # Create structured output matching markdown format
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

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"Written JSON summary to {output_file}")

    # Generate markdown summary
    markdown_file = output_dir / f"{prefix}_artifacts_summary.md"
    markdown_content = "# Release Artifacts Summary\n\n"

    # Add copied images section
    markdown_content += "## Images Promoted from Models CI\n\n"
    if copied_images:
        for dst, src in sorted(copied_images.items()):
            # Convert to HTTPS links for clickability
            dst_link = dst.replace("ghcr.io/", "https://ghcr.io/")
            src_link = src.replace("ghcr.io/", "https://ghcr.io/")
            markdown_content += f"- {dst_link}\n"
            markdown_content += f"  - from: {src_link}\n\n"
        markdown_content += f"**Total:** {len(copied_images)}\n\n"
    else:
        markdown_content += "No images were copied from Models CI.\n\n"

    # Add existing images with CI backing section
    markdown_content += "## Existing Images with Models CI reference\n\n"
    markdown_content += "Images that already exist on remote and have a valid Models CI image available.\n\n"
    if existing_with_ci_ref:
        for dst, src in sorted(existing_with_ci_ref.items()):
            # Convert to HTTPS links for clickability
            dst_link = dst.replace("ghcr.io/", "https://ghcr.io/")
            src_link = src.replace("ghcr.io/", "https://ghcr.io/")
            markdown_content += f"- {dst_link}\n"
            markdown_content += f"  - CI source: {src_link}\n\n"
        markdown_content += f"**Total:** {len(existing_with_ci_ref)}\n\n"
    else:
        markdown_content += "No existing images with Models CI reference.\n\n"

    # Add existing images without CI backing section
    markdown_content += "## Existing Images without Models CI reference\n\n"
    markdown_content += "Images that already exist on remote but have no valid Models CI reference (manually built/pushed).\n\n"
    if unique_existing_without_ci:
        for img in unique_existing_without_ci:
            # Convert to HTTPS link for clickability
            img_link = img.replace("ghcr.io/", "https://ghcr.io/")
            markdown_content += f"- {img_link}\n"
        markdown_content += f"\n**Total:** {len(unique_existing_without_ci)}\n\n"
    else:
        markdown_content += "No existing images without Models CI reference.\n\n"

    # Add unique images to build section
    markdown_content += "## Docker Images Requiring New Builds\n\n"
    note_content = "".join(
        [
            "**Note:** Model Specs added outside of Models CI will need to ",
            "have Docker images built manually and will show up here if not already ",
            "existing. This will happen by design when a release happens and the ",
            "VERSION file is incremented.\n\n",
        ]
    )
    markdown_content += note_content
    if unique_images:
        for img in unique_images:
            # Convert to HTTPS link for clickability
            img_link = img.replace("ghcr.io/", "https://ghcr.io/")
            markdown_content += f"- {img_link}\n"
        markdown_content += f"\n**Total:** {len(unique_images)}\n"
    else:
        markdown_content += "No images need to be built.\n"

    with open(markdown_file, "w") as f:
        f.write(markdown_content)
    logger.info(f"Written markdown summary to {markdown_file}")

    return output_data


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Create release image artifacts by copying CI docker images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "models_ci_last_good_json",
        help="Path to models_ci_last_good JSON file with CI results",
    )
    parser.add_argument("--dev", action="store_true", help="Target -dev- images")
    parser.add_argument(
        "--release", action="store_true", help="Target -release- images"
    )
    parser.add_argument(
        "--increment",
        choices=["major", "minor", "patch"],
        help="Increment VERSION file (optional): major (X.0.0), minor (x.X.0), patch (x.x.X)",
    )
    parser.add_argument(
        "--version-file",
        default="VERSION",
        help="Path to VERSION file (default: VERSION)",
    )
    parser.add_argument(
        "--output-dir",
        default="release_logs",
        help="Directory for output files (default: release_logs)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview actions without executing them"
    )

    args = parser.parse_args()

    # Validate mutually exclusive flags
    if args.dev and args.release:
        parser.error("--dev and --release are mutually exclusive")
    if not args.dev and not args.release:
        parser.error("Either --dev or --release must be specified")

    # Setup paths
    ci_json_path = Path(args.models_ci_last_good_json).resolve()
    version_file = Path(args.version_file).resolve()
    output_dir = Path(args.output_dir).resolve()

    # Determine image target type
    image_target = "dev" if args.dev else "release"

    # Log configuration
    logger.info("=" * 80)
    logger.info("RELEASE IMAGE ARTIFACTS SCRIPT")
    logger.info("=" * 80)
    logger.info(f"CI JSON:          {ci_json_path}")
    logger.info(f"Image target:     {image_target}")
    logger.info(f"VERSION increment: {args.increment if args.increment else 'None'}")
    logger.info(f"VERSION file:     {version_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Dry-run mode:     {args.dry_run}")
    logger.info("=" * 80 + "\n")

    # Check prerequisites
    if not check_docker_installed():
        return 1
    if not check_crane_installed():
        return 1

    logger.info("\nStep 0: Reading current VERSION and incrementing if requested...")
    # Get the old version (what MODEL_SPECS were imported with)
    if not version_file.exists():
        logger.error(f"VERSION file not found: {version_file}")
        return 1
    old_version = version_file.read_text().strip()

    # Increment version if requested
    new_version = increment_version(version_file, args.increment, args.dry_run)

    logger.info("\nStep 1: Merging CI data with MODEL_SPECS...")
    merged_spec = merge_specs_with_ci_data(
        ci_json_path, args.dev, old_version, new_version
    )

    logger.info("\nStep 2: Creating release artifacts...")
    (
        images_to_build,
        unique_images_count,
        copied_images,
        existing_with_ci_ref,
        existing_without_ci_ref,
    ) = make_release_artifacts(merged_spec, args.dry_run)

    logger.info("\nStep 3: Writing output files...")
    write_output(
        images_to_build,
        copied_images,
        existing_with_ci_ref,
        existing_without_ci_ref,
        output_dir,
        image_target,
        args.dry_run,
    )

    logger.info("\n" + "=" * 80)
    logger.info("COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    if args.dry_run:
        logger.info("This was a DRY-RUN - no changes were made")
        logger.info("=" * 80)

    logger.info(f"VERSION: {new_version}")
    logger.info(f"Output JSON: {output_dir / f'{image_target}_artifacts_summary.json'}")
    logger.info(
        f"Output Markdown: {output_dir / f'{image_target}_artifacts_summary.md'}"
    )
    logger.info(f"Unique images to build: {unique_images_count}")
    logger.info(f"Images promoted from Models CI: {len(copied_images)}")
    logger.info(f"Existing images with CI backing: {len(existing_with_ci_ref)}")
    logger.info(f"Existing images without CI backing: {len(existing_without_ci_ref)}")

    print(open(output_dir / f"{image_target}_artifacts_summary.md").read())

    return 0


if __name__ == "__main__":
    sys.exit(main())
