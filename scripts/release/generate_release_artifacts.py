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
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Tuple

# Add project root to Python path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
from workflows.model_spec import MODEL_SPECS
from workflows.utils import get_version

try:
    from generate_release_notes import (
        build_release_notes,
        load_optional_text,
    )
    from release_paths import get_versioned_release_logs_dir, resolve_release_output_dir
except ImportError:
    from scripts.release.generate_release_notes import (
        build_release_notes,
        load_optional_text,
    )
    from scripts.release.release_paths import (
        get_versioned_release_logs_dir,
        resolve_release_output_dir,
    )

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


def merge_specs_with_ci_data(ci_json_path: Path, is_dev: bool) -> Dict[str, dict]:
    """
    Merge models_ci_last_good JSON with MODEL_SPECS.

    Args:
        ci_json_path: Path to the CI results JSON file
        is_dev: If True, modifies docker_image to use '-dev-' instead of '-release-'

    Returns:
        Dictionary mapping model_id to merged data containing model_spec and ci_data
    """
    logger.info(f"Loading CI data from: {ci_json_path}")

    if not ci_json_path.exists():
        raise FileNotFoundError(f"CI JSON file not found: {ci_json_path}")

    with open(ci_json_path, "r") as f:
        ci_data = json.load(f)

    logger.info(f"Loaded CI data for {len(ci_data)} model entries")

    merged = {}
    for model_id, model_spec in MODEL_SPECS.items():
        ci_entry = ci_data.get(model_id, {})

        docker_image = model_spec.docker_image
        if is_dev:
            docker_image = docker_image.replace("-release-", "-dev-")

        object.__setattr__(model_spec, "docker_image", docker_image)
        merged[model_id] = {"model_spec": model_spec, "ci_data": ci_entry}

    logger.info(f"Merged {len(merged)} model specs with CI data")
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
    if cache is not None and image in cache:
        logger.debug(f"Cache HIT for image: {image}")
        return cache[image]

    logger.debug(f"Cache MISS for image: {image}")

    try:
        result = subprocess.run(
            ["docker", "manifest", "inspect", image],
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

    if cache is not None:
        cache[image] = exists

    return exists


def extract_commits_from_tag(docker_image: str) -> Optional[Tuple[str, Optional[str]]]:
    """
    Extract tt-metal and vllm commit hashes from docker image tag.

    Expected format: registry/name:version-tt_metal_commit-vllm_commit[-build_id]

    Args:
        docker_image: Full docker image name with registry and tag

    Returns:
        Tuple of (tt_metal_commit, vllm_commit) or None if parsing fails
        vllm_commit can be None for images that don't have a vllm component
    """
    try:
        if ":" not in docker_image:
            logger.debug(f"No tag found in docker image: {docker_image}")
            return None

        tag = docker_image.split(":")[-1]
        parts = tag.split("-")

        if len(parts) < 2:
            logger.debug(f"Tag format unexpected (too few parts): {tag}")
            return None

        tt_metal_commit = parts[1]
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

    try:
        logger.info(f"Copying image: {src} -> {dst}")
        result = subprocess.run(
            ["crane", "copy", src, dst],
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.returncode == 0:
            return True

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


def generate_release_artifacts(
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

    Returns:
        Tuple of image build/copy state needed for output files.
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

        if docker_image in processed_images:
            logger.info("  Image already processed, using cached result")
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

        if check_image_exists(docker_image, cache=image_exists_cache):
            logger.info("  Found image on remote container registry")
            has_ci_backing = False

            if ci_data:
                ci_docker_image = ci_data.get("docker_image")
                if ci_docker_image:
                    logger.info(f"  CI image: {ci_docker_image}")
                    if check_image_exists(ci_docker_image, cache=image_exists_cache):
                        release_commits = extract_commits_from_tag(docker_image)
                        ci_commits = extract_commits_from_tag(ci_docker_image)

                        if commits_match(release_commits, ci_commits, model_spec):
                            logger.info("  Has valid Models CI reference")
                            existing_with_ci_ref[docker_image] = ci_docker_image
                            processed_images.add(docker_image)
                            image_results[docker_image] = "exists_with_ci"
                            found_with_ci += 1
                            has_ci_backing = True
                        else:
                            logger.info(
                                "  Commit mismatch between release and CI images"
                            )
                    else:
                        logger.info("  CI image not found on remote")
                else:
                    logger.info("  No CI docker_image in CI data")
            else:
                logger.info("  No CI data available")

            if not has_ci_backing:
                logger.info("  Existing image without Models CI reference")
                existing_without_ci_ref[docker_image].append(model_id)
                processed_images.add(docker_image)
                image_results[docker_image] = "exists_without_ci"
                found_without_ci += 1

            continue

        if not ci_data:
            if docker_image in copied_images:
                logger.warning(
                    f"  Image already copied from {copied_images[docker_image]}, skipping build list"
                )
                continue
            images_to_build[docker_image].append(model_id)
            image_results[docker_image] = "needs_building"
            logger.info("  No CI data available, added to images_to_be_built.json")
            needs_building += 1
            continue

        ci_docker_image = ci_data.get("docker_image")
        if not ci_docker_image:
            if docker_image in copied_images:
                logger.warning(
                    f"  Image already copied from {copied_images[docker_image]}, skipping build list"
                )
                continue
            images_to_build[docker_image].append(model_id)
            processed_images.add(docker_image)
            image_results[docker_image] = "needs_building"
            logger.info(
                "  No CI docker_image in CI data, added to images_to_be_built.json"
            )
            needs_building += 1
            continue

        logger.info(f"  CI image: {ci_docker_image}")

        if not check_image_exists(ci_docker_image, cache=image_exists_cache):
            logger.error("  CI image not found on remote container registry")
            if docker_image in copied_images:
                logger.warning(
                    f"  Image already copied from {copied_images[docker_image]}, skipping build list"
                )
                continue
            images_to_build[docker_image].append(model_id)
            processed_images.add(docker_image)
            image_results[docker_image] = "needs_building"
            needs_building += 1
            continue

        release_commits = extract_commits_from_tag(docker_image)
        ci_commits = extract_commits_from_tag(ci_docker_image)

        if not commits_match(release_commits, ci_commits, model_spec):
            logger.warning(
                f"  Commit mismatch between release and CI images for model: {model_id}"
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
                "     This happens when multiple models in MODEL_SPECS share the same docker_image"
            )
            logger.warning(
                "        but have different commits in CI data. Skipping copy to avoid mismatch."
            )
            if docker_image in copied_images:
                logger.warning(
                    f"  Image already copied from {copied_images[docker_image]}, skipping build list"
                )
                continue
            images_to_build[docker_image].append(model_id)
            processed_images.add(docker_image)
            image_results[docker_image] = "needs_building"
            needs_building += 1
            continue

        logger.info(
            "  Copying from Models CI container registry to release container registry"
        )
        if copy_docker_image(ci_docker_image, docker_image, dry_run):
            logger.info("  Successfully copied to release container registry")
            copied_images[docker_image] = ci_docker_image

            if docker_image in images_to_build:
                del images_to_build[docker_image]

            processed_images.add(docker_image)
            image_results[docker_image] = "copied"
            copied_from_ci += 1
        else:
            logger.error("  Failed to copy image")
            if docker_image in copied_images:
                logger.warning(
                    f"  Image already copied from {copied_images[docker_image]}, skipping build list"
                )
                continue
            images_to_build[docker_image].append(model_id)
            processed_images.add(docker_image)
            image_results[docker_image] = "needs_building"
            needs_building += 1

    unique_images_count = len(images_to_build)
    unique_images_processed = len(processed_images)

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
    logger.info("=" * 80)
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

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
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

    with open(markdown_file, "w") as f:
        f.write(markdown_content)
    logger.info(f"Written markdown summary to {markdown_file}")

    return output_data


def write_release_notes(output_dir: Path, version: str) -> Path:
    """Write versioned release notes using the pre-release diff and artifact summary."""
    notes_path = output_dir / f"release_notes_v{version}.md"
    model_diff_markdown = load_optional_text(output_dir / "pre_release_models_diff.md")
    artifacts_summary_markdown = load_optional_text(
        output_dir / "release_artifacts_summary.md"
    )

    notes = build_release_notes(
        version=version,
        model_diff_markdown=model_diff_markdown,
        artifacts_summary_markdown=artifacts_summary_markdown,
    )
    notes_path.write_text(notes)
    logger.info(f"Written release notes to {notes_path}")
    return notes_path


def main():
    """Main entry point for the script."""
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
        logger.info("\nStep 4: Writing release notes...")
        notes_path = write_release_notes(output_dir, get_version())

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

    print(open(output_dir / f"{image_target}_artifacts_summary.md").read())

    return 0


if __name__ == "__main__":
    sys.exit(main())
