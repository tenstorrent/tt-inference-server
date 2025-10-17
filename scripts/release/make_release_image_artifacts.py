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
- Incrementing VERSION file appropriately

Usage:
    python3 make_release_image_artifacts.py <models_ci_last_good_json> --release {dev|major|minor|patch}
    python3 make_release_image_artifacts.py --help

Examples:
    # Do a major release (increments VERSION)
    python3 make_release_image_artifacts.py release_logs/models_ci_last_good_*.json --release major
    
    # Update dev images without incrementing VERSION
    python3 make_release_image_artifacts.py release_logs/models_ci_last_good_*.json --release dev
    
    # Dry-run to preview actions
    python3 make_release_image_artifacts.py release_logs/models_ci_last_good_*.json --release patch --dry-run
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
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def check_crane_installed() -> bool:
    """Check if crane tool is installed and available."""
    if not shutil.which("crane"):
        logger.error("crane tool not found in PATH")
        logger.error("Please install crane: https://github.com/google/go-containerregistry/blob/main/cmd/crane/README.md")
        return False
    return True


def check_docker_installed() -> bool:
    """Check if docker is installed and available."""
    if not shutil.which("docker"):
        logger.error("docker not found in PATH")
        logger.error("Please install docker: https://docs.docker.com/get-docker/")
        return False
    return True


def merge_specs_with_ci_data(ci_json_path: Path, release_type: str) -> Dict[str, dict]:
    """
    Merge models_ci_last_good JSON with MODEL_SPECS.
    
    Args:
        ci_json_path: Path to the CI results JSON file
        release_type: Release type ('dev', 'major', 'minor', or 'patch'). 
                      For 'dev' releases, modifies docker_image to use '-dev-' instead of '-release-'
    
    Returns:
        Dictionary mapping model_id to merged data containing model_spec and ci_data
    """
    logger.info(f"Loading CI data from: {ci_json_path}")
    
    if not ci_json_path.exists():
        raise FileNotFoundError(f"CI JSON file not found: {ci_json_path}")
    
    with open(ci_json_path, 'r') as f:
        ci_data = json.load(f)
    
    logger.info(f"Loaded CI data for {len(ci_data)} model entries")
    
    # Merge with MODEL_SPECS
    merged = {}
    for model_id, model_spec in MODEL_SPECS.items():
        ci_entry = ci_data.get(model_id, {})

        if release_type == "dev":
            object.__setattr__(model_spec, 'docker_image', model_spec.docker_image.replace("-release-", "-dev-"))

        merged[model_id] = {
            "model_spec": model_spec,
            "ci_data": ci_entry
        }
  
    logger.info(f"Merged {len(merged)} model specs with CI data")
    logger.info(f"Models with CI data: {len([m for m in merged.values() if m['ci_data']])}")
    logger.info(f"Models without CI data: {len([m for m in merged.values() if not m['ci_data']])}")
    
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
            ["docker", "manifest", "inspect", image],
            capture_output=True,
            text=True,
            timeout=30
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
            timeout=600  # 10 minute timeout for large images
        )
        
        if result.returncode == 0:
            return True
        else:
            logger.error(f"crane copy failed with exit code {result.returncode}")
            if result.stderr:
                logger.error(f"Error output: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout copying image (10 minutes exceeded)")
        return False
    except Exception as e:
        logger.error(f"Exception during crane copy: {e}")
        return False


def make_release_artifacts(merged_spec: Dict, dry_run: bool) -> Tuple[DefaultDict[str, List[str]], int, Dict[str, str]]:
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
                  dictionary mapping destination to source for successfully copied images)
    """
    images_to_build = defaultdict(list)
    copied_images = {}
    processed = 0
    found_existing = 0
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
        
        logger.info(f"\n[{processed}/{len(merged_spec)}] Processing {model_id}")
        logger.info(f"  Release image: {docker_image}")
        
        # Check if this image has already been processed
        if docker_image in processed_images:
            logger.info(f"  ⚡ Image already processed, using cached result")
            cached_result = image_results.get(docker_image)
            if cached_result == 'needs_building':
                images_to_build[docker_image].append(model_id)
                needs_building += 1
            elif cached_result == 'exists':
                found_existing += 1
            elif cached_result == 'copied':
                copied_from_ci += 1
            continue
        
        # Check if release image already exists
        if check_image_exists(docker_image, cache=image_exists_cache):
            logger.info(f"  ✓ Found image on remote container registry")
            processed_images.add(docker_image)
            image_results[docker_image] = 'exists'
            found_existing += 1
            continue
        
        # Check if we have CI data
        if not ci_data:
            if docker_image in copied_images:
                logger.warning(f"  ⚠ Image already copied from {copied_images[docker_image]}, skipping build list")
                logger.debug(f"  → Checked copied_images, found existing entry")
                continue
            logger.debug(f"  → Checked copied_images (size={len(copied_images)}), not found")
            images_to_build[docker_image].append(model_id)
            # processed_images.add(docker_image)
            image_results[docker_image] = 'needs_building'
            logger.info(f"  ⚠ No CI data available, added to images_to_be_built.json")
            needs_building += 1
            continue
        
        # Get CI docker image
        ci_docker_image = ci_data.get("docker_image")
        if not ci_docker_image:
            if docker_image in copied_images:
                logger.warning(f"  ⚠ Image already copied from {copied_images[docker_image]}, skipping build list")
                continue
            images_to_build[docker_image].append(model_id)
            processed_images.add(docker_image)
            image_results[docker_image] = 'needs_building'
            logger.info(f"  ⚠ No CI docker_image in CI data, added to images_to_be_built.json")
            needs_building += 1
            continue
        
        logger.info(f"  CI image: {ci_docker_image}")
        
        # Check if CI image exists
        if not check_image_exists(ci_docker_image, cache=image_exists_cache):
            logger.error(f"  ✗ CI image not found on remote container registry")
            if docker_image in copied_images:
                logger.warning(f"  ⚠ Image already copied from {copied_images[docker_image]}, skipping build list")
                continue
            images_to_build[docker_image].append(model_id)
            processed_images.add(docker_image)
            image_results[docker_image] = 'needs_building'
            needs_building += 1
            continue
        
        # Copy CI image to release location
        logger.info(f"  → Copying from Models CI container registry to release container registry")
        if copy_docker_image(ci_docker_image, docker_image, dry_run):
            logger.info(f"  ✓ Successfully copied to release container registry")
            copied_images[docker_image] = ci_docker_image
            logger.debug(f"  → Added to copied_images dict: {docker_image} <- {ci_docker_image}")
            
            # Remove this image from images_to_build if it was added by a previous model
            if docker_image in images_to_build:
                removed_models = images_to_build[docker_image]
                del images_to_build[docker_image]
                logger.debug(f"  → Removed {len(removed_models)} model(s) from images_to_build (image was copied): {removed_models}")
            
            processed_images.add(docker_image)
            image_results[docker_image] = 'copied'
            copied_from_ci += 1
        else:
            logger.error(f"  ✗ Failed to copy image")
            if docker_image in copied_images:
                logger.warning(f"  ⚠ Image already copied from {copied_images[docker_image]}, skipping build list")
                continue
            images_to_build[docker_image].append(model_id)
            processed_images.add(docker_image)
            image_results[docker_image] = 'needs_building'
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
    logger.info(f"Efficiency gain: {processed - unique_images_processed} redundant checks avoided")
    logger.info(f"Images found on remote: {found_existing}")
    logger.info(f"Images copied from CI: {copied_from_ci}")
    logger.info(f"Images that need building: {needs_building}")
    logger.info(f"Unique images to build: {unique_images_count}")
    logger.info(f"Tracked copied images (dict size): {len(copied_images)}")
    logger.info(f"Image existence cache entries: {len(image_exists_cache)}")
    logger.info("=" * 80)
    return images_to_build, unique_images_count, copied_images


def increment_version(version_file: Path, release_type: str, dry_run: bool) -> str:
    """
    Read, increment, and write VERSION file based on release type.
    
    Args:
        version_file: Path to VERSION file
        release_type: One of 'dev', 'major', 'minor', 'patch'
        dry_run: If True, don't write changes
    
    Returns:
        New version string
    """
    if not version_file.exists():
        raise FileNotFoundError(f"VERSION file not found: {version_file}")
    
    current = version_file.read_text().strip()
    logger.info(f"Current VERSION: {current}")
    
    if release_type == "dev":
        logger.info("Dev mode: VERSION remains unchanged")
        return current
    
    # Parse version (major.minor.patch)
    try:
        parts = current.split('.')
        if len(parts) != 3:
            raise ValueError(f"VERSION format must be major.minor.patch, got: {current}")
        major, minor, patch = map(int, parts)
    except Exception as e:
        raise ValueError(f"Invalid VERSION format '{current}': {e}")
    
    # Increment based on release type
    if release_type == "major":
        new_version = f"{major + 1}.0.0"
    elif release_type == "minor":
        new_version = f"{major}.{minor + 1}.0"
    elif release_type == "patch":
        new_version = f"{major}.{minor}.{patch + 1}"
    else:
        raise ValueError(f"Invalid release_type: {release_type}")
    
    # Write new version
    if dry_run:
        logger.info(f"[DRY-RUN] Would update VERSION: {current} -> {new_version}")
    else:
        version_file.write_text(new_version + "\n")
        logger.info(f"Updated VERSION: {current} -> {new_version}")
    
    return new_version


def write_output(images_to_build: DefaultDict[str, List[str]], copied_images: Dict[str, str], output_dir: Path, dry_run: bool):
    """
    Write release_artifacts_summary.json and release_artifacts_summary.md files.
    
    Args:
        images_to_build: DefaultDict mapping docker_image to list of model_ids
        copied_images: Dictionary mapping destination to source for successfully copied images
        output_dir: Directory for output files
        dry_run: If True, don't write files
    
    Returns:
        Dictionary containing 'images_to_build', 'copied_images', and 'summary' keys
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "release_artifacts_summary.json"
    
    # Calculate unique images
    unique_images = sorted(images_to_build.keys())
    
    # Create structured output matching markdown format
    output_data = {
        "images_to_build": unique_images,
        "copied_images": copied_images,
        "summary": {
            "total_to_build": len(unique_images),
            "total_copied": len(copied_images)
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    logger.info(f"Written JSON summary to {output_file}")

    # Generate markdown summary
    markdown_file = output_dir / "release_artifacts_summary.md"
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
    
    # Add unique images to build section
    markdown_content += "## Docker Images Requiring New Builds\n\n"
    if unique_images:
        for img in unique_images:
            # Convert to HTTPS link for clickability
            img_link = img.replace("ghcr.io/", "https://ghcr.io/")
            markdown_content += f"- {img_link}\n"
        markdown_content += f"\n**Total:** {len(unique_images)}\n"
    else:
        markdown_content += "No images need to be built.\n"

    with open(markdown_file, 'w') as f:
        f.write(markdown_content)
    logger.info(f"Written markdown summary to {markdown_file}")

    return output_data


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Create release image artifacts by copying CI docker images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'models_ci_last_good_json',
        help='Path to models_ci_last_good JSON file with CI results'
    )
    parser.add_argument(
        '--release',
        required=True,
        choices=['dev', 'major', 'minor', 'patch'],
        help='Release type: dev (no VERSION change), major (X.0.0), minor (x.X.0), patch (x.x.X)'
    )
    parser.add_argument(
        '--version-file',
        default='VERSION',
        help='Path to VERSION file (default: VERSION)'
    )
    parser.add_argument(
        '--output-dir',
        default='release_logs',
        help='Directory for output files (default: release_logs)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview actions without executing them'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    ci_json_path = Path(args.models_ci_last_good_json).resolve()
    version_file = Path(args.version_file).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    # Log configuration
    logger.info("=" * 80)
    logger.info("RELEASE IMAGE ARTIFACTS SCRIPT")
    logger.info("=" * 80)
    logger.info(f"CI JSON:          {ci_json_path}")
    logger.info(f"Release type:     {args.release}")
    logger.info(f"VERSION file:     {version_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Dry-run mode:     {args.dry_run}")
    logger.info("=" * 80 + "\n")
    
    # Check prerequisites
    if not check_docker_installed():
        return 1
    if not check_crane_installed():
        return 1
    
    logger.info("\nStep 0: Merging CI data with MODEL_SPECS...")
    merged_spec = merge_specs_with_ci_data(ci_json_path, args.release)
    
    logger.info("\nStep 1: Creating release artifacts...")
    images_to_build, unique_images_count, copied_images = make_release_artifacts(merged_spec, args.dry_run)

    logger.info("\nStep 2: Incrementing VERSION file...")
    new_version = increment_version(version_file, args.release, args.dry_run)
    
    logger.info("\nStep 3: Writing output files...")
    output_data = write_output(images_to_build, copied_images, output_dir, args.dry_run)

    logger.info("\n" + "=" * 80)
    logger.info("COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    if args.dry_run:
        logger.info("This was a DRY-RUN - no changes were made")
        logger.info("=" * 80)
    
    logger.info(f"VERSION: {new_version}")
    logger.info(f"Output JSON: {output_dir / 'release_artifacts_summary.json'}")
    logger.info(f"Output Markdown: {output_dir / 'release_artifacts_summary.md'}")
    logger.info(f"Unique images to build: {unique_images_count}")
    logger.info(f"Images promoted from Models CI: {len(copied_images)}")

    print(open(output_dir / 'release_artifacts_summary.md').read())

    return 0
        



if __name__ == '__main__':
    sys.exit(main())

