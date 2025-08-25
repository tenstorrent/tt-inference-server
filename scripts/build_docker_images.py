# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import logging
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
import argparse
import os
import multiprocessing
from datetime import datetime

# Add the script's directory to the Python path
# this for 0 setup python setup script
project_root = Path(__file__).resolve().parent.parent
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))

from workflows.model_spec import MODEL_SPECS
from workflows.utils import get_repo_root_path
from workflows.log_setup import setup_workflow_script_logger

logger = logging.getLogger(__file__)


def setup_individual_logger(tt_metal_commit, vllm_commit, log_dir):
    """
    Set up individual logger for each combination with file logging only.
    Console output is handled by the main process to avoid overlapping.
    """
    combination_id = f"{tt_metal_commit[:16]}-{vllm_commit[:16]}"
    logger_name = f"process_{combination_id}"

    # Create log directory if it doesn't exist
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create log file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"build_{timestamp}_{combination_id}.log"

    # Create logger
    process_logger = logging.getLogger(logger_name)
    process_logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers
    if process_logger.handlers:
        process_logger.handlers.clear()

    # File handler for this combination (NO console handler to avoid overlap)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter(
        f"[{combination_id}] %(asctime)s - %(levelname)s: %(message)s"
    )
    file_handler.setFormatter(formatter)

    # Add only file handler - no console handler to prevent overlapping output
    process_logger.addHandler(file_handler)

    return process_logger, log_file


def get_build_logs_dir():
    """
    Get the directory for storing individual build logs.
    """
    from workflows.utils import get_default_workflow_root_log_dir

    return get_default_workflow_root_log_dir() / "docker_build_logs"


def get_physical_cpu_count():
    """
    Get the number of physical CPU cores available on the system.
    """
    try:
        # Try to get physical cores first
        physical_cores = multiprocessing.cpu_count()

        # On Linux, try to get actual physical cores (not hyperthreaded)
        if hasattr(os, "sched_getaffinity"):
            # Get the CPU cores available to this process
            available_cores = len(os.sched_getaffinity(0))
            physical_cores = min(physical_cores, available_cores)

        # Try to read from /proc/cpuinfo for more accurate physical core count
        try:
            with open("/proc/cpuinfo", "r") as f:
                content = f.read()
                physical_ids = set()
                for line in content.split("\n"):
                    if line.startswith("physical id"):
                        physical_ids.add(line.split(":")[1].strip())

                cores_per_socket = 0
                for line in content.split("\n"):
                    if line.startswith("cpu cores"):
                        cores_per_socket = int(line.split(":")[1].strip())
                        break

                if physical_ids and cores_per_socket > 0:
                    physical_cores = len(physical_ids) * cores_per_socket

        except (FileNotFoundError, ValueError, IndexError):
            # Fall back to multiprocessing.cpu_count()
            pass

        logger.info(f"Detected {physical_cores} physical CPU cores")
        return physical_cores

    except Exception as e:
        logger.warning(
            f"Could not determine physical CPU count: {e}, falling back to 1"
        )
        return 1


def process_sha_combination(args_tuple):
    """
    Process a single (tt_metal_commit, vllm_commit) combination.
    This function is designed to be run in parallel with individual logging.
    """
    (
        tt_metal_commit,
        vllm_commit,
        ubuntu_version,
        force_build,
        release,
        push,
        container_app_uid,
    ) = args_tuple

    # Set up individual logging for this combination
    logs_dir = get_build_logs_dir()
    process_logger, log_file = setup_individual_logger(
        tt_metal_commit, vllm_commit, logs_dir
    )

    combination_id = f"{tt_metal_commit[:16]}-{vllm_commit[:16]}"

    try:
        process_logger.info(f"=== STARTING BUILD FOR COMBINATION {combination_id} ===")
        process_logger.info(f"tt_metal_commit: {tt_metal_commit}")
        process_logger.info(f"vllm_commit: {vllm_commit}")
        process_logger.info(f"ubuntu_version: {ubuntu_version}")
        process_logger.info(f"Log file: {log_file}")

        # Generate image tags
        image_tags = get_image_tags(
            tt_metal_commit=tt_metal_commit,
            vllm_commit=vllm_commit,
            ubuntu_version=ubuntu_version,
        )

        process_logger.info(f"Generated image tags: {image_tags}")

        # Determine what images need to be built
        build_cloud_image_flag = True
        build_dev_image_flag = True
        build_release_image_flag = True

        if not force_build:
            process_logger.info("Checking if images already exist...")
            if check_image_exists_remote(
                image_tags["cloud"]
            ) or check_image_exists_local(image_tags["cloud"]):
                build_cloud_image_flag = False
                process_logger.info("Cloud image already exists, skipping build")

            if check_image_exists_remote(image_tags["dev"]) or check_image_exists_local(
                image_tags["dev"]
            ):
                build_dev_image_flag = False
                process_logger.info("Dev image already exists, skipping build")

            if check_image_exists_remote(
                image_tags["release"]
            ) or check_image_exists_local(image_tags["release"]):
                build_release_image_flag = False
                process_logger.info("Release image already exists, skipping build")

        # Build tt-metal base image
        process_logger.info("Building tt-metal base image...")
        build_tt_metal_base_image(tt_metal_commit, ubuntu_version, process_logger)

        # Build cloud image
        if build_cloud_image_flag:
            process_logger.info("Building cloud image...")
            build_cloud_image(
                image_tags,
                tt_metal_commit,
                vllm_commit,
                container_app_uid,
                process_logger,
            )
        else:
            process_logger.info(f"Skipping cloud image build: {image_tags['cloud']}")

        # Build dev image
        if build_dev_image_flag:
            process_logger.info("Building dev image...")
            build_dev_image(image_tags, process_logger)
        else:
            process_logger.info(f"Skipping dev image build: {image_tags['dev']}")

        # Build release image (only if release=True)
        if release and build_release_image_flag:
            process_logger.info("Building release image...")
            build_release_image(image_tags, process_logger)
        else:
            process_logger.info(
                f"Skipping release image build: {image_tags['release']}"
            )

        # Push images if requested
        if push:
            process_logger.info("Pushing images to registry...")

            for image_type in ["cloud", "dev", "release"]:
                image_tag = image_tags[image_type]

                # Skip release image unless explicitly marked as a release
                if image_type == "release" and not release:
                    process_logger.info(
                        f"Skipping push for {image_tag}, release={release}"
                    )
                    continue

                if should_push_image(image_tag, force_push=False):
                    push_image(image_tag, process_logger)
                else:
                    process_logger.info(f"Skipping push for {image_tag}")

        process_logger.info(f"=== COMPLETED BUILD FOR COMBINATION {combination_id} ===")
        return {
            "success": True,
            "combination_id": combination_id,
            "log_file": str(log_file),
        }

    except Exception as e:
        process_logger.error(f"=== FAILED BUILD FOR COMBINATION {combination_id} ===")
        process_logger.error(f"Error: {str(e)}")
        process_logger.error(f"Exception type: {type(e).__name__}")

        # Log the full traceback
        import traceback

        process_logger.error(f"Full traceback:\n{traceback.format_exc()}")

        return {
            "success": False,
            "combination_id": combination_id,
            "log_file": str(log_file),
            "error": str(e),
        }


def run_command_with_logging(command, logger, check=True, cwd=None):
    """
    Run a command and write output directly to the process log file.
    """
    if isinstance(command, str):
        command = command.split()

    logger.info(f"Running command: {' '.join(command)}")

    # Get the file handler from the logger to write raw output
    log_file_handler = None
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            log_file_handler = handler
            break

    try:
        # Use Popen to stream output in real-time
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr with stdout
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True,
            cwd=cwd,
        )

        # Capture output while streaming it
        stdout_lines = []

        # Read output line by line as it's produced
        for line in iter(process.stdout.readline, ""):
            line = line.rstrip()
            if line:
                stdout_lines.append(line)
                # Write raw output directly to log file if available
                if log_file_handler:
                    log_file_handler.stream.write(line + "\n")
                    log_file_handler.stream.flush()

        # Wait for process to complete
        process.wait()

        stdout_text = "\n".join(stdout_lines)

        if process.returncode != 0:
            logger.error(f"Command failed with exit code {process.returncode}")
            if check:
                raise subprocess.CalledProcessError(
                    process.returncode, command, output=stdout_text
                )

        return process.returncode, stdout_text

    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        if check:
            raise
        return e.returncode, e.output if hasattr(e, "output") else ""
    except Exception as e:
        logger.error(f"Error running command: {e}")
        if check:
            raise
        return 1, str(e)


def check_image_exists_remote(image_tag):
    """
    Check if a Docker image exists in the remote registry.
    Returns True if image exists, False otherwise.
    """
    try:
        result = subprocess.run(
            ["docker", "manifest", "inspect", image_tag],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            logger.info(f"✅ The image exists on GHCR: {image_tag}")
            return True
        else:
            logger.info(f"The image does NOT exist on GHCR: {image_tag}")
            return False
    except Exception as e:
        logger.error(f"Error checking remote image {image_tag}: {e}")
        return False


def check_image_exists_local(image_tag):
    """
    Check if a Docker image exists locally.
    Returns True if image exists, False otherwise.
    """
    try:
        result = subprocess.run(
            ["docker", "inspect", "--type=image", image_tag],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            logger.info(f"✅ The image exists locally: {image_tag}")
            return True
        else:
            logger.info(f"The image does NOT exist locally: {image_tag}")
            return False
    except Exception as e:
        logger.error(f"Error checking local image {image_tag}: {e}")
        return False


def validate_inputs(ubuntu_version, container_app_uid):
    """
    Validate input parameters.
    """
    repo_root = get_repo_root_path()
    expected_suffix = "tt-inference-server"

    if not str(repo_root).endswith(expected_suffix):
        raise ValueError(
            f"Script must be run in tt-inference-server repo root, found: {repo_root}"
        )

    if ubuntu_version not in ["22.04", "20.04"]:
        raise ValueError(
            f"Unsupported UBUNTU_VERSION: {ubuntu_version}. Only 22.04 and 20.04 are supported."
        )

    if not (
        isinstance(container_app_uid, (int, str))
        and 1000 <= int(container_app_uid) < 60000
    ):
        raise ValueError(
            f"CONTAINER_APP_UID={container_app_uid} is not within expected range of 1000 to 59999."
        )

    logger.info(f"CONTAINER_APP_UID={container_app_uid} is within expected range.")


def get_image_tags(
    tt_metal_commit,
    vllm_commit,
    ubuntu_version,
    tag_suffix="",
    image_repo="ghcr.io/tenstorrent/tt-inference-server",
):
    """
    Generate Docker image tags for all image types.
    """
    repo_root = get_repo_root_path()
    version_file = repo_root / "VERSION"
    image_version = version_file.read_text().strip()

    os_version = f"ubuntu-{ubuntu_version}-amd64"
    tt_metal_tag = tt_metal_commit
    vllm_tag = vllm_commit

    suffix = f"-{tag_suffix}" if tag_suffix else ""

    cloud_image_tag = f"{image_repo}/vllm-tt-metal-src-cloud-{os_version}:{image_version}-{tt_metal_tag}-{vllm_tag}{suffix}"
    dev_image_tag = f"{image_repo}/vllm-tt-metal-src-dev-{os_version}:{image_version}-{tt_metal_tag}-{vllm_tag}{suffix}"
    release_image_tag = f"{image_repo}/vllm-tt-metal-src-release-{os_version}:{image_version}-{tt_metal_tag}-{vllm_tag}{suffix}"
    tt_metal_base_tag = f"local/tt-metal/tt-metalium/{os_version}:{tt_metal_commit}"

    return {
        "cloud": cloud_image_tag,
        "dev": dev_image_tag,
        "release": release_image_tag,
        "tt_metal_base": tt_metal_base_tag,
    }


def resolve_commit_to_full_sha(tt_metal_commit):
    """
    Resolve a commit reference (tag, branch, or short SHA) to a full SHA using git commands.
    Returns the full SHA if found, otherwise returns the original reference.
    """
    try:
        # Try to resolve using ls-remote to get the full SHA from origin
        # need to use shell=True and cmd string because of the pipe
        result = subprocess.run(
            f"git ls-remote https://github.com/tenstorrent/tt-metal.git | grep {tt_metal_commit}",
            shell=True,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            # Parse the output: "SHA\trefs/..."
            lines = result.stdout.strip().split("\n")

            # First, look for a line that ends with ^{} (actual commit object)
            for line in lines:
                if line and line.endswith("^{}"):
                    sha, ref = line.split("\t", 1)
                    logger.info(
                        f"Resolved {tt_metal_commit} to full SHA via ls-remote (^{{}}): {sha}"
                    )
                    return sha

            # If no line with ^{} found, use the first line
            for line in lines:
                if line:
                    sha, ref = line.split("\t", 1)
                    logger.info(
                        f"Resolved {tt_metal_commit} to full SHA via ls-remote (first match): {sha}"
                    )
                    return sha
    except Exception as e:
        logger.debug(f"ls-remote failed for {tt_metal_commit}: {e}")
    # If we can't resolve it, return the original reference
    logger.info(f"Could not resolve {tt_metal_commit} to full SHA, using as-is")
    return tt_metal_commit


def build_tt_metal_base_image(tt_metal_commit, ubuntu_version, logger=logger):
    """
    Build the tt-metal base image if it doesn't exist.
    """
    os_version = f"ubuntu-{ubuntu_version}-amd64"
    tt_metal_base_tag = f"local/tt-metal/tt-metalium/{os_version}:{tt_metal_commit}"

    if check_image_exists_local(tt_metal_base_tag):
        logger.info(f"TT-Metal base image already exists: {tt_metal_base_tag}")
        return True

    logger.info(f"Building TT-Metal base image: {tt_metal_base_tag}")

    # Resolve the commit to a full SHA before cloning
    resolved_commit = resolve_commit_to_full_sha(tt_metal_commit)

    # Create temporary directory for building
    temp_dir = Path(tempfile.mkdtemp(prefix=f"tt_metal_build_{tt_metal_commit}_"))

    try:
        # Clone tt-metal repository
        logger.info("Cloning tt-metal repository...")
        run_command_with_logging(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "https://github.com/tenstorrent/tt-metal.git",
            ],
            logger=logger,
            check=True,
            cwd=temp_dir,
        )

        tt_metal_dir = temp_dir / "tt-metal"

        # Fetch and checkout the specific commit/tag
        logger.info(f"Fetching and checking out {resolved_commit}")
        try:
            run_command_with_logging(
                ["git", "fetch", "--depth", "1", "origin", "tag", resolved_commit],
                logger=logger,
                check=True,
                cwd=tt_metal_dir,
            )
            logger.info("Fetched as tag.")
        except subprocess.CalledProcessError:
            try:
                run_command_with_logging(
                    ["git", "fetch", "--depth", "1", "origin", resolved_commit],
                    logger=logger,
                    check=True,
                    cwd=tt_metal_dir,
                )
                logger.info("Fetched as commit SHA.")
            except subprocess.CalledProcessError:
                raise RuntimeError(
                    f"Could not fetch {resolved_commit} as either a tag or commit SHA."
                )

        run_command_with_logging(
            ["git", "checkout", resolved_commit],
            logger=logger,
            check=True,
            cwd=tt_metal_dir,
        )

        # Build the Docker image
        logger.info("Building tt-metal Docker image...")
        build_command = [
            "docker",
            "build",
            "-t",
            tt_metal_base_tag,
            "--build-arg",
            f"UBUNTU_VERSION={ubuntu_version}",
            "--target",
            "ci-build",
            "-f",
            "dockerfile/Dockerfile",
            ".",
        ]

        run_command_with_logging(
            build_command, logger=logger, check=True, cwd=tt_metal_dir
        )

        logger.info(f"Successfully built tt-metal base image: {tt_metal_base_tag}")

        return True

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)


def should_push_image(image_tag, force_push=False):
    """
    Determine if an image should be pushed to the registry.
    """
    logger.info(f"Checking if should push image: {image_tag}")

    local_exists = check_image_exists_local(image_tag)
    remote_exists = check_image_exists_remote(image_tag)

    logger.info(
        f"local_exists={local_exists}, remote_exists={remote_exists}, force_push={force_push}"
    )

    return local_exists and (not remote_exists or force_push)


def build_cloud_image(
    image_tags, tt_metal_commit, vllm_commit, container_app_uid, logger
):
    """
    Build the cloud Docker image.
    """
    repo_root = get_repo_root_path()
    cloud_image_tag = image_tags["cloud"]
    tt_metal_base_tag = image_tags["tt_metal_base"]

    logger.info(f"Building cloud image: {cloud_image_tag}")

    build_command = [
        "docker",
        "build",
        "-t",
        cloud_image_tag,
        "--build-arg",
        f"TT_METAL_DOCKERFILE_URL={tt_metal_base_tag}",
        "--build-arg",
        f"TT_METAL_COMMIT_SHA_OR_TAG={tt_metal_commit}",
        "--build-arg",
        f"TT_VLLM_COMMIT_SHA_OR_TAG={vllm_commit}",
        "--build-arg",
        f"CONTAINER_APP_UID={container_app_uid}",
        "-f",
        "vllm-tt-metal-llama3/vllm.tt-metal.src.cloud.Dockerfile",
        ".",
    ]

    run_command_with_logging(build_command, logger=logger, check=True, cwd=repo_root)
    logger.info(f"Successfully built cloud image: {cloud_image_tag}")


def build_dev_image(image_tags, logger):
    """
    Build the dev Docker image.
    """
    repo_root = get_repo_root_path()
    dev_image_tag = image_tags["dev"]
    cloud_image_tag = image_tags["cloud"]

    logger.info(f"Building dev image: {dev_image_tag}")

    build_command = [
        "docker",
        "build",
        "-t",
        dev_image_tag,
        "--build-arg",
        f"CLOUD_DOCKERFILE_URL={cloud_image_tag}",
        "-f",
        "vllm-tt-metal-llama3/vllm.tt-metal.src.dev.Dockerfile",
        ".",
    ]

    run_command_with_logging(build_command, logger=logger, check=True, cwd=repo_root)
    logger.info(f"Successfully built dev image: {dev_image_tag}")


def build_release_image(image_tags, logger):
    """
    Build the release Docker image.
    """
    repo_root = get_repo_root_path()
    release_image_tag = image_tags["release"]
    cloud_image_tag = image_tags["cloud"]

    logger.info(f"Building release image: {release_image_tag}")

    build_command = [
        "docker",
        "build",
        "-t",
        release_image_tag,
        "--build-arg",
        f"CLOUD_DOCKERFILE_URL={cloud_image_tag}",
        "-f",
        "vllm-tt-metal-llama3/vllm.tt-metal.src.dev.Dockerfile",
        ".",
    ]

    run_command_with_logging(build_command, logger=logger, check=True, cwd=repo_root)
    logger.info(f"Successfully built release image: {release_image_tag}")


def push_image(image_tag, logger):
    """
    Push a Docker image to the registry.
    """
    logger.info(f"Pushing image: {image_tag}")

    push_command = ["docker", "push", image_tag]
    run_command_with_logging(push_command, logger=logger, check=True)
    logger.info(f"Successfully pushed image: {image_tag}")


def build_docker_images(
    model_configs,
    force_build=False,
    release=False,
    push=False,
    ubuntu_version="20.04",
    build_metal_commit=None,
    max_workers=None,
    single_threaded=False,
):
    """
    Builds all Docker images required by the provided ModelConfigs.

    Args:
        model_configs: Dictionary of model configurations
        force_build: Force rebuild even if images exist
        release: Mark build as release
        push: Push containers to registry
        ubuntu_version: Ubuntu version to use for base images
        build_metal_commit: Only build containers with this exact tt-metal commit
        max_workers: Maximum number of parallel workers (defaults to physical CPU cores)
        single_threaded: Run builds sequentially instead of in parallel (for debugging)
    """
    # Validate inputs
    container_app_uid = 1000
    validate_inputs(ubuntu_version, container_app_uid)

    # Filter combinations if build_metal_commit is specified
    unique_sha_combinations = {
        (config.tt_metal_commit, config.vllm_commit)
        for config in model_configs.values()
    }

    if build_metal_commit:
        unique_sha_combinations = {
            combo for combo in unique_sha_combinations if combo[0] == build_metal_commit
        }

    if not unique_sha_combinations:
        logger.warning(
            f"No configurations found with tt_metal_commit={build_metal_commit}"
        )
        return
    unique_sha_combinations_str = "\n".join(
        [f"{combo[0]}-{combo[1]}" for combo in unique_sha_combinations]
    )
    logger.info(
        f"Unique SHA combinations to build if needed:\n{unique_sha_combinations_str}"
    )

    # Get physical CPU count for multiprocessing
    physical_cpu_count = get_physical_cpu_count()

    # Use max_workers if specified, otherwise use physical CPU count
    if max_workers is not None:
        workers = max(1, min(max_workers, physical_cpu_count))
        logger.info(
            f"Using {workers} workers (max_workers={max_workers}, physical_cores={physical_cpu_count})"
        )
    else:
        workers = max(1, physical_cpu_count)
        logger.info(f"Using {workers} workers (physical cores detected)")

    # Create argument tuples for each combination
    args_tuples = [
        (
            tt_metal_commit,
            vllm_commit,
            ubuntu_version,
            force_build,
            release,
            push,
            container_app_uid,
        )  # container_app_uid=1000
        for tt_metal_commit, vllm_commit in unique_sha_combinations
    ]

    if not args_tuples:
        logger.warning("No combinations to process")
        return

    # Use multiprocessing.Pool to process combinations in parallel
    logger.info(
        f"Processing {len(args_tuples)} docker builds in parallel with {workers} workers"
    )

    if single_threaded:
        results = []
        for args_tuple in args_tuples:
            results.append(process_sha_combination(args_tuple))
    else:
        with multiprocessing.Pool(processes=workers) as pool:
            results = pool.map(process_sha_combination, args_tuples)

    # Process results and show individual log files
    success_count = 0
    failure_count = 0
    failed_combinations = []

    for result in results:
        if result["success"]:
            success_count += 1
            logger.info(
                f"✅ Success: {result['combination_id']} - Log: {result['log_file']}"
            )
        else:
            failure_count += 1
            failed_combinations.append(result)
            logger.error(
                f"❌ Failed: {result['combination_id']} - Log: {result['log_file']}"
            )
            logger.error(f"   Error: {result['error']}")

    logger.info(
        f"Completed processing: {success_count} successful, {failure_count} failed"
    )

    if failed_combinations:
        logger.error("Failed combinations and their log files:")
        for failed in failed_combinations:
            logger.error(f"  - {failed['combination_id']}: {failed['log_file']}")
        logger.error("Use the log files above to debug the specific failures.")

    logger.info("Done building Docker images.")


if __name__ == "__main__":
    setup_workflow_script_logger(logger)
    parser = argparse.ArgumentParser(
        description="Build Docker images for model configs."
    )
    parser.add_argument(
        "--force-build", action="store_true", help="Force rebuild even if image exists."
    )
    parser.add_argument("--release", action="store_true", help="Mark build as release.")
    parser.add_argument("--push", action="store_true", help="Push containers.")
    parser.add_argument(
        "--ubuntu-version",
        type=str,
        default="22.04",
        help="Ubuntu version to use for the base image.",
        choices={"20.04", "22.04"},
    )
    parser.add_argument(
        "--build-metal-commit",
        type=str,
        help="Only build containers with this exact tt-metal commit",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Maximum number of parallel workers (defaults to physical CPU cores)",
    )
    parser.add_argument(
        "--single-threaded",
        action="store_true",
        help="Run builds sequentially instead of in parallel (for debugging)",
    )
    args = parser.parse_args()
    logger.info(f"ubuntu_version: {args.ubuntu_version}")
    logger.info(f"build_metal_commit: {args.build_metal_commit}")
    logger.info(f"max_workers: {args.max_workers}")
    logger.info(f"force_build: {args.force_build}")
    logger.info(f"release: {args.release}")
    logger.info(f"push: {args.push}")
    logger.info(f"single_threaded: {args.single_threaded}")

    build_docker_images(
        MODEL_SPECS,
        force_build=args.force_build,
        release=args.release,
        push=args.push,
        ubuntu_version=args.ubuntu_version,
        build_metal_commit=args.build_metal_commit,
        max_workers=args.max_workers,
        single_threaded=args.single_threaded,
    )
