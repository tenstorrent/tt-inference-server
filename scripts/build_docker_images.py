# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import argparse
import json
import logging
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

# Add the script's directory to the Python path
# this for 0 setup python setup script
project_root = Path(__file__).resolve().parent.parent
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))

from workflows.log_setup import setup_workflow_script_logger
from workflows.model_spec import MODEL_SPECS
from workflows.utils import get_repo_root_path

logger = logging.getLogger(__file__)


def _format_commit_for_id(commit, fallback):
    """
    Safely format a commit-like value for inclusion in combination IDs.
    Ensures None values don't cause slicing errors and keeps IDs concise.
    """
    return (commit if commit else fallback)[:16]


def _build_combination_id(tt_metal_commit, vllm_commit):
    """
    Build a safe combination identifier from tt_metal and vllm commits.
    """
    tm = _format_commit_for_id(tt_metal_commit, "unknown-tt")
    vc = _format_commit_for_id(vllm_commit, "no-vllm")
    return f"{tm}-{vc}"


def setup_individual_logger(tt_metal_commit, vllm_commit, log_dir, stdout_only=False):
    """
    Set up individual logger for each combination.
    Default: logs to file only.
    With stdout_only=True: logs to stdout only (no file).

    Args:
        tt_metal_commit: tt-metal commit hash
        vllm_commit: vllm commit hash
        log_dir: Directory for log files
        stdout_only: If True, only log to stdout (no file logging)
    """
    combination_id = _build_combination_id(tt_metal_commit, vllm_commit)
    logger_name = f"process_{combination_id}"

    # Create log directory if it doesn't exist (only if we're logging to file)
    log_dir = Path(log_dir)
    if not stdout_only:
        log_dir.mkdir(parents=True, exist_ok=True)

    # Create log file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = (
        log_dir / f"build_{timestamp}_{combination_id}.log" if not stdout_only else None
    )

    # Create logger
    process_logger = logging.getLogger(logger_name)
    process_logger.setLevel(logging.DEBUG)

    # Prevent duplicate handlers
    if process_logger.handlers:
        process_logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(
        f"[{combination_id}] %(asctime)s - %(levelname)s: %(message)s"
    )

    # File handler (default behavior - only if not stdout_only)
    if not stdout_only:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        process_logger.addHandler(file_handler)
    else:
        # Console handler for stdout-only mode (with combination_id prefix for clarity)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        process_logger.addHandler(console_handler)

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
        dry_run,
        stdout_only,
    ) = args_tuple

    # Set up individual logging for this combination
    logs_dir = get_build_logs_dir()
    process_logger, log_file = setup_individual_logger(
        tt_metal_commit, vllm_commit, logs_dir, stdout_only=stdout_only
    )

    combination_id = _build_combination_id(tt_metal_commit, vllm_commit)

    try:
        process_logger.info(f"=== STARTING BUILD FOR COMBINATION {combination_id} ===")
        process_logger.info(f"tt_metal_commit: {tt_metal_commit}")
        process_logger.info(f"vllm_commit: {vllm_commit}")
        process_logger.info(f"ubuntu_version: {ubuntu_version}")
        if log_file:
            process_logger.info(f"Log file: {log_file}")
        else:
            process_logger.info("Logging to stdout only (no file logging)")

        # Resolve tt_metal_commit to full SHA early in the process
        process_logger.info("Resolving tt_metal_commit to full SHA...")
        resolved_tt_metal_commit = resolve_commit_to_full_sha(tt_metal_commit)
        process_logger.info(f"Resolved tt_metal_commit: {resolved_tt_metal_commit}")

        # Generate image tags using provided commit
        image_tags = get_image_tags(
            tt_metal_commit=tt_metal_commit,
            vllm_commit=vllm_commit,
            ubuntu_version=ubuntu_version,
        )

        process_logger.info(f"Generated image tags: {image_tags}")

        # Track image status for all image types
        image_status = {}

        # Check existence of all images
        process_logger.info("Checking if images already exist...")
        for image_type in ["tt_metal_base", "cloud", "dev", "release"]:
            image_tag = image_tags[image_type]
            local_exists = check_image_exists_local(image_tag)
            remote_exists = check_image_exists_remote(image_tag)

            image_status[image_type] = {
                "tag": image_tag,
                "local_exists": local_exists,
                "remote_exists": remote_exists,
                "build_attempted": False,
                "build_succeeded": None,
            }

        # Determine what images need to be built
        build_tt_metal_base_flag = True
        build_cloud_image_flag = True
        build_dev_image_flag = True
        build_release_image_flag = True

        if not force_build:
            if (
                image_status["cloud"]["local_exists"]
                or image_status["cloud"]["remote_exists"]
            ):
                build_cloud_image_flag = False
                process_logger.info("Cloud image already exists, skipping build")

            if (
                image_status["dev"]["local_exists"]
                or image_status["dev"]["remote_exists"]
            ):
                build_dev_image_flag = False
                process_logger.info("Dev image already exists, skipping build")
                # Dev image is built FROM cloud image, so cloud must exist too
                if build_cloud_image_flag:
                    build_cloud_image_flag = False
                    process_logger.info(
                        "Dev image exists, cloud image must exist too, skipping cloud build"
                    )

            if (
                image_status["release"]["local_exists"]
                or image_status["release"]["remote_exists"]
            ):
                build_release_image_flag = False
                process_logger.info("Release image already exists, skipping build")
                # Release image is built FROM cloud image, so cloud must exist too
                if build_cloud_image_flag:
                    build_cloud_image_flag = False
                    process_logger.info(
                        "Release image exists, cloud image must exist too, skipping cloud build"
                    )

            if (
                image_status["tt_metal_base"]["local_exists"]
                or image_status["tt_metal_base"]["remote_exists"]
            ):
                build_tt_metal_base_flag = False

            if (
                release
                and build_release_image_flag
                and not image_status["cloud"]["local_exists"]
            ):
                # NOTE: copying a dev image into a release image is not guaranteed
                # to have correct code in it, so it is required to build the cloud image
                # as part of release process.
                process_logger.info(
                    "Cloud image does not exist locally, building cloud image to safely build release image"
                )
                build_cloud_image_flag = True
                # might as well build the dev image too, just a different tag
                build_dev_image_flag = True

        # Build tt-metal base image only if needed
        if (
            build_cloud_image_flag or build_dev_image_flag
        ) and build_tt_metal_base_flag:
            image_status["tt_metal_base"]["build_attempted"] = True
            if dry_run:
                process_logger.info("[DRY-RUN] Would build tt-metal base image...")
            else:
                process_logger.info("Building tt-metal base image...")
                try:
                    build_tt_metal_base_image(
                        image_tags["tt_metal_base"],
                        resolved_tt_metal_commit,
                        ubuntu_version,
                        process_logger,
                    )
                    image_status["tt_metal_base"]["build_succeeded"] = True
                except Exception as e:
                    process_logger.error(f"Failed to build tt-metal base image: {e}")
                    image_status["tt_metal_base"]["build_succeeded"] = False
                    raise
        else:
            process_logger.info(
                "All final images exist, skipping tt-metal base image build"
            )

        # Build cloud image
        if build_cloud_image_flag:
            image_status["cloud"]["build_attempted"] = True
            if dry_run:
                process_logger.info(
                    f"[DRY-RUN] Would build cloud image: {image_tags['cloud']}"
                )
            else:
                process_logger.info("Building cloud image...")
                try:
                    build_cloud_image(
                        image_tags,
                        resolved_tt_metal_commit,
                        vllm_commit,
                        container_app_uid,
                        process_logger,
                    )
                    image_status["cloud"]["build_succeeded"] = True
                except Exception as e:
                    process_logger.error(f"Failed to build cloud image: {e}")
                    image_status["cloud"]["build_succeeded"] = False
                    raise
        else:
            process_logger.info(f"Skipping cloud image build: {image_tags['cloud']}")

        # Build dev image
        if build_dev_image_flag:
            image_status["dev"]["build_attempted"] = True
            if dry_run:
                process_logger.info(
                    f"[DRY-RUN] Would build dev image: {image_tags['dev']}"
                )
            else:
                process_logger.info("Building dev image...")
                try:
                    build_dev_image(image_tags, process_logger)
                    image_status["dev"]["build_succeeded"] = True
                except Exception as e:
                    process_logger.error(f"Failed to build dev image: {e}")
                    image_status["dev"]["build_succeeded"] = False
                    raise
        else:
            process_logger.info(f"Skipping dev image build: {image_tags['dev']}")

        # Build release image (only if release=True)
        if release and build_release_image_flag:
            image_status["release"]["build_attempted"] = True
            if dry_run:
                process_logger.info(
                    f"[DRY-RUN] Would build release image: {image_tags['release']}"
                )
            else:
                process_logger.info("Building release image...")
                try:
                    build_release_image(image_tags, process_logger)
                    image_status["release"]["build_succeeded"] = True
                except Exception as e:
                    process_logger.error(f"Failed to build release image: {e}")
                    image_status["release"]["build_succeeded"] = False
                    raise
        else:
            process_logger.info(
                f"Skipping release image build: {image_tags['release']}"
            )

        # Push images if requested
        if push:
            if dry_run:
                process_logger.info("[DRY-RUN] Would push images to registry...")
            else:
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
                    if dry_run:
                        process_logger.info(f"[DRY-RUN] Would push image: {image_tag}")
                    else:
                        push_image(image_tag, process_logger)
                else:
                    process_logger.info(f"Skipping push for {image_tag}")

        process_logger.info(f"=== COMPLETED BUILD FOR COMBINATION {combination_id} ===")
        return {
            "success": True,
            "combination_id": combination_id,
            "tt_metal_commit": tt_metal_commit,
            "vllm_commit": vllm_commit,
            "log_file": str(log_file) if log_file else None,
            "images": image_status,
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
            "tt_metal_commit": tt_metal_commit,
            "vllm_commit": vllm_commit,
            "log_file": str(log_file) if log_file else None,
            "error": str(e),
            "images": image_status if "image_status" in locals() else {},
        }


def run_command_with_logging(command, logger, check=True, cwd=None):
    """
    Run a command and write output to log file (default) or stdout (if stdout-only mode).
    Default behavior: writes to log file only.
    stdout-only mode: writes to stdout in real-time.
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

        # Check if there's a console handler (stdout-only mode)
        has_console_handler = any(
            isinstance(handler, logging.StreamHandler) for handler in logger.handlers
        )

        # Read output line by line as it's produced
        for line in iter(process.stdout.readline, ""):
            line = line.rstrip()
            if line:
                stdout_lines.append(line)
                # Write raw output directly to log file if available
                if log_file_handler:
                    log_file_handler.stream.write(line + "\n")
                    log_file_handler.stream.flush()
                # Print to stdout only if console handler is present (stdout-only mode)
                if has_console_handler:
                    print(line, flush=True)

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
        cmd = ["docker", "manifest", "inspect", image_tag]
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
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

    # Make sure we are in the git root
    try:
        git_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        current_dir = os.getcwd()
        if current_dir != git_root:
            raise ValueError(
                f"Script must be run in tt-inference-server repo root. "
                f"Current directory: {current_dir}, git root: {git_root}"
            )
    except subprocess.CalledProcessError:
        raise ValueError(
            "Script must be run in a git repository. Could not determine git root directory."
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

    # Fallback: Try GitHub API for short SHA resolution
    try:
        logger.info(f"Trying GitHub commits API fallback for {tt_metal_commit}...")
        api_url = f"https://api.github.com/repos/tenstorrent/tt-metal/commits/{tt_metal_commit}"

        with urllib.request.urlopen(api_url, timeout=10) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                full_sha = data["sha"]
                logger.info(
                    f"Resolved {tt_metal_commit} to full SHA via GitHub API: {full_sha}"
                )
                return full_sha
            else:
                logger.debug(
                    f"GitHub API returned status {response.status} for {tt_metal_commit}"
                )

    except urllib.error.HTTPError as e:
        if e.code == 404:
            logger.debug(f"GitHub API: commit {tt_metal_commit} not found (404)")
        else:
            logger.debug(f"GitHub API HTTP error for {tt_metal_commit}: {e.code}")
    except urllib.error.URLError as e:
        logger.debug(f"GitHub API URL error for {tt_metal_commit}: {e}")
    except json.JSONDecodeError as e:
        logger.debug(f"GitHub API JSON decode error for {tt_metal_commit}: {e}")
    except Exception as e:
        logger.debug(f"GitHub API fallback failed for {tt_metal_commit}: {e}")

    # If we can't resolve it, return the original reference
    logger.info(f"Could not resolve {tt_metal_commit} to full SHA, using as-is")
    return tt_metal_commit


def build_tt_metal_base_image(
    tt_metal_base_tag, tt_metal_commit, ubuntu_version, logger=logger
):
    """
    Build the tt-metal base image if it doesn't exist.

    Args:
        tt_metal_base_tag: Docker image tag to build tt-metal base image with
        tt_metal_commit: Already resolved full SHA commit hash
        ubuntu_version: Ubuntu version to use
        logger: Logger instance
    """
    # note: note used
    # os_version = f"ubuntu-{ubuntu_version}-amd64"

    if check_image_exists_local(tt_metal_base_tag):
        logger.info(f"TT-Metal base image already exists: {tt_metal_base_tag}")
        return True

    logger.info(f"Building TT-Metal base image: {tt_metal_base_tag}")

    # tt_metal_commit is already resolved to full SHA by the caller
    resolved_commit = tt_metal_commit

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
            logger.info("Trying to fetch as tag ...")
            run_command_with_logging(
                ["git", "fetch", "--depth", "1", "origin", "tag", resolved_commit],
                logger=logger,
                check=True,
                cwd=tt_metal_dir,
            )
            logger.info("Fetched as tag.")
        except subprocess.CalledProcessError:
            try:
                logger.info(
                    "Trying to fetch as commit SHA from shallow repo history ..."
                )
                run_command_with_logging(
                    ["git", "fetch", "--depth", "1", "origin", resolved_commit],
                    logger=logger,
                    check=True,
                    cwd=tt_metal_dir,
                )
                logger.info("Fetched as commit SHA.")
            except subprocess.CalledProcessError:
                logger.info(
                    "Trying to fetch in unshallow repo history, this make take a minute ..."
                )
                try:
                    run_command_with_logging(
                        ["git", "fetch", "--unshallow"],
                        logger=logger,
                        check=True,
                        cwd=tt_metal_dir,
                    )
                    logger.info("Fetched full history successfully.")
                except subprocess.CalledProcessError:
                    logger.info(
                        "Trying full repo history fetch, this make take a minute ..."
                    )
                    run_command_with_logging(
                        ["git", "fetch", "origin"],
                        logger=logger,
                        check=True,
                        cwd=tt_metal_dir,
                    )
                    logger.info("Full fetch completed.")

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
            "--platform",
            "linux/amd64",
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

    Args:
        image_tags: Dictionary of image tags
        tt_metal_commit: Already resolved full SHA commit hash
        vllm_commit: VLLM commit hash
        container_app_uid: Container application UID
        logger: Logger instance
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


def list_image_combinations(model_configs, build_metal_commit=None):
    """
    Get unique Docker image commit combinations that would be built.

    Args:
        model_configs: Dictionary of model configurations
        build_metal_commit: Only return combinations with this exact tt-metal commit

    Returns:
        Set of tuples (tt_metal_commit, vllm_commit) representing unique combinations
    """
    unique_sha_combinations = {
        (config.tt_metal_commit, config.vllm_commit)
        for config in model_configs.values()
        if config.vllm_commit is not None
    }

    skipped_count = sum(
        1 for config in model_configs.values() if config.vllm_commit is None
    )

    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} model config(s) with vllm_commit=None")

    if build_metal_commit:
        unique_sha_combinations = {
            combo for combo in unique_sha_combinations if combo[0] == build_metal_commit
        }

    if not unique_sha_combinations:
        logger.warning(
            f"No configurations found with tt_metal_commit={build_metal_commit}"
        )
        return set()

    return unique_sha_combinations


def build_docker_images(
    model_configs,
    force_build=False,
    release=False,
    push=False,
    ubuntu_version="20.04",
    build_metal_commit=None,
    max_workers=None,
    single_threaded=False,
    dry_run=False,
    stdout_only=False,
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
        dry_run: Print summary of what would be built without building
        stdout_only: If True, only log to stdout (no file logging)
    """
    # Validate inputs
    container_app_uid = 1000
    validate_inputs(ubuntu_version, container_app_uid)

    # Get unique combinations using the shared function
    unique_sha_combinations = list_image_combinations(
        model_configs,
        build_metal_commit=build_metal_commit,
    )

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
            dry_run,
            stdout_only,
        )
        for tt_metal_commit, vllm_commit in unique_sha_combinations
    ]

    if not args_tuples:
        logger.warning("No combinations to process")
        return

    # Log execution mode
    if dry_run:
        logger.info("=" * 80)
        logger.info("DRY-RUN MODE: Checking image status without building")
        logger.info("=" * 80)

    # Use multiprocessing.Pool to process combinations in parallel
    logger.info(f"Processing {len(args_tuples)} combinations with {workers} workers")

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
            log_info = (
                f" - Log: {result['log_file']}"
                if result.get("log_file")
                else " (stdout only)"
            )
            logger.info(f"✅ Success: {result['combination_id']}{log_info}")
        else:
            failure_count += 1
            failed_combinations.append(result)
            log_info = (
                f" - Log: {result['log_file']}"
                if result.get("log_file")
                else " (stdout only)"
            )
            logger.error(f"❌ Failed: {result['combination_id']}{log_info}")
            logger.error(f"   Error: {result['error']}")

    logger.info(
        f"Completed processing: {success_count} successful, {failure_count} failed"
    )

    if failed_combinations:
        logger.error("Failed combinations:")
        for failed in failed_combinations:
            log_info = (
                f" - Log: {failed['log_file']}"
                if failed.get("log_file")
                else " (stdout only)"
            )
            logger.error(f"  - {failed['combination_id']}{log_info}")
        if any(f.get("log_file") for f in failed_combinations):
            logger.error("Use the log files above to debug the specific failures.")

    # Aggregate image build status across all combinations
    build_attempted = {"cloud": [], "dev": [], "release": []}
    build_succeeded = {"cloud": [], "dev": [], "release": []}
    remote_exists = {"cloud": [], "dev": [], "release": []}

    for result in results:
        images = result.get("images", {})
        for image_type in ["cloud", "dev", "release"]:
            if image_type in images:
                image_info = images[image_type]
                if image_info.get("build_attempted", False):
                    build_attempted[image_type].append(image_info.get("tag", ""))
                if image_info.get("build_succeeded", False):
                    build_succeeded[image_type].append(image_info.get("tag", ""))
                if image_info.get("remote_exists", False):
                    remote_exists[image_type].append(image_info.get("tag", ""))

    # Generate JSON summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = get_build_logs_dir()
    json_file = logs_dir / f"build_summary_{timestamp}.json"

    summary_data = {
        "timestamp": timestamp,
        "dry_run": dry_run,
        "ubuntu_version": ubuntu_version,
        "force_build": force_build,
        "release": release,
        "push": push,
        "success_count": success_count,
        "failure_count": failure_count,
        "build_attempted": build_attempted,
        "build_succeeded": build_succeeded,
        "remote_exists": remote_exists,
        "combinations": results,
    }

    with open(json_file, "w") as f:
        json.dump(summary_data, f, indent=2)

    logger.info(f"Build summary saved to: {json_file}")
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary of what would be built without building",
    )
    parser.add_argument(
        "--stdout-only",
        action="store_true",
        help="Only log to stdout (no file logging)",
    )
    args = parser.parse_args()
    logger.info(f"ubuntu_version: {args.ubuntu_version}")
    logger.info(f"build_metal_commit: {args.build_metal_commit}")
    logger.info(f"max_workers: {args.max_workers}")
    logger.info(f"force_build: {args.force_build}")
    logger.info(f"release: {args.release}")
    logger.info(f"push: {args.push}")
    logger.info(f"single_threaded: {args.single_threaded}")
    logger.info(f"dry_run: {args.dry_run}")
    logger.info(f"stdout_only: {args.stdout_only}")

    build_docker_images(
        MODEL_SPECS,
        force_build=args.force_build,
        release=args.release,
        push=args.push,
        ubuntu_version=args.ubuntu_version,
        build_metal_commit=args.build_metal_commit,
        max_workers=args.max_workers,
        single_threaded=args.single_threaded,
        dry_run=args.dry_run,
        stdout_only=args.stdout_only,
    )
