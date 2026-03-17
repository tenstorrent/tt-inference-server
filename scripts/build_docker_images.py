# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import argparse
import concurrent.futures
import json
import logging
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
import time
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
from workflows.model_spec import MODEL_SPECS, export_model_specs_json
from workflows.utils import get_repo_root_path

logger = logging.getLogger(__file__)

MEMORY_PER_BUILD_GB = 16
MEMORY_RESERVE_GB = 16
DISK_PER_BUILD_GB = 40
DISK_RESERVE_GB = 20
RESOURCE_POLL_INTERVAL_SECONDS = 10
# ~4 CPU cores needed per concurrent Docker build
CPU_CORES_PER_BUILD = 4
DRY_RUN_BUILD_DURATION_SECONDS = 5


def get_available_memory_gb():
    """Get available system memory in GB from /proc/meminfo (MemAvailable).

    MemAvailable accounts for reclaimable buffers/cache, so it reflects
    what is actually usable. Available since Linux 3.14.
    """
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024)
    except (FileNotFoundError, ValueError, IndexError):
        pass

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        avail_pages = os.sysconf("SC_AVPHYS_PAGES")
        if page_size > 0 and avail_pages > 0:
            return (page_size * avail_pages) / (1024**3)
    except (ValueError, OSError):
        pass

    raise RuntimeError("Could not determine available memory")


def get_docker_root_dir():
    """Get Docker's storage root directory via `docker info`."""
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.DockerRootDir}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return "/var/lib/docker"


def get_available_disk_gb(path=None):
    """Get available disk space in GB for the filesystem containing path.

    Defaults to the Docker storage root directory.
    """
    if path is None:
        path = get_docker_root_dir()
    try:
        usage = shutil.disk_usage(path)
        return usage.free / (1024**3)
    except OSError:
        usage = shutil.disk_usage("/")
        return usage.free / (1024**3)


def get_max_concurrent_builds(
    memory_per_build_gb=MEMORY_PER_BUILD_GB,
    memory_reserve_gb=MEMORY_RESERVE_GB,
    disk_per_build_gb=DISK_PER_BUILD_GB,
    disk_reserve_gb=DISK_RESERVE_GB,
    max_workers=None,
):
    """Compute safe max concurrent builds based on host RAM, disk, and CPU.

    Returns:
        Tuple of (limit, details_dict) where details_dict contains the
        individual constraint values for logging.
    """
    available_mem_gb = get_available_memory_gb()
    docker_root = get_docker_root_dir()
    available_disk_gb = get_available_disk_gb(docker_root)
    physical_cpu_count = get_physical_cpu_count()

    max_by_memory = int((available_mem_gb - memory_reserve_gb) // memory_per_build_gb)
    max_by_disk = int((available_disk_gb - disk_reserve_gb) // disk_per_build_gb)
    max_by_cpu = physical_cpu_count // CPU_CORES_PER_BUILD

    resource_limit = max(1, min(max_by_memory, max_by_disk, max_by_cpu))

    if max_workers is not None:
        limit = min(resource_limit, max_workers)
    else:
        limit = resource_limit

    limit = max(1, limit)

    constraints = {
        "memory": max_by_memory,
        "disk": max_by_disk,
        "cpu": max_by_cpu,
    }
    binding = min(constraints, key=constraints.get)

    details = {
        "available_memory_gb": round(available_mem_gb, 1),
        "available_disk_gb": round(available_disk_gb, 1),
        "docker_root": docker_root,
        "physical_cpu_count": physical_cpu_count,
        "memory_per_build_gb": memory_per_build_gb,
        "memory_reserve_gb": memory_reserve_gb,
        "disk_per_build_gb": disk_per_build_gb,
        "disk_reserve_gb": disk_reserve_gb,
        "max_by_memory": max_by_memory,
        "max_by_disk": max_by_disk,
        "max_by_cpu": max_by_cpu,
        "resource_limit": resource_limit,
        "max_workers_override": max_workers,
        "effective_limit": limit,
        "binding_constraint": binding,
    }
    return limit, details


def check_resources_for_new_build(
    memory_per_build_gb=MEMORY_PER_BUILD_GB,
    memory_reserve_gb=MEMORY_RESERVE_GB,
    disk_per_build_gb=DISK_PER_BUILD_GB,
    disk_reserve_gb=DISK_RESERVE_GB,
):
    """Check if there are sufficient resources to start one more build.

    Returns:
        Tuple of (ok, available_mem_gb, available_disk_gb)
    """
    available_mem_gb = get_available_memory_gb()
    available_disk_gb = get_available_disk_gb()
    mem_ok = (available_mem_gb - memory_reserve_gb) >= memory_per_build_gb
    disk_ok = (available_disk_gb - disk_reserve_gb) >= disk_per_build_gb
    return (mem_ok and disk_ok), available_mem_gb, available_disk_gb


def log_resource_summary(details, total_builds):
    """Log a detailed resource summary before builds begin."""
    d = details
    logger.info("=" * 60)
    logger.info("BUILD RESOURCE SUMMARY")
    logger.info("=" * 60)
    logger.info("Host resources:")
    logger.info(f"  Available memory: {d['available_memory_gb']} GB")
    logger.info(f"  Available disk ({d['docker_root']}): {d['available_disk_gb']} GB")
    logger.info(f"  Physical CPU cores: {d['physical_cpu_count']}")
    logger.info("Per-build requirements:")
    logger.info(f"  Memory per build: {d['memory_per_build_gb']} GB")
    logger.info(f"  Memory reserve: {d['memory_reserve_gb']} GB")
    logger.info(f"  Disk per build: {d['disk_per_build_gb']} GB")
    logger.info(f"  Disk reserve: {d['disk_reserve_gb']} GB")
    logger.info("Concurrency limits:")
    logger.info(
        f"  By memory: {d['max_by_memory']}"
        f" (({d['available_memory_gb']} - {d['memory_reserve_gb']}) GB"
        f" / {d['memory_per_build_gb']} GB)"
    )
    logger.info(
        f"  By disk: {d['max_by_disk']}"
        f" (({d['available_disk_gb']} - {d['disk_reserve_gb']}) GB"
        f" / {d['disk_per_build_gb']} GB)"
    )
    logger.info(
        f"  By CPU: {d['max_by_cpu']}"
        f" ({d['physical_cpu_count']} cores / {CPU_CORES_PER_BUILD})"
    )
    if d["max_workers_override"] is not None:
        logger.info(f"  --max-workers override: {d['max_workers_override']}")
    logger.info(
        f"  Effective max concurrent builds: {d['effective_limit']}"
        f" (limited by {d['binding_constraint']})"
    )
    logger.info("Build queue:")
    start_now = min(d["effective_limit"], total_builds)
    queued = total_builds - start_now
    logger.info(f"  Total builds queued: {total_builds}")
    logger.info(
        f"  Will start {start_now} immediately, {queued} queued waiting for resources"
    )
    logger.info("=" * 60)

    if d["effective_limit"] < total_builds:
        logger.warning(
            f"Throttling active: only {d['effective_limit']} of {total_builds}"
            " builds can run concurrently due to resource constraints"
        )

    usable_mem = d["available_memory_gb"] - d["memory_reserve_gb"]
    if usable_mem < d["memory_per_build_gb"]:
        logger.error(
            f"Available memory after reserve ({usable_mem:.1f} GB) is below the"
            f" minimum per-build requirement ({d['memory_per_build_gb']} GB)."
            " Builds may fail due to OOM."
        )

    usable_disk = d["available_disk_gb"] - d["disk_reserve_gb"]
    if usable_disk < d["disk_per_build_gb"]:
        logger.error(
            f"Available disk after reserve ({usable_disk:.1f} GB) is below the"
            f" minimum per-build requirement ({d['disk_per_build_gb']} GB)."
            " Builds may fail due to insufficient disk space."
        )


def generate_model_specs_json(output_path: Path = None) -> Path:
    """Generate model_spec.json by serializing all MODEL_SPECS.

    This function generates a JSON file containing all model specifications
    that will be embedded in the Docker image. The JSON can be used at runtime
    to look up model configurations by hf_model_repo and device_type.

    Args:
        output_path: Path where the JSON file should be written.
                    Defaults to repo_root / "model_spec.json"

    Returns:
        Path to the generated JSON file
    """
    if output_path is None:
        output_path = get_repo_root_path() / "model_spec.json"

    num_specs = export_model_specs_json(MODEL_SPECS, output_path)
    logger.info(f"Generated model_spec.json with {num_specs} specs at {output_path}")
    return output_path


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
        multihost,
        push,
        container_app_uid,
        dry_run,
        stdout_only,
        dry_run_build_duration,
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
        for image_type in ["tt_metal_base", "dev", "release", "multihost"]:
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
        build_dev_image_flag = True
        build_release_image_flag = True
        build_multihost_image_flag = True

        if not force_build:
            if (
                image_status["dev"]["local_exists"]
                or image_status["dev"]["remote_exists"]
            ):
                build_dev_image_flag = False
                process_logger.info("Dev image already exists, skipping build")

            if (
                image_status["release"]["local_exists"]
                or image_status["release"]["remote_exists"]
            ):
                build_release_image_flag = False
                process_logger.info("Release image already exists, skipping build")

            if (
                image_status["multihost"]["local_exists"]
                or image_status["multihost"]["remote_exists"]
            ):
                build_multihost_image_flag = False
                process_logger.info("Multihost image already exists, skipping build")

            if (
                image_status["tt_metal_base"]["local_exists"]
                or image_status["tt_metal_base"]["remote_exists"]
            ):
                build_tt_metal_base_flag = False

            if (
                release
                and build_release_image_flag
                and not image_status["dev"]["local_exists"]
            ):
                # Release image is tagged from dev, so dev must be built
                process_logger.info(
                    "Dev image does not exist locally, building dev image to safely build release image"
                )
                build_dev_image_flag = True

            if (
                multihost
                and build_multihost_image_flag
                and not image_status["release"]["local_exists"]
            ):
                # Multihost image is built from release, so release must exist
                process_logger.info(
                    "Release image does not exist locally, building release image to build multihost image"
                )
                build_release_image_flag = True
                if not image_status["dev"]["local_exists"]:
                    build_dev_image_flag = True

        # Build tt-metal base image only if needed
        if build_dev_image_flag and build_tt_metal_base_flag:
            image_status["tt_metal_base"]["build_attempted"] = True
            if dry_run:
                process_logger.info(
                    f"[DRY-RUN] Would build tt-metal base image... "
                    f"(simulating {dry_run_build_duration}s build)"
                )
                time.sleep(dry_run_build_duration)
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

        # Build dev image
        if build_dev_image_flag:
            image_status["dev"]["build_attempted"] = True
            if dry_run:
                process_logger.info(
                    f"[DRY-RUN] Would build dev image: {image_tags['dev']} "
                    f"(simulating {dry_run_build_duration}s build)"
                )
                time.sleep(dry_run_build_duration)
            else:
                process_logger.info("Building dev image...")
                try:
                    build_dev_image(
                        image_tags,
                        resolved_tt_metal_commit,
                        vllm_commit,
                        container_app_uid,
                        process_logger,
                    )
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
                    f"[DRY-RUN] Would build release image: {image_tags['release']} "
                    f"(simulating {dry_run_build_duration}s build)"
                )
                time.sleep(dry_run_build_duration)
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

        # Build multihost image (only if multihost=True)
        if multihost and build_multihost_image_flag:
            image_status["multihost"]["build_attempted"] = True
            if dry_run:
                process_logger.info(
                    f"[DRY-RUN] Would build multihost image: {image_tags['multihost']}"
                )
            else:
                process_logger.info("Building multihost image...")
                try:
                    build_multihost_image(image_tags, process_logger)
                    image_status["multihost"]["build_succeeded"] = True
                except Exception as e:
                    process_logger.error(f"Failed to build multihost image: {e}")
                    image_status["multihost"]["build_succeeded"] = False
                    raise
        else:
            process_logger.info(
                f"Skipping multihost image build: {image_tags['multihost']}"
            )

        # Push images if requested
        if push:
            if dry_run:
                process_logger.info(
                    f"[DRY-RUN] Would push images to registry... "
                    f"(simulating {dry_run_build_duration}s push)"
                )
                time.sleep(dry_run_build_duration)
            else:
                process_logger.info("Pushing images to registry...")

            for image_type in ["dev", "release", "multihost"]:
                image_tag = image_tags[image_type]

                # Skip release image unless explicitly marked as a release
                if image_type == "release" and not release:
                    process_logger.info(
                        f"Skipping push for {image_tag}, release={release}"
                    )
                    continue

                # Skip multihost image unless explicitly marked as multihost
                if image_type == "multihost" and not multihost:
                    process_logger.info(
                        f"Skipping push for {image_tag}, multihost={multihost}"
                    )
                    continue

                if should_push_image(image_tag, force_push=False):
                    if dry_run:
                        process_logger.info(
                            f"[DRY-RUN] Would push image: {image_tag} "
                            f"(simulating {dry_run_build_duration}s push)"
                        )
                        time.sleep(dry_run_build_duration)
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
        result = subprocess.run(cmd, capture_output=True, text=True)
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

    dev_image_tag = f"{image_repo}/vllm-tt-metal-src-dev-{os_version}:{image_version}-{tt_metal_tag}-{vllm_tag}{suffix}"
    release_image_tag = f"{image_repo}/vllm-tt-metal-src-release-{os_version}:{image_version}-{tt_metal_tag}-{vllm_tag}{suffix}"
    multihost_image_tag = f"{image_repo}/vllm-tt-metal-src-multihost-{os_version}:{image_version}-{tt_metal_tag}-{vllm_tag}{suffix}"
    tt_metal_base_tag = f"local/tt-metal/tt-metalium/{os_version}:{tt_metal_commit}"

    return {
        "dev": dev_image_tag,
        "release": release_image_tag,
        "multihost": multihost_image_tag,
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


def build_dev_image(
    image_tags, tt_metal_commit, vllm_commit, container_app_uid, logger
):
    """
    Build the dev Docker image from the Dockerfile.

    Args:
        image_tags: Dictionary of image tags
        tt_metal_commit: Already resolved full SHA commit hash
        vllm_commit: VLLM commit hash
        container_app_uid: Container application UID
        logger: Logger instance
    """
    repo_root = get_repo_root_path()
    dev_image_tag = image_tags["dev"]
    tt_metal_base_tag = image_tags["tt_metal_base"]

    # Generate model_spec.json before building (COPY'd into image)
    model_specs_json_path = generate_model_specs_json()
    logger.info(f"Generated model specs JSON at: {model_specs_json_path}")

    logger.info(f"Building dev image: {dev_image_tag}")

    build_command = [
        "docker",
        "build",
        "-t",
        dev_image_tag,
        "--build-arg",
        f"TT_METAL_DOCKERFILE_URL={tt_metal_base_tag}",
        "--build-arg",
        f"TT_METAL_COMMIT_SHA_OR_TAG={tt_metal_commit}",
        "--build-arg",
        f"TT_VLLM_COMMIT_SHA_OR_TAG={vllm_commit}",
        "--build-arg",
        f"CONTAINER_APP_UID={container_app_uid}",
        "-f",
        "vllm-tt-metal/vllm.tt-metal.src.dev.Dockerfile",
        ".",
    ]

    run_command_with_logging(build_command, logger=logger, check=True, cwd=repo_root)
    logger.info(f"Successfully built dev image: {dev_image_tag}")


def build_release_image(image_tags, logger):
    """
    Tag the dev image as the release image.

    Release image is identical to the dev image — just a different tag.
    """
    release_image_tag = image_tags["release"]
    dev_image_tag = image_tags["dev"]

    logger.info(f"Tagging dev image as release: {dev_image_tag} -> {release_image_tag}")

    tag_command = ["docker", "tag", dev_image_tag, release_image_tag]
    run_command_with_logging(tag_command, logger=logger, check=True)
    logger.info(f"Successfully tagged release image: {release_image_tag}")


def build_multihost_image(image_tags, logger):
    """
    Build the multihost Docker image from the release image.

    Multihost image extends the release image with SSH server and
    multihost_entrypoint.sh for distributed MPI-based inference.
    """
    repo_root = get_repo_root_path()
    multihost_image_tag = image_tags["multihost"]
    release_image_tag = image_tags["release"]

    logger.info(f"Building multihost image: {multihost_image_tag}")
    logger.info(f"Base image: {release_image_tag}")

    build_command = [
        "docker",
        "build",
        "-t",
        multihost_image_tag,
        "--build-arg",
        f"BASE_IMAGE={release_image_tag}",
        "-f",
        "vllm-tt-metal/vllm.tt-metal.src.multihost.Dockerfile",
        ".",
    ]

    run_command_with_logging(build_command, logger=logger, check=True, cwd=repo_root)
    logger.info(f"Successfully built multihost image: {multihost_image_tag}")


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


def _run_resource_aware_queue(
    args_tuples,
    max_concurrent,
    memory_per_build_gb,
    memory_reserve_gb,
    disk_per_build_gb,
    disk_reserve_gb,
):
    """Submit builds via a resource-gated queue using ProcessPoolExecutor.

    Submits up to max_concurrent builds initially, then after each completion
    polls host resources before submitting the next queued build.
    """
    pending_queue = list(args_tuples)
    results = []
    active_futures = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_concurrent) as executor:

        def _submit_next():
            if not pending_queue:
                return
            args = pending_queue.pop(0)
            combo_id = _build_combination_id(args[0], args[1])
            future = executor.submit(process_sha_combination, args)
            active_futures[future] = combo_id
            active = len(active_futures)
            queued = len(pending_queue)
            logger.info(
                f"Starting build {combo_id}"
                f" ({active}/{max_concurrent} slots, {queued} queued)"
            )

        def _wait_for_resources_and_submit():
            logger.info("Polling resources for next build...")
            ok, avail_mem, avail_disk = check_resources_for_new_build(
                memory_per_build_gb,
                memory_reserve_gb,
                disk_per_build_gb,
                disk_reserve_gb,
            )
            while not ok:
                logger.warning(
                    f"Waiting for resources: {avail_mem:.1f} GB memory,"
                    f" {avail_disk:.1f} GB disk available"
                    f" (need {memory_per_build_gb} GB mem"
                    f" + {memory_reserve_gb} GB reserve,"
                    f" {disk_per_build_gb} GB disk"
                    f" + {disk_reserve_gb} GB reserve;"
                    f" polling every {RESOURCE_POLL_INTERVAL_SECONDS}s)..."
                )
                time.sleep(RESOURCE_POLL_INTERVAL_SECONDS)
                ok, avail_mem, avail_disk = check_resources_for_new_build(
                    memory_per_build_gb,
                    memory_reserve_gb,
                    disk_per_build_gb,
                    disk_reserve_gb,
                )
            logger.info("Resources available, starting next build from queue")
            _submit_next()

        while pending_queue and len(active_futures) < max_concurrent:
            ok, avail_mem, avail_disk = check_resources_for_new_build(
                memory_per_build_gb,
                memory_reserve_gb,
                disk_per_build_gb,
                disk_reserve_gb,
            )
            if not ok:
                logger.info(
                    f"Pausing initial submissions: {avail_mem:.1f} GB memory,"
                    f" {avail_disk:.1f} GB disk available"
                )
                break
            _submit_next()

        while active_futures:
            done, _ = concurrent.futures.wait(
                active_futures,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                combo_id = active_futures.pop(future)
                result = future.result()
                results.append(result)
                status = "succeeded" if result["success"] else "FAILED"
                logger.info(f"Completed build {combo_id} ({status})")

                if pending_queue:
                    _wait_for_resources_and_submit()

    return results


def build_docker_images(
    model_configs,
    force_build=False,
    release=False,
    multihost=False,
    push=False,
    ubuntu_version="20.04",
    build_metal_commit=None,
    max_workers=None,
    single_threaded=False,
    dry_run=False,
    stdout_only=False,
    memory_per_build_gb=MEMORY_PER_BUILD_GB,
    disk_per_build_gb=DISK_PER_BUILD_GB,
    dry_run_build_duration=DRY_RUN_BUILD_DURATION_SECONDS,
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
        max_workers: Ceiling on parallel workers (resource-based limit applies first)
        single_threaded: Run builds sequentially instead of in parallel (for debugging)
        dry_run: Print summary of what would be built without building
        stdout_only: If True, only log to stdout (no file logging)
        memory_per_build_gb: GB of RAM required per concurrent build
        disk_per_build_gb: GB of disk required per concurrent build
        dry_run_build_duration: Seconds each mock build sleeps in dry-run mode
    """
    container_app_uid = 1000
    validate_inputs(ubuntu_version, container_app_uid)

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

    args_tuples = [
        (
            tt_metal_commit,
            vllm_commit,
            ubuntu_version,
            force_build,
            release,
            multihost,
            push,
            container_app_uid,
            dry_run,
            stdout_only,
            dry_run_build_duration,
        )
        for tt_metal_commit, vllm_commit in unique_sha_combinations
    ]

    if not args_tuples:
        logger.warning("No combinations to process")
        return

    max_concurrent, resource_details = get_max_concurrent_builds(
        memory_per_build_gb=memory_per_build_gb,
        memory_reserve_gb=MEMORY_RESERVE_GB,
        disk_per_build_gb=disk_per_build_gb,
        max_workers=max_workers,
    )
    log_resource_summary(resource_details, total_builds=len(args_tuples))

    if dry_run:
        logger.info("=" * 80)
        logger.info("DRY-RUN MODE: Checking image status without building")
        logger.info("=" * 80)

    logger.info(
        f"Processing {len(args_tuples)} combinations (max concurrent: {max_concurrent})"
    )

    if single_threaded:
        results = []
        for args_tuple in args_tuples:
            results.append(process_sha_combination(args_tuple))
    else:
        results = _run_resource_aware_queue(
            args_tuples,
            max_concurrent,
            memory_per_build_gb,
            MEMORY_RESERVE_GB,
            disk_per_build_gb,
            DISK_RESERVE_GB,
        )

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
    build_attempted = {"dev": [], "release": [], "multihost": []}
    build_succeeded = {"dev": [], "release": [], "multihost": []}
    remote_exists = {"dev": [], "release": [], "multihost": []}

    for result in results:
        images = result.get("images", {})
        for image_type in ["dev", "release", "multihost"]:
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
    logs_dir.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument(
        "--multihost",
        action="store_true",
        help="Build multihost image for distributed inference.",
    )
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
        help="Ceiling on parallel workers (resource-based auto-limit applies first)",
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
        "--dry-run-build-duration",
        type=float,
        default=DRY_RUN_BUILD_DURATION_SECONDS,
        help=f"Seconds each mock build sleeps in dry-run mode (default: {DRY_RUN_BUILD_DURATION_SECONDS})",
    )
    parser.add_argument(
        "--stdout-only",
        action="store_true",
        help="Only log to stdout (no file logging)",
    )
    parser.add_argument(
        "--memory-per-build",
        type=float,
        default=MEMORY_PER_BUILD_GB,
        help=f"GB of RAM required per concurrent build (default: {MEMORY_PER_BUILD_GB})",
    )
    parser.add_argument(
        "--disk-per-build",
        type=float,
        default=DISK_PER_BUILD_GB,
        help=f"GB of disk required per concurrent build (default: {DISK_PER_BUILD_GB})",
    )
    args = parser.parse_args()
    logger.info(f"ubuntu_version: {args.ubuntu_version}")
    logger.info(f"build_metal_commit: {args.build_metal_commit}")
    logger.info(f"max_workers: {args.max_workers}")
    logger.info(f"force_build: {args.force_build}")
    logger.info(f"release: {args.release}")
    logger.info(f"multihost: {args.multihost}")
    logger.info(f"push: {args.push}")
    logger.info(f"single_threaded: {args.single_threaded}")
    logger.info(f"dry_run: {args.dry_run}")
    logger.info(f"dry_run_build_duration: {args.dry_run_build_duration}")
    logger.info(f"stdout_only: {args.stdout_only}")
    logger.info(f"memory_per_build: {args.memory_per_build}")
    logger.info(f"disk_per_build: {args.disk_per_build}")

    build_docker_images(
        MODEL_SPECS,
        force_build=args.force_build,
        release=args.release,
        multihost=args.multihost,
        push=args.push,
        ubuntu_version=args.ubuntu_version,
        build_metal_commit=args.build_metal_commit,
        max_workers=args.max_workers,
        single_threaded=args.single_threaded,
        dry_run=args.dry_run,
        stdout_only=args.stdout_only,
        memory_per_build_gb=args.memory_per_build,
        disk_per_build_gb=args.disk_per_build,
        dry_run_build_duration=args.dry_run_build_duration,
    )
