# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# run_genai_benchmarks.py - Host-side script for launching genai-perf benchmarks
# This script runs on the host and launches a Docker container with the
# genai_benchmark.py script inside it.

import argparse
import json
import logging
import os
import sys
import uuid
from pathlib import Path

import jwt

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from workflows.log_setup import setup_workflow_script_logger
from workflows.model_spec import ModelSpec
from workflows.utils import get_repo_root_path, run_command
from workflows.workflow_types import WorkflowVenvType
from workflows.workflow_venvs import VENV_CONFIGS

logger = logging.getLogger(__name__)

# Default Docker image for genai-perf
DEFAULT_DOCKER_IMAGE = "nvcr.io/nvidia/tritonserver"
DEFAULT_RELEASE = "25.11"


def short_uuid():
    """Generate 8-character random UUID for container naming."""
    return str(uuid.uuid4())[:8]


def get_docker_image(release: str = None) -> str:
    """Get the Docker image name with release tag."""
    release = release or os.getenv("RELEASE", DEFAULT_RELEASE)
    return f"{DEFAULT_DOCKER_IMAGE}:{release}-py3-sdk"


def generate_auth_token(jwt_secret: str) -> str:
    """Generate AUTH_TOKEN from JWT_SECRET using the same logic as other benchmarks."""
    if not jwt_secret:
        return ""
    json_payload = json.loads('{"team_id": "tenstorrent", "token_id": "debug-test"}')
    encoded_jwt = jwt.encode(json_payload, jwt_secret, algorithm="HS256")
    return encoded_jwt


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run genai-perf benchmarks via Docker")
    parser.add_argument(
        "--model-spec-json",
        type=str,
        help="Use model specification from JSON file",
        required=True,
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path for benchmark output",
        required=True,
    )
    parser.add_argument(
        "--jwt-secret",
        type=str,
        help="JWT secret for generating token to set AUTH_TOKEN",
        default=os.getenv("JWT_SECRET", ""),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (2 benchmarks only, verbose logging)",
    )
    return parser.parse_args()


def run_genai_benchmarks(
    model_spec: ModelSpec,
    output_path: str,
    jwt_secret: str,
    service_port: str = "8000",
    debug: bool = False,
    raw_output: bool = False,
) -> int:
    """
    Run genai-perf benchmarks using Docker container.

    Args:
        model_spec: Model specification with configuration
        output_path: Path to store benchmark results
        jwt_secret: JWT secret for authentication
        service_port: Service port for the inference server
        debug: If True, run in debug mode with verbose logging
        raw_output: If True, print and save original genai-perf JSON output

    Returns:
        Return code (0 for success, non-zero for failure)
    """
    logger.info("Starting genai-perf benchmarks...")

    # Get venv config for artifacts directory
    venv_config = VENV_CONFIGS[WorkflowVenvType.BENCHMARKS_GENAI_PERF]
    artifacts_dir = venv_config.venv_path / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Get paths
    repo_root = get_repo_root_path()
    genai_script = repo_root / "benchmarking" / "genai_benchmark.py"

    # Get benchmarks output directory for direct file output
    from workflows.workflow_config import get_default_workflow_root_log_dir

    benchmarks_output_dir = get_default_workflow_root_log_dir() / "benchmarks_output"
    benchmarks_output_dir.mkdir(parents=True, exist_ok=True)

    # Generate AUTH_TOKEN from JWT_SECRET
    auth_token = generate_auth_token(jwt_secret)
    if not auth_token:
        logger.warning("No JWT_SECRET provided, AUTH_TOKEN will be empty")

    # Get Docker image
    release = os.getenv("RELEASE", DEFAULT_RELEASE)
    docker_image = get_docker_image(release)
    logger.info(f"Using Docker image: {docker_image}")

    # Generate unique container name for tracking and management
    container_name = f"genai-tritonserver-{short_uuid()}"
    logger.info(f"Container name: {container_name}")

    # Get current user UID:GID for proper file permissions
    uid = os.getuid()
    gid = os.getgid()
    user_spec = f"{uid}:{gid}"

    # Get host HF cache directory (reuse system's HF cache instead of creating new one)
    from workflows.utils import get_default_hf_home_path

    host_hf_cache = get_default_hf_home_path()
    logger.info(f"Using host HF cache: {host_hf_cache}")

    # Get model configuration
    device_model_spec = model_spec.device_model_spec
    hf_model_repo = model_spec.hf_model_repo
    max_context = device_model_spec.max_context
    max_concurrency = device_model_spec.max_concurrency

    # Prepare config JSON for the container
    config = {
        "model_name": hf_model_repo,
        "model_id": model_spec.model_id,
        "tokenizer": "hf-internal-testing/llama-tokenizer",
        "url": f"localhost:{service_port}",
        "max_context": max_context,
        "model_max_concurrency": max_concurrency,
        "auth_token": auth_token,
        "artifact_base": "/workspace/artifacts",
    }

    # Write config to temp file
    config_path = artifacts_dir / "genai_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Config written to: {config_path}")

    # Build Docker command
    # fmt: off
    cmd = [
        "docker", "run", "--rm", "--net", "host",
        "--name", container_name,
        "--user", user_spec,  # Run as host user for proper file permissions
        "-v", f"{genai_script}:/workspace/genai_benchmark.py:ro",
        "-v", f"{artifacts_dir}:/workspace/artifacts",
        "-v", f"{benchmarks_output_dir}:/workspace/benchmarks_output",  # Direct output mount
        "-v", f"{config_path}:/workspace/config.json:ro",
        "-v", f"{host_hf_cache}:/workspace/.cache/huggingface",  # Mount host HF cache (reuse system cache)
        "-e", f"AUTH_TOKEN={auth_token}",
        "-e", f"MODEL_NAME={hf_model_repo}",
        "-e", f"URL=localhost:{service_port}",
        "-e", f"MAX_CONTEXT={max_context}",
        "-e", f"MODEL_MAX_CONCURRENCY={max_concurrency}",
        "-e", "BENCHMARKS_OUTPUT_DIR=/workspace/benchmarks_output",
        "-e", "PYTHONUNBUFFERED=1",  # Force unbuffered output for real-time logs
        "-e", "HF_HOME=/workspace/.cache/huggingface",  # Point to mounted HF cache
        "-e", "TRANSFORMERS_CACHE=/workspace/.cache/huggingface",  # Point to mounted HF cache
        docker_image,
        "python", "/workspace/genai_benchmark.py",
        "--config-json", "/workspace/config.json",
    ]
    # fmt: on

    if debug:
        cmd.append("--debug")
    if raw_output:
        cmd.append("--raw-output")

    logger.info(f"Running Docker command: {' '.join(cmd)}")

    # Run the Docker container
    return_code = run_command(cmd, logger=logger)

    # Results are already in benchmarks_output_dir via direct mount
    # Individual result files saved per benchmark run

    if return_code == 0:
        logger.info("[OK] genai-perf benchmarks completed successfully")
        logger.info(f"Individual results saved to: {benchmarks_output_dir}")
    else:
        logger.error(
            f"[FAIL] genai-perf benchmarks failed with return code: {return_code}"
        )

    return return_code


def main():
    """Main entry point for the script."""
    setup_workflow_script_logger(logger)
    logger.info(f"Running {__file__} ...")

    args = parse_args()
    model_spec = ModelSpec.from_json(args.model_spec_json)

    # Extract CLI args from model_spec
    cli_args = model_spec.cli_args
    service_port = cli_args.get("service_port", os.getenv("SERVICE_PORT", "8000"))

    logger.info(f"Model: {model_spec.model_name}")
    logger.info(f"Device: {model_spec.device_type}")
    logger.info(f"Service Port: {service_port}")
    logger.info(f"Output Path: {args.output_path}")

    return_code = run_genai_benchmarks(
        model_spec=model_spec,
        output_path=args.output_path,
        jwt_secret=args.jwt_secret,
        service_port=service_port,
        debug=args.debug,
    )

    if return_code == 0:
        logger.info("[OK] Completed genai-perf benchmarks")
    else:
        logger.error(
            f"[FAIL] genai-perf benchmarks failed with return code: {return_code}"
        )

    return return_code


if __name__ == "__main__":
    sys.exit(main())
