#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import argparse
import subprocess
import logging
import sys
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

project_root = Path(__file__).resolve().parent.parent
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))
from workflows.log_setup import setup_workflow_script_logger
from workflows.model_spec import get_model_id
from workflows.utils import get_default_workflow_root_log_dir, ensure_readwriteable_dir


def find_container_by_port(service_port):
    """
    Find Docker container ID by service port mapping.

    Args:
        service_port: Port number the container is listening on

    Returns:
        Container ID if found, None otherwise
    """
    logger.info(f"Searching for Docker container with port {service_port}")

    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.ID}}\t{{.Ports}}"],
            capture_output=True,
            text=True,
            check=True,
        )

        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            container_id, ports = line.split("\t", 1)
            if f":{service_port}->" in ports or f"->{service_port}/" in ports:
                logger.info(f"Found container {container_id} with port {service_port}")
                return container_id

        logger.error(f"No container found with port {service_port}")
        return None

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to list Docker containers: {e}")
        return None


def run_command_in_container(container_id, command):
    """
    Execute a command in the Docker container.

    Args:
        container_id: Docker container ID
        command: Command to execute

    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    logger.info(f"Executing command in container {container_id}: {command}")

    try:
        result = subprocess.run(
            ["docker", "exec", container_id, "bash", "-c", command],
            capture_output=True,
            text=True,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to execute command: {e}")
        return (
            e.returncode,
            e.stdout if hasattr(e, "stdout") else "",
            e.stderr if hasattr(e, "stderr") else "",
        )


def run_debugger_in_container(container_id, output_file):
    """
    Run the debugging sequence in the Docker container.

    Args:
        container_id: Docker container ID
        output_file: Path to save triage output

    Returns:
        True if successful, False otherwise
    """
    logger.info("Detecting tt-triage directory location")

    # Detect which directory structure exists in the container
    # Old commit: /home/container_app_user/tt-metal/scripts/debugging_scripts
    # New commit: /home/container_app_user/tt-metal/tools/triage
    check_new_path_cmd = "test -d /home/container_app_user/tt-metal/tools/triage && echo 'new' || echo 'old'"
    return_code, stdout, stderr = run_command_in_container(
        container_id, check_new_path_cmd
    )

    if return_code != 0:
        logger.error(f"Failed to detect tt-triage directory: {stderr}")
        return False

    uses_new_path = stdout.strip() == "new"

    if uses_new_path:
        triage_dir = "/home/container_app_user/tt-metal/tools/triage"
        requirements_path = (
            "/home/container_app_user/tt-metal/tools/triage/requirements.txt"
        )
        logger.info("Using new tt-metal structure: tools/triage")
    else:
        triage_dir = "/home/container_app_user/tt-metal/scripts/debugging_scripts"
        requirements_path = "/home/container_app_user/tt-metal/scripts/debugging_scripts/requirements.txt"
        logger.info("Using old tt-metal structure: scripts/debugging_scripts")

    logger.info("Building chained command sequence")

    # Chain all commands together with && to run in a single shell session
    # This ensures environment changes and directory changes persist
    chained_command = f"""
cd {triage_dir} && \
source /home/container_app_user/tt-metal/python_env/bin/activate && \
unset LD_LIBRARY_PATH && \
echo "=== Environment configured ===" && \
/home/container_app_user/tt-metal/scripts/install_debugger.sh && \
echo "=== Debugger tools installed ===" && \
pip install -r {requirements_path} && \
echo "=== Python dependencies installed ===" && \
python triage.py
"""

    combined_output = []
    combined_output.append(f"{'=' * 80}")
    combined_output.append("TT-Metal Debugger Triage Report")
    combined_output.append(f"Container ID: {container_id}")
    combined_output.append(f"Timestamp: {datetime.now().isoformat()}")
    combined_output.append(f"{'=' * 80}\n")

    logger.info("Executing chained command in container")
    combined_output.append(f"\n{'=' * 80}")
    combined_output.append("Executing debugging sequence")
    combined_output.append(f"{'=' * 80}\n")

    return_code, stdout, stderr = run_command_in_container(
        container_id, chained_command
    )

    if stdout:
        combined_output.append("STDOUT:")
        combined_output.append(stdout)

    if stderr:
        combined_output.append("\nSTDERR:")
        combined_output.append(stderr)

    combined_output.append(f"\nReturn code: {return_code}\n")

    if return_code != 0:
        logger.warning(f"Command sequence returned non-zero exit code: {return_code}")

    output_content = "\n".join(combined_output)

    try:
        with open(output_file, "w") as f:
            f.write(output_content)
        logger.info(f"✅ Debugger output saved to: {output_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to write output file: {e}")
        return False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run TT-Metal debugger triage in a Docker container identified by service port"
    )
    parser.add_argument(
        "--service-port",
        type=int,
        required=True,
        help="Service port number to identify the Docker container",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--device", type=str, required=True, help="Device type (e.g., n300, n150)"
    )
    parser.add_argument(
        "--impl",
        type=str,
        default="tt-transformers",
        help="Implementation name (default: tt-transformers)",
    )
    return parser.parse_args()


def main():
    setup_workflow_script_logger(logger)
    logger.info(f"Running {__file__} ...")

    args = parse_args()

    # Generate model spec ID from model, device, and impl
    model_spec_id = get_model_id(args.impl, args.model, args.device)
    logger.info(f"Model spec ID: {model_spec_id}")

    # Create triage output directory
    workflow_root_log_dir = get_default_workflow_root_log_dir()
    triage_output_dir = workflow_root_log_dir / "triage_output"
    ensure_readwriteable_dir(triage_output_dir)
    logger.info(f"Triage output directory: {triage_output_dir}")

    # Generate output filename with model spec ID and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"triage_{model_spec_id}_{timestamp}.txt"
    output_file = triage_output_dir / output_filename
    logger.info(f"Output file: {output_file}")

    container_id = find_container_by_port(args.service_port)
    if container_id is None:
        logger.error(
            f"⛔ Could not find container with service port {args.service_port}"
        )
        return 1

    success = run_debugger_in_container(container_id, output_file)

    if success:
        logger.info("✅ Debugger triage completed successfully")
        return 0
    else:
        logger.error("⛔ Debugger triage failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
