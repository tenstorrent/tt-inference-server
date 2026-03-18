# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations  # fix for older python versions
import logging
import signal
import sys
import os
from typing import Any
import subprocess
import json
import re
import argparse

logger = logging.getLogger(__name__)


class SystemResourceService:
    """Service for monitoring system resources and TT device telemetry"""

    @staticmethod
    def get_tt_smi_metal_data(timeout: int = 600) -> str | None:
        """Get raw tt-smi-metal data"""

        executable_path = os.path.join("/usr/local", "bin", "tt-smi-metal")

        stdout: str | None = SystemResourceService._run_command(
            [executable_path, "-s"], timeout
        )

        if stdout is None:
            return None

        try:
            # tt-smi-metal may output a prefix like "starting tt-smi" before the JSON
            # Find the start of the JSON object
            json_start = stdout.find("{")
            if json_start == -1:
                raise ValueError(
                    f"No JSON object found in {executable_path} stdout: {stdout}"
                )
            json_str = stdout[json_start:]
            data = json.loads(json_str)
            logger.info(f"Successfully parsed {executable_path} data")
            return data
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Failed to parse {executable_path} JSON output: {e}, stdout: {stdout}"
            )

    @staticmethod
    def get_tt_topology_data(timeout=10) -> str | None:
        """Get raw tt-topology data"""
        executable_path = os.path.join("/usr/local", "bin", "tt-topology")

        stdout: str | None = SystemResourceService._run_command(
            [executable_path, "-ls"], timeout
        )

        if stdout is None:
            return None

        # Parse output
        # remove ANSI colour, terminal control sequences
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        clean_stdout = ansi_escape.sub("", stdout)

        def parse_configuration(text: str) -> str | None:
            match = re.search(r"Configuration:\s*(.+)", text)
            if match:
                # Clean up any trailing whitespace or newlines
                return match.group(1).strip()
            return None

        topology_configuration = parse_configuration(clean_stdout)
        if topology_configuration is None:
            raise ValueError(f"Failed to parse tt-topology output: {stdout}")

        logger.info("Successfully parsed tt-topology data")
        return topology_configuration

    @staticmethod
    def _run_command(args: Any, timeout: int = 10):
        """
        Run a command and return stdout.

        Args:
            args: Command arguments to run
            timeout: Timeout in seconds for command execution

        Returns:
            Command stdout or None if command failed
        """
        try:
            logger.info(f"Running command: {' '.join(args)} to get device telemetry")

            process = subprocess.Popen(
                # [executable_path, "-s"],
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                text=True,
                preexec_fn=os.setsid,  # Create new process group
            )

            try:
                stdout, stderr = process.communicate(timeout=timeout)

                if process.returncode != 0:
                    logger.error(
                        f"Command {' '.join(args)} failed with return code {process.returncode}, stderr: {stderr}"
                    )
                    return None

                return stdout

            except subprocess.TimeoutExpired:
                logger.error(
                    f"Command {' '.join(args)} timed out after {timeout} seconds"
                )
                # Kill the process group to ensure cleanup
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    process.wait(timeout=2)
                except Exception as e:
                    logger.warning(
                        f"Failed to kill process {' '.join(args)}, error: {e}"
                    )
                    logger.warning("Trying with SIGKILL")
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    except Exception as e:
                        logger.warning(f"Failed, error: {e}")
                return None

        except FileNotFoundError:
            raise RuntimeError(
                f"Command {' '.join(args)} not found. Please ensure it is installed and in PATH."
            )
        except Exception as e:
            raise RuntimeError(f"Error getting command {' '.join(args)} data: {str(e)}")


def gather_args() -> Any:
    """Gather command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch Tenstorrent system information."
    )
    parser.add_argument(
        "-o",
        "--output-path",
        dest="output_path",
        default=None,
        help="Optional output path for saving system info.",
    )

    parser.add_argument(
        "--runner-label",
        dest="runner_label",
        default=None,
        help="Runner label.",
    )

    return parser.parse_args()


def fetch_system_info(
    output_path: str | None = None, runner_label: str | None = None
) -> dict:
    """
    Fetch system information using various commands.

    Args:
        output_path: Optional output path for saving system info.

    Returns:
        dict: Parsed device information

    """

    system_info_data = {}
    try:

        # Get tt-smi data
        try:
            tt_smi_data = SystemResourceService.get_tt_smi_metal_data(timeout=600)
            if tt_smi_data:
                system_info_data["host_info"] = tt_smi_data.get("host_info", {})
                system_info_data["host_sw_vers"] = tt_smi_data.get("host_sw_vers", {})

                # Extract per-device info
                devices = []
                for device in tt_smi_data.get("device_info", []):
                    devices.append(
                        {
                            "board_info": device.get("board_info", {}),
                            "firmwares": device.get("firmwares", {}),
                            "telemetry": device.get("telemetry", {}),
                        }
                    )
                system_info_data["devices"] = devices

        except Exception as e:
            logger.warning(f"WARNING: Failed to collect tt-smi data: {e}")

        # Get topology info
        try:
            topology = SystemResourceService.get_tt_topology_data(timeout=30)
            if topology:
                system_info_data["topology"] = (
                    topology.value if hasattr(topology, "value") else str(topology)
                )
        except Exception as e:
            logger.warning(f"WARNING: Failed to collect tt-topology data: {e}")

        # Add runner data
        if runner_label:
            system_info_data["runner"] = {
                "label": runner_label,
            }

    except Exception as e:
        logger.error(f"Failed to fetch system info: {e}")

    if output_path:
        save_system_info(system_info_data, output_path)

    return system_info_data


def save_system_info(system_info_data: dict, output_path: str) -> None:
    """Save system information to a file.

    Args:
        system_info_data: System information data to save
        output_path: Path to save the system information
    """
    with open(output_path, "w") as f:
        json.dump(system_info_data, f, indent=4)


def main() -> None:
    try:

        args = gather_args()

        output_path = args.output_path
        runner_label = args.runner_label

        system_info = fetch_system_info(output_path, runner_label)
        logger.info("System information fetched successfully.")
        logger.info("\nTenstorrent Device Information:")
        logger.info(system_info)

    except Exception as e:
        logger.error(f"ERROR: Unable to fetch system info: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
