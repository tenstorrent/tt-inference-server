# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import argparse
import subprocess
import json
import logging
import os
from packaging.specifiers import SpecifierSet
from packaging.version import Version
from pathlib import Path
import re
import signal
import sys

logger = logging.getLogger(__name__)

# Add the script's directory to the Python path
# this for 0 setup python setup script
project_root = Path(__file__).resolve().parent.parent
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))
from workflows.log_setup import setup_workflow_script_logger
from workflows.model_spec import ModelSpec, VersionRequirement
from workflows.workflow_types import SystemTopology, VersionMode


class SystemResourceService:
    """Service for monitoring system resources and TT device telemetry"""

    @staticmethod
    def get_tt_smi_data(timeout=10):
        """Get raw tt-smi data with timeout handling"""
        import sys

        tt_smi_executable = os.path.join(sys.prefix, "bin", "tt-smi")
        try:
            logger.info("Running tt-smi -s to get device telemetry")

            process = subprocess.Popen(
                [tt_smi_executable, "-s"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                text=True,
                preexec_fn=os.setsid,  # Create new process group
            )

            try:
                # Wait for process with timeout
                stdout, stderr = process.communicate(timeout=timeout)

                if process.returncode != 0:
                    logger.error(
                        f"tt-smi -s failed with return code {process.returncode}, stderr: {stderr}"
                    )
                    return None

                # Parse JSON output
                try:
                    data = json.loads(stdout)
                    logger.info("Successfully parsed tt-smi data")
                    return data
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Failed to parse tt-smi JSON output: {e}, stdout: {stdout}"
                    )

            except subprocess.TimeoutExpired:
                logger.error(f"tt-smi -s command timed out after {timeout} seconds")
                # Kill the process group to ensure cleanup
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    process.wait(timeout=2)
                except Exception:
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    except Exception:
                        pass
                return None

        except FileNotFoundError:
            raise RuntimeError("tt-smi command not found")
        except Exception as e:
            raise RuntimeError(f"Error getting tt-smi data: {str(e)}")

    @staticmethod
    def get_tt_topology_data(timeout=10):
        """Get raw tt-topology data with timeout handling"""
        import sys

        tt_topology_executable = os.path.join(sys.prefix, "bin", "tt-topology")
        try:
            logger.info("Running tt-topology -ls to get system topology")

            process = subprocess.Popen(
                [tt_topology_executable, "-ls"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                text=True,
                preexec_fn=os.setsid,  # Create new process group
            )

            try:
                # Wait for process with timeout
                stdout, stderr = process.communicate(timeout=timeout)

                if process.returncode != 0:
                    logger.error(
                        f"tt-topology -ls failed with return code {process.returncode}, stderr: {stderr}"
                    )
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

            except subprocess.TimeoutExpired:
                logger.error(
                    f"tt-topology -ls command timed out after {timeout} seconds"
                )
                # Kill the process group to ensure cleanup
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    process.wait(timeout=2)
                except Exception:
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    except Exception:
                        pass
                return None

        except FileNotFoundError:
            raise RuntimeError("tt-topology command not found")
        except Exception as e:
            raise RuntimeError(f"Error getting tt-topology data: {str(e)}")

    @classmethod
    def get_system_topology(cls, timeout=10):
        """Parse tt-topology data and enumerate the system-level topology"""
        # enumerate topology configuration
        topology_configuration = cls.get_tt_topology_data(timeout)
        topology = SystemTopology.from_topology_string(topology_configuration)
        return topology


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run FW & KMD Version checking against a ModelSpec"
    )
    parser.add_argument(
        "--model-spec-json",
        type=str,
        help="Use model specification from JSON file",
        required=True,
    )
    ret_args = parser.parse_args()
    return ret_args


def main():
    # Setup logging configuration.
    setup_workflow_script_logger(logger)
    logger.info(f"Running {__file__} ...")
    args = parse_args()

    # create ModelSpec
    model_spec = ModelSpec.from_json(args.model_spec_json)

    # parse tt-smi FW & KMD version data
    tt_smi_data = SystemResourceService.get_tt_smi_data()
    fw_bundle_versions = []
    for info in tt_smi_data["device_info"]:
        fw_versions = info["firmwares"]
        fw_bundle_versions.append(fw_versions["fw_bundle_version"])

    # parse system topology
    topology = SystemResourceService.get_system_topology()

    # enforce matching FW bundle versions across all devices
    cli_args = model_spec.cli_args
    if (device_ids := cli_args.get("device_id")) is not None:
        fw_bundle_versions = [fw_bundle_versions[i] for i in device_ids]
    if len(set(fw_bundle_versions)) != 1:
        raise ValueError(
            f"FW bundle versions must match among all devices: {fw_bundle_versions}"
        )

    kmd_version = tt_smi_data["host_info"]["Driver"]
    prefix, kmd_version = kmd_version.split(" ")
    system_info = (
        f"System info:\n"
        f"{'='*80}\n"
        f"FW bundle versions (across all devices): {fw_bundle_versions}\n"
        f"KMD version (on host): {kmd_version}\n"
        f"Topology: {topology}\n"
        f"{'='*80}"
    )
    logger.info(system_info)

    # enforce ModelSpec system requirements
    system_requirements = model_spec.system_requirements
    if system_requirements is not None:

        def _enforce_requirement(
            requirement_name: str, version: str, version_requirement: VersionRequirement
        ) -> None:
            version_specifier = version_requirement.specifier
            enforcement_mode = version_requirement.mode
            meets_requirement = Version(version) in SpecifierSet(version_specifier)
            if meets_requirement:
                return
            message = f"{requirement_name} version '{version}' does not satisfy the requirement '{version_specifier}'."
            if enforcement_mode == VersionMode.STRICT:
                raise ValueError(message)
            elif enforcement_mode == VersionMode.SUGGESTED:
                logger.warning(message)

        # enforce firmware versioning
        # by default, ModelSpecs have no FW requirement
        firmware_requirement = model_spec.system_requirements.firmware
        if firmware_requirement is not None:
            for fw_bundle_version in fw_bundle_versions:
                _enforce_requirement(
                    "FW bundle", fw_bundle_version, firmware_requirement
                )

        # enforce KMD versioning
        # by default, ModelSpecs have no KMD requirement
        kmd_requirement = model_spec.system_requirements.kmd
        if kmd_requirement is not None:
            _enforce_requirement("KMD", kmd_version, kmd_requirement)

    # enforce system-level topology
    device = model_spec.device_model_spec.device
    topology_requirement = device.get_topology_requirement()
    if topology_requirement is None:
        return
    if topology != topology_requirement:
        raise ValueError(
            f"ModelSpec requires a system-level topology of {topology_requirement}, detected {topology}"
        )


if __name__ == "__main__":
    main()
