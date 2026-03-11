# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""Validation functions for multi-host deployment.

This module provides pre-flight validation checks for distributed vLLM
inference across multiple hosts. These checks ensure all hosts are properly
configured before starting any containers.

Validation Categories:
- SSH connectivity (required for all other checks)
- Docker availability
- Network interface for MPI
- Shared storage accessibility
- Tenstorrent device availability
- Hugepages configuration
- Bind mount permissions for shared storage
- System software versions (FW/KMD)
"""

import json
import logging
import os
import subprocess
from pathlib import Path

from workflows.validate_permissions import check_path_permissions_for_uid

logger = logging.getLogger("run_log")


# =============================================================================
# SSH Helper Function
# =============================================================================


def _run_ssh_command(
    host: str, command: list[str], timeout: int = 10
) -> tuple[bool, str]:
    """Run a command on a remote host via SSH.

    Args:
        host: Target hostname
        command: Command to run as list of strings
        timeout: Timeout in seconds

    Returns:
        Tuple of (success: bool, output: str)
    """
    ssh_cmd = ["ssh", "-p", "22", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host]
    ssh_cmd.extend(command)

    try:
        result = subprocess.run(
            ssh_cmd, capture_output=True, text=True, timeout=timeout
        )
        return result.returncode == 0, result.stdout.strip() or result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "SSH connection timed out"
    except Exception as e:
        return False, str(e)


# =============================================================================
# Bind Mount Permission Validation
# =============================================================================


def validate_multihost_bind_mount_permissions(
    multihost_config,
    container_uid: int = 1000,
) -> None:
    """Validate bind mount permissions for multi-host deployment.

    Checks that the container user (UID 1000) can access required paths:
    - SHARED_STORAGE_ROOT: read only (just needs to traverse)
    - CONFIG_PKL_DIR: read+write (vLLM writes pickle files here)
    - DEEPSEEK_V3_HF_MODEL: read (if configured via environment)
    - DEEPSEEK_V3_CACHE: read+write (if configured via environment)

    Unlike single-host validation, this does NOT attempt auto-fix because
    modifying permissions on shared storage could affect other users/hosts.

    Args:
        multihost_config: MultihostConfig object with paths to validate
        container_uid: UID that needs access (default: 1000 for container_app_user)

    Raises:
        ValueError: If any permission check fails, with actionable fix instructions
    """
    errors = []

    # SHARED_STORAGE_ROOT only needs read access (for traversal)
    shared_root = Path(multihost_config.shared_storage_root)
    ok, reason = check_path_permissions_for_uid(shared_root, container_uid, need_write=False)
    if not ok:
        errors.append(
            f"SHARED_STORAGE_ROOT={shared_root}: Container user (UID {container_uid}) needs read access.\n"
            f"  {reason}\n"
            f"  Fix: chmod o+rx {shared_root}"
        )
    else:
        logger.info(f"✅ Bind mount permission check passed for SHARED_STORAGE_ROOT={shared_root}")

    # CONFIG_PKL_DIR needs read+write access
    config_pkl = Path(multihost_config.config_pkl_dir)
    ok, reason = check_path_permissions_for_uid(config_pkl, container_uid, need_write=True)
    if not ok:
        errors.append(
            f"CONFIG_PKL_DIR={config_pkl}: Container user (UID {container_uid}) needs read+write access.\n"
            f"  {reason}\n"
            f"  Fix: sudo chown -R {container_uid}:{container_uid} {config_pkl}\n"
            f"       or: chmod -R o+rwx {config_pkl}"
        )
    else:
        logger.info(f"✅ Bind mount permission check passed for CONFIG_PKL_DIR={config_pkl}")

    # Optional paths from multihost_config (set via .env or interactive input)
    # Use getattr to handle both MultihostConfig variants
    deepseek_model = getattr(multihost_config, "deepseek_hf_model", None)
    if deepseek_model:
        path = Path(deepseek_model)
        ok, reason = check_path_permissions_for_uid(path, container_uid, need_write=False)
        if not ok:
            errors.append(
                f"DEEPSEEK_V3_HF_MODEL={path}: Container user (UID {container_uid}) needs read access.\n"
                f"  {reason}\n"
                f"  Fix: chmod -R o+rx {path}"
            )
        else:
            logger.info(f"✅ Bind mount permission check passed for DEEPSEEK_V3_HF_MODEL={path}")

    deepseek_cache = getattr(multihost_config, "deepseek_cache", None)
    if deepseek_cache:
        path = Path(deepseek_cache)
        if not path.exists():
            # Directory doesn't exist - will be created during deployment
            logger.warning(
                f"⚠️ DEEPSEEK_V3_CACHE={path} does not exist. "
                f"Directory will be created during deployment."
            )
        else:
            # If cache exists, only read access is needed (pre-built cache)
            ok, reason = check_path_permissions_for_uid(path, container_uid, need_write=False)
            if not ok:
                errors.append(
                    f"DEEPSEEK_V3_CACHE={path}: Container user (UID {container_uid}) needs read access.\n"
                    f"  {reason}\n"
                    f"  Fix: chmod -R o+rx {path}"
                )
            else:
                logger.info(f"✅ Bind mount permission check passed for DEEPSEEK_V3_CACHE={path}")

    if errors:
        raise ValueError(
            "\n⛔ Multi-host bind mount permission check failed:\n\n"
            + "\n\n".join(f"  - {e}" for e in errors)
            + "\n"
        )


# =============================================================================
# Remote Host Validation Functions
# =============================================================================


def validate_host_ssh_connectivity(hosts: list[str]) -> list[str]:
    """Validate SSH connectivity to all hosts.

    Args:
        hosts: List of hostnames to check

    Returns:
        List of error messages (empty if all checks pass)
    """
    errors = []
    for host in hosts:
        success, output = _run_ssh_command(host, ["echo", "SSH_OK"])
        if success and "SSH_OK" in output:
            logger.info(f"✅ SSH connectivity to {host}: OK")
        else:
            errors.append(f"Cannot SSH to {host}: {output}")
            logger.error(f"❌ SSH connectivity to {host}: FAILED - {output}")
    return errors


def validate_host_docker_availability(hosts: list[str]) -> list[str]:
    """Validate Docker is available and running on all hosts.

    Args:
        hosts: List of hostnames to check

    Returns:
        List of error messages (empty if all checks pass)
    """
    errors = []
    for host in hosts:
        success, output = _run_ssh_command(
            host, ["docker", "info", "--format", "{{.ServerVersion}}"]
        )
        if success and output:
            logger.info(f"✅ Docker on {host}: version {output}")
        else:
            errors.append(f"Docker not available on {host}: {output}")
            logger.error(f"❌ Docker on {host}: FAILED - {output}")
    return errors


def validate_host_network_interface(hosts: list[str], interface: str) -> list[str]:
    """Validate MPI network interface exists on all hosts.

    Args:
        hosts: List of hostnames to check
        interface: Network interface name (e.g., 'cnx1')

    Returns:
        List of error messages (empty if all checks pass)
    """
    errors = []
    for host in hosts:
        success, output = _run_ssh_command(host, ["ip", "link", "show", interface])
        if success:
            logger.info(f"✅ Network interface {interface} on {host}: OK")
        else:
            errors.append(f"Network interface {interface} not found on {host}")
            logger.error(f"❌ Network interface {interface} on {host}: FAILED")
    return errors


def validate_host_shared_directory(
    hosts: list[str], shared_storage_root: str
) -> list[str]:
    """Validate shared storage root is accessible on all hosts.

    Args:
        hosts: List of hostnames to check
        shared_storage_root: Path to shared storage root directory

    Returns:
        List of error messages (empty if all checks pass)
    """
    errors = []
    for host in hosts:
        # Check directory exists and is readable
        success, output = _run_ssh_command(
            host,
            [
                "test",
                "-d",
                shared_storage_root,
                "-a",
                "-r",
                shared_storage_root,
                "&&",
                "echo",
                "OK",
            ],
        )
        if success and "OK" in output:
            logger.info(f"✅ Shared storage root {shared_storage_root} on {host}: OK")
        else:
            errors.append(
                f"Shared storage root {shared_storage_root} not accessible on {host}"
            )
            logger.error(
                f"❌ Shared storage root {shared_storage_root} on {host}: FAILED"
            )
    return errors


def validate_host_tt_device(hosts: list[str]) -> list[str]:
    """Validate Tenstorrent device is available on all hosts.

    Args:
        hosts: List of hostnames to check

    Returns:
        List of error messages (empty if all checks pass)
    """
    errors = []
    for host in hosts:
        success, output = _run_ssh_command(
            host, ["test", "-e", "/dev/tenstorrent", "&&", "echo", "OK"]
        )
        if success and "OK" in output:
            logger.info(f"✅ TT device /dev/tenstorrent on {host}: OK")
        else:
            errors.append(f"TT device /dev/tenstorrent not found on {host}")
            logger.error(f"❌ TT device /dev/tenstorrent on {host}: FAILED")
    return errors


def validate_host_tt_smi(hosts: list[str], tt_smi_path: str = "tt-smi") -> list[str]:
    """Validate tt-smi command is available on all hosts.

    Args:
        hosts: List of hostnames to check
        tt_smi_path: Path to tt-smi binary on remote hosts

    Returns:
        List of error messages (empty if all checks pass)
    """
    errors = []
    for host in hosts:
        success, output = _run_ssh_command(host, ["which", tt_smi_path])
        if success and output:
            logger.info(f"✅ tt-smi on {host}: {output}")
        else:
            errors.append(f"tt-smi ({tt_smi_path}) not found on {host}")
            logger.error(f"❌ tt-smi on {host}: FAILED")
    return errors


def validate_host_hugepages(hosts: list[str]) -> list[str]:
    """Validate hugepages are available on all hosts.

    Args:
        hosts: List of hostnames to check

    Returns:
        List of error messages (empty if all checks pass)
    """
    errors = []
    for host in hosts:
        success, output = _run_ssh_command(
            host, ["test", "-d", "/dev/hugepages-1G", "&&", "echo", "OK"]
        )
        if success and "OK" in output:
            logger.info(f"✅ Hugepages /dev/hugepages-1G on {host}: OK")
        else:
            errors.append(f"Hugepages /dev/hugepages-1G not available on {host}")
            logger.error(f"❌ Hugepages /dev/hugepages-1G on {host}: FAILED")
    return errors


# =============================================================================
# System Software Validation
# =============================================================================


def validate_host_system_software(
    hosts: list[str],
    model_spec,
    tt_smi_path: str = "tt-smi",
) -> list[str]:
    """Validate FW/KMD versions on all hosts via SSH.

    Runs 'tt-smi -s' on each host and validates:
    - FW bundle versions match across all devices/hosts
    - KMD version meets model_spec requirements (if specified)

    Args:
        hosts: List of hostnames
        model_spec: ModelSpec with system_requirements (can be None)
        tt_smi_path: Path to tt-smi binary on remote hosts

    Returns:
        List of error messages (empty if all checks pass)
    """
    errors = []
    all_fw_versions = []
    all_kmd_versions = []
    host_fw_info = {}

    for host in hosts:
        # Run tt-smi -s via SSH and parse JSON
        success, output = _run_ssh_command(host, [tt_smi_path, "-s"], timeout=30)
        if not success:
            errors.append(f"Failed to run tt-smi on {host}: {output}")
            continue

        try:
            tt_smi_data = json.loads(output)

            # Extract FW bundle versions for all devices on this host
            fw_versions_on_host = []
            for info in tt_smi_data.get("device_info", []):
                fw_versions = info.get("firmwares", {})
                fw_bundle = fw_versions.get("fw_bundle_version")
                if fw_bundle:
                    fw_versions_on_host.append(fw_bundle)
                    all_fw_versions.append(fw_bundle)

            host_fw_info[host] = fw_versions_on_host

            # Extract KMD version
            kmd_version = tt_smi_data.get("host_info", {}).get("Driver", "")
            if kmd_version:
                # Format is "tenstorrent <version>" - extract version part
                parts = kmd_version.split(" ", 1)
                if len(parts) == 2:
                    all_kmd_versions.append((host, parts[1]))
                else:
                    all_kmd_versions.append((host, kmd_version))

            logger.info(f"✅ tt-smi on {host}: FW={fw_versions_on_host}, KMD={kmd_version}")

        except json.JSONDecodeError as e:
            errors.append(f"Failed to parse tt-smi JSON from {host}: {e}")
            continue

    # Validate all FW versions match across all hosts/devices
    unique_fw_versions = set(all_fw_versions)
    if len(unique_fw_versions) > 1:
        version_details = ", ".join(
            f"{host}: {versions}" for host, versions in host_fw_info.items()
        )
        errors.append(
            f"FW bundle versions mismatch across hosts. "
            f"All devices must have identical FW versions.\n"
            f"  Found versions: {version_details}"
        )
    elif unique_fw_versions:
        logger.info(f"✅ FW bundle version consistent across all hosts: {unique_fw_versions.pop()}")

    # Validate FW/KMD versions meet requirements (if model_spec has requirements)
    if model_spec and hasattr(model_spec, "system_requirements"):
        system_requirements = model_spec.system_requirements
        if system_requirements:
            try:
                from packaging.specifiers import SpecifierSet
                from packaging.version import Version

                # Check firmware requirements
                if hasattr(system_requirements, "firmware") and system_requirements.firmware:
                    fw_requirement = system_requirements.firmware
                    for fw_version in all_fw_versions:
                        try:
                            parsed_version = Version(fw_version)
                            specifier = SpecifierSet(fw_requirement.specifier)
                            if parsed_version not in specifier:
                                errors.append(
                                    f"FW bundle version ({fw_version}) does not meet "
                                    f"requirement {fw_requirement.specifier}"
                                )
                                break  # Only report once since all FW versions should match
                        except Exception as e:
                            logger.warning(f"Could not parse FW version {fw_version}: {e}")

                # Check KMD requirements
                if hasattr(system_requirements, "kmd") and system_requirements.kmd:
                    kmd_requirement = system_requirements.kmd
                    for host, kmd_version in all_kmd_versions:
                        try:
                            parsed_version = Version(kmd_version)
                            specifier = SpecifierSet(kmd_requirement.specifier)
                            if parsed_version not in specifier:
                                errors.append(
                                    f"KMD version on {host} ({kmd_version}) does not meet "
                                    f"requirement {kmd_requirement.specifier}"
                                )
                        except Exception as e:
                            logger.warning(f"Could not parse KMD version {kmd_version} on {host}: {e}")
            except ImportError:
                logger.warning("packaging module not available, skipping FW/KMD version validation")

    return errors


# =============================================================================
# Composite Validation Functions
# =============================================================================


def validate_multihost_environment(
    hosts: list[str],
    mpi_interface: str,
    shared_storage_root: str,
    model_spec=None,
    tt_smi_path: str = "tt-smi",
) -> None:
    """Run all pre-flight validation checks for multi-host deployment.

    This function validates that all hosts are properly configured for
    multi-host deployment before starting any containers.

    Args:
        hosts: List of hostnames
        mpi_interface: Network interface for MPI
        shared_storage_root: Shared storage root path
        model_spec: Optional ModelSpec for system software version validation
        tt_smi_path: Path to tt-smi binary on remote hosts

    Raises:
        ValueError: If any validation check fails
    """
    logger.info("=" * 60)
    logger.info("Multi-host Pre-flight Validation")
    logger.info("=" * 60)

    all_errors = []

    # 1. SSH connectivity (required for all other checks)
    logger.info("Checking SSH connectivity...")
    errors = validate_host_ssh_connectivity(hosts)
    if errors:
        # SSH is critical - fail fast
        raise ValueError(
            "SSH connectivity validation failed:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    # 2. Docker availability
    logger.info("Checking Docker availability...")
    all_errors.extend(validate_host_docker_availability(hosts))

    # 3. Network interface
    logger.info(f"Checking network interface {mpi_interface}...")
    all_errors.extend(validate_host_network_interface(hosts, mpi_interface))

    # 4. Shared storage root
    logger.info(f"Checking shared storage root {shared_storage_root}...")
    all_errors.extend(validate_host_shared_directory(hosts, shared_storage_root))

    # 5. TT device
    logger.info("Checking TT device availability...")
    all_errors.extend(validate_host_tt_device(hosts))

    # 6. Hugepages
    logger.info("Checking hugepages availability...")
    all_errors.extend(validate_host_hugepages(hosts))

    # 7. tt-smi availability
    logger.info("Checking tt-smi availability...")
    all_errors.extend(validate_host_tt_smi(hosts, tt_smi_path))

    # 8. System software validation
    # FW consistency check always runs, KMD requirement check needs model_spec
    logger.info("Checking system software versions (FW/KMD)...")
    all_errors.extend(validate_host_system_software(hosts, model_spec, tt_smi_path))

    logger.info("=" * 60)

    if all_errors:
        raise ValueError(
            "Multi-host environment validation failed:\n"
            + "\n".join(f"  - {e}" for e in all_errors)
        )

    logger.info("✅ All multi-host pre-flight checks passed")


def validate_multihost_args(
    multihost_config,
    model_spec=None,
    skip_environment_check: bool = False,
    tt_smi_path: str = "tt-smi",
    dry_run: bool = False,
) -> list[str]:
    """Validate multi-host configuration.

    Args:
        multihost_config: MultihostConfig object from setup_multihost_config()
        model_spec: Optional ModelSpec for system software version validation
        skip_environment_check: If True, skip remote host environment validation
        tt_smi_path: Path to tt-smi binary on remote hosts
        dry_run: If True, skip directory existence and permission checks
                 (for --print-docker-cmd mode where directories may not exist yet)

    Raises:
        ValueError: If configuration is invalid or environment check fails

    Returns:
        List of hostnames
    """
    hosts = multihost_config.hosts

    # Validate host count
    if len(hosts) < 2:
        raise ValueError("Multi-host deployment requires at least 2 hosts")
    if len(hosts) not in (2, 4):
        raise ValueError(
            f"Unsupported number of hosts: {len(hosts)}. Supported: 2 (Dual Galaxy), 4 (Quad Galaxy)"
        )

    # Validate shared storage root exists
    shared_root = Path(multihost_config.shared_storage_root)
    if not shared_root.exists():
        raise ValueError(f"Shared storage root not found: {shared_root}")

    # In dry_run mode, skip directory existence and permission checks
    # (used for --print-docker-cmd where auto-generated directories don't exist yet)
    if not dry_run:
        # Validate config_pkl_dir exists and is under shared_storage_root
        config_pkl_dir = Path(multihost_config.config_pkl_dir)
        if not config_pkl_dir.exists():
            raise ValueError(
                f"CONFIG_PKL_DIR does not exist: {config_pkl_dir}\n"
                f"  Create it with: mkdir -p {config_pkl_dir}"
            )

        # Validate bind mount permissions for container user (UID 1000)
        # This uses the shared permission checking utilities but does NOT auto-fix
        # (auto-fix is dangerous for shared storage as it could affect other users)
        validate_multihost_bind_mount_permissions(multihost_config)

    # Run comprehensive environment validation on all hosts
    if not skip_environment_check:
        validate_multihost_environment(
            hosts=hosts,
            mpi_interface=multihost_config.mpi_interface,
            shared_storage_root=multihost_config.shared_storage_root,
            model_spec=model_spec,
            tt_smi_path=tt_smi_path,
        )

    return hosts


# =============================================================================
# CLI Entry Point (Dry-Run)
# =============================================================================


def main():
    """CLI entry point for dry-run validation of multi-host configuration.

    Usage:
        python -m workflows.validate_multihost --hosts host1,host2 \\
            --shared-storage-root /mnt/shared \\
            --mpi-interface cnx1 \\
            --model DeepSeek-V3

    Note: --config-pkl-dir is optional. If not specified, a temporary directory
    will be auto-generated under shared-storage-root for validation purposes.
    """
    import argparse
    import shutil
    import sys

    from workflows.log_setup import setup_workflow_script_logger

    parser = argparse.ArgumentParser(
        description="Dry-run validation for multi-host deployment"
    )
    parser.add_argument(
        "--hosts",
        required=True,
        help="Comma-separated list of hostnames (e.g., 'host1,host2')",
    )
    parser.add_argument(
        "--shared-storage-root",
        required=True,
        help="Path to shared storage root directory",
    )
    parser.add_argument(
        "--config-pkl-dir",
        help="Path to config pickle directory (auto-generated if not specified)",
    )
    parser.add_argument(
        "--mpi-interface",
        default="cnx1",
        help="MPI network interface (default: cnx1)",
    )
    parser.add_argument(
        "--model",
        help="Model name for system software validation (e.g., 'DeepSeek-V3')",
    )
    parser.add_argument(
        "--tt-smi-path",
        default="tt-smi",
        help="Path to tt-smi binary on remote hosts (default: tt-smi)",
    )
    parser.add_argument(
        "--skip-ssh-checks",
        action="store_true",
        help="Skip SSH-based remote host checks (permissions only)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_workflow_script_logger(logger)

    # Build config
    from workflows.multihost_config import MultiHostConfig
    from workflows.multihost_orchestrator import (
        _create_config_pkl_dir_with_permissions,
        _generate_config_pkl_dir,
    )

    hosts = [h.strip() for h in args.hosts.split(",")]

    # Auto-generate config_pkl_dir if not specified
    auto_generated = False
    if args.config_pkl_dir:
        config_pkl_dir = args.config_pkl_dir
        # Validation only - don't create, just check
    else:
        config_pkl_dir = _generate_config_pkl_dir(args.shared_storage_root)
        auto_generated = True
        # Create temp directory for validation (will be cleaned up)
        _create_config_pkl_dir_with_permissions(config_pkl_dir)

    config = MultiHostConfig(
        hosts=hosts,
        mpi_interface=args.mpi_interface,
        shared_storage_root=args.shared_storage_root,
        config_pkl_dir=config_pkl_dir,
        _auto_generated_config_pkl_dir=auto_generated,
    )

    # Load model_spec if --model is provided
    model_spec = None
    if args.model:
        from workflows.model_spec import get_runtime_model_spec

        # Derive device type from number of hosts
        device_map = {2: "dual-galaxy", 4: "quad-galaxy"}
        if len(hosts) not in device_map:
            logger.error(f"Unsupported number of hosts: {len(hosts)}. Supported: 2 or 4")
            sys.exit(1)
        device = device_map[len(hosts)]

        try:
            model_spec, _, _ = get_runtime_model_spec(model=args.model, device=device)
            logger.info(f"Loaded model spec for {args.model} on {device}")
        except Exception as e:
            logger.error(f"Failed to load model spec for {args.model}: {e}")
            sys.exit(1)

    logger.info("=" * 60)
    logger.info("Multi-host Validation (Dry-Run)")
    logger.info("=" * 60)
    logger.info(f"Hosts: {hosts}")
    logger.info(f"Shared storage root: {args.shared_storage_root}")
    logger.info(f"Config pkl dir: {config_pkl_dir}" + (" (auto-generated)" if auto_generated else ""))
    logger.info(f"MPI interface: {args.mpi_interface}")
    if args.model:
        logger.info(f"Model: {args.model}")
    logger.info("=" * 60)

    try:
        validate_multihost_args(
            config,
            model_spec=model_spec,
            skip_environment_check=args.skip_ssh_checks,
            tt_smi_path=args.tt_smi_path,
        )
        logger.info("✅ All validation checks passed")
        exit_code = 0
    except ValueError as e:
        logger.error(f"❌ Validation failed: {e}")
        exit_code = 1
    finally:
        # Clean up auto-generated directory
        if auto_generated:
            config_pkl_path = Path(config_pkl_dir)
            if config_pkl_path.exists():
                shutil.rmtree(config_pkl_path, ignore_errors=True)
                # Also try to remove parent session directory if empty
                try:
                    session_dir = config_pkl_path.parent
                    if session_dir.exists() and not any(session_dir.iterdir()):
                        session_dir.rmdir()
                except OSError:
                    pass

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
