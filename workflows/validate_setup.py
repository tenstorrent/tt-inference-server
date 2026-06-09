# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import logging
import os
import stat
from pathlib import Path

from benchmarking.benchmark_config import BENCHMARK_CONFIGS
from evals.eval_config import EVAL_CONFIGS
from server_tests.test_config import TEST_CONFIGS
from workflows.model_spec import MODEL_SPECS
from workflows.utils import (
    MIN_SUPPORTED_IMAGE_VERSION,
    check_path_permissions_for_uid,
    ensure_readwriteable_dir,
    get_default_workflow_root_log_dir,
    get_groups_for_uid,
    get_repo_root_path,
    parse_version_tuple,
    resolve_hf_snapshot_dir,
    run_command,
    user_error,
)
from workflows.workflow_types import (
    DeviceTypes,
    InferenceEngine,
    WorkflowType,
    WorkflowVenvType,
)
from workflows.workflow_venvs import VENV_CONFIGS

logger = logging.getLogger("run_log")


def _check_image_version_supported(model_spec):
    """Refuse to run a pre-0.11 vLLM image with this run.py.

    The vLLM docker image interface was reshaped in v0.11.0 (commit 50db8ac7
    "Simplify and improve vLLM Docker image interface"): ENTRYPOINT changed
    from docker-entrypoint.sh + gosu to bash -c, the script's CLI argument
    contract changed, and shared-memory + env-var conventions changed. main
    only emits the new contract, so an older vLLM image won't start.

    Scoped to vLLM only — media-inference-server and forge images have
    different Dockerfiles and aren't affected by this interface change
    (the docker command for them is also simpler and stable across versions).

    apply_overrides re-parses model_spec.version from --override-docker-image
    when present, so this check covers both template-pinned versions and
    override paths.
    """
    if model_spec.inference_engine != InferenceEngine.VLLM.value:
        return
    parsed = parse_version_tuple(model_spec.version)
    if parsed is None:
        # Unparseable versions (`dev`, `latest`, etc.) default to "newest
        # contract" — let the runtime decide, matches main's behaviour.
        return
    if parsed < MIN_SUPPORTED_IMAGE_VERSION:
        min_str = ".".join(str(p) for p in MIN_SUPPORTED_IMAGE_VERSION)
        tag = f"v{model_spec.version}"
        raise RuntimeError(
            f"ERROR: Docker image v{model_spec.version} is not supported by this version of run.py.\n"
            f"This run.py requires vLLM image v{min_str} or newer (the Docker interface changed in v{min_str}).\n"
            f"\nTo fix this:\n"
            f"  1. Check out the matching release tag: git checkout {tag}\n"
            f"  2. Re-run this script\n"
            f"\nIf you need help, see https://docs.tenstorrent.com/getting-started/README.html#before-you-begin"
        )


def validate_runtime_args(model_spec, runtime_config):
    args = runtime_config
    workflow_type = WorkflowType.from_string(args.workflow)

    if not args.device:
        # TODO: detect phy device
        user_error(
            "ERROR: No device specified and automatic device detection is not yet implemented.\n"
            "You must tell run.py which Tenstorrent device to target.\n"
            "\nTo fix this:\n"
            "  1. Add --tt-device <device> to your command (e.g. --tt-device n150)\n"
            "  2. Re-run this script\n"
            "\nIf you need help, see https://docs.tenstorrent.com/getting-started/README.html#before-you-begin"
        )

    model_id = model_spec.model_id

    # Check if the model_id exists in MODEL_SPECS (this validates device support)
    if model_id not in MODEL_SPECS:
        user_error(
            f"ERROR: Model '{runtime_config.model}' does not support device '{runtime_config.device}'.\n"
            "This combination is not listed in the supported model specs.\n"
            "\nTo fix this:\n"
            "  1. Run: python run.py --help  to see the list of supported --model / --tt-device pairs\n"
            "  2. Choose a supported combination and re-run this script\n"
            "\nIf you need help, see https://docs.tenstorrent.com/getting-started/README.html#before-you-begin"
        )

    _check_image_version_supported(model_spec)

    if args.docker_server and args.local_server:
        user_error(
            "ERROR: --docker-server and --local-server cannot be used at the same time.\n"
            "These flags are mutually exclusive — pick one mode.\n"
            "\nTo fix this:\n"
            "  1. Use --docker-server to run inside a Docker container, OR\n"
            "  2. Use --local-server to run directly on the host\n"
            "  3. Re-run this script with only one of those flags\n"
            "\nIf you need help, see https://docs.tenstorrent.com/getting-started/README.html#before-you-begin"
        )

    if workflow_type == WorkflowType.EVALS:
        if model_spec.model_name not in EVAL_CONFIGS:
            user_error(
                f"ERROR: Model '{model_spec.model_name}' does not have an eval configuration.\n"
                "The --workflow evals mode requires a config entry in EVAL_CONFIGS.\n"
                "\nTo fix this:\n"
                "  1. Choose a model that supports evals (run python run.py --help for the list)\n"
                "  2. Or use a different --workflow\n"
                "\nIf you need help, see https://docs.tenstorrent.com/getting-started/README.html#before-you-begin"
            )
    if workflow_type == WorkflowType.BENCHMARKS:
        if os.getenv("OVERRIDE_BENCHMARKS"):
            logger.warning("OVERRIDE_BENCHMARKS is active, using override benchmarks")
        if model_spec.model_id not in BENCHMARK_CONFIGS:
            user_error(
                f"ERROR: Model '{model_spec.model_name}' does not have a benchmark configuration.\n"
                "The --workflow benchmarks mode requires a config entry in BENCHMARK_CONFIGS.\n"
                "\nTo fix this:\n"
                "  1. Choose a model that supports benchmarks (run python run.py --help for the list)\n"
                "  2. Or use a different --workflow\n"
                "\nIf you need help, see https://docs.tenstorrent.com/getting-started/README.html#before-you-begin"
            )
    if workflow_type == WorkflowType.STRESS_TESTS:
        pass  # Model support already validated via MODEL_SPECS check

    if workflow_type == WorkflowType.TESTS:
        if model_spec.model_name not in TEST_CONFIGS:
            user_error(
                f"ERROR: Model '{model_spec.model_name}' does not have a test configuration.\n"
                "The --workflow tests mode requires a config entry in TEST_CONFIGS.\n"
                "\nTo fix this:\n"
                "  1. Choose a model that supports tests (run python run.py --help for the list)\n"
                "  2. Or use a different --workflow\n"
                "\nIf you need help, see https://docs.tenstorrent.com/getting-started/README.html#before-you-begin"
            )
    if workflow_type == WorkflowType.REPORTS:
        pass
    if workflow_type == WorkflowType.SERVER:
        if not (args.docker_server or args.local_server):
            user_error(
                f"ERROR: The '{args.workflow}' workflow requires a server mode flag.\n"
                "You must tell run.py where to run the inference server.\n"
                "\nTo fix this:\n"
                "  1. Add --docker-server to run inside a Docker container, OR\n"
                "  2. Add --local-server to run directly on the host\n"
                "  3. Re-run this script\n"
                "\nIf you need help, see https://docs.tenstorrent.com/getting-started/README.html#before-you-begin"
            )
        if (
            args.local_server
            and model_spec.inference_engine != InferenceEngine.VLLM.value
        ):
            user_error(
                f"ERROR: --local-server only supports vLLM-backed models, but '{model_spec.model_name}' uses engine '{model_spec.inference_engine}'.\n"
                "Local server mode is not yet implemented for other inference engines.\n"
                "\nTo fix this:\n"
                "  1. Use --docker-server instead, OR\n"
                "  2. Choose a vLLM-backed model\n"
                "\nIf you need help, see https://docs.tenstorrent.com/getting-started/README.html#before-you-begin"
            )

        # For partitioning Galaxy per tray as T3K
        # TODO: Add a check to verify whether these devices belong to the same tray
        if DeviceTypes.from_string(args.device) == DeviceTypes.GALAXY_T3K:
            if not args.device_id or len(args.device_id) != 8:
                user_error(
                    "ERROR: Galaxy T3K requires exactly 8 device IDs but the wrong number was specified.\n"
                    "Each T3K tray contains 8 Tenstorrent devices that must all be listed.\n"
                    "\nTo fix this:\n"
                    "  1. Run: tt-smi  to list the PCI device indices available on this host\n"
                    "  2. Add --device-id 0,1,2,3,4,5,6,7  (using the 8 indices from the same tray)\n"
                    "  3. Re-run this script\n"
                    "\nIf you need help, see https://docs.tenstorrent.com/getting-started/README.html#before-you-begin"
                )

    if workflow_type == WorkflowType.RELEASE:
        # NOTE: fail fast for models without both defined evals and benchmarks
        # today this will stop models defined in MODEL_SPECS
        # but not in EVAL_CONFIGS or BENCHMARK_CONFIGS, e.g. non-instruct models
        # a run_*.log fill will be made for the failed combination indicating this
        if model_spec.model_name not in EVAL_CONFIGS:
            user_error(
                f"ERROR: Model '{model_spec.model_name}' is missing an eval configuration required by the release workflow.\n"
                "The release workflow requires both EVAL_CONFIGS and BENCHMARK_CONFIGS entries.\n"
                "\nTo fix this:\n"
                "  1. Use a model that has a complete release configuration\n"
                "  2. Or use a different --workflow (e.g. --workflow server)\n"
                "\nIf you need help, see https://docs.tenstorrent.com/getting-started/README.html#before-you-begin"
            )
        if model_spec.model_id not in BENCHMARK_CONFIGS:
            user_error(
                f"ERROR: Model '{model_spec.model_name}' is missing a benchmark configuration required by the release workflow.\n"
                "The release workflow requires both EVAL_CONFIGS and BENCHMARK_CONFIGS entries.\n"
                "\nTo fix this:\n"
                "  1. Use a model that has a complete release configuration\n"
                "  2. Or use a different --workflow (e.g. --workflow server)\n"
                "\nIf you need help, see https://docs.tenstorrent.com/getting-started/README.html#before-you-begin"
            )

    if DeviceTypes.from_string(args.device) == DeviceTypes.GPU:
        if args.docker_server or args.local_server:
            user_error(
                "ERROR: Running the inference server on GPU is not yet implemented.\n"
                "GPU device support for --docker-server and --local-server is a planned feature.\n"
                "\nTo fix this:\n"
                "  1. Use a Tenstorrent device (e.g. --tt-device n150) instead of GPU\n"
                "  2. Or remove --docker-server / --local-server to run workflows without a server\n"
                "\nIf you need help, see https://docs.tenstorrent.com/getting-started/README.html#before-you-begin"
            )

    if args.local_server and not args.tt_metal_home:
        user_error(
            "ERROR: --local-server requires a tt-metal build directory but none was specified.\n"
            "The local server needs the compiled tt-metal libraries to run.\n"
            "\nTo fix this:\n"
            "  1. Set the environment variable: export TT_METAL_HOME=/path/to/tt-metal\n"
            "  2. Or pass the flag: --tt-metal-home /path/to/tt-metal\n"
            "  3. Re-run this script\n"
            "\nIf you need help, see https://docs.tenstorrent.com/getting-started/README.html#before-you-begin"
        )

    # Validate mutual exclusivity of weight source options
    weight_source_args = [
        args.host_volume,
        args.host_hf_cache,
        getattr(args, "host_weights_dir", None),
    ]
    if sum(1 for a in weight_source_args if a) > 1:
        user_error(
            "ERROR: More than one weight source flag was specified.\n"
            "Only one of --host-volume, --host-hf-cache, or --host-weights-dir may be used at a time.\n"
            "\nTo fix this:\n"
            "  1. Choose one weight source:\n"
            "     --host-volume       for a Docker named volume\n"
            "     --host-hf-cache     to reuse an existing HuggingFace cache directory\n"
            "     --host-weights-dir  to point at a pre-downloaded weights directory\n"
            "  2. Remove the other flags and re-run this script\n"
            "\nIf you need help, see https://docs.tenstorrent.com/getting-started/README.html#before-you-begin"
        )

    if "ENABLE_AUTO_TOOL_CHOICE" in os.environ:
        user_error(
            "ERROR: The ENABLE_AUTO_TOOL_CHOICE environment variable is no longer supported.\n"
            "This setting was deprecated and has been replaced by --vllm-override-args.\n"
            "\nTo fix this:\n"
            "  1. Unset the variable: unset ENABLE_AUTO_TOOL_CHOICE\n"
            "  2. Pass the setting via the CLI flag instead:\n"
            '     --vllm-override-args \'{"enable-auto-tool-choice": true, "tool-call-parser": <parser-name>}\'\n'
            "  3. Re-run this script\n"
            "\nIf you need help, see https://docs.tenstorrent.com/getting-started/README.html#before-you-begin"
        )


def _get_local_server_python_env_dir(runtime_config) -> Path:
    tt_metal_home = Path(runtime_config.tt_metal_home).expanduser().resolve()
    if runtime_config.tt_metal_python_venv_dir:
        return Path(runtime_config.tt_metal_python_venv_dir).expanduser().resolve()
    return tt_metal_home / "python_env"


def _validate_local_vllm_installation(runtime_config):
    venv_python = _get_local_server_python_env_dir(runtime_config) / "bin" / "python"
    if not venv_python.exists():
        user_error(
            f"ERROR: The Python virtual environment for tt-metal was not found.\n"
            f"Expected interpreter at: {venv_python}\n"
            "\nTo fix this:\n"
            "  1. Build the tt-metal Python environment by following:\n"
            "     https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md\n"
            "  2. Verify the path is correct with --tt-metal-home or TT_METAL_HOME\n"
            "  3. Re-run this script\n"
            "\nIf you need help, see https://docs.tenstorrent.com/getting-started/README.html#before-you-begin"
        )

    return_code = run_command([str(venv_python), "-c", "import vllm"], logger=logger)
    if return_code != 0:
        user_error(
            "ERROR: The vllm Python package is not installed in the tt-metal virtual environment.\n"
            f"Tried to import vllm using: {venv_python}\n"
            "\nTo fix this:\n"
            "  1. Activate the tt-metal venv and install vLLM:\n"
            f"     {venv_python} -m pip install vllm\n"
            "  2. Or follow the full local install guide: vllm-tt-metal/README.md\n"
            "  3. Re-run this script\n"
            "\nIf you need help, see https://docs.tenstorrent.com/getting-started/README.html#before-you-begin"
        )
    logger.info(f"✅ validated vLLM Python package import with: {venv_python}")

    _validate_local_vllm_tt_plugin(runtime_config, venv_python)


def _validate_local_vllm_tt_plugin(runtime_config, venv_python: Path):
    """Ensure the vllm-tt-plugin package is installed and registered when the
    user's vLLM checkout requires it.

    Since tenstorrent/vllm commit a072e40a6 (2026-05-04, "Extract TT backend
    into plugin package (Phase 1)") the TT platform lives in a separate
    editable package at ``$VLLM_DIR/plugins/vllm-tt-plugin``. Without that
    package installed, ``vllm/platforms/tt.py`` raises
    ``ModuleNotFoundError: No module named 'vllm_tt_plugin'`` when
    ``vllm serve`` starts up.

    When the plugin source directory exists in the vLLM checkout this helper
    will:

    1. Editable-install it into the tt-metal venv (idempotent, mirrors the
       dev Dockerfile behaviour added by tt-inference-server PR #3370).
    2. Run a strict in-process check that ``import vllm_tt_plugin`` works AND
       that the ``tt`` entry is registered under the ``vllm.platform_plugins``
       entry-point group.

    When the plugin source is absent (older vLLM checkouts that still ship
    TT support inside vllm core) both steps are skipped.
    """
    # Local import to avoid a module-level circular dependency:
    # workflows.run_local_server -> workflows.setup_host -> workflows.validate_setup
    from workflows.run_local_server import (  # noqa: PLC0415
        install_vllm_tt_plugin_if_present,
        vllm_tt_plugin_source_path,
    )

    plugin_path = vllm_tt_plugin_source_path(_get_local_vllm_dir(runtime_config))
    if not (plugin_path / "pyproject.toml").exists():
        logger.info(
            f"Skipping vllm-tt-plugin validation: source not present at {plugin_path}"
        )
        return

    install_vllm_tt_plugin_if_present(runtime_config)

    check_script = (
        "import vllm_tt_plugin; "
        "from importlib.metadata import entry_points; "
        "eps = {ep.name for ep in entry_points(group='vllm.platform_plugins')}; "
        "assert 'tt' in eps, "
        "f'tt platform plugin not registered in vllm.platform_plugins entry points, got: {eps}'"
    )
    return_code = run_command([str(venv_python), "-c", check_script], logger=logger)
    if return_code != 0:
        vllm_dir = _get_local_vllm_dir(runtime_config)
        user_error(
            "ERROR: The vllm-tt-plugin package is not installed or not registered in the tt-metal venv.\n"
            "This plugin is required for the TT platform backend since vLLM extracted it into a separate package.\n"
            f"Plugin source detected at: {plugin_path}\n"
            "\nTo fix this:\n"
            f"  1. Run the install script from the vLLM repo:\n"
            f"     cd {vllm_dir} && bash tt_metal/install-vllm-tt.sh\n"
            "  2. Or see the full local install guide: vllm-tt-metal/README.md\n"
            "  3. Re-run this script\n"
            "\nIf you need help, see https://docs.tenstorrent.com/getting-started/README.html#before-you-begin"
        )
    logger.info(
        f"✅ validated vllm-tt-plugin install and `tt` platform_plugins entry "
        f"point registration with: {venv_python}"
    )


def _get_local_vllm_dir(runtime_config) -> Path:
    """Resolve $VLLM_DIR the same way workflows.run_local_server does."""
    tt_metal_home = Path(runtime_config.tt_metal_home).expanduser().resolve()
    vllm_dir = getattr(runtime_config, "vllm_dir", None)
    if vllm_dir:
        return Path(vllm_dir).expanduser().resolve()
    return (tt_metal_home / "vllm").resolve()


def validate_local_setup(model_spec, runtime_config, json_fpath):
    logger.info("Starting local setup validation")
    workflow_root_log_dir = get_default_workflow_root_log_dir()
    ensure_readwriteable_dir(workflow_root_log_dir)

    if (
        WorkflowType.from_string(runtime_config.workflow)
        in (WorkflowType.SERVER, WorkflowType.RELEASE)
    ) and (not runtime_config.skip_system_sw_validation):
        # check, and enforce if necessary, system software dependency versions
        venv_config = VENV_CONFIGS[WorkflowVenvType.SYSTEM_SOFTWARE_VALIDATION]
        venv_config.setup(model_spec=model_spec)

        # fmt: off
        cmd = [
            str(venv_config.venv_python),
            str(get_repo_root_path() / "workflows" / "run_system_software_validation.py"),
            "--runtime-model-spec-json", str(json_fpath),
        ]
        # fmt: on

        return_code = run_command(cmd, logger=logger)

        if return_code != 0:
            user_error(
                "ERROR: System software validation failed.\n"
                "One or more required software versions (drivers, firmware, tt-smi) did not meet the minimum requirements.\n"
                "\nTo fix this:\n"
                "  1. Check the errors above for the specific version that failed\n"
                "  2. Install or upgrade the flagged software to the required version\n"
                "  3. Re-run this script\n"
                "\nIf you are debugging and want to skip this check temporarily, add:\n"
                "  --skip-system-sw-validation\n"
                "\nIf you need help, see https://docs.tenstorrent.com/getting-started/README.html#before-you-begin"
            )
        logger.info("✅ validating system software dependencies completed")

    if (
        runtime_config.local_server
        and model_spec.inference_engine == InferenceEngine.VLLM.value
    ):
        _validate_local_vllm_installation(runtime_config)

    logger.info("✅ validating local setup completed")


def run_multihost_validation_subprocess(
    multihost_config, model_spec, json_fpath, dry_run=False
):
    """Run multihost validation via subprocess with dedicated venv.

    This aligns multihost validation with single-host validation pattern:
    - Uses SYSTEM_SOFTWARE_VALIDATION venv (with packaging library)
    - Runs run_multihost_validation.py as subprocess
    - Returns validated hosts list

    Args:
        multihost_config: MultiHostConfig object with hosts, paths, etc.
        model_spec: ModelSpec for system software version validation
        json_fpath: Path to runtime model spec JSON file
        dry_run: If True, skip directory existence and permission checks

    Returns:
        List of validated hostnames

    Raises:
        ValueError: If validation fails
    """
    venv_config = VENV_CONFIGS[WorkflowVenvType.SYSTEM_SOFTWARE_VALIDATION]
    venv_config.setup(model_spec=model_spec)

    cmd = [
        str(venv_config.venv_python),
        str(get_repo_root_path() / "workflows" / "run_multihost_validation.py"),
        "--hosts",
        ",".join(multihost_config.hosts),
        "--shared-storage-root",
        str(multihost_config.shared_storage_root),
        "--config-pkl-dir",
        str(multihost_config.config_pkl_dir),
        "--mpi-interface",
        multihost_config.mpi_interface,
        "--tt-smi-path",
        multihost_config.tt_smi_path,
    ]

    if json_fpath is not None:
        cmd.extend(["--runtime-model-spec-json", str(json_fpath)])

    if dry_run:
        cmd.append("--dry-run")

    return_code = run_command(cmd, logger=logger)

    if return_code != 0:
        user_error(
            "ERROR: Multi-host validation failed.\n"
            "One or more remote hosts failed software or connectivity checks.\n"
            "\nTo fix this:\n"
            "  1. Check the errors above for the specific host and check that failed\n"
            "  2. Ensure all hosts are reachable over SSH and meet the software requirements\n"
            "  3. Re-run this script\n"
            "\nIf you are debugging and want to skip this check temporarily, add:\n"
            "  --skip-system-sw-validation\n"
            "\nIf you need help, see https://docs.tenstorrent.com/getting-started/README.html#before-you-begin"
        )

    logger.info("✅ Multi-host validation completed")
    return multihost_config.hosts


def _try_fix_path_permissions_for_uid(path, uid, need_write=False):
    """Best-effort chmod to grant the target UID the required access bits.

    Determines which POSIX scope (owner/group/other) the UID falls into and
    adds read (+execute for directories, +write if need_write) bits for that
    scope.  Only succeeds when the current process has permission to chmod
    (i.e. current user owns the path or is root) -- no sudo required.

    Returns True if chmod was applied, False on failure.
    """
    path = Path(path)
    if not path.exists():
        return False

    st = path.stat()
    mode = st.st_mode
    gids = get_groups_for_uid(uid)

    if uid == st.st_uid:
        new_bits = stat.S_IRUSR
        if path.is_dir():
            new_bits |= stat.S_IXUSR
        if need_write:
            new_bits |= stat.S_IWUSR
    elif st.st_gid in gids:
        new_bits = stat.S_IRGRP
        if path.is_dir():
            new_bits |= stat.S_IXGRP
        if need_write:
            new_bits |= stat.S_IWGRP
    else:
        new_bits = stat.S_IROTH
        if path.is_dir():
            new_bits |= stat.S_IXOTH
        if need_write:
            new_bits |= stat.S_IWOTH

    target_mode = mode | new_bits
    if target_mode == mode:
        return False

    try:
        os.chmod(path, target_mode)
        logger.info(f"Fixed permissions on {path}: {oct(mode)} -> {oct(target_mode)}")
        return True
    except OSError as e:
        logger.debug(f"Cannot chmod {path}: {e}")
        return False


def validate_bind_mount_permissions(args):
    """Validate that --image-user UID can access bind-mounted host paths.

    Checks read permission for --host-hf-cache and --host-weights-dir (readonly mounts),
    and read+write permission for --host-volume (read-write mount).

    If a check fails, attempts to fix permissions via chmod (no sudo).
    Raises ValueError with actionable guidance if the fix is not possible.
    """
    uid = int(args.image_user)
    checks = []

    if args.host_volume:
        host_volume_path = Path(args.host_volume)
        if not host_volume_path.exists():
            logger.info(f"Creating host volume directory: {host_volume_path}")
            host_volume_path.mkdir(parents=True, exist_ok=True)
        checks.append(("--host-volume", args.host_volume, True))
    if args.host_hf_cache:
        checks.append(("--host-hf-cache", args.host_hf_cache, False))
    if getattr(args, "host_weights_dir", None):
        checks.append(("--host-weights-dir", args.host_weights_dir, False))

    for flag, host_path, need_write in checks:
        ok, reason = check_path_permissions_for_uid(
            host_path, uid, need_write=need_write
        )
        if not ok:
            _try_fix_path_permissions_for_uid(host_path, uid, need_write=need_write)
            ok, reason = check_path_permissions_for_uid(
                host_path, uid, need_write=need_write
            )
        if not ok:
            access_type = "read+write" if need_write else "read"
            user_error(
                f"ERROR: The Docker container user cannot access a bind-mounted host path.\n"
                f"  Path: {host_path}  (flag: {flag})\n"
                f"  Container user UID {uid} needs {access_type} access but was denied.\n"
                f"  Reason: {reason}\n"
                f"\nTo fix this:\n"
                f"  Option A — match the UID: add --image-user $(id -u) to your run.py command\n"
                f"  Option B — fix the path permissions: sudo chown -R {uid}:{uid} {host_path}\n"
                f"  Option C — use chmod: chmod o+{'rw' if need_write else 'r'} {host_path}\n"
                f"\nIf you need help, see https://docs.tenstorrent.com/getting-started/README.html#before-you-begin"
            )
        logger.info(
            f"✅ Bind mount permission check passed for {flag}={host_path} "
            f"(uid={uid}, write={need_write})"
        )


def validate_local_server_paths(args):
    """Validate required host paths for --local-server execution."""
    if not args.local_server:
        return
    if not args.tt_metal_home:
        user_error(
            "ERROR: --local-server requires a tt-metal build directory but none was specified.\n"
            "The local server needs the compiled tt-metal libraries to run.\n"
            "\nTo fix this:\n"
            "  1. Set the environment variable: export TT_METAL_HOME=/path/to/tt-metal\n"
            "  2. Or pass the flag: --tt-metal-home /path/to/tt-metal\n"
            "  3. Re-run this script\n"
            "\nIf you need help, see https://docs.tenstorrent.com/getting-started/README.html#before-you-begin"
        )

    tt_metal_home = Path(args.tt_metal_home).expanduser().resolve()
    if not tt_metal_home.exists():
        user_error(
            f"ERROR: The --tt-metal-home path does not exist: {tt_metal_home}\n"
            "This path must point to a built tt-metal repository.\n"
            "\nTo fix this:\n"
            "  1. Build tt-metal following: https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md\n"
            "  2. Update TT_METAL_HOME or --tt-metal-home to the correct path\n"
            "  3. Re-run this script\n"
            "\nIf you need help, see https://docs.tenstorrent.com/getting-started/README.html#before-you-begin"
        )
    if not tt_metal_home.is_dir():
        user_error(
            f"ERROR: The --tt-metal-home path exists but is not a directory: {tt_metal_home}\n"
            "This path must point to the root directory of a built tt-metal repository.\n"
            "\nTo fix this:\n"
            "  1. Check that TT_METAL_HOME or --tt-metal-home is set to a directory path, not a file\n"
            "  2. Re-run this script\n"
            "\nIf you need help, see https://docs.tenstorrent.com/getting-started/README.html#before-you-begin"
        )

    python_env_dir = _get_local_server_python_env_dir(args)
    vllm_dir = (
        Path(args.vllm_dir).expanduser().resolve()
        if getattr(args, "vllm_dir", None)
        else (tt_metal_home / "vllm").resolve()
    )
    venv_python = python_env_dir / "bin" / "python"
    build_lib_dir = tt_metal_home / "build" / "lib"
    entrypoint_path = (
        get_repo_root_path() / "vllm-tt-metal" / "src" / "run_vllm_api_server.py"
    )

    required_paths = [
        ("python venv interpreter", venv_python),
        ("tt-metal build/lib", build_lib_dir),
        ("vLLM source dir", vllm_dir),
        ("local server entrypoint", entrypoint_path),
    ]
    for label, path in required_paths:
        if not path.exists():
            user_error(
                f"ERROR: A required path for --local-server is missing.\n"
                f"  Missing: {label}\n"
                f"  Expected at: {path}\n"
                "\nTo fix this:\n"
                "  1. Ensure tt-metal is fully built and --tt-metal-home points to the right directory\n"
                "  2. If using a custom vLLM directory, check --vllm-dir is correct\n"
                "  3. Re-run this script\n"
                "\nIf you need help, see https://docs.tenstorrent.com/getting-started/README.html#before-you-begin"
            )

    if args.host_hf_cache:
        host_hf_cache = Path(args.host_hf_cache).expanduser().resolve()
        if not host_hf_cache.exists():
            user_error(
                f"ERROR: The --host-hf-cache directory does not exist: {host_hf_cache}\n"
                "This path must point to an existing HuggingFace cache directory.\n"
                "\nTo fix this:\n"
                f"  1. Create the directory: mkdir -p {host_hf_cache}\n"
                "  2. Or point to your existing HuggingFace cache (usually ~/.cache/huggingface)\n"
                "  3. Re-run this script\n"
                "\nIf you need help, see https://docs.tenstorrent.com/getting-started/README.html#before-you-begin"
            )
        snapshot_dir = resolve_hf_snapshot_dir(
            args.runtime_model_spec["hf_weights_repo"], host_hf_cache
        )
        if snapshot_dir is None:
            hf_repo = args.runtime_model_spec["hf_weights_repo"]
            user_error(
                f"ERROR: No cached snapshot found for '{hf_repo}' in the HuggingFace cache.\n"
                f"Cache directory: {host_hf_cache}\n"
                "\nTo fix this:\n"
                f"  1. Download the model weights first: huggingface-cli download {hf_repo}\n"
                "  2. Or remove --host-hf-cache to let the server download weights on first start\n"
                "  3. Re-run this script\n"
                "\nIf you need help, see https://docs.tenstorrent.com/getting-started/README.html#before-you-begin"
            )

    if args.host_weights_dir:
        host_weights_dir = Path(args.host_weights_dir).expanduser().resolve()
        if not host_weights_dir.exists():
            user_error(
                f"ERROR: The --host-weights-dir directory does not exist: {host_weights_dir}\n"
                "This path must point to a directory containing pre-downloaded model weights.\n"
                "\nTo fix this:\n"
                f"  1. Download the model weights to that directory, or\n"
                "  2. Update --host-weights-dir to point to where your weights are stored\n"
                "  3. Re-run this script\n"
                "\nIf you need help, see https://docs.tenstorrent.com/getting-started/README.html#before-you-begin"
            )


def validate_setup(model_spec, runtime_config, json_fpath):
    """Top-level validation orchestrator called from run.py main().

    Runs all pre-flight validation checks in order:
    1. validate_runtime_args - CLI arg consistency and model/workflow support
    2. validate_local_setup - system software dependencies
    3. validate_bind_mount_permissions - Docker bind mount UID access (docker-server only)
    """
    validate_runtime_args(model_spec, runtime_config)
    validate_local_setup(model_spec, runtime_config, json_fpath)
    if runtime_config.docker_server:
        validate_bind_mount_permissions(runtime_config)
    elif runtime_config.local_server:
        validate_local_server_paths(runtime_config)
