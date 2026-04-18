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
    check_path_permissions_for_uid,
    ensure_readwriteable_dir,
    get_default_workflow_root_log_dir,
    get_groups_for_uid,
    get_repo_root_path,
    resolve_hf_snapshot_dir,
    run_command,
)
from workflows.workflow_types import (
    DeviceTypes,
    InferenceEngine,
    WorkflowType,
    WorkflowVenvType,
)
from workflows.workflow_venvs import VENV_CONFIGS

logger = logging.getLogger("run_log")


def validate_runtime_args(model_spec, runtime_config):
    args = runtime_config
    workflow_type = WorkflowType.from_string(args.workflow)

    if not args.device:
        # TODO: detect phy device
        raise NotImplementedError("Device detection not implemented yet")

    model_id = model_spec.model_id

    # Check if the model_id exists in MODEL_SPECS (this validates device support)
    if model_id not in MODEL_SPECS:
        raise ValueError(
            f"model:={runtime_config.model} does not support device:={runtime_config.device}"
        )

    assert not (args.docker_server and args.local_server), (
        "Cannot run --docker-server and --local-server"
    )

    if workflow_type == WorkflowType.EVALS:
        assert model_spec.model_name in EVAL_CONFIGS, (
            f"Model:={model_spec.model_name} not found in EVAL_CONFIGS"
        )
    if workflow_type == WorkflowType.BENCHMARKS:
        if os.getenv("OVERRIDE_BENCHMARKS"):
            logger.warning("OVERRIDE_BENCHMARKS is active, using override benchmarks")
        assert model_spec.model_id in BENCHMARK_CONFIGS, (
            f"Model:={model_spec.model_name} not found in BENCHMARKS_CONFIGS"
        )
    if workflow_type == WorkflowType.STRESS_TESTS:
        pass  # Model support already validated via MODEL_SPECS check

    if workflow_type == WorkflowType.TESTS:
        assert model_spec.model_name in TEST_CONFIGS, (
            f"Model:={model_spec.model_name} not found in TEST_CONFIGS"
        )
    if workflow_type == WorkflowType.REPORTS:
        pass
    if workflow_type == WorkflowType.SERVER:
        if not (args.docker_server or args.local_server):
            raise ValueError(
                f"Workflow {args.workflow} requires --docker-server or --local-server"
            )
        if (
            args.local_server
            and model_spec.inference_engine != InferenceEngine.VLLM.value
        ):
            raise NotImplementedError(
                "--local-server currently supports only vLLM-backed model specs"
            )

        # For partitioning Galaxy per tray as T3K
        # TODO: Add a check to verify whether these devices belong to the same tray
        if DeviceTypes.from_string(args.device) == DeviceTypes.GALAXY_T3K:
            if not args.device_id or len(args.device_id) != 8:
                raise ValueError(
                    "Galaxy T3K requires exactly 8 device IDs specified with --device-id (e.g. '0,1,2,3,4,5,6,7'). These must be devices within the same tray."
                )

    if workflow_type == WorkflowType.RELEASE:
        # NOTE: fail fast for models without both defined evals and benchmarks
        # today this will stop models defined in MODEL_SPECS
        # but not in EVAL_CONFIGS or BENCHMARK_CONFIGS, e.g. non-instruct models
        # a run_*.log fill will be made for the failed combination indicating this
        assert model_spec.model_name in EVAL_CONFIGS, (
            f"Model:={model_spec.model_name} not found in EVAL_CONFIGS"
        )
        assert model_spec.model_id in BENCHMARK_CONFIGS, (
            f"Model:={model_spec.model_name} not found in BENCHMARKS_CONFIGS"
        )

    if DeviceTypes.from_string(args.device) == DeviceTypes.GPU:
        if args.docker_server or args.local_server:
            raise NotImplementedError(
                "GPU support for running inference server not implemented yet"
            )

    if args.local_server and not args.tt_metal_home:
        raise ValueError(
            "--local-server requires --tt-metal-home or TT_METAL_HOME to be set"
        )

    # Validate mutual exclusivity of weight source options
    weight_source_args = [
        args.host_volume,
        args.host_hf_cache,
        getattr(args, "host_weights_dir", None),
    ]
    if sum(1 for a in weight_source_args if a) > 1:
        raise ValueError(
            "Only one of --host-volume, --host-hf-cache, --host-weights-dir can be specified."
        )

    if "ENABLE_AUTO_TOOL_CHOICE" in os.environ:
        raise AssertionError(
            "Setting ENABLE_AUTO_TOOL_CHOICE has been deprecated, use the VLLM_OVERRIDE_ARGS env var directly or via --vllm-override-args in run.py CLI.\n"
            'Enable auto tool choice by adding --vllm-override-args \'{"enable-auto-tool-choice": true, "tool-call-parser": <parser-name>}\' when calling run.py'
        )


def _get_local_server_python_env_dir(runtime_config) -> Path:
    tt_metal_home = Path(runtime_config.tt_metal_home).expanduser().resolve()
    if runtime_config.tt_metal_python_venv_dir:
        return Path(runtime_config.tt_metal_python_venv_dir).expanduser().resolve()
    return tt_metal_home / "python_env"


def _validate_local_vllm_installation(runtime_config):
    venv_python = _get_local_server_python_env_dir(runtime_config) / "bin" / "python"
    if not venv_python.exists():
        raise ValueError(f"⛔ Missing required python venv interpreter: {venv_python}")

    return_code = run_command([str(venv_python), "-c", "import vllm"], logger=logger)
    if return_code != 0:
        raise ValueError(
            "⛔ --local-server with inference engine vLLM requires the `vllm` Python "
            f"package to be installed in the tt-metal python environment. Could not "
            f"import `vllm` with: {venv_python}"
        )
    logger.info(f"✅ validated vLLM Python package import with: {venv_python}")


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
            raise ValueError(
                "⛔ validating system software dependencies failed. See errors above for "
                "required version, and System Info section above for current system versions."
                "\nTo skip system software validation, use the flag: --skip-system-sw-validation"
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
        raise ValueError(
            "⛔ Multi-host validation failed. See errors above.\n"
            "To skip system software validation, use the flag: --skip-system-sw-validation"
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
            raise ValueError(
                f"⛔ Bind mount permission check failed for {flag}={host_path}\n"
                f"  Container user (--image-user={uid}) needs {access_type} access.\n"
                f"  {reason}\n"
                f"  Fix: set --image-user to match the path owner UID, or adjust "
                f"permissions with chmod/chown on the host path."
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
        raise ValueError(
            "--local-server requires --tt-metal-home or TT_METAL_HOME to be set"
        )

    tt_metal_home = Path(args.tt_metal_home).expanduser().resolve()
    if not tt_metal_home.exists():
        raise ValueError(f"⛔ --tt-metal-home path does not exist: {tt_metal_home}")
    if not tt_metal_home.is_dir():
        raise ValueError(f"⛔ --tt-metal-home is not a directory: {tt_metal_home}")

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
            raise ValueError(f"⛔ Missing required {label}: {path}")

    if args.host_hf_cache:
        host_hf_cache = Path(args.host_hf_cache).expanduser().resolve()
        if not host_hf_cache.exists():
            raise ValueError(f"⛔ --host-hf-cache path does not exist: {host_hf_cache}")
        snapshot_dir = resolve_hf_snapshot_dir(
            args.runtime_model_spec["hf_weights_repo"], host_hf_cache
        )
        if snapshot_dir is None:
            raise ValueError(
                f"⛔ --host-hf-cache did not contain a cached snapshot for "
                f"{args.runtime_model_spec['hf_weights_repo']}: {host_hf_cache}"
            )

    if args.host_weights_dir:
        host_weights_dir = Path(args.host_weights_dir).expanduser().resolve()
        if not host_weights_dir.exists():
            raise ValueError(
                f"⛔ --host-weights-dir path does not exist: {host_weights_dir}"
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
