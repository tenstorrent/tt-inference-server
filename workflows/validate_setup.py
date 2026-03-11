# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import grp
import logging
import os
import re
import stat
from pathlib import Path
from typing import Optional, Tuple

from benchmarking.benchmark_config import BENCHMARK_CONFIGS
from evals.eval_config import EVAL_CONFIGS
from tests.test_config import TEST_CONFIGS
from workflows.model_spec import MODEL_SPECS
from workflows.utils import (
    ensure_readwriteable_dir,
    get_default_workflow_root_log_dir,
    get_repo_root_path,
    get_version,
    run_command,
)
from workflows.workflow_types import (
    DeviceTypes,
    WorkflowType,
    WorkflowVenvType,
)
from workflows.workflow_venvs import VENV_CONFIGS

logger = logging.getLogger("run_log")


def _parse_semantic_version(version: Optional[str]) -> Optional[Tuple[int, int, int]]:
    """Return a comparable semantic version tuple or None."""
    if not version:
        return None

    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", version)
    if not match:
        return None

    return tuple(int(part) for part in match.groups())


def warn_if_release_version_is_older(model_spec) -> None:
    """Warn when a ModelSpec was last released before the current repo version."""
    release_version = getattr(model_spec, "release_version", None)
    current_version = get_version()
    release_parts = _parse_semantic_version(release_version)
    current_parts = _parse_semantic_version(current_version)

    if not release_parts or not current_parts:
        return

    if release_parts < current_parts:
        logger.warning(
            f"ModelSpec {model_spec.model_id} has release_version {release_version}, "
            f"which is older than the current VERSION {current_version}."
        )


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
        if args.local_server:
            raise NotImplementedError(
                f"Workflow {args.workflow} not implemented for --local-server"
            )
        if not (args.docker_server or args.local_server):
            raise ValueError(
                f"Workflow {args.workflow} requires --docker-server argument"
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

    assert not (args.docker_server and args.local_server), (
        "Cannot run --docker-server and --local-server"
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

    logger.info("✅ validating local setup completed")


def _get_groups_for_uid(uid):
    """Return the set of GIDs that a given UID belongs to on this host."""
    gids = set()
    try:
        import pwd

        pw = pwd.getpwuid(uid)
        gids.add(pw.pw_gid)
        username = pw.pw_name
        for group in grp.getgrall():
            if username in group.gr_mem:
                gids.add(group.gr_gid)
    except KeyError:
        # UID doesn't exist on the host; can only rely on "other" bits
        pass
    return gids


def _check_path_permissions_for_uid(path, uid, need_write=False):
    """Check whether the given UID can access a path based on POSIX permission bits.

    Best-effort pre-flight check. Cannot detect ACLs, SELinux, or other
    security modules, but catches common UID/permission mismatches.

    Args:
        path: Filesystem path to check.
        uid: Numeric UID that will access the path (i.e. --image-user).
        need_write: If True, also check write permission.

    Returns:
        Tuple of (ok: bool, reason: str). reason is empty when ok is True.
    """
    path = Path(path)
    if not path.exists():
        return False, f"path does not exist: {path}"

    st = path.stat()
    mode = st.st_mode
    gids = _get_groups_for_uid(uid)

    if uid == st.st_uid:
        has_read = bool(mode & stat.S_IRUSR)
        has_write = bool(mode & stat.S_IWUSR)
        has_exec = bool(mode & stat.S_IXUSR)
        scope = "owner"
    elif st.st_gid in gids:
        has_read = bool(mode & stat.S_IRGRP)
        has_write = bool(mode & stat.S_IWGRP)
        has_exec = bool(mode & stat.S_IXGRP)
        scope = "group"
    else:
        has_read = bool(mode & stat.S_IROTH)
        has_write = bool(mode & stat.S_IWOTH)
        has_exec = bool(mode & stat.S_IXOTH)
        scope = "other"

    if not has_read:
        return False, (
            f"UID {uid} lacks read permission ({scope}) on {path} "
            f"(owner={st.st_uid}, gid={st.st_gid}, mode={oct(mode)})"
        )

    if path.is_dir() and not has_exec:
        return False, (
            f"UID {uid} lacks execute/traverse permission ({scope}) on directory {path} "
            f"(owner={st.st_uid}, gid={st.st_gid}, mode={oct(mode)})"
        )

    if need_write and not has_write:
        return False, (
            f"UID {uid} lacks write permission ({scope}) on {path} "
            f"(owner={st.st_uid}, gid={st.st_gid}, mode={oct(mode)})"
        )

    return True, ""


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
    gids = _get_groups_for_uid(uid)

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
        ok, reason = _check_path_permissions_for_uid(
            host_path, uid, need_write=need_write
        )
        if not ok:
            _try_fix_path_permissions_for_uid(host_path, uid, need_write=need_write)
            ok, reason = _check_path_permissions_for_uid(
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


def validate_setup(model_spec, runtime_config, json_fpath):
    """Top-level validation orchestrator called from run.py main().

    Runs all pre-flight validation checks in order:
    1. validate_runtime_args - CLI arg consistency and model/workflow support
    2. validate_local_setup - system software dependencies
    3. validate_bind_mount_permissions - Docker bind mount UID access (docker-server only)
    """
    validate_runtime_args(model_spec, runtime_config)
    warn_if_release_version_is_older(model_spec)
    validate_local_setup(model_spec, runtime_config, json_fpath)
    if runtime_config.docker_server:
        validate_bind_mount_permissions(runtime_config)
