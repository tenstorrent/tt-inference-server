# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Host-side tensor cache monitoring used by benchmark and eval startup waits.

`PromptClient.wait_for_healthy()` uses this module to decide whether startup is
still generating tensor cache, whether cache files are growing, and whether
generation appears stalled.

Cache location resolution order:
1. Use the explicit `cache_dir` argument when provided.
2. Otherwise use the host `TT_CACHE_PATH` environment variable when set.
3. Otherwise derive a persistent cache target from `model_spec` and
   `runtime_config` using the cache suffix
   `tt_metal_cache/cache_<model_name>/<mesh_device>`:
   - `host_volume`: monitor
     `<host_volume>/volume_id_<impl>-<model>-v<version>/<cache suffix>`
   - `local_server` without `host_volume`: monitor the repo default
     `<repo_root>/persistent_volume/...`
   - Docker named volume mode: inspect `volume_id_<impl>-<model>`. If the
     mountpoint is readable by the current host user, monitor the resolved host
     path directly. If the volume exists but the mountpoint is not readable,
     fall back to Docker CLI and inspect the same cache subtree through a
     read-only helper container mounted at `/cache`.
4. If none of the above can be resolved, monitoring is disabled.

Permissions:
- Direct-path modes require the host process to read the cache directory. When
  marker files are writable, explicit lifecycle methods can create
  `TT_METAL_CACHE_FIRST_RUN_STARTED` and `TT_METAL_CACHE_COMPLETED` in the
  cache directory.
- Docker CLI fallback is used specifically when the named volume exists but the
  host user cannot read its mountpoint directly. It does not change host
  permissions or write real marker files into the volume. Instead it reads file
  count and total size via `docker run` and keeps in-memory started/completed
  marker state for the lifetime of the host-side process.
"""

import errno
import logging
import os
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

from utils.device_utils import get_mesh_device_name

logger = logging.getLogger(__name__)

DEFAULT_CACHE_ROOT = Path("/home/container_app_user/cache_root")
DEFAULT_DOCKER_CACHE_MONITOR_ROOT = Path("/cache")
DEFAULT_DOCKER_CACHE_MONITOR_HELPER_IMAGE = "python:3.8-slim"

DOCKER_VOLUME_SNAPSHOT_SCRIPT = """
import json
import sys
from pathlib import Path

target = Path("/cache") / sys.argv[1]
marker_files = set(sys.argv[2:])
total_size_bytes = 0
file_count = 0

if target.exists():
    for cache_file in target.rglob("*"):
        if not cache_file.is_file() or cache_file.name in marker_files:
            continue
        try:
            total_size_bytes += cache_file.stat().st_size
            file_count += 1
        except OSError:
            continue

print(
    json.dumps(
        {
            "file_count": file_count,
            "total_size_bytes": total_size_bytes,
        }
    )
)
""".strip()


_PERMISSION_DENIED_ERRNOS = {errno.EACCES, errno.EPERM}


def _get_value(obj, key: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _normalize_optional_path(path_value: Optional[Union[str, Path]]) -> Optional[Path]:
    if path_value is None:
        return None
    return Path(path_value).expanduser()


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_default_persistent_volume_root(repo_root: Optional[Path] = None) -> Path:
    root = repo_root or _resolve_repo_root()
    return Path(root).resolve() / "persistent_volume"


def get_docker_volume_name(model_spec) -> Optional[str]:
    impl = _get_value(model_spec, "impl")
    impl_id = _get_value(impl, "impl_id")
    model_name = _get_value(model_spec, "model_name")
    if not impl_id or not model_name:
        return None
    return f"volume_id_{impl_id}-{model_name}"


def get_host_model_volume_root(
    model_spec, host_volume: Optional[Union[str, Path]]
) -> Optional[Path]:
    host_volume_path = _normalize_optional_path(host_volume)
    impl = _get_value(model_spec, "impl")
    impl_id = _get_value(impl, "impl_id")
    model_name = _get_value(model_spec, "model_name")
    version = _get_value(model_spec, "version")
    if (
        host_volume_path is None
        or not impl_id
        or not model_name
        or version in (None, "")
    ):
        return None

    volume_dir_name = f"volume_id_{impl_id}-{model_name}-v{version}"
    return host_volume_path.resolve() / volume_dir_name


@dataclass(frozen=True)
class DockerVolumeInfo:
    volume_name: str
    mountpoint: Path
    is_readable: bool


@dataclass(frozen=True)
class CacheMonitorTarget:
    cache_dir: Optional[Path] = None
    docker_volume_name: Optional[str] = None
    docker_cache_dir: Optional[Path] = None
    docker_helper_image: Optional[str] = None

    def is_enabled(self) -> bool:
        return self.cache_dir is not None or self.uses_docker_cli()

    def uses_docker_cli(self) -> bool:
        return (
            self.cache_dir is None
            and self.docker_volume_name is not None
            and self.docker_cache_dir is not None
        )

    def get_display_cache_dir(self) -> Optional[Path]:
        return self.cache_dir or self.docker_cache_dir


def _is_permission_denied(error: OSError) -> bool:
    return (
        isinstance(error, PermissionError)
        or getattr(error, "errno", None) in _PERMISSION_DENIED_ERRNOS
    )


def _get_unreadable_docker_volume_info(
    volume_name: str, mountpoint: Path, error: OSError
) -> DockerVolumeInfo:
    logger.info(
        "Docker volume %s mountpoint could not be inspected by the current user: %s",
        volume_name,
        mountpoint,
    )
    return DockerVolumeInfo(
        volume_name=volume_name,
        mountpoint=mountpoint,
        is_readable=False,
    )


def get_cache_relative_dir(
    model_spec, runtime_config=None, device: Optional[str] = None
) -> Optional[Path]:
    model_name = _get_value(model_spec, "model_name")
    if not model_name:
        return None
    if device is None and runtime_config is not None:
        device = _get_value(runtime_config, "device")
    return (
        Path("tt_metal_cache")
        / f"cache_{model_name}"
        / get_mesh_device_name(model_spec, device=device)
    )


def inspect_docker_volume(volume_name: str) -> Optional[DockerVolumeInfo]:
    try:
        result = subprocess.run(
            [
                "docker",
                "volume",
                "inspect",
                volume_name,
                "--format",
                "{{ .Mountpoint }}",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except FileNotFoundError:
        logger.info(
            "Docker CLI is unavailable; host-side cache monitoring is disabled for volume %s.",
            volume_name,
        )
        return None
    except subprocess.TimeoutExpired:
        logger.info(
            "Timed out inspecting Docker volume %s; host-side cache monitoring is disabled.",
            volume_name,
        )
        return None
    except Exception as exc:
        logger.warning(
            "Could not inspect Docker volume %s for cache monitoring: %s",
            volume_name,
            exc,
        )
        return None

    if result.returncode != 0:
        reason = str(getattr(result, "stderr", "") or "").strip()
        reason = reason or "volume not found or daemon unavailable"
        logger.info(
            "Could not inspect Docker volume %s for cache monitoring: %s",
            volume_name,
            reason,
        )
        return None

    mountpoint_str = str(getattr(result, "stdout", "") or "").strip()
    if not mountpoint_str:
        logger.info(
            "Docker volume %s did not report a mountpoint; cache monitoring is disabled.",
            volume_name,
        )
        return None

    mountpoint = Path(mountpoint_str)
    try:
        mountpoint_exists = mountpoint.exists()
    except OSError as error:
        if _is_permission_denied(error):
            return _get_unreadable_docker_volume_info(volume_name, mountpoint, error)
        logger.info(
            "Could not inspect Docker volume %s mountpoint %s: %s",
            volume_name,
            mountpoint,
            error,
        )
        return None

    if not mountpoint_exists:
        logger.info(
            "Docker volume %s mountpoint is not accessible on this host: %s",
            volume_name,
            mountpoint,
        )
        return None

    try:
        mountpoint_is_dir = mountpoint.is_dir()
    except OSError as error:
        if _is_permission_denied(error):
            return _get_unreadable_docker_volume_info(volume_name, mountpoint, error)
        logger.info(
            "Could not inspect Docker volume %s mountpoint type %s: %s",
            volume_name,
            mountpoint,
            error,
        )
        return None

    if not mountpoint_is_dir:
        logger.info(
            "Docker volume %s mountpoint is not a directory: %s",
            volume_name,
            mountpoint,
        )
        return None

    try:
        resolved_mountpoint = mountpoint.resolve()
    except OSError as error:
        if _is_permission_denied(error):
            return _get_unreadable_docker_volume_info(volume_name, mountpoint, error)
        logger.info(
            "Could not resolve Docker volume %s mountpoint %s: %s",
            volume_name,
            mountpoint,
            error,
        )
        return None

    try:
        is_readable = os.access(resolved_mountpoint, os.R_OK | os.X_OK)
    except OSError as error:
        if _is_permission_denied(error):
            return _get_unreadable_docker_volume_info(
                volume_name, resolved_mountpoint, error
            )
        logger.info(
            "Could not inspect Docker volume %s mountpoint permissions %s: %s",
            volume_name,
            resolved_mountpoint,
            error,
        )
        return None

    if not is_readable:
        logger.info(
            "Docker volume %s mountpoint is not readable by the current user: %s",
            volume_name,
            resolved_mountpoint,
        )

    return DockerVolumeInfo(
        volume_name=volume_name,
        mountpoint=resolved_mountpoint,
        is_readable=is_readable,
    )


def inspect_docker_volume_mountpoint(volume_name: str) -> Optional[Path]:
    volume_info = inspect_docker_volume(volume_name)
    if volume_info is None or not volume_info.is_readable:
        return None
    return volume_info.mountpoint


def get_environment_cache_dir() -> Optional[Path]:
    return _normalize_optional_path(os.getenv("TT_CACHE_PATH"))


def get_container_cache_dir(
    model_spec,
    device: Optional[str] = None,
    cache_root: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    cache_root_path = _normalize_optional_path(
        cache_root or os.getenv("CACHE_ROOT") or DEFAULT_CACHE_ROOT
    )
    cache_relative_dir = get_cache_relative_dir(model_spec, device=device)
    if cache_relative_dir is None:
        return None
    return cache_root_path / cache_relative_dir


def get_host_cache_monitor_root(
    model_spec, runtime_config=None, repo_root: Optional[Path] = None
) -> Optional[Path]:
    host_volume = _normalize_optional_path(_get_value(runtime_config, "host_volume"))
    is_local_server = bool(_get_value(runtime_config, "local_server", False))

    if is_local_server and host_volume is None:
        host_volume = get_default_persistent_volume_root(repo_root=repo_root)

    host_model_volume_root = get_host_model_volume_root(model_spec, host_volume)
    if host_model_volume_root is not None:
        return host_model_volume_root

    if is_local_server:
        return None

    docker_volume_name = get_docker_volume_name(model_spec)
    if docker_volume_name is None:
        return None
    return inspect_docker_volume_mountpoint(docker_volume_name)


def _get_host_cache_monitor_target(
    model_spec, runtime_config=None, repo_root: Optional[Path] = None
) -> CacheMonitorTarget:
    cache_relative_dir = get_cache_relative_dir(
        model_spec, runtime_config=runtime_config
    )
    if cache_relative_dir is None:
        return CacheMonitorTarget()

    host_volume = _normalize_optional_path(_get_value(runtime_config, "host_volume"))
    is_local_server = bool(_get_value(runtime_config, "local_server", False))

    if is_local_server and host_volume is None:
        host_volume = get_default_persistent_volume_root(repo_root=repo_root)

    host_model_volume_root = get_host_model_volume_root(model_spec, host_volume)
    if host_model_volume_root is not None:
        return CacheMonitorTarget(cache_dir=host_model_volume_root / cache_relative_dir)

    if is_local_server:
        return CacheMonitorTarget()

    docker_volume_name = get_docker_volume_name(model_spec)
    if docker_volume_name is None:
        return CacheMonitorTarget()

    volume_info = inspect_docker_volume(docker_volume_name)
    if volume_info is None:
        return CacheMonitorTarget()
    if volume_info.is_readable:
        return CacheMonitorTarget(cache_dir=volume_info.mountpoint / cache_relative_dir)

    docker_cache_dir = DEFAULT_DOCKER_CACHE_MONITOR_ROOT / cache_relative_dir
    docker_helper_image = (
        _get_value(model_spec, "docker_image")
        or os.getenv("CACHE_MONITOR_DOCKER_HELPER_IMAGE")
        or DEFAULT_DOCKER_CACHE_MONITOR_HELPER_IMAGE
    )
    logger.info(
        "Using Docker CLI fallback to monitor unreadable volume %s at %s",
        docker_volume_name,
        docker_cache_dir,
    )
    return CacheMonitorTarget(
        docker_volume_name=docker_volume_name,
        docker_cache_dir=docker_cache_dir,
        docker_helper_image=docker_helper_image,
    )


def resolve_host_cache_monitor_dir(
    model_spec, runtime_config=None, repo_root: Optional[Path] = None
) -> Optional[Path]:
    return _get_host_cache_monitor_target(
        model_spec, runtime_config=runtime_config, repo_root=repo_root
    ).cache_dir


def detect_cache_monitor_target(
    model_spec=None,
    cache_dir: Optional[Union[str, Path]] = None,
    runtime_config=None,
    repo_root: Optional[Path] = None,
) -> CacheMonitorTarget:
    normalized_cache_dir = _normalize_optional_path(cache_dir)
    if normalized_cache_dir is not None:
        return CacheMonitorTarget(cache_dir=normalized_cache_dir)

    environment_cache_dir = get_environment_cache_dir()
    if environment_cache_dir is not None:
        return CacheMonitorTarget(cache_dir=environment_cache_dir)

    if model_spec is None:
        return CacheMonitorTarget()

    return _get_host_cache_monitor_target(
        model_spec, runtime_config=runtime_config, repo_root=repo_root
    )


def detect_cache_monitor_dir(
    model_spec=None,
    cache_dir: Optional[Union[str, Path]] = None,
    runtime_config=None,
    repo_root: Optional[Path] = None,
) -> Optional[Path]:
    return detect_cache_monitor_target(
        model_spec=model_spec,
        cache_dir=cache_dir,
        runtime_config=runtime_config,
        repo_root=repo_root,
    ).cache_dir


@dataclass
class CacheGenerationStatus:
    """Status of cache generation process."""

    is_generating: bool
    cache_dir: Optional[Path] = None
    container_id: Optional[str] = None
    last_activity_time: Optional[float] = None
    estimated_completion_time: Optional[float] = None
    is_first_run: bool = False
    has_existing_cache: bool = False
    is_stalled: bool = False
    file_count: int = 0
    total_size_bytes: int = 0
    no_progress_duration: float = 0.0


@dataclass(frozen=True)
class _CacheMarkerStatus:
    is_first_run: bool = False
    is_generating: bool = False
    is_completed: bool = False
    has_existing_cache: bool = False


@dataclass(frozen=True)
class _CacheSnapshot:
    total_size_bytes: int = 0
    file_count: int = 0

    def has_content(self) -> bool:
        return self.file_count > 0


@dataclass(frozen=True)
class _CacheProgressStatus:
    last_activity_time: Optional[float] = None
    no_progress_duration: float = 0.0
    is_stalled: bool = False


@dataclass
class _DockerMarkerState:
    started: bool = False
    completed: bool = False


class _CacheProgressTracker:
    def __init__(self, stall_timeout: float):
        self._stall_timeout = stall_timeout
        self.reset()

    def reset(self):
        self._last_snapshot: Optional[_CacheSnapshot] = None
        self._last_activity_time: Optional[float] = None
        self._has_seen_content = False

    def update(
        self, snapshot: _CacheSnapshot, is_generating: bool
    ) -> _CacheProgressStatus:
        if not is_generating:
            self.reset()
            return _CacheProgressStatus()

        current_time = time.time()
        snapshot_changed = snapshot != self._last_snapshot
        if snapshot_changed:
            self._last_snapshot = snapshot
            if snapshot.has_content() or self._has_seen_content:
                self._last_activity_time = current_time
            if snapshot.has_content():
                self._has_seen_content = True
            return _CacheProgressStatus(last_activity_time=self._last_activity_time)

        if not self._has_seen_content or self._last_activity_time is None:
            return _CacheProgressStatus(last_activity_time=self._last_activity_time)

        no_progress_duration = current_time - self._last_activity_time
        return _CacheProgressStatus(
            last_activity_time=self._last_activity_time,
            no_progress_duration=no_progress_duration,
            is_stalled=no_progress_duration >= self._stall_timeout,
        )


class CacheMonitor:
    """Monitor TT_METAL_CACHE generation progress using file markers."""

    TT_METAL_CACHE_FIRST_RUN_STARTED = "TT_METAL_CACHE_FIRST_RUN_STARTED"
    TT_METAL_CACHE_COMPLETED = "TT_METAL_CACHE_COMPLETED"
    DEFAULT_TENSOR_CACHE_TIMEOUT = 1200.0
    TENSOR_CACHE_NO_CHANGE_TIMEOUT = 900

    def __init__(
        self,
        model_spec=None,
        cache_dir: Optional[Union[str, Path]] = None,
        runtime_config=None,
    ):
        self.model_spec = model_spec
        self.runtime_config = runtime_config
        self.tensor_cache_timeout = self._get_tensor_cache_timeout(model_spec)
        self.cache_target = CacheMonitorTarget()
        self._docker_markers = _DockerMarkerState()
        self._progress_tracker = _CacheProgressTracker(
            self.TENSOR_CACHE_NO_CHANGE_TIMEOUT
        )
        self._saw_empty_cache_without_markers = False
        self._inferred_generation_without_markers = False

        if model_spec is not None and not self._uses_tensor_model_cache(model_spec):
            logger.info(
                "🔍 Model %s does not use tensor cache - cache monitoring disabled",
                _get_value(model_spec, "model_name", "unknown"),
            )
            self.cache_dir = None
            return

        self.cache_target = detect_cache_monitor_target(
            model_spec=model_spec,
            cache_dir=cache_dir,
            runtime_config=runtime_config,
        )
        self.cache_dir = self.cache_target.get_display_cache_dir()

    def _uses_tensor_model_cache(self, model_spec) -> bool:
        return bool(
            _get_value(
                model_spec,
                "uses_tensor_model_cache",
                _get_value(model_spec, "uses_model_cache", True),
            )
        )

    @classmethod
    def _coerce_tensor_cache_timeout(cls, timeout_value) -> float:
        try:
            timeout = float(timeout_value)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid tensor_cache_timeout=%r; using default timeout=%ss",
                timeout_value,
                cls.DEFAULT_TENSOR_CACHE_TIMEOUT,
            )
            return cls.DEFAULT_TENSOR_CACHE_TIMEOUT

        if timeout <= 0:
            logger.warning(
                "Non-positive tensor_cache_timeout=%r; using default timeout=%ss",
                timeout_value,
                cls.DEFAULT_TENSOR_CACHE_TIMEOUT,
            )
            return cls.DEFAULT_TENSOR_CACHE_TIMEOUT

        return timeout

    def _get_tensor_cache_timeout(self, model_spec) -> float:
        if model_spec is None:
            return self.DEFAULT_TENSOR_CACHE_TIMEOUT

        device_model_spec = _get_value(model_spec, "device_model_spec")
        timeout_value = _get_value(
            device_model_spec,
            "tensor_cache_timeout",
            self.DEFAULT_TENSOR_CACHE_TIMEOUT,
        )
        return self._coerce_tensor_cache_timeout(timeout_value)

    def get_tensor_cache_timeout(self) -> float:
        return self.tensor_cache_timeout

    def get_effective_timeout(
        self,
        default_timeout: float,
        cache_status: Optional[CacheGenerationStatus] = None,
    ) -> float:
        if cache_status is None:
            cache_status = self.get_cache_generation_status()

        if cache_status.is_generating or cache_status.is_first_run or cache_status.has_existing_cache:
            return self.get_tensor_cache_timeout()

        return float(default_timeout)

    def _is_monitoring_enabled(self) -> bool:
        return self.cache_target.is_enabled()

    def _supports_local_marker_files(self) -> bool:
        return self.cache_target.cache_dir is not None

    def _uses_docker_cli_fallback(self) -> bool:
        return self.cache_target.uses_docker_cli()

    def _reset_observed_first_run(self):
        self._saw_empty_cache_without_markers = False
        self._inferred_generation_without_markers = False

    @staticmethod
    def _build_marker_status(
        started_exists: bool, completed_exists: bool
    ) -> _CacheMarkerStatus:
        if completed_exists:
            return _CacheMarkerStatus(is_completed=True, has_existing_cache=True)
        if started_exists:
            return _CacheMarkerStatus(is_generating=True)
        return _CacheMarkerStatus(is_first_run=True)

    def get_cache_marker_files(self) -> Tuple[Optional[Path], Optional[Path]]:
        """Get paths to cache marker files."""
        if not self._supports_local_marker_files():
            return None, None

        started_file = (
            self.cache_target.cache_dir / self.TT_METAL_CACHE_FIRST_RUN_STARTED
        )
        completed_file = self.cache_target.cache_dir / self.TT_METAL_CACHE_COMPLETED
        return started_file, completed_file

    def _write_marker_file(self, marker_path: Path, message: str) -> bool:
        try:
            logger.info("Creating cache directory: %s", marker_path.parent)
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            with open(marker_path, "w") as marker_file:
                marker_file.write(f"{message} at: {time.time()}\n")
                marker_file.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            logger.info("Created cache marker: %s", marker_path)
            return True
        except (OSError, PermissionError) as error:
            logger.warning("Failed to create cache marker %s: %s", marker_path, error)
            return False

    def mark_cache_first_run_started(self) -> bool:
        """Create CACHE_FIRST_RUN_STARTED marker file."""
        started_file, _ = self.get_cache_marker_files()
        if started_file is None:
            if not self._uses_docker_cli_fallback():
                logger.warning(
                    "Cannot create cache first run marker: no cache directory configured"
                )
                return False

            self._docker_markers.started = True
            self._docker_markers.completed = False
            self._reset_observed_first_run()
            return True

        return self._write_marker_file(started_file, "Cache first run started")

    def mark_cache_completed(self) -> bool:
        """Create CACHE_COMPLETED marker file."""
        _, completed_file = self.get_cache_marker_files()
        if completed_file is None:
            if self._uses_docker_cli_fallback():
                self._docker_markers.completed = True
                self._docker_markers.started = False
                self._progress_tracker.reset()
                self._reset_observed_first_run()
                return True
            return False

        success = self._write_marker_file(completed_file, "Cache completed")
        if success:
            self._progress_tracker.reset()
            self._reset_observed_first_run()
        return success

    def _read_marker_status(self) -> _CacheMarkerStatus:
        started_file, completed_file = self.get_cache_marker_files()
        if started_file is not None and completed_file is not None:
            started_exists = started_file.exists()
            completed_exists = completed_file.exists()
            logger.debug(
                "Cache marker status: started_exists=%s, completed_exists=%s",
                started_exists,
                completed_exists,
            )
            logger.debug("Started file: %s", started_file)
            logger.debug("Completed file: %s", completed_file)
            return self._build_marker_status(started_exists, completed_exists)

        if self._uses_docker_cli_fallback():
            return self._build_marker_status(
                self._docker_markers.started,
                self._docker_markers.completed,
            )

        logger.debug("No cache marker files configured")
        return _CacheMarkerStatus()

    def check_cache_status_from_markers(self) -> Tuple[bool, bool, bool]:
        """
        Check cache status from marker files.

        Returns:
            Tuple[bool, bool, bool]: (is_first_run, is_generating, is_completed)
        """
        marker_status = self._read_marker_status()
        return (
            marker_status.is_first_run,
            marker_status.is_generating,
            marker_status.is_completed,
        )

    def _get_path_cache_snapshot(self, cache_dir: Path) -> Tuple[int, int]:
        if not cache_dir.exists():
            return 0, 0

        total_size_bytes = 0
        file_count = 0
        marker_files = {
            self.TT_METAL_CACHE_FIRST_RUN_STARTED,
            self.TT_METAL_CACHE_COMPLETED,
        }

        try:
            for cache_file in cache_dir.rglob("*"):
                if not cache_file.is_file() or cache_file.name in marker_files:
                    continue
                try:
                    total_size_bytes += cache_file.stat().st_size
                    file_count += 1
                except (OSError, PermissionError):
                    logger.debug("Skipping unreadable cache file: %s", cache_file)
        except (OSError, PermissionError) as error:
            logger.warning("Failed to inspect cache directory content: %s", error)
            return 0, 0

        return total_size_bytes, file_count

    def _get_docker_cache_snapshot(self) -> Tuple[int, int]:
        docker_volume_name = self.cache_target.docker_volume_name
        docker_cache_dir = self.cache_target.docker_cache_dir
        helper_image = self.cache_target.docker_helper_image
        if not docker_volume_name or not docker_cache_dir or not helper_image:
            return 0, 0

        try:
            relative_cache_dir = docker_cache_dir.relative_to(
                DEFAULT_DOCKER_CACHE_MONITOR_ROOT
            )
        except ValueError:
            logger.warning(
                "Invalid Docker cache directory for cache monitoring: %s",
                docker_cache_dir,
            )
            return 0, 0

        command = [
            "docker",
            "run",
            "--rm",
            "--user",
            "0:0",
            "--volume",
            f"{docker_volume_name}:{DEFAULT_DOCKER_CACHE_MONITOR_ROOT}:ro",
            "--entrypoint",
            "python3",
            helper_image,
            "-c",
            DOCKER_VOLUME_SNAPSHOT_SCRIPT,
            relative_cache_dir.as_posix(),
            self.TT_METAL_CACHE_FIRST_RUN_STARTED,
            self.TT_METAL_CACHE_COMPLETED,
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except FileNotFoundError:
            logger.info(
                "Docker CLI is unavailable; Docker cache fallback is disabled for volume %s.",
                docker_volume_name,
            )
            return 0, 0
        except subprocess.TimeoutExpired:
            logger.info(
                "Timed out collecting Docker cache snapshot for volume %s.",
                docker_volume_name,
            )
            return 0, 0
        except Exception as exc:
            logger.warning(
                "Could not collect Docker cache snapshot for volume %s: %s",
                docker_volume_name,
                exc,
            )
            return 0, 0

        if result.returncode != 0:
            reason = str(getattr(result, "stderr", "") or "").strip()
            reason = reason or "docker run failed"
            logger.info(
                "Docker cache snapshot command failed for volume %s: %s",
                docker_volume_name,
                reason,
            )
            return 0, 0

        output = str(getattr(result, "stdout", "") or "").strip()
        if not output:
            return 0, 0

        try:
            snapshot = json.loads(output)
        except json.JSONDecodeError:
            logger.warning(
                "Could not parse Docker cache snapshot output for volume %s: %s",
                docker_volume_name,
                output,
            )
            return 0, 0

        return (
            int(snapshot.get("total_size_bytes", 0)),
            int(snapshot.get("file_count", 0)),
        )

    def _read_cache_snapshot(self) -> _CacheSnapshot:
        total_size_bytes, file_count = self._get_tensor_cache_snapshot()
        return _CacheSnapshot(
            total_size_bytes=total_size_bytes,
            file_count=file_count,
        )

    def _get_tensor_cache_snapshot(self) -> Tuple[int, int]:
        if self.cache_target.cache_dir is not None:
            return self._get_path_cache_snapshot(self.cache_target.cache_dir)
        if self._uses_docker_cli_fallback():
            return self._get_docker_cache_snapshot()
        return 0, 0

    def _infer_unmarked_cache_status(
        self, snapshot: _CacheSnapshot
    ) -> _CacheMarkerStatus:
        # Distinguish an existing cache from a cold start without writing markers.
        if snapshot.has_content():
            if (
                self._saw_empty_cache_without_markers
                or self._inferred_generation_without_markers
            ):
                self._inferred_generation_without_markers = True
                return _CacheMarkerStatus(is_generating=True)

            self._reset_observed_first_run()
            return _CacheMarkerStatus(has_existing_cache=True)

        self._saw_empty_cache_without_markers = True
        self._inferred_generation_without_markers = False
        return _CacheMarkerStatus(is_first_run=True)

    def _resolve_cache_lifecycle(
        self, marker_status: _CacheMarkerStatus, snapshot: _CacheSnapshot
    ) -> _CacheMarkerStatus:
        if marker_status.is_completed:
            self._reset_observed_first_run()
            return marker_status

        if marker_status.is_generating:
            return marker_status

        return self._infer_unmarked_cache_status(snapshot)

    def _build_generation_status(
        self,
        marker_status: _CacheMarkerStatus,
        snapshot: _CacheSnapshot,
        progress_status: _CacheProgressStatus,
    ) -> CacheGenerationStatus:
        status = CacheGenerationStatus(
            is_generating=marker_status.is_generating,
            cache_dir=self.cache_dir,
            container_id=None,
            last_activity_time=progress_status.last_activity_time,
            estimated_completion_time=None,
            is_first_run=marker_status.is_first_run,
            has_existing_cache=marker_status.has_existing_cache,
            is_stalled=progress_status.is_stalled,
            file_count=snapshot.file_count,
            total_size_bytes=snapshot.total_size_bytes,
            no_progress_duration=progress_status.no_progress_duration,
        )
        status.estimated_completion_time = self.estimate_cache_completion_time(status)
        return status

    def check_cache_directory_has_content(self) -> bool:
        """Check if cache directory has actual cache content."""
        if not self._is_monitoring_enabled():
            return False

        try:
            snapshot = self._read_cache_snapshot()
            logger.debug(
                "Found %s cache files in %s", snapshot.file_count, self.cache_dir
            )
            return snapshot.has_content()
        except (OSError, PermissionError) as error:
            logger.warning("Failed to check cache directory content: %s", error)
            return False

    def get_cache_generation_status(self) -> CacheGenerationStatus:
        """Get comprehensive cache generation status without mutating markers."""

        if not self._is_monitoring_enabled():
            logger.info("🔍 No cache directory configured - cache monitoring disabled")
            return CacheGenerationStatus(
                is_generating=False,
                cache_dir=None,
                container_id=None,
                last_activity_time=None,
            )

        snapshot = self._read_cache_snapshot()
        marker_status = self._resolve_cache_lifecycle(
            self._read_marker_status(),
            snapshot,
        )
        if marker_status.is_first_run:
            logger.info("🔍 No cache content found - treating as first run")

        progress_status = self._progress_tracker.update(
            snapshot,
            is_generating=marker_status.is_generating,
        )
        if progress_status.is_stalled:
            logger.error(
                "⛔ Tensor cache generation stalled in %s after %.1fs without file size change",
                self.cache_dir,
                progress_status.no_progress_duration,
            )

        return self._build_generation_status(
            marker_status,
            snapshot,
            progress_status,
        )

    def estimate_cache_completion_time(
        self, current_status: CacheGenerationStatus
    ) -> Optional[float]:
        """
        Estimate cache completion time based on historical data.

        This is a simple heuristic that can be improved later.
        """
        if not current_status.is_generating:
            return None

        base_estimate = 50 * 60
        return time.time() + base_estimate
