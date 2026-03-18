# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import time
import logging
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CacheGenerationStatus:
    """Status of cache generation process"""

    is_generating: bool
    cache_dir: Optional[Path] = None
    container_id: Optional[str] = None
    last_activity_time: Optional[float] = None
    estimated_completion_time: Optional[float] = None
    is_first_run: bool = False
    is_stalled: bool = False
    file_count: int = 0
    total_size_bytes: int = 0
    no_progress_duration: float = 0.0


class CacheMonitor:
    """Monitor TT_METAL_CACHE generation progress using file markers on host filesystem"""

    TT_METAL_CACHE_FIRST_RUN_STARTED = "TT_METAL_CACHE_FIRST_RUN_STARTED"
    TT_METAL_CACHE_COMPLETED = "TT_METAL_CACHE_COMPLETED"
    DEFAULT_TENSOR_CACHE_TIMEOUT = 1200.0
    TENSOR_CACHE_NO_CHANGE_TIMEOUT = 180

    def __init__(self, model_spec=None, cache_dir: Optional[Path] = None):
        self.model_spec = model_spec
        self.tensor_cache_timeout = self._get_tensor_cache_timeout(model_spec)
        self._last_progress_time: Optional[float] = None
        self._last_cache_size_bytes: Optional[int] = None
        self._last_cache_file_count: Optional[int] = None

        if cache_dir is not None:
            self.cache_dir = cache_dir
        elif model_spec is not None and self._uses_tensor_model_cache(model_spec):
            self.cache_dir = self._detect_cache_directory(model_spec)
        elif model_spec is not None and not self._uses_tensor_model_cache(model_spec):
            logger.info(
                f"🔍 Model {getattr(model_spec, 'model_name', 'unknown')} does not use tensor cache - cache monitoring disabled"
            )
            self.cache_dir = None
        else:
            self.cache_dir = cache_dir

    def _uses_tensor_model_cache(self, model_spec) -> bool:
        return bool(
            getattr(
                model_spec,
                "uses_tensor_model_cache",
                getattr(model_spec, "uses_model_cache", True),
            )
        )

    def _get_tensor_cache_timeout(self, model_spec) -> float:
        if model_spec is None:
            return self.DEFAULT_TENSOR_CACHE_TIMEOUT

        device_model_spec = getattr(model_spec, "device_model_spec", None)
        timeout = getattr(
            device_model_spec, "tensor_cache_timeout", self.DEFAULT_TENSOR_CACHE_TIMEOUT
        )
        return float(timeout)

    def get_tensor_cache_timeout(self) -> float:
        return self.tensor_cache_timeout

    def get_effective_timeout(
        self,
        default_timeout: float,
        cache_status: Optional[CacheGenerationStatus] = None,
    ) -> float:
        if cache_status is None:
            cache_status = self.get_cache_generation_status()

        if cache_status.is_generating:
            return self.get_tensor_cache_timeout()

        return float(default_timeout)

    def _detect_cache_directory(self, model_spec):
        """
        Detect cache directory for cache monitoring.
        Uses the correct persistent volume structure with device-specific paths.

        Returns:
            Optional[Path]: cache_dir
        """
        cache_dir = None

        try:
            # Import here to avoid circular imports
            from workflows.setup_host import SetupConfig
            from workflows.workflow_types import DeviceTypes

            # Try to create a SetupConfig to get cache directory info
            setup_config = SetupConfig(model_spec=model_spec)

            device_str = model_spec.device_type.name.lower()
            device = DeviceTypes.from_string(device_str)
            mesh_device_str = device.to_mesh_device_str()
            device_cache_str = (
                DeviceTypes.to_mesh_device_str(model_spec.subdevice_type)
                if model_spec.subdevice_type
                else mesh_device_str
            )

            # Build the full cache path: .../tt_metal_cache/cache_{model_name}/{device_str}
            base_cache_dir = setup_config.host_tt_metal_cache_dir
            cache_dir = base_cache_dir / device_cache_str

            logger.info(f"Detected cache directory: {cache_dir}")
            logger.info(f"Device cache string: {device_cache_str}")

        except Exception as e:
            logger.warning(f"Could not detect cache directory from SetupConfig: {e}")

        return cache_dir

    def get_cache_marker_files(self) -> Tuple[Optional[Path], Optional[Path]]:
        """Get paths to cache marker files"""
        if not self.cache_dir:
            return None, None

        started_file = self.cache_dir / self.TT_METAL_CACHE_FIRST_RUN_STARTED
        completed_file = self.cache_dir / self.TT_METAL_CACHE_COMPLETED

        return started_file, completed_file

    def mark_cache_first_run_started(self) -> bool:
        """Create CACHE_FIRST_RUN_STARTED marker file"""
        started_file, _ = self.get_cache_marker_files()
        if not started_file:
            logger.warning(
                "Cannot create cache first run marker: no cache directory configured"
            )
            return False

        try:
            # Ensure cache directory exists
            logger.info(f"Creating cache directory: {started_file.parent}")
            started_file.parent.mkdir(parents=True, exist_ok=True)

            # Create marker file with timestamp
            with open(started_file, "w") as f:
                f.write(f"Cache first run started at: {time.time()}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

            logger.info(f"✅ Created cache first run marker: {started_file}")
            return True

        except (OSError, PermissionError) as e:
            logger.error(f"❌ Failed to create cache first run marker: {e}")
            logger.error(f"   Cache directory: {started_file.parent}")
            logger.error(f"   Marker file: {started_file}")
            return False

    def mark_cache_completed(self) -> bool:
        """Create CACHE_COMPLETED marker file"""
        _, completed_file = self.get_cache_marker_files()
        if not completed_file:
            return False

        try:
            # Ensure cache directory exists
            completed_file.parent.mkdir(parents=True, exist_ok=True)

            # Create marker file with timestamp
            with open(completed_file, "w") as f:
                f.write(f"Cache completed at: {time.time()}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

            logger.info(f"Created cache completed marker: {completed_file}")
            return True

        except (OSError, PermissionError) as e:
            logger.warning(f"Failed to create cache completed marker: {e}")
            return False

    def check_cache_status_from_markers(self) -> Tuple[bool, bool, bool]:
        """
        Check cache status from marker files

        Returns:
            Tuple[bool, bool, bool]: (is_first_run, is_generating, is_completed)
        """
        started_file, completed_file = self.get_cache_marker_files()

        if not started_file or not completed_file:
            logger.debug("No cache marker files configured")
            return False, False, False

        started_exists = started_file.exists()
        completed_exists = completed_file.exists()

        logger.debug(
            f"Cache marker status: started_exists={started_exists}, completed_exists={completed_exists}"
        )
        logger.debug(f"Started file: {started_file}")
        logger.debug(f"Completed file: {completed_file}")

        # Markers have strict precedence:
        # completed -> done
        # started without completed -> generation in progress
        # no markers -> determine first-run using cache content
        if completed_exists:
            return False, False, True
        if started_exists:
            return False, True, False

        return True, False, False

    def _get_tensor_cache_snapshot(self) -> Tuple[int, int]:
        if not self.cache_dir or not self.cache_dir.exists():
            return 0, 0

        total_size_bytes = 0
        file_count = 0
        marker_files = {
            self.TT_METAL_CACHE_FIRST_RUN_STARTED,
            self.TT_METAL_CACHE_COMPLETED,
        }

        try:
            for cache_file in self.cache_dir.rglob("*"):
                if not cache_file.is_file() or cache_file.name in marker_files:
                    continue

                try:
                    total_size_bytes += cache_file.stat().st_size
                    file_count += 1
                except (OSError, PermissionError):
                    logger.debug(f"Skipping unreadable cache file: {cache_file}")
        except (OSError, PermissionError) as e:
            logger.warning(f"Failed to inspect cache directory content: {e}")
            return 0, 0

        return total_size_bytes, file_count

    def _reset_progress_tracking(self):
        self._last_progress_time = None
        self._last_cache_size_bytes = None
        self._last_cache_file_count = None

    def _get_progress_state(
        self, is_generating: bool
    ) -> Tuple[int, int, Optional[float], float, bool]:
        if not is_generating:
            self._reset_progress_tracking()
            return 0, 0, None, 0.0, False

        current_time = time.time()
        total_size_bytes, file_count = self._get_tensor_cache_snapshot()
        cache_changed = (
            self._last_cache_size_bytes is None
            or self._last_cache_file_count is None
            or total_size_bytes != self._last_cache_size_bytes
            or file_count != self._last_cache_file_count
        )

        if cache_changed:
            self._last_cache_size_bytes = total_size_bytes
            self._last_cache_file_count = file_count
            self._last_progress_time = current_time
            return total_size_bytes, file_count, self._last_progress_time, 0.0, False

        no_progress_duration = current_time - self._last_progress_time
        is_stalled = no_progress_duration >= self.TENSOR_CACHE_NO_CHANGE_TIMEOUT
        return (
            total_size_bytes,
            file_count,
            self._last_progress_time,
            no_progress_duration,
            is_stalled,
        )

    def check_cache_directory_has_content(self) -> bool:
        """Check if cache directory has actual cache content (not just empty directories)"""
        if not self.cache_dir or not self.cache_dir.exists():
            return False

        try:
            _, file_count = self._get_tensor_cache_snapshot()
            logger.debug(f"Found {file_count} cache files in {self.cache_dir}")
            return file_count > 0
        except (OSError, PermissionError) as e:
            logger.warning(f"Failed to check cache directory content: {e}")
            return False

    def get_cache_generation_status(self) -> CacheGenerationStatus:
        """Get comprehensive cache generation status using file markers"""

        # If no cache directory is configured, return non-generating status
        if not self.cache_dir:
            logger.info("🔍 No cache directory configured - cache monitoring disabled")
            return CacheGenerationStatus(
                is_generating=False,
                cache_dir=None,
                container_id=None,
                last_activity_time=None,
            )

        # Check cache status from marker files (primary method)
        is_first_run, is_generating_from_markers, is_completed_from_markers = (
            self.check_cache_status_from_markers()
        )

        # If there are no marker files, inspect the tensor cache directory directly.
        if not is_generating_from_markers and not is_completed_from_markers:
            has_cache_content = self.check_cache_directory_has_content()
            is_first_run = not has_cache_content
            if is_first_run:
                logger.info("🔍 No cache content found - treating as first run")

        # If this is a first run and no started marker exists, create it
        if is_first_run and self.cache_dir:
            success = self.mark_cache_first_run_started()
            if success:
                is_generating_from_markers = True

        # Determine final status based on marker files and cache content
        if is_completed_from_markers:
            is_generating = False
        elif is_generating_from_markers:
            is_generating = True
        else:
            # If no markers, assume first run if no cache content
            is_generating = is_first_run

        (
            total_size_bytes,
            file_count,
            last_progress_time,
            no_progress_duration,
            is_stalled,
        ) = self._get_progress_state(is_generating=is_generating)

        if is_stalled:
            logger.error(
                "⛔ Tensor cache generation stalled in %s after %.1fs without file size change",
                self.cache_dir,
                no_progress_duration,
            )

        status = CacheGenerationStatus(
            is_generating=is_generating,
            cache_dir=self.cache_dir,
            container_id=None,
            last_activity_time=last_progress_time,
            is_first_run=is_first_run,
            is_stalled=is_stalled,
            file_count=file_count,
            total_size_bytes=total_size_bytes,
            no_progress_duration=no_progress_duration,
        )
        status.estimated_completion_time = self.estimate_cache_completion_time(status)
        return status

    def estimate_cache_completion_time(
        self, current_status: CacheGenerationStatus
    ) -> Optional[float]:
        """
        Estimate cache completion time based on historical data
        This is a simple heuristic - could be improved with more sophisticated modeling
        """
        if not current_status.is_generating:
            return None

        # Simple heuristic: if cache is actively being generated,
        # estimate 40-60 minutes total time (as mentioned in docs)
        # This could be made more sophisticated by tracking progress
        base_estimate = 50 * 60  # 50 minutes in seconds

        return time.time() + base_estimate
