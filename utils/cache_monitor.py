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


class CacheMonitor:
    """Monitor TT_METAL_CACHE generation progress using file markers on host filesystem"""

    TT_METAL_CACHE_FIRST_RUN_STARTED = "TT_METAL_CACHE_FIRST_RUN_STARTED"
    TT_METAL_CACHE_COMPLETED = "TT_METAL_CACHE_COMPLETED"

    def __init__(self, model_spec=None, cache_dir: Optional[Path] = None):
        if model_spec is not None and getattr(model_spec, "uses_model_cache", True):
            self.cache_dir = self._detect_cache_directory(model_spec)
        elif model_spec is not None and not getattr(
            model_spec, "uses_model_cache", True
        ):
            logger.info(
                f"🔍 Model {getattr(model_spec, 'model_name', 'unknown')} does not use model cache - cache monitoring disabled"
            )
            self.cache_dir = None
        else:
            self.cache_dir = cache_dir

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

            device_str = model_spec.cli_args.get("device")
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

        # Determine status based on marker files
        is_first_run = not started_exists
        is_generating = started_exists and not completed_exists
        is_completed = completed_exists

        return is_first_run, is_generating, is_completed

    def check_cache_directory_has_content(self) -> bool:
        """Check if cache directory has actual cache content (not just empty directories)"""
        if not self.cache_dir or not self.cache_dir.exists():
            return False

        try:
            # Look for actual cache files (not just directories)
            # Cache files typically have extensions like .bin, .cache, etc.
            cache_files = list(self.cache_dir.rglob("*"))
            actual_files = [
                f
                for f in cache_files
                if f.is_file() and not f.name.startswith("CACHE_")
            ]

            logger.debug(f"Found {len(actual_files)} cache files in {self.cache_dir}")
            return len(actual_files) > 0

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

        # Fallback: if no marker files but cache directory exists, check for actual cache content
        if not is_generating_from_markers and not is_completed_from_markers:
            has_cache_content = self.check_cache_directory_has_content()
            if not has_cache_content:
                # No cache content found, this is likely a first run
                is_first_run = True
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

        return CacheGenerationStatus(
            is_generating=is_generating,
            cache_dir=self.cache_dir,
            container_id=None,
            last_activity_time=time.time() if is_generating else None,
        )

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
