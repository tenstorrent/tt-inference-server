# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""
Device discovery: tt-smi for (1,1) and (2,4), test_system_health for Galaxy (2,1).
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

SYSTEM_HEALTH_BINARY_ENV = "TT_SYSTEM_HEALTH_BINARY"
TT_METAL_HOME_ENV = "TT_METAL_HOME"
TT_SMI_TIMEOUT = int(os.environ.get("TT_SMI_TIMEOUT", "30"))
SYSTEM_HEALTH_TIMEOUT = int(os.environ.get("TT_SYSTEM_HEALTH_TIMEOUT", "60"))
SYSTEM_HEALTH_RELATIVE_PATH = "build/test/tt_metal/tt_fabric/test_system_health"
SYSTEM_HEALTH_GTEST_FILTER = "Cluster.ReportSystemHealth"
N_PER_TRAY = 8
N_LOC_PAIRS = (("1", "2"), ("3", "4"), ("5", "6"), ("7", "8"))

CHIP_PATTERN = re.compile(
    r"Chip:\s+(\d+)\s+PCIe:\s+(\d+)\s+Unique ID:\s+(\w+)\s+Tray:\s+(\d+)\s+N(\d+)"
)


class DeviceDiscoveryError(RuntimeError):
    """Device discovery failed."""


@dataclass(frozen=True)
class ChipInfo:
    chip_id: str
    pcie_id: str
    unique_id: str
    tray: str
    n_loc: str


class DeviceManager:
    """
    Unified device discovery.

    - tt-smi: get_single_devices(), get_device_groups_of_eight()
    - test_system_health: get_device_pairs()
    """

    def __init__(self, system_health_binary: Optional[str] = None):
        self._system_health_binary = system_health_binary

    def get_single_devices(self) -> list[int]:
        """(1,1) mesh - flat list via tt-smi. Returns [] on discovery failure."""
        try:
            tray_mapping = self._run_tt_smi()
        except DeviceDiscoveryError as e:
            logger.warning("tt-smi discovery failed (will use env device_ids): %s", e)
            return []
        devices = self._create_single_devices(tray_mapping)
        logger.info("Single devices: %s", devices)
        return devices

    def get_device_pairs(self) -> list[tuple[int, int]]:
        """(2,1) Galaxy - N1-N2, N3-N4 pairs via test_system_health. Returns [] on failure."""
        try:
            chips = self._run_system_health()
            pairs = self._build_pairs_from_chips(chips)
            logger.info("Device pairs: %s", pairs)
            return pairs
        except DeviceDiscoveryError as e:
            logger.error("Galaxy device discovery failed: %s", e)
            return []

    def get_device_groups_of_eight(self) -> list[tuple[int, ...]]:
        """(2,4) mesh - groups of 8 via tt-smi. Returns [] on failure."""
        try:
            tray_mapping = self._run_tt_smi()
        except DeviceDiscoveryError as e:
            logger.error("tt-smi discovery failed: %s", e)
            return []
        try:
            groups = self._create_device_groups_of_eight(tray_mapping)
        except DeviceDiscoveryError as e:
            logger.error("%s", e)
            return []
        logger.info("Device groups of 8: %s", groups)
        return groups

    def _run_tt_smi(self) -> dict[int, list[int]]:
        """Run tt-smi, return tray -> [device_ids]. Raises DiscoveryError on failure."""
        try:
            result = subprocess.run(
                ["tt-smi", "-glx_list_tray_to_device"],
                capture_output=True,
                text=True,
                timeout=TT_SMI_TIMEOUT,
            )
        except FileNotFoundError:
            raise DeviceDiscoveryError("tt-smi not found in PATH")
        except subprocess.TimeoutExpired:
            raise DeviceDiscoveryError(f"tt-smi timed out after {TT_SMI_TIMEOUT}s")

        if result.returncode != 0:
            raise DeviceDiscoveryError(f"tt-smi failed: {result.stderr}")

        tray_mapping = self._parse_tt_smi_output(result.stdout or "")
        if not tray_mapping:
            raise DeviceDiscoveryError("tt-smi: no trays parsed")

        logger.info("Tray mapping: %s", tray_mapping)
        return tray_mapping

    @staticmethod
    def _parse_tt_smi_output(output: str) -> dict[int, list[int]]:
        """Parse tt-smi tray table."""
        tray_mapping: dict[int, list[int]] = {}
        for line in output.strip().split("\n"):
            if not line.strip().startswith("│") or not any(c.isdigit() for c in line):
                continue
            parts = [p.strip() for p in line.split("│") if p.strip()]
            if len(parts) >= 3:
                try:
                    tray_mapping[int(parts[0])] = [
                        int(d.strip()) for d in parts[2].split(",")
                    ]
                except (ValueError, IndexError):
                    continue
        return tray_mapping

    def get_tray_mapping_from_system(self) -> dict[int, list[int]]:
        """Run tt-smi and return tray mapping. Returns {} on failure."""
        try:
            return self._run_tt_smi()
        except Exception as e:
            logger.error("tt-smi discovery failed: %s", e)
            return {}

    @staticmethod
    def _create_single_devices(tray_mapping: dict[int, list[int]]) -> list[int]:
        """Flatten tray mapping to sorted list of device ids."""
        return [d for tray in sorted(tray_mapping) for d in sorted(tray_mapping[tray])]

    @staticmethod
    def _create_device_groups_of_eight(
        tray_mapping: dict[int, list[int]],
    ) -> list[tuple[int, ...]]:
        """Build one group of 8 per tray. Raises DiscoveryError if any tray has < 8."""
        groups: list[tuple[int, ...]] = []
        for tray in sorted(tray_mapping):
            ids = sorted(tray_mapping[tray])  # Maybe not mandatory to sort
            if len(ids) < N_PER_TRAY:
                raise DeviceDiscoveryError(
                    f"Tray {tray}: need {N_PER_TRAY} devices, got {len(ids)}"
                )
            groups.append(tuple(ids[:N_PER_TRAY]))
        return groups

    def _run_system_health(self) -> list[ChipInfo]:
        """Run test_system_health, return parsed chips. Raises DiscoveryError on failure."""
        binary = self._resolve_binary()
        result = self._exec_system_health(binary)

        output = result.stdout or ""
        if not output:
            raise DeviceDiscoveryError("test_system_health: no output")

        chips = self._parse_system_health_output(output)
        if not chips:
            raise DeviceDiscoveryError("test_system_health: no chips parsed")

        return chips

    def _exec_system_health(self, binary: str) -> subprocess.CompletedProcess:
        """Execute the binary, retrying once after chmod +x on permission failure."""
        cmd = [binary, f"--gtest_filter={SYSTEM_HEALTH_GTEST_FILTER}"]
        env = self._build_env()

        logger.info("Running test_system_health (~10-15s): %s", binary)
        self._flush_streams()

        start = time.monotonic()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=SYSTEM_HEALTH_TIMEOUT,
                env=env,
                start_new_session=True,
            )
        except subprocess.TimeoutExpired:
            raise DeviceDiscoveryError(
                f"test_system_health timed out after {SYSTEM_HEALTH_TIMEOUT}s"
            )
        except OSError as e:
            result = self._retry_after_chmod(binary, cmd, env, e)

        logger.info(
            "Completed in %.1fs, rc=%s",
            time.monotonic() - start,
            result.returncode,
        )
        return result

    def _retry_after_chmod(
        self,
        binary: str,
        cmd: list[str],
        env: dict,
        original_error: OSError,
    ) -> subprocess.CompletedProcess:
        """chmod +x the binary and retry once. Raises DeviceDiscoveryError if retry also fails."""
        logger.warning(
            "test_system_health exec failed: %s — attempting chmod +x %s",
            original_error,
            binary,
        )
        try:
            os.chmod(binary, os.stat(binary).st_mode | 0o111)
        except OSError as chmod_err:
            raise DeviceDiscoveryError(
                f"test_system_health not executable and chmod +x failed: {chmod_err}"
            ) from original_error

        try:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=SYSTEM_HEALTH_TIMEOUT,
                env=env,
                start_new_session=True,
            )
        except subprocess.TimeoutExpired:
            raise DeviceDiscoveryError(
                f"test_system_health timed out after {SYSTEM_HEALTH_TIMEOUT}s (after chmod +x)"
            )
        except OSError as e:
            raise DeviceDiscoveryError(
                f"test_system_health failed even after chmod +x: {e}"
            ) from original_error

    def _resolve_binary(self) -> str:
        """Resolve test_system_health binary path."""
        if self._system_health_binary:
            return self._system_health_binary
        env_binary = os.environ.get(SYSTEM_HEALTH_BINARY_ENV)
        if env_binary:
            return env_binary
        home = os.environ.get(TT_METAL_HOME_ENV)
        if home:
            return os.path.join(home, SYSTEM_HEALTH_RELATIVE_PATH)
        return os.path.join(".", SYSTEM_HEALTH_RELATIVE_PATH)

    @staticmethod
    def _build_env() -> dict[str, str]:
        """Build subprocess environment."""
        env = os.environ.copy()
        home = os.environ.get(TT_METAL_HOME_ENV)
        if home:
            env["TT_METAL_RUNTIME_ROOT"] = home
        return env

    @staticmethod
    def _flush_streams() -> None:
        """Flush streams before subprocess."""
        sys.stdout.flush()
        sys.stderr.flush()
        for handler in logging.getLogger().handlers:
            handler.flush()

    @staticmethod
    def _parse_system_health_output(output: str) -> list[ChipInfo]:
        """Parse test_system_health output."""
        chips: list[ChipInfo] = []
        for line in output.split("\n"):
            if "PCIe:" not in line:
                continue
            match = CHIP_PATTERN.search(line)
            if match is None:
                continue
            chips.append(
                ChipInfo(
                    chip_id=match.group(1),
                    pcie_id=match.group(2),
                    unique_id=match.group(3),
                    tray=match.group(4),
                    n_loc=match.group(5),
                )
            )
        return chips

    @staticmethod
    def _build_pairs_from_chips(chips: list[ChipInfo]) -> list[tuple[int, int]]:
        """Build N1-N2, N3-N4, N5-N6, N7-N8 pairs per tray."""
        tray_map: dict[str, dict[str, str]] = {}
        for c in chips:
            tray_map.setdefault(c.tray, {})[c.n_loc] = c.chip_id

        pairs: list[tuple[int, int]] = []
        for tray in sorted(tray_map, key=int):
            n_devices = tray_map[tray]
            for n1, n2 in N_LOC_PAIRS:
                dev1 = n_devices.get(n1)
                dev2 = n_devices.get(n2)
                if dev1 and dev2:
                    pairs.append((int(dev1), int(dev2)))
        return pairs
