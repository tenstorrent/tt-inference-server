# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""
Galaxy device discovery via test_system_health binary.

Runs test_system_health and parses the output to build device mapping.
Primary use: mesh (2, 1) — device pairs via get_device_pairs_from_discovery().
Tray mapping and groups-of-8 are used for (1, 1) and (2, 4) via DeviceManager for now.

C++ binary source (tt-metal):
  tests/tt_metal/tt_fabric/system_health/test_system_health.cpp
  TEST(Cluster, ReportSystemHealth) uses cluster.get_ethernet_connections().
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

# Binary and env
ENV_TEST_BINARY = "TT_SYSTEM_HEALTH_BINARY"
ENV_TT_METAL_HOME = "TT_METAL_HOME"
RELATIVE_TEST_BINARY_PATH = "build/test/tt_metal/tt_fabric/test_system_health"
GTEST_FILTER = "Cluster.ReportSystemHealth"
RUN_TIMEOUT_SEC = 60

# Parsing
CHIP_LINE_PATTERN = re.compile(
    r"Chip:\s+(\d+)\s+PCIe:\s+(\d+)\s+Unique ID:\s+(\w+)\s+Tray:\s+(\d+)\s+N(\d+)"
)
N_LOC_PAIRS = [("1", "2"), ("3", "4"), ("5", "6"), ("7", "8")]
MAX_N_LOC = 8


class GalaxyDiscoveryError(RuntimeError):
    """Raised when discovery binary fails or output cannot be parsed."""


@dataclass(frozen=True)
class ChipInfo:
    """Parsed chip line from test_system_health output."""

    chip_id: str
    pcie_id: str
    unique_id: str
    tray: str
    n_loc: str

    @property
    def tray_int(self) -> int:
        return int(self.tray)

    @property
    def n_loc_int(self) -> int:
        return int(self.n_loc)


def _resolve_binary(test_binary: Optional[str] = None) -> str:
    """Resolve test_system_health binary path: explicit > TT_SYSTEM_HEALTH_BINARY > TT_METAL_HOME/rel > ./rel."""
    if test_binary:
        return test_binary
    env_binary = os.environ.get(ENV_TEST_BINARY)
    if env_binary:
        return env_binary
    root = os.environ.get(ENV_TT_METAL_HOME)
    if root:
        return os.path.join(root, RELATIVE_TEST_BINARY_PATH)
    return os.path.join(".", RELATIVE_TEST_BINARY_PATH)


def _flush_logging() -> None:
    """Flush logging and stdio before subprocess that might kill the process."""
    for handler in logging.getLogger().handlers:
        handler.flush()
    sys.stdout.flush()
    sys.stderr.flush()


def _build_subprocess_env() -> dict[str, str]:
    """Build env for test_system_health; set TT_METAL_RUNTIME_ROOT when TT_METAL_HOME is set."""
    env = os.environ.copy()
    if os.environ.get(ENV_TT_METAL_HOME):
        env["TT_METAL_RUNTIME_ROOT"] = os.environ[ENV_TT_METAL_HOME]
    return env


def _run_discovery_raw(binary: str) -> str:
    """Run test_system_health binary and return stdout. Returns empty string on timeout or exception."""
    cmd = [binary, f"--gtest_filter={GTEST_FILTER}"]
    env = _build_subprocess_env()
    logger.info(
        "Running test_system_health (cluster discovery, ~10-15s): %s",
        binary,
    )
    _flush_logging()
    start = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=RUN_TIMEOUT_SEC,
            env=env,
            start_new_session=True,
        )
        elapsed = time.monotonic() - start
        logger.info(
            "test_system_health completed in %.1fs, returncode=%s",
            elapsed,
            result.returncode,
        )
        stdout = result.stdout or ""
        return stdout
    except subprocess.TimeoutExpired:
        if sys.stderr:
            print("Error: Timeout while running test_system_health", file=sys.stderr)
        return ""
    except Exception as e:
        if sys.stderr:
            print(f"Error running test_system_health: {e}", file=sys.stderr)
        return ""


def run_test_system_health(test_binary: str) -> str:
    """Run the test_system_health binary and capture stdout. Public alias for _run_discovery_raw."""
    return _run_discovery_raw(test_binary)


def parse_chip_line(line: str) -> Optional[ChipInfo]:
    """Parse a chip info line. Example: 'Chip: 31 PCIe: 15 ... Tray: 2 N8'. Returns None if no match."""
    match = CHIP_LINE_PATTERN.search(line)
    if not match:
        return None
    return ChipInfo(
        chip_id=match.group(1),
        pcie_id=match.group(2),
        unique_id=match.group(3),
        tray=match.group(4),
        n_loc=match.group(5),
    )


def create_device_mapping(output: str) -> list[ChipInfo]:
    """Build list of ChipInfo from test_system_health stdout (lines containing 'PCIe:')."""
    chip_lines = [line for line in output.split("\n") if "PCIe:" in line]
    results: list[ChipInfo] = []
    for line in chip_lines:
        info = parse_chip_line(line)
        if info is not None:
            results.append(info)
    return results


def _validate_non_empty_results(
    results: list[ChipInfo],
    context: str = "Galaxy device discovery",
) -> None:
    """Raise GalaxyDiscoveryError if results are empty."""
    if not results:
        raise GalaxyDiscoveryError(
            f"{context}: no chip lines parsed from test_system_health"
        )


def _run_discovery_and_parse(
    test_binary: Optional[str] = None,
) -> list[ChipInfo]:
    """Resolve binary, run discovery, parse output. Raises GalaxyDiscoveryError on empty output or no chips."""
    binary = _resolve_binary(test_binary)
    output = _run_discovery_raw(binary)
    if not output:
        raise GalaxyDiscoveryError(
            "Galaxy device discovery: no output from test_system_health"
        )
    results = create_device_mapping(output)
    _validate_non_empty_results(results)
    return results


def _build_tray_mapping(chips: list[ChipInfo]) -> dict[int, list[int]]:
    """Build tray -> [N1..N8] pcie_ids from parsed chips (pcie_id order by n_loc 1..8)."""
    tray_to_n: dict[int, dict[int, int]] = {}
    for c in chips:
        tray = c.tray_int
        n_loc = c.n_loc_int
        pcie_id = int(c.pcie_id)
        if tray not in tray_to_n:
            tray_to_n[tray] = {}
        tray_to_n[tray][n_loc] = pcie_id
    return {
        tray: [n_to_id[i] for i in range(1, MAX_N_LOC + 1) if i in n_to_id]
        for tray, n_to_id in tray_to_n.items()
    }


def get_tray_mapping_from_discovery(
    test_binary: Optional[str] = None,
) -> dict[int, list[int]]:
    """
    Run test_system_health and return tray -> [N1..N8] pcie_ids.
    Same format as DeviceManager.get_tray_mapping_from_system().
    """
    chips = _run_discovery_and_parse(test_binary)
    return _build_tray_mapping(chips)


def _build_pairs_from_chips(chips: list[ChipInfo]) -> list[tuple[int, int]]:
    """Extract (chip_id, chip_id) pairs N1-N2, N3-N4, N5-N6, N7-N8 per tray for TT_VISIBLE_DEVICES."""
    tray_map: dict[str, dict[str, str]] = {}
    for c in chips:
        if c.tray not in tray_map:
            tray_map[c.tray] = {}
        tray_map[c.tray][c.n_loc] = c.chip_id
    pairs: list[tuple[int, int]] = []
    for tray in sorted(tray_map.keys(), key=int):
        n_devices = tray_map[tray]
        for n1, n2 in N_LOC_PAIRS:
            dev1 = n_devices.get(n1)
            dev2 = n_devices.get(n2)
            if dev1 and dev2:
                pairs.append((int(dev1), int(dev2)))
    return pairs


def get_device_pairs(results: list[ChipInfo]) -> list[tuple[int, int]]:
    """Extract (chip_id, chip_id) pairs (N1-N2, N3-N4, N5-N6, N7-N8) per tray. Public for tests."""
    return _build_pairs_from_chips(results)


def get_device_pairs_from_discovery(
    test_binary: Optional[str] = None,
) -> list[tuple[int, int]]:
    """
    Run test_system_health and return list of (chip_id1, chip_id2) pairs
    in N1-N2, N3-N4, N5-N6, N7-N8 order per tray for TT_VISIBLE_DEVICES.
    """
    binary = _resolve_binary(test_binary)
    logger.info("get_device_pairs_from_discovery: binary=%s", binary)
    _flush_logging()
    chips = _run_discovery_and_parse(test_binary)
    pairs = _build_pairs_from_chips(chips)
    logger.info("GALAXY_DISCOVERY_PAIRS=%s", pairs)
    return pairs


def get_device_groups_of_eight_from_discovery(
    test_binary: Optional[str] = None,
) -> list[tuple[int, ...]]:
    """
    Run test_system_health and return list of 8-tuples (N1..N8) per tray
    in correct wiring order, one group per tray.
    """
    mapping = get_tray_mapping_from_discovery(test_binary)
    out: list[tuple[int, ...]] = []
    for tray in sorted(mapping.keys()):
        ids = mapping[tray]
        if len(ids) < MAX_N_LOC:
            raise GalaxyDiscoveryError(
                f"Galaxy device discovery: tray {tray} has {len(ids)} devices, need {MAX_N_LOC}"
            )
        out.append(tuple(ids[:MAX_N_LOC]))
    return out


def print_device_mapping(results: list[ChipInfo]) -> None:
    """Print device mapping by tray (N1-N2, N3-N4, etc.)."""
    if not results:
        print("No results found")
        return
    tray_map: dict[str, dict[str, str]] = {}
    for c in results:
        if c.tray not in tray_map:
            tray_map[c.tray] = {}
        tray_map[c.tray][c.n_loc] = c.pcie_id
    print("\nDevice Mapping by Tray:")
    print("=" * 80)
    for tray in sorted(tray_map.keys(), key=int):
        print(f"\nTray {tray}:")
        print("-" * 80)
        n_devices = tray_map[tray]
        for n1, n2 in N_LOC_PAIRS:
            dev1 = n_devices.get(n1, "N/A")
            dev2 = n_devices.get(n2, "N/A")
            print(f"  N{n1}-N{n2}: ({dev1},{dev2})")
    print("\n" + "=" * 80)
    print(f"Total entries: {len(results)}")
