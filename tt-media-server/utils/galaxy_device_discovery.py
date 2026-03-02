# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""
Galaxy device discovery via test_system_health.
Runs test_system_health and parses the output to create a device mapping.

C++ binary source (tt-metal):
  tests/tt_metal/tt_fabric/system_health/test_system_health.cpp
  TEST(Cluster, ReportSystemHealth) uses cluster.get_ethernet_connections()
  (same data as PhysicalSystemDescriptor::run_local_discovery for topology mapper).
Diagnostic: scripts/diagnose_pair_topology.py runs discovery for one pair and
  reports ethernet connectivity vs n300 MGD expectation.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import time

logger = logging.getLogger(__name__)


def _flush_logging() -> None:
    """Flush logging and stdio so logs are visible before a subprocess that might kill the process."""
    root = logging.getLogger()
    for h in root.handlers:
        h.flush()
    sys.stdout.flush()
    sys.stderr.flush()


def _default_test_binary() -> str:
    if os.environ.get("TT_METAL_HOME"):
        return os.path.join(
            os.environ["TT_METAL_HOME"],
            "build/test/tt_metal/tt_fabric/test_system_health",
        )
    return "./build/test/tt_metal/tt_fabric/test_system_health"


DEFAULT_TEST_BINARY = _default_test_binary()
ENV_TEST_BINARY = "TT_SYSTEM_HEALTH_BINARY"
RUN_TIMEOUT_SEC = 60
CHIP_LINE_PATTERN = re.compile(
    r"Chip:\s+(\d+)\s+PCIe:\s+(\d+)\s+Unique ID:\s+(\w+)\s+Tray:\s+(\d+)\s+N(\d+)"
)
N_PAIRS = [("1", "2"), ("3", "4"), ("5", "6"), ("7", "8")]


def run_test_system_health(test_binary: str) -> str:
    """Run the test_system_health binary and capture output."""
    cmd = [test_binary, "--gtest_filter=Cluster.ReportSystemHealth"]
    env = os.environ.copy()
    if os.environ.get("TT_METAL_HOME"):
        env["TT_METAL_RUNTIME_ROOT"] = os.environ["TT_METAL_HOME"]
    logger.info("Running test_system_health binary (cluster discovery, ~10-15s): %s", test_binary)
    _flush_logging()
    start = time.monotonic()
    try:
        # New session so a fatal crash/signal in the binary is less likely to kill this process
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=RUN_TIMEOUT_SEC,
            env=env,
            start_new_session=True,
        )
        elapsed = time.monotonic() - start
        logger.info("test_system_health completed in %.1fs, returncode=%s", elapsed, result.returncode)
        stdout = result.stdout or ""
        logger.info("GALAXY_DISCOVERY_RAW_LEN=%s", len(stdout))
        logger.info("GALAXY_DISCOVERY_RAW_SAMPLE=%s", stdout[:600] if stdout else "(empty)")
        return stdout
    except subprocess.TimeoutExpired:
        if sys.stderr:
            print("Error: Timeout while running test", file=sys.stderr)
        return ""
    except Exception as e:
        if sys.stderr:
            print(f"Error running test: {e}", file=sys.stderr)
        return ""


def parse_chip_line(line: str) -> dict[str, str]:
    """Parse a chip info line. Example: 'Chip: 31 PCIe: 15 ... Tray: 2 N8'. Returns empty dict if no match."""
    match = CHIP_LINE_PATTERN.search(line)
    if not match:
        return {}
    return {
        "chip": match.group(1),
        "pcie_id": match.group(2),
        "unique_id": match.group(3),
        "tray": match.group(4),
        "n_loc": match.group(5),
    }


def create_device_mapping(output: str) -> list[dict[str, str]]:
    """Create device mapping from test_system_health output."""
    chip_lines = [line for line in output.split("\n") if "PCIe:" in line]
    results = []
    for line in chip_lines:
        info = parse_chip_line(line)
        if info:
            results.append(
                {
                    "chip": info["chip"],
                    "pcie_id": info["pcie_id"],
                    "unique_id": info["unique_id"],
                    "tray": info["tray"],
                    "n_loc": info["n_loc"],
                }
            )
    return results


def get_tray_mapping_from_discovery(
    test_binary: str | None = None,
) -> dict[int, list[int]]:
    """
    Run test_system_health and return tray -> [N1, N2, N3, N4, N5, N6, N7, N8] pcie_ids.
    Same format as DeviceManager.get_tray_mapping_from_system().
    """
    binary = test_binary or os.environ.get(ENV_TEST_BINARY, DEFAULT_TEST_BINARY)
    output = run_test_system_health(binary)
    if not output:
        raise RuntimeError("Galaxy device discovery: no output from test_system_health")
    results = create_device_mapping(output)
    if not results:
        raise RuntimeError(
            "Galaxy device discovery: no chip lines parsed from test_system_health"
        )
    tray_to_n_to_id: dict[int, dict[int, int]] = {}
    for r in results:
        tray = int(r["tray"])
        n_loc = int(r["n_loc"])
        pcie_id = int(r["pcie_id"])
        if tray not in tray_to_n_to_id:
            tray_to_n_to_id[tray] = {}
        tray_to_n_to_id[tray][n_loc] = pcie_id
    return {
        tray: [n_to_id[i] for i in range(1, 9) if i in n_to_id]
        for tray, n_to_id in tray_to_n_to_id.items()
    }


def get_device_pairs(results: list[dict[str, str]]) -> list[tuple[str, str]]:
    """Extract (chip_id, chip_id) pairs (N1-N2, N3-N4, N5-N6, N7-N8) per tray for TT_VISIBLE_DEVICES."""
    tray_map: dict[str, dict[str, str]] = {}
    for result in results:
        tray = result["tray"]
        if tray not in tray_map:
            tray_map[tray] = {}
        tray_map[tray][result["n_loc"]] = result["chip"]
    pairs_list: list[tuple[str, str]] = []
    for tray in sorted(tray_map.keys(), key=int):
        n_devices = tray_map[tray]
        for n1, n2 in N_PAIRS:
            dev1 = n_devices.get(n1)
            dev2 = n_devices.get(n2)
            if dev1 and dev2:
                pairs_list.append((dev1, dev2))
    return pairs_list


def get_device_pairs_from_discovery(
    test_binary: str | None = None,
) -> list[tuple[int, int]]:
    """
    Run test_system_health and return list of (chip_id1, chip_id2) pairs
    in N1-N2, N3-N4, N5-N6, N7-N8 order per tray. Use chip_id (not pcie_id)
    for TT_VISIBLE_DEVICES.
    """
    binary = test_binary or os.environ.get(ENV_TEST_BINARY, DEFAULT_TEST_BINARY)
    logger.info("get_device_pairs_from_discovery: binary=%s", binary)
    _flush_logging()
    output = run_test_system_health(binary)
    if not output:
        raise RuntimeError("Galaxy device discovery: no output from test_system_health")
    results = create_device_mapping(output)
    if not results:
        raise RuntimeError(
            "Galaxy device discovery: no chip lines parsed from test_system_health"
        )
    pairs = get_device_pairs(results)
    out = [(int(a), int(b)) for a, b in pairs]
    logger.info("GALAXY_DISCOVERY_PAIRS=%s", repr(out))
    return out


def get_device_groups_of_eight_from_discovery(
    test_binary: str | None = None,
) -> list[tuple[int, ...]]:
    """
    Run test_system_health and return list of 8-tuples (N1..N8) per tray
    in correct wiring order, one group per tray.
    """
    mapping = get_tray_mapping_from_discovery(test_binary)
    out: list[tuple[int, ...]] = []
    for tray in sorted(mapping.keys()):
        ids = mapping[tray]
        if len(ids) < 8:
            raise RuntimeError(
                f"Galaxy device discovery: tray {tray} has {len(ids)} devices, need 8"
            )
        out.append(tuple(ids[:8]))
    return out


def print_device_mapping(results: list[dict[str, str]]) -> None:
    """Print device mapping by tray (N1-N2, N3-N4, etc.)."""
    if not results:
        print("No results found")
        return
    tray_map: dict[str, dict[str, str]] = {}
    for result in results:
        tray = result["tray"]
        if tray not in tray_map:
            tray_map[tray] = {}
        tray_map[tray][result["n_loc"]] = result["pcie_id"]
    print("\nDevice Mapping by Tray:")
    print("=" * 80)
    for tray in sorted(tray_map.keys(), key=int):
        print(f"\nTray {tray}:")
        print("-" * 80)
        n_devices = tray_map[tray]
        for n1, n2 in N_PAIRS:
            dev1 = n_devices.get(n1, "N/A")
            dev2 = n_devices.get(n2, "N/A")
            print(f"  N{n1}-N{n2}: ({dev1},{dev2})")
    print("\n" + "=" * 80)
    print(f"Total entries: {len(results)}")
