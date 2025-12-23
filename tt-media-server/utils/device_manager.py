# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import re
import subprocess
from typing import Dict, List, Tuple

from utils.logger import TTLogger


class DeviceManager:
    def __init__(self):
        self.logger = TTLogger()

    def get_tray_mapping_from_system(self):
        """Execute tt-smi command and return tray mapping dictionary"""
        try:
            # Execute the system command
            result = subprocess.run(
                ["tt-smi", "-glx_list_tray_to_device"],
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout
            )

            if result.returncode != 0:
                self.logger.error(
                    f"tt-smi command failed with return code {result.returncode}"
                )
                self.logger.error(f"stderr: {result.stderr}")
                return {}

            # Parse the output using existing method
            tray_mapping = self.parse_tray_mapping(result.stdout)
            self.logger.info(f"Successfully parsed tray mapping: {tray_mapping}")
            return tray_mapping

        except subprocess.TimeoutExpired:
            self.logger.error("tt-smi command timed out after 30 seconds")
            return {}
        except FileNotFoundError:
            self.logger.error(
                "tt-smi command not found. Make sure it's installed and in PATH"
            )
            return {}
        except Exception as e:
            self.logger.error(f"Error executing tt-smi command: {e}")
            return {}

    @staticmethod
    def parse_tray_mapping(table_text):
        """Parse the tray mapping table and return a dictionary of tray -> device IDs"""

        lines = table_text.strip().split("\n")
        tray_mapping = {}

        # Find the data rows (skip header and separator lines)
        data_rows = []
        for line in lines:
            # Look for lines that start with │ and contain actual data (not headers)
            if line.strip().startswith("│") and any(char.isdigit() for char in line):
                data_rows.append(line)

        for row in data_rows:
            # Split by │ and clean up the parts
            parts = [part.strip() for part in row.split("│") if part.strip()]

            if len(parts) >= 3:
                try:
                    # First column is tray number
                    tray_number = int(parts[0])

                    # Third column contains the device IDs
                    device_ids_str = parts[2]

                    # Parse the device IDs (comma-separated)
                    device_ids = [int(id.strip()) for id in device_ids_str.split(",")]

                    tray_mapping[tray_number] = device_ids

                except (ValueError, IndexError) as e:
                    print(f"Error parsing row: {row}, Error: {e}")
                    continue

        return tray_mapping

    def create_device_pairs(self, tray_mapping):
        """Create device pairs from tray mapping. Each pair contains adjacent device IDs from the same tray"""
        device_pairs = []

        for tray_number, device_ids in tray_mapping.items():
            # Sort device IDs to ensure consistent pairing
            sorted_device_ids = sorted(device_ids)

            # Create pairs from adjacent devices in the same tray
            for i in range(0, len(sorted_device_ids), 2):
                if i + 1 < len(sorted_device_ids):
                    pair = (sorted_device_ids[i], sorted_device_ids[i + 1])
                    device_pairs.append(pair)
                else:
                    # Handle odd number of devices - log warning
                    self.logger.warning(
                        f"Tray {tray_number} has odd number of devices. Device {sorted_device_ids[i]} will not be paired."
                    )

        self.logger.info(f"Created {len(device_pairs)} device pairs: {device_pairs}")

        return device_pairs

    def get_device_pairs_from_system(self):
        """Convenience method to get tray mapping and create device pairs in one call"""
        tray_mapping = self.get_tray_mapping_from_system()
        if not tray_mapping:
            self.logger.error("Failed to get tray mapping, cannot create device pairs")
            return []

        return self.create_device_pairs(tray_mapping)

    def create_single_devices(self, tray_mapping):
        """Create single devices from tray mapping. Each device is returned as individual integer"""
        single_devices = []

        for tray_number, device_ids in tray_mapping.items():
            # Sort device IDs to ensure consistent ordering
            sorted_device_ids = sorted(device_ids)

            # Add each device individually
            single_devices.extend(sorted_device_ids)

        self.logger.info(
            f"Created {len(single_devices)} single devices: {single_devices}"
        )
        return single_devices

    def get_single_devices_from_system(self):
        """Convenience method to get tray mapping and create single device tuples in one call"""
        tray_mapping = self.get_tray_mapping_from_system()
        if not tray_mapping:
            self.logger.error(
                "Failed to get tray mapping, cannot create single device tuples"
            )
            return []

        return self.create_single_devices(tray_mapping)

    def create_device_groups_of_eight(self, tray_mapping):
        """Create device groups from tray mapping. Each group contains 8 device IDs from the same tray"""
        device_groups = []

        for tray_number, device_ids in tray_mapping.items():
            # Sort device IDs to ensure consistent grouping
            sorted_device_ids = sorted(device_ids)

            # Check if we have at least 8 devices in this tray
            if len(sorted_device_ids) < 8:
                error_msg = f"Tray {tray_number} has only {len(sorted_device_ids)} devices, but 8 are required for grouping"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            # Create groups of 8 devices from the same tray
            for i in range(0, len(sorted_device_ids), 8):
                if i + 7 < len(sorted_device_ids):
                    # Get 8 consecutive devices
                    group = tuple(sorted_device_ids[i : i + 8])
                    device_groups.append(group)
                else:
                    # Handle remaining devices (less than 8)
                    remaining = len(sorted_device_ids) - i
                    error_msg = f"Tray {tray_number} has {remaining} remaining devices that cannot form a group of 8"
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)

        self.logger.info(
            f"Created {len(device_groups)} device groups of 8 chips each: {device_groups}"
        )
        return device_groups

    def get_device_groups_of_eight_from_system(self):
        """Convenience method to get tray mapping and create device groups of 8 in one call"""
        tray_mapping = self.get_tray_mapping_from_system()
        if not tray_mapping:
            self.logger.error("Failed to get tray mapping, cannot create device groups")
            return []

        return self.create_device_groups_of_eight(tray_mapping)

    def run_test_system_health_pcie(
        self,
        test_binary: str = "/home/ubuntu/tt-metal/build/test/tt_metal/tt_fabric/test_system_health",
    ) -> str:
        """
        Run the test_system_health binary and capture output.

        Args:
            test_binary: Path to the test binary

        Returns:
            The output containing chip information
        """
        cmd = [test_binary, "--gtest_filter=Cluster.ReportSystemHealth"]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            return result.stdout
        except subprocess.TimeoutExpired:
            self.logger.debug("Error: Timeout while running test")
            return ""
        except Exception as e:
            self.logger.debug(f"Error running test: {e}")
            return ""

    def parse_chip_line(self, line: str) -> Dict[str, str]:
        """
        Parse a chip info line to extract relevant information.

        Args:
            line: A line containing chip information
            Example: "Chip: 31 PCIe: 15 Unique ID: 835323530303144 Tray: 2 N8"

        Returns:
            Dictionary with parsed information or empty dict if parsing fails
        """
        # Pattern to match the chip line
        pattern = r"Chip:\s+(\d+)\s+PCIe:\s+(\d+)\s+Unique ID:\s+(\w+)\s+Tray:\s+(\d+)\s+N(\d+)"
        match = re.search(pattern, line)

        if match:
            return {
                "chip": match.group(1),
                "pcie_id": match.group(2),
                "unique_id": match.group(3),
                "tray": match.group(4),
                "n_loc": match.group(5),
            }
        return {}

    def create_device_mapping_pcie(self, output: str) -> List[Dict[str, str]]:
        """
        Create device mapping from test output.

        Args:
            output: The output from test_system_health

        Returns:
            List of dictionaries containing device mapping information
        """
        # Filter lines containing "PCIe:"
        chip_lines = [line for line in output.split("\n") if "PCIe:" in line]

        # Create list of device info
        results = []
        for line in chip_lines:
            info = self.parse_chip_line(line)
            if info:
                results.append(
                    {
                        "pcie_id": info["pcie_id"],
                        "unique_id": info["unique_id"],
                        "tray": info["tray"],
                        "n_loc": info["n_loc"],
                    }
                )

        return results

    def get_device_pairs_pcie(
        self, results: List[Dict[str, str]]
    ) -> List[Tuple[str, str]]:
        """Extract device ID pairs from results.

        Args:
            results: List of device mapping information

        Returns:
            List of (pcie_id1, pcie_id2) tuples
        """
        # Group results by tray
        tray_map = {}
        for result in results:
            tray = result["tray"]
            if tray not in tray_map:
                tray_map[tray] = {}
            n_loc = result["n_loc"]
            tray_map[tray][n_loc] = result["pcie_id"]

        # Extract pairs
        pairs_list = []
        for tray in sorted(tray_map.keys(), key=int):
            n_devices = tray_map[tray]

            # Get pairs: N1-N2, N3-N4, N5-N6, N7-N8
            pairs = [("1", "2"), ("3", "4"), ("5", "6"), ("7", "8")]
            for n1, n2 in pairs:
                dev1 = n_devices.get(n1)
                dev2 = n_devices.get(n2)
                if dev1 and dev2:
                    pairs_list.append((int(dev1), int(dev2)))

        return pairs_list

    def print_device_mapping(self, results: List[Dict[str, str]]):
        """
        self.logger.debug device mapping by tray in the requested format.

        Args:
            results: List of device mapping information
        """
        if not results:
            self.logger.debug("No results found")
            return

        # Group results by tray
        tray_map = {}
        for result in results:
            tray = result["tray"]
            if tray not in tray_map:
                tray_map[tray] = {}
            n_loc = result["n_loc"]
            tray_map[tray][n_loc] = result["pcie_id"]

        self.logger.debug("\nDevice Mapping by Tray:")
        self.logger.debug("=" * 80)

        for tray in sorted(tray_map.keys(), key=int):
            self.logger.debug(f"\nTray {tray}:")
            self.logger.debug("-" * 80)

            n_devices = tray_map[tray]

            pairs = [("1", "2"), ("3", "4"), ("5", "6"), ("7", "8")]
            for n1, n2 in pairs:
                dev1 = n_devices.get(n1, "N/A")
                dev2 = n_devices.get(n2, "N/A")
                self.logger.debug(f"  N{n1}-N{n2}: ({dev1},{dev2})")

        self.logger.debug("\n" + "=" * 80)
        self.logger.debug(f"Total entries: {len(results)}")
