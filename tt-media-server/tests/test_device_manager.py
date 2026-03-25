# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import logging
import subprocess
import unittest
from unittest.mock import MagicMock, patch

from utils.device_manager import ChipInfo, DeviceDiscoveryError, DeviceManager


class TestDeviceManager(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures"""
        self.device_manager = DeviceManager()

        # Sample tray mapping data for testing
        self.sample_tray_mapping = {
            1: [0, 1, 2, 3, 4, 5, 6, 7],
            2: [8, 9, 10, 11, 12, 13, 14, 15],
            3: [16, 17, 18, 19, 20, 21, 22, 23],
            4: [24, 25, 26, 27, 28, 29, 30, 31],
        }

        # Sample tt-smi output
        self.sample_tt_smi_output = """
Mapping of trays to devices on the galaxy:
┏━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Tray Number ┃ Tray Bus ID ┃ PCI Dev ID              ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 1           │ 0xc0        │ 0,1,2,3,4,5,6,7         │
│ 2           │ 0x80        │ 8,9,10,11,12,13,14,15   │
│ 3           │ 0x00        │ 16,17,18,19,20,21,22,23 │
│ 4           │ 0x40        │ 24,25,26,27,28,29,30,31 │
└─────────────┴─────────────┴─────────────────────────┘
"""

    def test_parse_tray_mapping_valid_input(self):
        """Test parsing valid tray mapping table"""
        result = DeviceManager._parse_tt_smi_output(self.sample_tt_smi_output)

        expected = {
            1: [0, 1, 2, 3, 4, 5, 6, 7],
            2: [8, 9, 10, 11, 12, 13, 14, 15],
            3: [16, 17, 18, 19, 20, 21, 22, 23],
            4: [24, 25, 26, 27, 28, 29, 30, 31],
        }

        self.assertEqual(result, expected)

    def test_parse_tray_mapping_empty_input(self):
        """Test parsing empty input"""
        result = DeviceManager._parse_tt_smi_output("")
        self.assertEqual(result, {})

    def test_parse_tray_mapping_malformed_input(self):
        """Test parsing malformed input"""
        malformed_input = """
        Some random text
        Not a table format
        """
        result = DeviceManager._parse_tt_smi_output(malformed_input)
        self.assertEqual(result, {})

    def test_build_pairs_from_chips_n1_n2_n3_n4(self):
        """Test N1-N2, N3-N4 pairs from test_system_health chips"""
        chips = [
            ChipInfo("0", "0", "a", "1", "1"),
            ChipInfo("1", "0", "b", "1", "2"),
            ChipInfo("2", "0", "c", "1", "3"),
            ChipInfo("3", "0", "d", "1", "4"),
        ]
        result = DeviceManager._build_pairs_from_chips(chips)
        self.assertEqual(result, [(0, 1), (2, 3)])

    def test_build_pairs_from_chips_empty(self):
        """Test _build_pairs_from_chips with no chips"""
        result = DeviceManager._build_pairs_from_chips([])
        self.assertEqual(result, [])

    @patch("subprocess.run")
    def test_get_tray_mapping_from_system_success(self, mock_subprocess):
        """Test successful execution of tt-smi command"""
        # Mock successful subprocess call
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = self.sample_tt_smi_output
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        result = self.device_manager.get_tray_mapping_from_system()

        expected = {
            1: [0, 1, 2, 3, 4, 5, 6, 7],
            2: [8, 9, 10, 11, 12, 13, 14, 15],
            3: [16, 17, 18, 19, 20, 21, 22, 23],
            4: [24, 25, 26, 27, 28, 29, 30, 31],
        }

        self.assertEqual(result, expected)

        # Verify subprocess was called with correct arguments
        mock_subprocess.assert_called_once_with(
            ["tt-smi", "-glx_list_tray_to_device"],
            capture_output=True,
            text=True,
            timeout=30,
        )

    @patch("subprocess.run")
    def test_get_tray_mapping_from_system_command_failure(self, mock_subprocess):
        """Test tt-smi command failure"""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Command failed"
        mock_subprocess.return_value = mock_result

        with patch("utils.device_manager.logger") as mock_logger:
            result = self.device_manager.get_tray_mapping_from_system()
            self.assertEqual(result, {})
            mock_logger.error.assert_called()

    @patch("subprocess.run")
    def test_get_tray_mapping_from_system_timeout(self, mock_subprocess):
        """Test tt-smi command timeout"""
        mock_subprocess.side_effect = subprocess.TimeoutExpired("tt-smi", 30)

        with patch("utils.device_manager.logger") as mock_logger:
            result = self.device_manager.get_tray_mapping_from_system()
            self.assertEqual(result, {})
            mock_logger.error.assert_called()
            self.assertIn("timed out", str(mock_logger.error.call_args))

    @patch("subprocess.run")
    def test_get_tray_mapping_from_system_file_not_found(self, mock_subprocess):
        """Test tt-smi command not found"""
        mock_subprocess.side_effect = FileNotFoundError("tt-smi not found")

        with patch("utils.device_manager.logger") as mock_logger:
            result = self.device_manager.get_tray_mapping_from_system()
            self.assertEqual(result, {})
            mock_logger.error.assert_called()
            self.assertIn("not found", str(mock_logger.error.call_args))

    @patch("subprocess.run")
    def test_get_tray_mapping_from_system_generic_exception(self, mock_subprocess):
        """Test generic exception during tt-smi command execution"""
        mock_subprocess.side_effect = Exception("Generic error")

        with patch("utils.device_manager.logger") as mock_logger:
            result = self.device_manager.get_tray_mapping_from_system()
            self.assertEqual(result, {})
            mock_logger.error.assert_called()

    @patch.object(DeviceManager, "_run_system_health")
    def test_get_device_pairs_success(self, mock_run_system_health):
        """Test get_device_pairs when test_system_health returns chips"""
        chips = [
            ChipInfo("0", "0", "a", "1", "1"),
            ChipInfo("1", "0", "b", "1", "2"),
            ChipInfo("2", "0", "c", "1", "3"),
            ChipInfo("3", "0", "d", "1", "4"),
        ]
        mock_run_system_health.return_value = chips
        result = self.device_manager.get_device_pairs()
        self.assertEqual(result, [(0, 1), (2, 3)])

    @patch.object(DeviceManager, "_run_system_health")
    def test_get_device_pairs_failure(self, mock_run_system_health):
        """Test get_device_pairs when discovery raises"""
        mock_run_system_health.side_effect = DeviceDiscoveryError(
            "test_system_health failed"
        )

        with patch("utils.device_manager.logger") as mock_logger:
            result = self.device_manager.get_device_pairs()
            self.assertEqual(result, [])
            mock_logger.error.assert_called()

    def test_create_device_groups_of_eight_valid_input(self):
        """Test creating device groups of 8 from valid tray mapping"""
        result = self.device_manager._create_device_groups_of_eight(
            self.sample_tray_mapping
        )

        expected = [
            (0, 1, 2, 3, 4, 5, 6, 7),  # Tray 1
            (8, 9, 10, 11, 12, 13, 14, 15),  # Tray 2
            (16, 17, 18, 19, 20, 21, 22, 23),  # Tray 3
            (24, 25, 26, 27, 28, 29, 30, 31),  # Tray 4
        ]

        self.assertEqual(result, expected)

    def test_create_device_groups_of_eight_insufficient_devices(self):
        """Test creating device groups when tray has less than 8 devices"""
        insufficient_tray_mapping = {
            1: [0, 1, 2, 3, 4, 5, 6]  # Only 7 devices
        }

        with self.assertRaises(DeviceDiscoveryError) as context:
            self.device_manager._create_device_groups_of_eight(
                insufficient_tray_mapping
            )

        self.assertIn("need 8 devices", str(context.exception))
        self.assertIn("Tray 1", str(context.exception))

    def test_create_device_groups_of_eight_non_divisible_devices(self):
        """Test creating device groups when tray has more than 8 - takes first 8"""
        non_divisible_tray_mapping = {
            1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 10 devices
        }

        result = self.device_manager._create_device_groups_of_eight(
            non_divisible_tray_mapping
        )
        self.assertEqual(result, [(0, 1, 2, 3, 4, 5, 6, 7)])

    def test_create_device_groups_of_eight_multiple_groups_per_tray(self):
        """Test creating device groups when tray has exactly 16 devices (takes first 8)"""
        double_tray_mapping = {
            1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # 16 devices
        }

        result = self.device_manager._create_device_groups_of_eight(double_tray_mapping)
        expected = [(0, 1, 2, 3, 4, 5, 6, 7)]
        self.assertEqual(result, expected)

    def test_create_device_groups_of_eight_unsorted_input(self):
        """Test creating device groups with unsorted device IDs"""
        unsorted_tray_mapping = {
            1: [7, 1, 5, 3, 0, 6, 2, 4]  # Unsorted 8 devices
        }

        result = self.device_manager._create_device_groups_of_eight(
            unsorted_tray_mapping
        )
        expected = [(0, 1, 2, 3, 4, 5, 6, 7)]
        self.assertEqual(result, expected)

    def test_create_device_groups_of_eight_empty_input(self):
        """Test creating device groups from empty tray mapping"""
        result = self.device_manager._create_device_groups_of_eight({})
        self.assertEqual(result, [])

    def test_create_device_groups_of_eight_exactly_eight_devices(self):
        """Test creating device groups with exactly 8 devices in tray"""
        exact_tray_mapping = {
            1: [0, 1, 2, 3, 4, 5, 6, 7]  # Exactly 8 devices
        }

        result = self.device_manager._create_device_groups_of_eight(exact_tray_mapping)
        expected = [(0, 1, 2, 3, 4, 5, 6, 7)]
        self.assertEqual(result, expected)

    @patch.object(DeviceManager, "_run_tt_smi")
    def test_get_device_groups_of_eight_success(self, mock_run_tt_smi):
        """Test get_device_groups_of_eight when _run_tt_smi succeeds."""
        mock_run_tt_smi.return_value = self.sample_tray_mapping

        result = self.device_manager.get_device_groups_of_eight()
        expected = [
            (0, 1, 2, 3, 4, 5, 6, 7),
            (8, 9, 10, 11, 12, 13, 14, 15),
            (16, 17, 18, 19, 20, 21, 22, 23),
            (24, 25, 26, 27, 28, 29, 30, 31),
        ]
        self.assertEqual(result, expected)

    @patch.object(DeviceManager, "_run_tt_smi")
    def test_get_device_groups_of_eight_failure(self, mock_run_tt_smi):
        """Test get_device_groups_of_eight when _run_tt_smi fails."""
        mock_run_tt_smi.side_effect = DeviceDiscoveryError("no trays")

        with patch("utils.device_manager.logger") as mock_logger:
            result = self.device_manager.get_device_groups_of_eight()
            self.assertEqual(result, [])
            mock_logger.error.assert_called()

    @patch.object(DeviceManager, "_run_tt_smi")
    def test_get_device_groups_of_eight_insufficient_devices(self, mock_run_tt_smi):
        """Test get_device_groups_of_eight when tray has < 8 devices."""
        mock_run_tt_smi.return_value = {1: [0, 1, 2, 3, 4, 5, 6]}

        with patch("utils.device_manager.logger") as mock_logger:
            result = self.device_manager.get_device_groups_of_eight()
            self.assertEqual(result, [])
            mock_logger.error.assert_called()

    def test_create_single_devices_valid_input(self):
        """Test creating single device list from valid tray mapping"""
        result = self.device_manager._create_single_devices(self.sample_tray_mapping)

        expected = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,  # Tray 1
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,  # Tray 2
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,  # Tray 3
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,  # Tray 4
        ]

        self.assertEqual(result, expected)

    def test_create_single_devices_empty_input(self):
        """Test creating single device list from empty tray mapping"""
        result = self.device_manager._create_single_devices({})
        self.assertEqual(result, [])

    def test_create_single_devices_single_device(self):
        """Test creating single device list with single device in tray"""
        single_device_mapping = {1: [5]}

        result = self.device_manager._create_single_devices(single_device_mapping)
        self.assertEqual(result, [5])

    def test_create_single_devices_unsorted_input(self):
        """Test creating single device list with unsorted device IDs"""
        unsorted_tray_mapping = {
            1: [7, 1, 5, 3, 0, 6, 2, 4]  # Unsorted device IDs
        }

        result = self.device_manager._create_single_devices(unsorted_tray_mapping)
        expected = [0, 1, 2, 3, 4, 5, 6, 7]
        self.assertEqual(result, expected)

    def test_create_single_devices_multiple_trays(self):
        """Test creating single device list from multiple trays"""
        multi_tray_mapping = {
            1: [0, 1, 2],
            2: [3, 4, 5],
            3: [6, 7],
        }

        result = self.device_manager._create_single_devices(multi_tray_mapping)
        expected = [0, 1, 2, 3, 4, 5, 6, 7]
        self.assertEqual(result, expected)

    @patch.object(DeviceManager, "_run_tt_smi")
    def test_get_single_devices_success(self, mock_run_tt_smi):
        """Test get_single_devices when _run_tt_smi succeeds."""
        mock_run_tt_smi.return_value = self.sample_tray_mapping

        result = self.device_manager.get_single_devices()
        expected = list(range(32))
        self.assertEqual(result, expected)

    @patch.object(DeviceManager, "_run_tt_smi")
    def test_get_single_devices_failure(self, mock_run_tt_smi):
        """Test get_single_devices when _run_tt_smi fails."""
        mock_run_tt_smi.side_effect = DeviceDiscoveryError("tt-smi failed")

        with patch("utils.device_manager.logger") as mock_logger:
            result = self.device_manager.get_single_devices()
            self.assertEqual(result, [])
            mock_logger.warning.assert_called()


if __name__ == "__main__":
    # Configure logging for tests
    logging.basicConfig(level=logging.DEBUG)

    unittest.main()
