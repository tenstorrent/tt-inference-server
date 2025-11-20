# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import unittest
from unittest.mock import patch, MagicMock
import subprocess
import logging
from utils.device_manager import DeviceManager


class TestDeviceManager(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.device_manager = DeviceManager()
        
        # Sample tray mapping data for testing
        self.sample_tray_mapping = {
            1: [0, 1, 2, 3, 4, 5, 6, 7],
            2: [8, 9, 10, 11, 12, 13, 14, 15],
            3: [16, 17, 18, 19, 20, 21, 22, 23],
            4: [24, 25, 26, 27, 28, 29, 30, 31]
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
        result = DeviceManager.parse_tray_mapping(self.sample_tt_smi_output)
        
        expected = {
            1: [0, 1, 2, 3, 4, 5, 6, 7],
            2: [8, 9, 10, 11, 12, 13, 14, 15],
            3: [16, 17, 18, 19, 20, 21, 22, 23],
            4: [24, 25, 26, 27, 28, 29, 30, 31]
        }
        
        self.assertEqual(result, expected)

    def test_parse_tray_mapping_empty_input(self):
        """Test parsing empty input"""
        result = DeviceManager.parse_tray_mapping("")
        self.assertEqual(result, {})

    def test_parse_tray_mapping_malformed_input(self):
        """Test parsing malformed input"""
        malformed_input = """
        Some random text
        Not a table format
        """
        result = DeviceManager.parse_tray_mapping(malformed_input)
        self.assertEqual(result, {})

    def test_create_device_pairs_valid_input(self):
        """Test creating device pairs from valid tray mapping"""
        result = self.device_manager.create_device_pairs(self.sample_tray_mapping)
        
        expected = [
            (0, 1), (2, 3), (4, 5), (6, 7),           # Tray 1
            (8, 9), (10, 11), (12, 13), (14, 15),     # Tray 2  
            (16, 17), (18, 19), (20, 21), (22, 23),   # Tray 3
            (24, 25), (26, 27), (28, 29), (30, 31)    # Tray 4
        ]
        
        self.assertEqual(result, expected)

    def test_create_device_pairs_odd_number_devices(self):
        """Test creating device pairs with odd number of devices in a tray"""
        odd_tray_mapping = {
            1: [0, 1, 2, 3, 4]  # 5 devices - odd number
        }
        
        with patch.object(self.device_manager.logger, 'warning') as mock_warning:
            result = self.device_manager.create_device_pairs(odd_tray_mapping)
            
            expected = [(0, 1), (2, 3)]  # Device 4 should be unpaired
            self.assertEqual(result, expected)
            
            # Check that warning was logged
            mock_warning.assert_called_once()
            self.assertIn("odd number of devices", mock_warning.call_args[0][0])

    def test_create_device_pairs_empty_input(self):
        """Test creating device pairs from empty tray mapping"""
        result = self.device_manager.create_device_pairs({})
        self.assertEqual(result, [])

    def test_create_device_pairs_single_device_tray(self):
        """Test creating device pairs with single device in tray"""
        single_device_mapping = {
            1: [0]
        }
        
        with patch.object(self.device_manager.logger, 'warning') as mock_warning:
            result = self.device_manager.create_device_pairs(single_device_mapping)
            
            self.assertEqual(result, [])
            mock_warning.assert_called_once()

    @patch('subprocess.run')
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
            4: [24, 25, 26, 27, 28, 29, 30, 31]
        }
        
        self.assertEqual(result, expected)
        
        # Verify subprocess was called with correct arguments
        mock_subprocess.assert_called_once_with(
            ["tt-smi", "-glx_list_tray_to_device"],
            capture_output=True,
            text=True,
            timeout=30
        )

    @patch('subprocess.run')
    def test_get_tray_mapping_from_system_command_failure(self, mock_subprocess):
        """Test tt-smi command failure"""
        # Mock failed subprocess call
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Command failed"
        mock_subprocess.return_value = mock_result
        
        with patch.object(self.device_manager.logger, 'error') as mock_error:
            result = self.device_manager.get_tray_mapping_from_system()
            
            self.assertEqual(result, {})
            mock_error.assert_called()

    @patch('subprocess.run')
    def test_get_tray_mapping_from_system_timeout(self, mock_subprocess):
        """Test tt-smi command timeout"""
        # Mock timeout exception
        mock_subprocess.side_effect = subprocess.TimeoutExpired("tt-smi", 30)
        
        with patch.object(self.device_manager.logger, 'error') as mock_error:
            result = self.device_manager.get_tray_mapping_from_system()
            
            self.assertEqual(result, {})
            mock_error.assert_called()
            self.assertIn("timed out", mock_error.call_args[0][0])

    @patch('subprocess.run')
    def test_get_tray_mapping_from_system_file_not_found(self, mock_subprocess):
        """Test tt-smi command not found"""
        # Mock FileNotFoundError
        mock_subprocess.side_effect = FileNotFoundError("tt-smi not found")
        
        with patch.object(self.device_manager.logger, 'error') as mock_error:
            result = self.device_manager.get_tray_mapping_from_system()
            
            self.assertEqual(result, {})
            mock_error.assert_called()
            self.assertIn("not found", mock_error.call_args[0][0])

    @patch('subprocess.run')
    def test_get_tray_mapping_from_system_generic_exception(self, mock_subprocess):
        """Test generic exception during tt-smi command execution"""
        # Mock generic exception
        mock_subprocess.side_effect = Exception("Generic error")
        
        with patch.object(self.device_manager.logger, 'error') as mock_error:
            result = self.device_manager.get_tray_mapping_from_system()
            
            self.assertEqual(result, {})
            mock_error.assert_called()

    @patch.object(DeviceManager, 'get_tray_mapping_from_system')
    def test_get_device_pairs_from_system_success(self, mock_get_tray_mapping):
        """Test getting device pairs from system successfully"""
        # Mock successful tray mapping retrieval
        mock_get_tray_mapping.return_value = self.sample_tray_mapping
        
        result = self.device_manager.get_device_pairs_from_system()
        
        expected = [
            (0, 1), (2, 3), (4, 5), (6, 7),
            (8, 9), (10, 11), (12, 13), (14, 15),
            (16, 17), (18, 19), (20, 21), (22, 23),
            (24, 25), (26, 27), (28, 29), (30, 31)
        ]
        
        self.assertEqual(result, expected)

    @patch.object(DeviceManager, 'get_tray_mapping_from_system')
    def test_get_device_pairs_from_system_failure(self, mock_get_tray_mapping):
        """Test getting device pairs from system when tray mapping fails"""
        # Mock failed tray mapping retrieval
        mock_get_tray_mapping.return_value = {}
        
        with patch.object(self.device_manager.logger, 'error') as mock_error:
            result = self.device_manager.get_device_pairs_from_system()
            
            self.assertEqual(result, [])
            mock_error.assert_called()
            self.assertIn("Failed to get tray mapping", mock_error.call_args[0][0])

    def test_create_device_pairs_unsorted_input(self):
        """Test creating device pairs with unsorted device IDs"""
        unsorted_tray_mapping = {
            1: [7, 1, 5, 3, 0, 6, 2, 4]  # Unsorted device IDs
        }
        
        result = self.device_manager.create_device_pairs(unsorted_tray_mapping)
        
        # Should be sorted and paired correctly
        expected = [(0, 1), (2, 3), (4, 5), (6, 7)]
        self.assertEqual(result, expected)


if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(level=logging.DEBUG)
    
    unittest.main()