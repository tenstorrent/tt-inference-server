# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import subprocess
from utils.logger import TTLogger
import os
import csv
from pathlib import Path

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
                timeout=30  # 30 second timeout
            )
            
            if result.returncode != 0:
                self.logger.error(f"tt-smi command failed with return code {result.returncode}")
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
            self.logger.error("tt-smi command not found. Make sure it's installed and in PATH")
            return {}
        except Exception as e:
            self.logger.error(f"Error executing tt-smi command: {e}")
            return {}
    
    @staticmethod
    def parse_tray_mapping(table_text):
        """Parse the tray mapping table and return a dictionary of tray -> device IDs"""
        
        lines = table_text.strip().split('\n')
        tray_mapping = {}
        
        # Find the data rows (skip header and separator lines)
        data_rows = []
        for line in lines:
            # Look for lines that start with │ and contain actual data (not headers)
            if line.strip().startswith('│') and any(char.isdigit() for char in line):
                data_rows.append(line)
        
        for row in data_rows:
            # Split by │ and clean up the parts
            parts = [part.strip() for part in row.split('│') if part.strip()]
            
            if len(parts) >= 3:
                try:
                    # First column is tray number
                    tray_number = int(parts[0])
                    
                    # Third column contains the device IDs
                    device_ids_str = parts[2]
                    
                    # Parse the device IDs (comma-separated)
                    device_ids = [int(id.strip()) for id in device_ids_str.split(',')]
                    
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
                    self.logger.warning(f"Tray {tray_number} has odd number of devices. Device {sorted_device_ids[i]} will not be paired.")

        self.logger.info(f"Created {len(device_pairs)} device pairs: {device_pairs}")

        return device_pairs
    
    def get_device_pairs_from_system(self):
        """Convenience method to get device pairs from pairs.csv if it exists, otherwise from system"""

        # Check if TT_METAL_HOME is set and pairs.csv exists
        tt_metal_home = os.environ.get('TT_METAL_HOME')
        if tt_metal_home:
            pairs_csv_path = Path(tt_metal_home) / 'pairs.csv'
            if pairs_csv_path.exists():
                try:
                    self.logger.info(f"Reading device pairs from {pairs_csv_path}")
                    device_pairs = []
                    with open(pairs_csv_path, 'r') as csvfile:
                        reader = csv.DictReader(csvfile)
                        for row in reader:
                            device_id1 = int(row['device_id1'])
                            device_id2 = int(row['device_id2'])
                            device_pairs.append((device_id1, device_id2))
                    
                    self.logger.info(f"Loaded {len(device_pairs)} device pairs from CSV: {device_pairs}")
                    return device_pairs
                except Exception as e:
                    self.logger.error(f"Error reading pairs.csv: {e}, falling back to system detection")
                    raise AssertionError(f"pairs.csv exists at {pairs_csv_path} but failed to load")

    
    def create_single_devices(self, tray_mapping):
        """Create single devices from tray mapping. Each device is returned as individual integer"""
        single_devices = []
        
        for tray_number, device_ids in tray_mapping.items():
            # Sort device IDs to ensure consistent ordering
            sorted_device_ids = sorted(device_ids)
            
            # Add each device individually
            single_devices.extend(sorted_device_ids)
        
        self.logger.info(f"Created {len(single_devices)} single devices: {single_devices}")
        return single_devices
    
    def get_single_devices_from_system(self):
        """Convenience method to get tray mapping and create single device tuples in one call"""
        tray_mapping = self.get_tray_mapping_from_system()
        if not tray_mapping:
            self.logger.error("Failed to get tray mapping, cannot create single device tuples")
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
                    group = tuple(sorted_device_ids[i:i+8])
                    device_groups.append(group)
                else:
                    # Handle remaining devices (less than 8)
                    remaining = len(sorted_device_ids) - i
                    error_msg = f"Tray {tray_number} has {remaining} remaining devices that cannot form a group of 8"
                    self.logger.error(error_msg)
                    raise ValueError(error_msg)

        self.logger.info(f"Created {len(device_groups)} device groups of 8 chips each: {device_groups}")
        return device_groups
    
    def get_device_groups_of_eight_from_system(self):
        """Convenience method to get tray mapping and create device groups of 8 in one call"""
        tray_mapping = self.get_tray_mapping_from_system()
        if not tray_mapping:
            self.logger.error("Failed to get tray mapping, cannot create device groups")
            return []
        
        return self.create_device_groups_of_eight(tray_mapping)