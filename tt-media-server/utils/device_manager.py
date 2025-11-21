# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import subprocess

class DeviceManager:
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
                return {}
            
            # Parse the output using existing method
            tray_mapping = self.parse_tray_mapping(result.stdout)
            return tray_mapping
            
        except subprocess.TimeoutExpired:
            return {}
        except FileNotFoundError:
            return {}
        except Exception as e:
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

        return device_pairs
    
    def get_device_pairs_from_system(self):
        """Convenience method to get tray mapping and create device pairs in one call"""
        tray_mapping = self.get_tray_mapping_from_system()
        if not tray_mapping:
            return []
        
        return self.create_device_pairs(tray_mapping)