# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import json
import sys

def get_all_valid_devices(json_file):
    with open(json_file) as f:
        data = json.load(f)

    results = []
    pci_id = 0

    for device in data["device_info"]:
        board_type = device["board_info"]["board_type"]
        bus_id = device["board_info"]["bus_id"]

        if bus_id != "N/A":
            results.append({
                "pci_dev_id": pci_id,
                "board_type": board_type,
                "bus_id": bus_id
            })
            pci_id += 1

    return results

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python get_pci_id.py <out.json> <board_type>")
        sys.exit(1)

    json_path = sys.argv[1]
    board_type_input = sys.argv[2].lower()

    try:
        valid_devices = get_all_valid_devices(json_path)
    except Exception:
        print("not-specified")
        sys.exit(0)

    if not valid_devices:
        print("not-specified")
        sys.exit(0)

    unique_types = set(d["board_type"].lower() for d in valid_devices)

    if len(unique_types) == 1 and board_type_input in next(iter(unique_types)):
        print("not-specified")
        sys.exit(0)

    match = next((d for d in valid_devices if board_type_input in d["board_type"].lower()), None)

    if match:
        print(f"{match['pci_dev_id']}")
    else:
        print("not-specified")
