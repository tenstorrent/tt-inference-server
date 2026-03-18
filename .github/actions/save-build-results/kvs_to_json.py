#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import sys
import json


def kvs_to_dict(input_kvs: str) -> dict:
    """Convert key=value pairs to dictionary"""
    result = {}
    try:
        for line in input_kvs.splitlines():
            line = line.strip()

            if not line or line.startswith('#'):
                continue

            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                result[key] = value if value else "unknown"
        return result
    except Exception as e:
        print(f"Error processing input: {e}")
        sys.exit(1)

def save_to_json(input_kvs: str, output_json: str):
    """Save dictionary to JSON file"""
    data_dict = kvs_to_dict(input_kvs)
    try:
        with open(output_json, 'w') as json_f:
            json.dump(data_dict, json_f, indent=2)
    except Exception as e:
        print(f"Error writing JSON file: {e}")
        sys.exit(1)

def validate_json(output_json: str):
    """Validate the generated JSON file"""
    try:
        with open(output_json, 'r') as json_f:
            data = json.load(json_f)
        # check if json is empty, i.e. contains only {}
        if data == {}:
            raise ValueError("JSON content is empty!")
    except (ValueError, Exception) as e:
        print(f"Error validating JSON file: {e}")
        sys.exit(1)

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 kvs_to_json.py <input_kvs> <output_json>")
        sys.exit(1)

    input_kvs = sys.argv[1]
    output_json = sys.argv[2]
    save_to_json(input_kvs, output_json)
    validate_json(output_json)


if __name__ == "__main__":
    main()
